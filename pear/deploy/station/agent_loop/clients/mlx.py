from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pear.deploy.station.agent_loop.clients.client import ContextLimit, ModelClient, TurnProgress
from pear.deploy.station.agent_loop.clients.metrics import Performance, RequestStatus
from pear.deploy.station.agent_loop.clients.schemas import (
    CameraImage,
    EventHandler,
    HealthStatus,
    ModelEvent,
    ModelHealth,
    ModelInput,
    ModelOutput,
    ReasoningKind,
    ReferenceImage,
    Usage,
)
from pear.deploy.station.exceptions import OperatorPaused

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionChunk


@dataclass(kw_only=True)
class MLXProgress(TurnProgress):
    """One MLX-VLM request's streamed answer, reasoning and termination status."""

    answer: str = ""
    reasoning: str = ""
    finish_reason: str | None = None


class MLXClient(ModelClient):
    """MLX-VLM v0.7.12 Chat Completions, with complete image history owned by the station."""

    LABEL = "MLX-VLM"
    PROVIDER = "mlx"
    CHECK_TIMEOUT_S = 3
    SDK_RETRIES = 0
    UNUSED_API_KEY = "unused"
    RELATIVE_PATH_ARGUMENTS = ("api_key_file",)
    IMAGE_URL_PREFIX = "data:image/jpeg;base64,"
    MAX_OUTPUT_TOKENS = 4096
    CONTEXT_LIMIT = 16384
    MAX_IMAGES = 128

    def __init__(
        self,
        system_prompt: str,
        *,
        timeout_s: float,
        base_url: str,
        model: str,
        api_key_file: str | None = None,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        context_limit: int = CONTEXT_LIMIT,
        max_images: int = MAX_IMAGES,
        temperature: float = 1,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1,
        enable_thinking: bool = True,
    ) -> None:
        if type(model) is not str or not model.strip():
            raise ValueError("model must be a non-empty name listed by the MLX-VLM server")
        for name, count in dict(
            max_output_tokens=max_output_tokens, context_limit=context_limit, max_images=max_images
        ).items():
            if type(count) is not int or count <= 0:
                raise ValueError(f"{name} must be a whole number above 0")
        if max_output_tokens >= context_limit:
            raise ValueError("max_output_tokens must be below context_limit to leave room for the prompt")
        if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be a number from 0 to 2")
        if type(top_p) not in (int, float) or not 0 < top_p <= 1:
            raise ValueError("top_p must be a number above 0 and at most 1")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a whole number at least 0; 0 disables top-k sampling")
        if type(repetition_penalty) not in (int, float) or not 0 < repetition_penalty < math.inf:
            raise ValueError("repetition_penalty must be a finite number above 0")
        if type(enable_thinking) is not bool:
            raise ValueError("enable_thinking must be a boolean")

        super().__init__(system_prompt, timeout_s=timeout_s)
        self.base_url = self._service_url(base_url)
        self.model = model
        self.api_key_file = api_key_file
        self.max_output_tokens = max_output_tokens
        self.context_limit = context_limit
        self.max_images = max_images
        self.sampling = dict(temperature=temperature, top_p=top_p)
        self.extra_body = dict(
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            enable_thinking=enable_thinking,
            # MLX-VLM otherwise summarizes history automatically near its context limit.
            context_management=[],
        )
        self.history: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def check(self) -> ModelHealth:
        """Checks that the endpoint lists the configured model without running inference."""
        from openai import OpenAI

        with OpenAI(
            base_url=self.base_url, api_key=self._api_key(), max_retries=self.SDK_RETRIES, timeout=self.CHECK_TIMEOUT_S
        ) as client:
            if self.model not in {model.id for model in client.models.list().data}:
                raise ValueError(f"MLX-VLM model is not served: {self.model}")

        return {"status": HealthStatus.OK.value, "message": "MLX-VLM model endpoint connected"}

    async def _infer_async(
        self, model_input: ModelInput, on_event: EventHandler, cancellation: threading.Event
    ) -> ModelOutput:
        from openai import AsyncOpenAI, BadRequestError

        if any(isinstance(message["content"], ModelInput) for message in self.history):
            model_input = replace(model_input, reference_images=())

        try:
            async with AsyncOpenAI(
                base_url=self.base_url, api_key=self._api_key(), max_retries=self.SDK_RETRIES, timeout=self.timeout_s
            ) as client:
                output = await self._race(
                    self._stream(client, model_input, on_event, cancellation), cancellation, timeout_s=self.timeout_s
                )
        except BadRequestError as error:
            # The pinned server's explicit context-budget refusal, not an arbitrary generation failure.
            if "Request needs " in str(error) and "but MAX_KV_SIZE is " in str(error):
                raise ContextLimit(
                    "MLX-VLM context is full. History is preserved; clear the task to restart."
                ) from error
            raise

        # Only a complete, normally terminated reply becomes part of the next observation's history.
        self.history.extend(
            [
                {"role": "user", "content": model_input},
                {"role": "assistant", "content": output.answer, "reasoning_content": output.reasoning},
            ]
        )
        return output

    async def _stream(
        self,
        client: AsyncOpenAI,
        model_input: ModelInput,
        on_event: EventHandler,
        cancellation: threading.Event,
    ) -> ModelOutput:
        OperatorPaused.check_cancelled(cancellation)
        OperatorPaused.check_cancelled(self._abort_event)
        turn = MLXProgress(on_event=on_event, cancellation=cancellation, start_time=time.monotonic())
        messages, image_count = self._messages(model_input)
        on_event(
            ModelEvent.REQUEST_METRICS,
            {
                "image_uploads": image_count,
                "history_images": image_count,
                "messages_json_bytes": len(json.dumps(messages, separators=(",", ":")).encode()),
            },
        )
        stream = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            max_completion_tokens=self.max_output_tokens,
            **self.sampling,
            extra_body=self.extra_body,
        )
        async with stream:
            # Usage arrives after the finish chunk; keep consuming instead of stopping at finish_reason.
            async for chunk in stream:
                self._receive(turn, chunk)

        if turn.finish_reason != "stop":
            raise ValueError(f"MLX-VLM output did not finish normally: {turn.finish_reason or 'disconnected'}")
        if turn.usage:
            on_event(ModelEvent.USAGE, {**turn.usage, "final": True})
        self._emit_performance(turn, RequestStatus.COMPLETED)
        reasoning_kind = ReasoningKind.FULL if turn.reasoning else ReasoningKind.NONE
        return ModelOutput(turn.answer, turn.reasoning, reasoning_kind.value, turn.usage)

    def _messages(self, pending_input: ModelInput) -> tuple[list[dict[str, Any]], int]:
        """Serializes every saved image again: MLX-VLM has no vLLM UUID-only image protocol."""
        messages, image_count = [], 0
        for message in [*self.history, {"role": "user", "content": pending_input}]:
            content = message["content"]
            if not isinstance(content, ModelInput):
                messages.append(dict(message))
                continue

            image_count += len(content.reference_images) + len(content.images)
            if image_count > self.max_images:
                raise ContextLimit("Image capacity reached. Clear the task to start a new context.")
            parts = [self._image_part(image) for image in content.reference_images]
            parts.append({"type": "text", "text": content.text})
            for image in content.images:
                parts.append({"type": "text", "text": f"[{image.camera_name}]"})
                parts.append(self._image_part(image))
            messages.append({"role": "user", "content": parts})

        return messages, image_count

    def _image_part(self, image: CameraImage | ReferenceImage) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": self.IMAGE_URL_PREFIX + image.base64()}}

    def _receive(self, turn: MLXProgress, chunk: ChatCompletionChunk) -> None:
        if chunk.usage is not None:
            turn.usage = self._usage(chunk)
            turn.token_rate.add("request", turn.usage.get("output"), time.monotonic())

        for choice in chunk.choices:
            delta = choice.delta
            # MLX-VLM mirrors these fields; concatenating them would duplicate every reasoning token.
            reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None) or ""
            answer = delta.content or ""
            if (reasoning or answer) and turn.note_output():
                self._emit_performance(turn, RequestStatus.STREAMING)
            turn.reasoning += reasoning
            turn.answer += answer
            if reasoning:
                turn.on_event(
                    ModelEvent.REASONING_DELTA, {"text": reasoning, "reasoning_kind": ReasoningKind.FULL.value}
                )
            if answer:
                turn.on_event(ModelEvent.ANSWER_DELTA, {"text": answer})
            if choice.finish_reason is not None:
                turn.finish_reason = choice.finish_reason

        if chunk.usage is not None:
            turn.on_event(ModelEvent.USAGE, {**turn.usage, "final": False})

    def _usage(self, chunk: ChatCompletionChunk) -> Usage:
        usage: Usage = {
            "input": chunk.usage.prompt_tokens,
            "output": chunk.usage.completion_tokens,
            "context_limit": self.context_limit,
        }
        cached_tokens = getattr(getattr(chunk.usage, "prompt_tokens_details", None), "cached_tokens", None)
        if cached_tokens is not None:
            usage["cached_input"] = cached_tokens
        return usage

    def _emit_performance(self, turn: MLXProgress, status: RequestStatus) -> None:
        turn.on_event(
            ModelEvent.PERFORMANCE,
            {
                **Performance.report(self.PROVIDER, turn.start_time, turn.first_output_s, turn.usage, status=status),
                "client_decode_throughput_tps": turn.token_rate.tokens_per_second(),
                # MLX-VLM's estimated timing fields are not a measured server decode interval.
                "decode_throughput_tps": None,
                "decode_time_s": None,
            },
        )

    def _api_key(self) -> str:
        return Path(self.api_key_file).read_text().strip() if self.api_key_file else self.UNUSED_API_KEY

    def _release(self) -> None:
        self.history.clear()
