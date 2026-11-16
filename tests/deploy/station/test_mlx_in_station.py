import json
import threading
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import openai
import pytest

from pear.deploy.station.agent_loop.action_parser import ActionParser
from pear.deploy.station.agent_loop.clients.client import ContextLimit, ModelClient
from pear.deploy.station.agent_loop.clients.mlx import MLXClient
from pear.deploy.station.agent_loop.clients.schemas import CameraImage, ModelInput, ReferenceImage
from pear.deploy.station.agent_loop.config import OfferedModel
from pear.deploy.station.control.robots.so101.formats import SINGLE_ARM_POSE_FORMAT
from pear.deploy.station.exceptions import OperatorPaused

MODEL = "mlx-community/Qwen3.8-27B-4bit"
ACTION = {"arm": {"x": 0, "y": 0, "z": 0, "pitch": 0, "roll": 0}, "gripper": {"width": 0}}
ANSWER = "Hold the arm.\n```json\n" + json.dumps(ACTION) + "\n```"


@dataclass
class MLXService:
    """An HTTP/SSE stand-in for MLX-VLM, including the alias it emits for reasoning."""

    url: str = ""
    calls: list = field(default_factory=list)
    modes: list = field(default_factory=list)
    stream_closed: threading.Event = field(default_factory=threading.Event)

    @staticmethod
    def images(request):
        return [
            part
            for message in request["messages"]
            if isinstance(message["content"], list)
            for part in message["content"]
            if part["type"] == "image_url"
        ]


@pytest.fixture
def mlx_service():
    service = MLXService()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/v1/models"
            body = json.dumps(
                {"object": "list", "data": [{"id": MODEL, "object": "model", "created": 0, "owned_by": "test"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            service.calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            mode = service.modes.pop(0) if service.modes else "ok"
            if mode in ("http_error", "bad_request", "context_overflow"):
                message = (
                    "Request needs 16400 tokens, but MAX_KV_SIZE is 16384."
                    if mode == "context_overflow"
                    else "generation unavailable"
                )
                body = json.dumps({"error": {"message": message}}).encode()
                self.send_response(500 if mode == "http_error" else 400)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def send(value):
                self.wfile.write(("data: " + json.dumps(value) + "\n\n").encode())
                self.wfile.flush()

            def chunk(delta, finish=None):
                return {
                    "id": "mlx-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": MODEL,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }

            try:
                # Distinct texts make the priority of reasoning_content over its alias observable.
                send(chunk({"reasoning_content": "Inspect the scene.", "reasoning": "ignored alias"}))
                if mode == "stall":
                    self.connection.settimeout(2)
                    try:
                        if not self.connection.recv(1):
                            service.stream_closed.set()
                    except ConnectionResetError:
                        service.stream_closed.set()
                    except TimeoutError:
                        pass
                    return
                if mode == "sse_error":
                    send({"error": {"message": "generation failed during streaming", "type": "server_error"}})
                    return

                send(chunk({"content": ANSWER}))
                if mode != "disconnected":
                    send(chunk({}, "length" if mode == "length" else "stop"))
                if mode not in ("no_usage", "disconnected"):
                    # The server reports the total after the finish chunk, with no choices.
                    send(
                        {
                            "id": "mlx-test",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": MODEL,
                            "choices": [],
                            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                        }
                    )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    service.url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield service
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.fixture
def mlx_inputs(tmp_path):
    def make(number):
        # The stand-in checks the wire format and does not decode the JPEG bytes.
        path = tmp_path / f"{number}-wrist.jpg"
        path.write_bytes(b"jpeg" + str(number).encode())
        return ModelInput(f"observation {number}", (CameraImage("wrist", path),))

    return make


def infer(client, item, *, on_event=lambda *args: None):
    return client.infer(item, on_event=on_event, cancellation=threading.Event())


def test_mlx_request_history_reasoning_usage_and_action_contract(mlx_service, mlx_inputs, tmp_path):
    ref = tmp_path / "goal.jpg"
    ref.write_bytes(b"goal")
    references = (ReferenceImage(ref),)
    client = MLXClient("system", timeout_s=10, base_url=mlx_service.url, model=MODEL)
    events = []

    try:
        assert client.check()["status"] == "ok"
        for number in range(3):
            output = infer(
                client,
                replace(mlx_inputs(number), reference_images=references),
                on_event=lambda kind, data: events.append((kind, data)),
            )
            assert output.answer == ANSWER
            assert output.reasoning == "Inspect the scene."
            assert output.reasoning_kind == "full"
            assert output.usage == {"input": 100, "output": 20, "context_limit": 16384}
            parsed = ActionParser.parse(output.answer, SINGLE_ARM_POSE_FORMAT)
            assert parsed.arm.x == 0 and parsed.gripper.width == 0

        first, _, last = mlx_service.calls
        assert first["model"] == MODEL
        assert first["max_completion_tokens"] == 4096
        assert "max_tokens" not in first
        assert first["stream_options"] == {"include_usage": True}
        assert first["enable_thinking"] is True and first["context_management"] == []
        assert "chat_template_kwargs" not in first
        assert first["temperature"] == 1 and first["top_p"] == 0.95 and first["top_k"] == 0
        assert first["repetition_penalty"] == 1
        assert [len(MLXService.images(call)) for call in mlx_service.calls] == [2, 3, 4]
        assert all(
            set(image) == {"type", "image_url"} and image["image_url"]["url"].startswith("data:image/jpeg;base64,")
            for call in mlx_service.calls
            for image in MLXService.images(call)
        )
        assert MLXService.images(first)[0] == MLXService.images(last)[0]
        users = [message for message in last["messages"] if message["role"] == "user"]
        assert users[0]["content"][0]["type"] == "image_url"
        assert users[1]["content"][0] == {"type": "text", "text": "observation 1"}
        assert last["messages"][2] == {
            "role": "assistant",
            "content": ANSWER,
            "reasoning_content": "Inspect the scene.",
        }
        assert len([data for kind, data in events if kind == "reasoning_delta"]) == 3
        assert [data for kind, data in events if kind == "usage"][-1]["final"] is True
    finally:
        client.close()
    assert client.history == []


@pytest.mark.parametrize(
    "mode,error",
    [
        ("length", ValueError),
        ("disconnected", ValueError),
        ("sse_error", openai.APIError),
        ("http_error", openai.APIStatusError),
        ("bad_request", openai.BadRequestError),
        ("context_overflow", ContextLimit),
    ],
)
def test_failed_generation_is_not_retried_or_committed(mlx_service, mlx_inputs, mode, error):
    client = MLXClient("system", timeout_s=10, base_url=mlx_service.url, model=MODEL)
    try:
        infer(client, mlx_inputs(0))
        history = list(client.history)
        mlx_service.modes.append(mode)

        with pytest.raises(error):
            infer(client, mlx_inputs(1))

        assert client.history == history
        assert len(mlx_service.calls) == 2
    finally:
        client.close()


def test_failed_first_turn_does_not_consume_reference_images(mlx_service, mlx_inputs, tmp_path):
    ref = tmp_path / "goal.jpg"
    ref.write_bytes(b"goal")
    item = replace(mlx_inputs(0), reference_images=(ReferenceImage(ref),))
    client = MLXClient("system", timeout_s=10, base_url=mlx_service.url, model=MODEL)
    try:
        mlx_service.modes.append("http_error")
        with pytest.raises(openai.APIStatusError):
            infer(client, item)
        infer(client, item)
        assert mlx_service.calls[0]["messages"] == mlx_service.calls[1]["messages"]
    finally:
        client.close()


def test_missing_usage_stays_unknown(mlx_service, mlx_inputs):
    mlx_service.modes.append("no_usage")
    client = MLXClient("system", timeout_s=10, base_url=mlx_service.url, model=MODEL, enable_thinking=False)
    events = []
    try:
        output = infer(client, mlx_inputs(0), on_event=lambda kind, data: events.append((kind, data)))
        assert output.usage == {}
        assert mlx_service.calls[0]["enable_thinking"] is False
        assert not [data for kind, data in events if kind == "usage"]
        reports = [data for kind, data in events if kind == "performance"]
        assert reports and reports[-1]["final"] is True
        assert all(report["output_tokens"] is None and report["output_throughput_tps"] is None for report in reports)
    finally:
        client.close()


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_interrupted_stream_is_closed_without_committing_partial_turn(mlx_service, mlx_inputs, stop):
    client = MLXClient("system", timeout_s=10, base_url=mlx_service.url, model=MODEL)
    events = []
    try:
        infer(client, mlx_inputs(0))
        history = list(client.history)
        mlx_service.modes.append("stall")
        if stop == "timeout":
            client.timeout_s = 0.3

        def receive(kind, data):
            events.append(kind)
            if stop == "cancel" and kind == "reasoning_delta":
                client.cancel()

        expected = OperatorPaused if stop == "cancel" else (TimeoutError, openai.APITimeoutError)
        with pytest.raises(expected):
            infer(client, mlx_inputs(1), on_event=receive)

        assert "reasoning_delta" in events and "answer_delta" not in events
        assert client.history == history and len(mlx_service.calls) == 2
        assert mlx_service.stream_closed.wait(1), "the aborted response remained open on the server"
    finally:
        client.close()


def test_image_limit_counts_history_and_refuses_before_http(mlx_service, mlx_inputs):
    client = MLXClient("system", timeout_s=10, base_url=mlx_service.url, model=MODEL, max_images=1)
    try:
        infer(client, mlx_inputs(0))
        history = list(client.history)
        with pytest.raises(ContextLimit):
            infer(client, mlx_inputs(1))
        assert len(mlx_service.calls) == 1 and client.history == history
    finally:
        client.close()


@pytest.mark.parametrize("max_images", [True, 0, 1.5])
def test_invalid_image_capacity_is_rejected(max_images):
    with pytest.raises(ValueError, match="max_images"):
        MLXClient("system", timeout_s=10, base_url="http://server/v1", model=MODEL, max_images=max_images)


def test_station_configuration_can_construct_and_close_without_network_io(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Configuration validation attempted to create a network client")

    monkeypatch.setattr(openai, "OpenAI", forbidden)
    monkeypatch.setattr(openai, "AsyncOpenAI", forbidden)
    offered = OfferedModel(
        label="Qwen3.8 MLX",
        client_class="pear.deploy.station.agent_loop.clients.mlx.MLXClient",
        client={"base_url": "http://server/v1", "model": MODEL, "api_key_file": "key.txt"},
    )
    offered.check("mlx")
    offered.resolve_paths(tmp_path)
    assert offered.client["api_key_file"] == str(tmp_path / "key.txt")
    client = offered.new_client("system")
    assert isinstance(client, ModelClient)
    assert client.system_prompt == "system" and client.timeout_s == 300
    client.close()
