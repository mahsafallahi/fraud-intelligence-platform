"""Chat model client (Phase 7): the local Ollama REST API, and a scripted fake for tests and CI.

Runs are reproducible as far as Ollama allows: temperature 0, a fixed seed, a fixed
context size, and the model digest (from /api/tags) recorded with every result.
No Ollama Python client: plain HTTP with requests (already pinned).
"""

from dataclasses import dataclass, field

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.1:latest"
DEFAULT_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 400}


@dataclass
class Reply:
    content: str
    tool_calls: list = field(default_factory=list)
    prompt_tokens: int = -1
    output_tokens: int = -1
    seconds: float = 0.0          # total time reported by the server (load + prompt + generation)


class OllamaChat:
    name = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_URL, session=None,
                 options: dict | None = None, timeout: float = 900):
        import requests
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.options = {**DEFAULT_OPTIONS, **(options or {})}
        self.session = session or requests.Session()

    def chat(self, messages: list[dict], tools: list | None = None) -> Reply:
        body = {"model": self.model, "messages": messages, "stream": False, "options": self.options}
        if tools:
            body["tools"] = tools
        response = self.session.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        return Reply(content=message.get("content", ""), tool_calls=message.get("tool_calls") or [],
                     prompt_tokens=int(data.get("prompt_eval_count", -1)),
                     output_tokens=int(data.get("eval_count", -1)),
                     seconds=data.get("total_duration", 0) / 1e9)

    def describe(self) -> dict:
        response = self.session.get(f"{self.base_url}/api/tags", timeout=60)
        response.raise_for_status()
        digest = next((m.get("digest") for m in response.json().get("models", [])
                       if self.model in (m.get("name"), m.get("model"))), None)
        if digest is None:
            raise RuntimeError(f"Ollama has no model {self.model!r}")
        return {"name": self.name, "model": self.model, "digest": digest, "options": dict(self.options)}


class FakeLLM:
    """Returns scripted replies in order (a str, or a callable of the messages); records every call."""

    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list | None = None) -> Reply:
        self.calls.append([dict(m) for m in messages])
        if not self.replies:
            raise AssertionError("FakeLLM: no scripted reply left")
        reply = self.replies.pop(0)
        content = reply(messages) if callable(reply) else reply
        return Reply(content=content)

    def describe(self) -> dict:
        return {"name": self.name, "model": "scripted", "digest": None, "options": {}}
