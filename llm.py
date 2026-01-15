"""Ollama client - simple HTTP interface with streaming."""
import json
import requests
from typing import Generator, Optional

class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 300):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def chat(self, model: str, messages: list[dict], tools: Optional[list] = None,
             stream: bool = True) -> Generator[dict, None, None]:
        """Stream chat responses. Yields dicts with 'content' or 'tool_calls'."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools

        resp = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            stream=stream,
            timeout=self.timeout
        )
        resp.raise_for_status()

        if stream:
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk
                    if chunk.get("done"):
                        break
        else:
            yield resp.json()

    def chat_sync(self, model: str, messages: list[dict],
                  tools: Optional[list] = None,
                  tool_choice: Optional[str] = "auto",
                  options: Optional[dict] = None) -> dict:
        """Non-streaming chat. Returns full response."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        if options:
            payload["options"] = options

        resp = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=self.timeout
        )
        if resp.status_code != 200:
            try:
                error_body = resp.json()
            except:
                error_body = resp.text
            raise Exception(f"Ollama API error {resp.status_code}: {error_body}")
        return resp.json()

    def list_models(self) -> list[str]:
        """List available models."""
        resp = requests.get(f"{self.host}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            requests.get(f"{self.host}/api/tags", timeout=5)
            return True
        except:
            return False
