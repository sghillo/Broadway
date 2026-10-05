"""LLM clients. vLLM exposes an OpenAI-compatible server, so we talk plain HTTP to it."""
from __future__ import annotations

import os
from typing import Protocol

import requests


class LLM(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 1024,
                 temperature: float = 0.1) -> str: ...


class VLLMClient:
    """Client for a vLLM server started with ``vllm serve <model>`` (OpenAI-compatible API)."""

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        self.timeout = timeout

    def complete(self, system: str, user: str, max_tokens: int = 1024,
                 temperature: float = 0.1) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class ExtractiveLLM:
    """Offline fallback: emits the first sentence of each context block with its citation.

    Keeps the whole pipeline runnable (and testable) without a GPU or model server.
    """

    def complete(self, system: str, user: str, max_tokens: int = 1024,
                 temperature: float = 0.1) -> str:
        lines = []
        for block in user.split("\n\n"):
            if not block.startswith("[doi:"):
                continue
            header, _, body = block.partition("\n")
            first = body.strip().split(". ")[0].rstrip(".")
            if first:
                lines.append(f"{first}. {header.split(']')[0]}]")
        return "\n".join(lines)
