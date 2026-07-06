#!/usr/bin/env python3
"""LLM backends and environment loading."""

from __future__ import annotations

import json
import inspect
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent
LLAMA_ENV_PATH = BASE_DIR / ".llama.env"
DEFAULT_BACKEND = "llama_cpp"


class LLMBackend(Protocol):
    """Small interface so the local model can be replaced later."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw model text."""


@dataclass
class LlamaCppBackend:
    """Local llama-cpp-python backend.

    Configure with environment variables:
    - LLAMA_MODEL_PATH: required path to a local GGUF model.
    - LLAMA_CTX_SIZE: context window, default 131072.
    - LLAMA_N_SEQ_MAX: parallel sequence slots, default 1.
    - LLAMA_N_GPU_LAYERS: GPU-offloaded layers, default -1.
    - LLAMA_MAX_TOKENS: maximum generated tokens, default 8192.
    - LLAMA_TEMPERATURE: generation temperature, default 0.1.
    - LLAMA_VERBOSE: print llama.cpp loading logs when set to 1/true/yes.
    """

    model_path: str
    ctx_size: int = 131072
    n_seq_max: int = 1
    n_gpu_layers: int = -1
    max_tokens: int = 8192
    temperature: float = 0.1
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "LlamaCppBackend":
        model_path = os.getenv("LLAMA_MODEL_PATH")
        if not model_path:
            raise RuntimeError(
                "LLAMA_MODEL_PATH is not set. Point it to a local GGUF model file."
            )

        return cls(
            model_path=model_path,
            ctx_size=int(os.getenv("LLAMA_CTX_SIZE", "131072")),
            n_seq_max=int(os.getenv("LLAMA_N_SEQ_MAX", "1")),
            n_gpu_layers=int(os.getenv("LLAMA_N_GPU_LAYERS", "-1")),
            max_tokens=int(os.getenv("LLAMA_MAX_TOKENS", "8192")),
            temperature=float(os.getenv("LLAMA_TEMPERATURE", "0.1")),
            verbose=os.getenv("LLAMA_VERBOSE", "").lower() in {"1", "true", "yes", "on"},
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it or replace LlamaCppBackend."
            ) from exc

        llama_kwargs = {
            "model_path": self.model_path,
            "n_ctx": self.ctx_size,
            "n_gpu_layers": self.n_gpu_layers,
            "verbose": self.verbose,
        }
        llama_signature = inspect.signature(Llama.__init__)
        if "n_seq_max" in llama_signature.parameters:
            llama_kwargs["n_seq_max"] = self.n_seq_max

        llm = Llama(**llama_kwargs)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Llama response format: {result!r}") from exc


@dataclass
class LMStudioBackend:
    """LM Studio OpenAI-compatible chat completions backend.

    Configure with environment variables:
    - LM_STUDIO_BASE_URL: API base URL, default http://127.0.0.1:1234/v1.
    - LM_STUDIO_MODEL: model id sent to LM Studio, default local-model.
    - LM_STUDIO_API_KEY: optional bearer token for compatible servers.
    - LM_STUDIO_MAX_TOKENS: maximum generated tokens, default 8192.
    - LM_STUDIO_TEMPERATURE: generation temperature, default 0.1.
    - LM_STUDIO_TIMEOUT: request timeout in seconds, default 600.
    """

    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "local-model"
    api_key: str | None = None
    max_tokens: int = 8192
    temperature: float = 0.1
    timeout: float = 600.0

    @classmethod
    def from_env(cls) -> "LMStudioBackend":
        return cls(
            base_url=os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
            model=os.getenv("LM_STUDIO_MODEL", "local-model"),
            api_key=os.getenv("LM_STUDIO_API_KEY") or None,
            max_tokens=int(os.getenv("LM_STUDIO_MAX_TOKENS", "8192")),
            temperature=float(os.getenv("LM_STUDIO_TEMPERATURE", "0.1")),
            timeout=float(os.getenv("LM_STUDIO_TIMEOUT", "600")),
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LM Studio request failed with HTTP {exc.code}: {error_body}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not connect to LM Studio at {endpoint}. "
                "Start the LM Studio local server and check LM_STUDIO_BASE_URL."
            ) from exc

        try:
            result = json.loads(response_body)
            return result["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected LM Studio response format: {response_body!r}"
            ) from exc


def load_llama_env(path: Path = LLAMA_ENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        try:
            parsed = shlex.split(value)
        except ValueError:
            parsed = [value.strip("'\"")]

        if len(parsed) == 1:
            os.environ[key] = parsed[0]


def build_backend() -> LLMBackend:
    load_llama_env()
    backend_name = os.getenv("LLM_BACKEND", DEFAULT_BACKEND).strip().lower()

    if backend_name in {"llama_cpp", "llamacpp", "llama-cpp"}:
        return LlamaCppBackend.from_env()
    if backend_name in {"lm_studio", "lmstudio", "lm-studio"}:
        return LMStudioBackend.from_env()

    raise RuntimeError(
        "Unsupported LLM_BACKEND. Use 'llama_cpp' or 'lm_studio'."
    )
