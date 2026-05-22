#!/usr/bin/env python3
"""
Read a technical assignment from task.txt, ask a local Llama-compatible LLM
to implement it, validate the returned Python code, and write output.py.
"""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


BASE_DIR = Path(__file__).resolve().parent
TASK_PATH = BASE_DIR / "task.txt"
OUTPUT_PATH = BASE_DIR / "output.py"


SYSTEM_PROMPT = """\
You are a senior Python engineer. You receive a technical assignment from task.txt
and must return ONLY complete, runnable Python source code for output.py.

Hard rules:
- Return Python code only.
- Do not include explanations, comments outside the program, or prose.
- The code must be self-contained unless the assignment explicitly allows external files.
- The code must be runnable with Python 3.10+.
- Prefer the Python standard library unless the assignment explicitly requests dependencies.
- Include clear error handling where useful.
- If the assignment is ambiguous, make reasonable assumptions and encode them in code comments.
- Never output placeholders such as TODO, pass-only stubs, or pseudo-code.
"""


USER_PROMPT_TEMPLATE = """\
Implement the following technical assignment as a single Python file named output.py.

Technical assignment:
{task}
"""


class LLMBackend(Protocol):
    """Small interface so the local model can be replaced later."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw model text."""


@dataclass
class LlamaCppBackend:
    """Local llama-cpp-python backend.

    Configure with environment variables:
    - LLAMA_MODEL_PATH: required path to a local GGUF model.
    - LLAMA_CTX_SIZE: context window, default 8192.
    - LLAMA_MAX_TOKENS: maximum generated tokens, default 4096.
    - LLAMA_TEMPERATURE: generation temperature, default 0.1.
    """

    model_path: str
    ctx_size: int = 8192
    max_tokens: int = 4096
    temperature: float = 0.1

    @classmethod
    def from_env(cls) -> "LlamaCppBackend":
        model_path = os.getenv("LLAMA_MODEL_PATH")
        if not model_path:
            raise RuntimeError(
                "LLAMA_MODEL_PATH is not set. Point it to a local GGUF model file."
            )

        return cls(
            model_path=model_path,
            ctx_size=int(os.getenv("LLAMA_CTX_SIZE", "8192")),
            max_tokens=int(os.getenv("LLAMA_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("LLAMA_TEMPERATURE", "0.1")),
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it or replace LlamaCppBackend."
            ) from exc

        llm = Llama(
            model_path=self.model_path,
            n_ctx=self.ctx_size,
            verbose=False,
        )

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


def read_task(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")

    task = path.read_text(encoding="utf-8").strip()
    if not task:
        raise ValueError(f"Task file is empty: {path}")

    return task


def trim_to_python_code(text: str) -> str:
    """Remove simple quote-like wrappers around code returned by an LLM."""

    code = text.strip()
    if not code:
        return "\n"

    code_starts = (
        "#!",
        "#",
        "from ",
        "import ",
        "def ",
        "class ",
        "async def ",
        "@",
        "if ",
        "for ",
        "while ",
        "try:",
        "with ",
        "print(",
    )

    def marker_matches(line: str, marker: str) -> bool:
        stripped = line.strip()
        if marker == "```":
            return stripped.startswith(marker)
        return stripped == marker

    def finish(source: str) -> str:
        return source.strip() + "\n"

    def remove_fence_language(source: str) -> str:
        lines = source.splitlines()
        if lines and lines[0].strip().lower() in {"python", "py"}:
            return "\n".join(lines[1:])
        return source

    def looks_like_code_payload(source: str) -> bool:
        stripped = source.lstrip()
        return (
            "\n" in source
            or stripped.startswith(code_starts)
            or "=" in stripped
        )

    def looks_like_code_boundary(source: str) -> bool:
        stripped = source.lstrip()
        return bool(stripped) and (stripped.startswith(code_starts) or "=" in stripped)

    lines = code.splitlines()
    for marker in ("```", "'''", '"""', "'", '"'):
        start = None
        for index, line in enumerate(lines):
            if marker_matches(line, marker):
                start = index
                break

        if start is None:
            continue

        for end in range(len(lines) - 1, start, -1):
            if marker_matches(lines[end], marker):
                inner = "\n".join(lines[start + 1 : end])
                if marker == "```":
                    inner = remove_fence_language(inner)
                return finish(inner)

    for marker in ("```", "'''", '"""'):
        start = code.find(marker)
        end = code.rfind(marker)
        if start == -1 or end <= start:
            continue

        suffix = code[end + len(marker) :].strip()
        if start == 0 and suffix:
            continue

        inner = code[start + len(marker) : end]
        if marker == "```":
            inner = remove_fence_language(inner)
        return finish(inner)

    for marker in ("'", '"'):
        start = code.find(marker)
        end = code.rfind(marker)
        if start == -1 or end <= start:
            continue
        if code[start : start + 3] == marker * 3:
            continue
        if code[end - 2 : end + 1] == marker * 3:
            continue

        prefix = code[:start].strip()
        suffix = code[end + 1 :].strip()
        if looks_like_code_boundary(prefix) or looks_like_code_boundary(suffix):
            continue

        inner = code[start + 1 : end]
        if looks_like_code_payload(inner):
            return finish(inner)

    return finish(code)


def validate_python_code(code: str, filename: Path = OUTPUT_PATH) -> None:
    if not code.strip():
        raise ValueError("Model returned empty code.")

    try:
        tree = ast.parse(code, filename=str(filename))
        compile(tree, str(filename), "exec")
    except SyntaxError as exc:
        raise ValueError(
            f"Model returned invalid Python at line {exc.lineno}: {exc.msg}"
        ) from exc


def write_output(path: Path, code: str) -> None:
    path.write_text(code, encoding="utf-8")


def build_backend() -> LLMBackend:
    return LlamaCppBackend.from_env()


def main() -> int:
    try:
        task = read_task(TASK_PATH)
        backend = build_backend()
        user_prompt = USER_PROMPT_TEMPLATE.format(task=task)
        raw_result = backend.generate(SYSTEM_PROMPT, user_prompt)
        code = trim_to_python_code(raw_result)
        validate_python_code(code)
        write_output(OUTPUT_PATH, code)
    except Exception as exc:
        print(f"worker.py failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote valid Python code to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
