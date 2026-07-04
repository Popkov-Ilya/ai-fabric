#!/usr/bin/env python3
"""
Read a technical assignment from artifacts/task.txt, ask a local
Llama-compatible LLM to implement it, validate the returned Python code, and
write artifacts/output.py.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from llm_backend import build_backend


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
TASK_PATH = ARTIFACTS_DIR / "task.txt"
OUTPUT_PATH = ARTIFACTS_DIR / "output.py"
API_SIGNATURE_PATH = ARTIFACTS_DIR / "output_api.json"


SYSTEM_PROMPT = """\
You are a senior Python engineer. You receive a technical assignment from task.txt
and must return ONLY complete, runnable Python source code for output.py.

Hard rules:
- Return Python code only.
- Do not include explanations, comments outside the program, or prose.
- The code must be self-contained unless the assignment explicitly allows external files.
- The code must be runnable with Python 3.10+.
- Structure the solution like a coding challenge / LeetCode task:
  define one clear public target function that implements the task logic.
- Put all core logic inside that target function or small helper functions it calls.
- Do not put task-solving logic directly in main(), top-level code, or the
  if __name__ == "__main__" block.
- If command-line, stdin, or file I/O is needed, keep it in a thin main()
  function that only parses input, calls the target function, and prints or
  writes the returned result.
- The if __name__ == "__main__" block must only call main() or the target
  function; it must not contain business logic.
- Prefer a target function with explicit input parameters and a return value
  so tests can import output.py and call it directly.
- Choose a descriptive target function name from the assignment when no exact
  function name is specified.
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")


def describe_arguments(arguments: ast.arguments) -> list[dict[str, object]]:
    described: list[dict[str, object]] = []

    positional_args = list(arguments.posonlyargs) + list(arguments.args)
    required_count = len(positional_args) - len(arguments.defaults)

    for index, arg in enumerate(positional_args):
        kind = "positional_only" if index < len(arguments.posonlyargs) else "positional_or_keyword"
        described.append(
            {
                "name": arg.arg,
                "kind": kind,
                "required": index < required_count,
            }
        )

    if arguments.vararg is not None:
        described.append(
            {
                "name": arguments.vararg.arg,
                "kind": "var_positional",
                "required": False,
            }
        )

    for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        described.append(
            {
                "name": arg.arg,
                "kind": "keyword_only",
                "required": default is None,
            }
        )

    if arguments.kwarg is not None:
        described.append(
            {
                "name": arguments.kwarg.arg,
                "kind": "var_keyword",
                "required": False,
            }
        )

    return described


def describe_python_api(code: str) -> dict[str, object]:
    tree = ast.parse(code, filename=str(OUTPUT_PATH))
    functions = []

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        functions.append(
            {
                "name": node.name,
                "async": isinstance(node, ast.AsyncFunctionDef),
                "arguments": describe_arguments(node.args),
            }
        )

    return {
        "module": "output",
        "source_file": OUTPUT_PATH.name,
        "functions": functions,
    }


def write_api_signature(path: Path, code: str) -> None:
    api_signature = describe_python_api(code)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(api_signature, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    try:
        task = read_task(TASK_PATH)
        backend = build_backend()
        user_prompt = USER_PROMPT_TEMPLATE.format(task=task)
        raw_result = backend.generate(SYSTEM_PROMPT, user_prompt)
        code = trim_to_python_code(raw_result)
        validate_python_code(code)
        write_output(OUTPUT_PATH, code)
        write_api_signature(API_SIGNATURE_PATH, code)
    except Exception as exc:
        print(f"worker.py failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote valid Python code to {OUTPUT_PATH}")
    print(f"Wrote API signature to {API_SIGNATURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
