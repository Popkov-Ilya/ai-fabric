#!/usr/bin/env python3
"""
Read a technical assignment from task.txt, ask a local Llama-compatible LLM
to write Python tests for output.py, validate them, and write test_output.py.
"""

from __future__ import annotations

import sys

from worker import (
    API_SIGNATURE_PATH,
    BASE_DIR,
    TASK_PATH,
    build_backend,
    read_task,
    trim_to_python_code,
    validate_python_code,
    write_output,
)


TEST_OUTPUT_PATH = BASE_DIR / "test_output.py"


TEST_SYSTEM_PROMPT = """\
You are a senior Python QA engineer. You receive a technical assignment from
task.txt and a public API summary extracted from output.py. You must return
ONLY complete, runnable Python test code for test_output.py.

Hard rules:
- Return Python code only.
- Do not include explanations, comments outside the test program, or prose.
- The tests must verify output.py against the technical assignment.
- Use the public API summary to decide which functions and arguments can be tested.
- Do not assume or reference implementation details that are not present in the task or public API summary.
- The tests must be runnable with Python 3.10+ as python test_output.py.
- Use only the Python standard library unless the assignment explicitly requires dependencies.
- Prefer unittest for test structure.
- If output.py is a command-line program, test it through subprocess using sys.executable.
- Include happy-path tests, edge-case tests, and error-handling tests when applicable.
- Avoid network access, hidden files, real user input, and environment-specific assumptions.
- Never output placeholders such as TODO, pass-only stubs, or pseudo-code.
"""


TEST_USER_PROMPT_TEMPLATE = """\
Write a Python test file named test_output.py for the program that will be
implemented in output.py according to the following technical assignment.

The tests should be practical, deterministic, and directly runnable.

Technical assignment:
{task}

Public API summary extracted from output.py without implementation:
{api_signature}
"""


def read_api_signature() -> str:
    if not API_SIGNATURE_PATH.exists():
        raise FileNotFoundError(
            f"API signature file not found: {API_SIGNATURE_PATH}. Run worker.py first."
        )

    return API_SIGNATURE_PATH.read_text(encoding="utf-8").strip()


def main() -> int:
    try:
        task = read_task(TASK_PATH)
        api_signature = read_api_signature()
        backend = build_backend()
        user_prompt = TEST_USER_PROMPT_TEMPLATE.format(
            task=task,
            api_signature=api_signature,
        )
        raw_result = backend.generate(TEST_SYSTEM_PROMPT, user_prompt)
        code = trim_to_python_code(raw_result)
        validate_python_code(code, TEST_OUTPUT_PATH)
        write_output(TEST_OUTPUT_PATH, code)
    except Exception as exc:
        print(f"tester.py failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote valid Python tests to {TEST_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
