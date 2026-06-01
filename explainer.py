#!/usr/bin/env python3
"""
Read a raw task from input_task.txt, format it according to task_template.txt
with a local Llama-compatible LLM, and write the result to task.txt.
"""

from __future__ import annotations

import sys

from worker import BASE_DIR, TASK_PATH, build_backend


INPUT_TASK_PATH = BASE_DIR / "input_task.txt"
TASK_TEMPLATE_PATH = BASE_DIR / "task_template.txt"


EXPLAINER_SYSTEM_PROMPT = """\
You are a senior technical analyst. You receive an unstructured programming
task and a plain-text task template. Return ONLY the completed task text in the
same structure as the template.

Hard rules:
- Return plain text only.
- Do not return Python code.
- Do not include explanations outside the completed task.
- Preserve every section from the template.
- Write in clear English.
- Make the task precise enough for another LLM to implement in output.py.
- If details are missing, make reasonable assumptions and state them inside the
  relevant section.
- Keep examples small, concrete, and internally consistent.
"""


EXPLAINER_USER_PROMPT_TEMPLATE = """\
Raw task:
{raw_task}

Task template:
{template}

Complete the template using the raw task above.
"""


def read_text_file(path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"File is empty: {path}")

    return text


def normalize_task_text(text: str) -> str:
    return text.strip() + "\n"


def validate_task_text(text: str, template: str) -> None:
    if not text.strip():
        raise ValueError("LLM returned empty task text.")

    section_names = [
        line.strip()
        for line in template.splitlines()
        if line.strip() and line.strip().upper() == line.strip()
    ]

    missing_sections = [
        section for section in section_names if section not in text
    ]
    if missing_sections:
        joined = ", ".join(missing_sections)
        raise ValueError(f"Formatted task is missing template sections: {joined}")


def main() -> int:
    try:
        raw_task = read_text_file(INPUT_TASK_PATH)
        template = read_text_file(TASK_TEMPLATE_PATH)
        backend = build_backend()
        user_prompt = EXPLAINER_USER_PROMPT_TEMPLATE.format(
            raw_task=raw_task,
            template=template,
        )
        formatted_task = normalize_task_text(
            backend.generate(EXPLAINER_SYSTEM_PROMPT, user_prompt)
        )
        validate_task_text(formatted_task, template)
        TASK_PATH.write_text(formatted_task, encoding="utf-8")
    except Exception as exc:
        print(f"explainer.py failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote formatted task to {TASK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
