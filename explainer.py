#!/usr/bin/env python3
"""
Read a raw task from artifacts/input_task.txt, format it according to
task_template.txt with a local Llama-compatible LLM, and write the result to
artifacts/task.txt and artifacts/contract.json.
"""

from __future__ import annotations

import json
import sys

from llm_backend import build_backend
from worker import ARTIFACTS_DIR, BASE_DIR, CONTRACT_PATH, TASK_PATH, validate_contract


INPUT_TASK_PATH = ARTIFACTS_DIR / "input_task.txt"
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
- Include every template section header exactly as written
- Do not omit any section, even if the raw task does not mention it.
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


CONTRACT_SYSTEM_PROMPT = """\
You are a senior Python API designer. You receive a normalized programming
task and must return ONLY a JSON object that specifies the public API contract
for output.py.

Hard rules:
- Return JSON only.
- Do not include Markdown, prose, comments, or code fences.
- The contract must describe functions that worker.py must implement and
  tester.py must test.
- Prefer exactly one public target function unless the task clearly requires
  more.
- The function name must be descriptive, snake_case, and derived from the task.
- Use only this JSON shape:
  {
    "module": "output",
    "source_file": "output.py",
    "functions": [
      {
        "name": "function_name",
        "async": false,
        "arguments": [
          {
            "name": "argument_name",
            "kind": "positional_or_keyword",
            "required": true
          }
        ],
        "returns": "short human-readable return type"
      }
    ]
  }
- Allowed argument kind values are: positional_only, positional_or_keyword,
  var_positional, keyword_only, var_keyword.
- Include zero arguments when the task requires no function input.
- Do not include a main function in the contract.
- Do not include private helpers in the contract.
"""


CONTRACT_USER_PROMPT_TEMPLATE = """\
Create the public API contract for output.py from this normalized task:

{task}
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


def normalize_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        if stripped.lower().startswith("json\n"):
            stripped = stripped.split("\n", 1)[1].strip()
    return stripped


def parse_contract(text: str) -> dict[str, object]:
    try:
        contract = json.loads(normalize_json_text(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid contract JSON: {exc.msg}") from exc

    if not isinstance(contract, dict):
        raise ValueError("LLM returned a contract that is not a JSON object.")
    return contract


def write_contract(path, contract: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def extract_template_sections(template: str) -> list[str]:
    sections: list[str] = []

    for line in template.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-"):
            continue
        if stripped.endswith(":"):
            continue
        if not stripped.replace(" ", "").isalpha():
            continue
        if stripped.upper() != stripped:
            continue

        sections.append(stripped)

    return sections


def validate_task_text(text: str, template: str) -> None:
    if not text.strip():
        raise ValueError("LLM returned empty task text.")

    section_names = extract_template_sections(template)
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
        contract_prompt = CONTRACT_USER_PROMPT_TEMPLATE.format(task=formatted_task)
        contract = parse_contract(
            backend.generate(CONTRACT_SYSTEM_PROMPT, contract_prompt)
        )
        validate_contract(contract, CONTRACT_PATH)
        TASK_PATH.parent.mkdir(parents=True, exist_ok=True)
        TASK_PATH.write_text(formatted_task, encoding="utf-8")
        write_contract(CONTRACT_PATH, contract)
    except Exception as exc:
        print(f"explainer.py failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote formatted task to {TASK_PATH}")
    print(f"Wrote API contract to {CONTRACT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
