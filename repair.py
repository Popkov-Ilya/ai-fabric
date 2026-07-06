#!/usr/bin/env python3
"""
Run generated tests and append repair feedback to artifacts/task.txt on failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tester import TEST_OUTPUT_PATH
from worker import ARTIFACTS_DIR, OUTPUT_PATH, TASK_PATH


TEST_RESULT_PATH = ARTIFACTS_DIR / "test_result.json"
TEST_STDOUT_PATH = ARTIFACTS_DIR / "test_stdout.txt"
TEST_STDERR_PATH = ARTIFACTS_DIR / "test_stderr.txt"


def truncate_text(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... truncated ...\n"


def run_tests(timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Generated implementation not found: {OUTPUT_PATH}")
    if not TEST_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Generated tests not found: {TEST_OUTPUT_PATH}")

    return subprocess.run(
        [sys.executable, TEST_OUTPUT_PATH.name],
        cwd=ARTIFACTS_DIR,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )


def write_test_artifacts(result: subprocess.CompletedProcess[str]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_STDOUT_PATH.write_text(result.stdout, encoding="utf-8")
    TEST_STDERR_PATH.write_text(result.stderr, encoding="utf-8")
    TEST_RESULT_PATH.write_text(
        json.dumps(
            {
                "command": [sys.executable, TEST_OUTPUT_PATH.name],
                "cwd": str(ARTIFACTS_DIR),
                "returncode": result.returncode,
                "stdout_path": TEST_STDOUT_PATH.name,
                "stderr_path": TEST_STDERR_PATH.name,
                "passed": result.returncode == 0,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def append_repair_feedback(
    task_path: Path,
    result: subprocess.CompletedProcess[str],
    attempt: int,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    stdout = truncate_text(result.stdout.strip())
    stderr = truncate_text(result.stderr.strip())
    feedback = f"""

REPAIR NOTES - ATTEMPT {attempt}
Generated tests failed on {timestamp}.

The next implementation must keep the existing public API contract unchanged,
but fix the behavior exposed by this test failure.

Test command:
{sys.executable} {TEST_OUTPUT_PATH.name}

Exit code:
{result.returncode}

Test stdout:
{stdout or "(empty)"}

Test stderr:
{stderr or "(empty)"}
"""
    task_path.write_text(
        task_path.read_text(encoding="utf-8").rstrip() + feedback + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run generated tests and append repair feedback on failure."
    )
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Run tests and write result artifacts without editing task.txt.",
    )
    args = parser.parse_args()

    try:
        result = run_tests(args.timeout)
        write_test_artifacts(result)
        if result.returncode == 0:
            print(f"Tests passed. Wrote result to {TEST_RESULT_PATH}")
            return 0

        if not args.no_append:
            append_repair_feedback(TASK_PATH, result, args.attempt)
            print(f"Tests failed. Appended repair notes to {TASK_PATH}")
        else:
            print("Tests failed. Repair notes were not appended.")
        return 2
    except subprocess.TimeoutExpired as exc:
        result = subprocess.CompletedProcess(
            exc.cmd,
            124,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + f"\nTimed out after {args.timeout} seconds.",
        )
        write_test_artifacts(result)
        if not args.no_append:
            append_repair_feedback(TASK_PATH, result, args.attempt)
        print(f"Tests timed out after {args.timeout} seconds.")
        return 2
    except Exception as exc:
        print(f"repair.py failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
