#!/usr/bin/env python3
"""Run the AI Fabric pipeline with a repair loop."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from worker import ARTIFACTS_DIR, BASE_DIR, TASK_PATH


PIPELINE_RESULT_PATH = ARTIFACTS_DIR / "pipeline_result.json"


def run_stage(name: str, command: list[str]) -> dict[str, object]:
    started_at = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(
        command,
        cwd=BASE_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "name": name,
        "command": command,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "ok": result.returncode == 0,
    }


def append_stage_feedback(stage: dict[str, object], attempt: int) -> None:
    stdout = str(stage.get("stdout", "")).strip()
    stderr = str(stage.get("stderr", "")).strip()
    feedback = f"""

REPAIR NOTES - ATTEMPT {attempt}
Pipeline stage failed before tests could pass.

Stage:
{stage["name"]}

Exit code:
{stage["returncode"]}

Stage stdout:
{stdout or "(empty)"}

Stage stderr:
{stderr or "(empty)"}
"""
    TASK_PATH.write_text(
        TASK_PATH.read_text(encoding="utf-8").rstrip() + feedback + "\n",
        encoding="utf-8",
    )


def write_pipeline_result(report: dict[str, object]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_RESULT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AI Fabric end to end.")
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--test-timeout", type=int, default=60)
    parser.add_argument(
        "--skip-explainer",
        action="store_true",
        help="Use existing artifacts/task.txt and artifacts/contract.json.",
    )
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "max_repairs": args.max_repairs,
        "stages": [],
        "passed": False,
    }
    stages = report["stages"]

    if not args.skip_explainer:
        stage = run_stage("explainer", [sys.executable, "explainer.py"])
        stages.append(stage)
        if not stage["ok"]:
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_pipeline_result(report)
            print(f"Pipeline failed at explainer. See {PIPELINE_RESULT_PATH}")
            return 1

    for attempt in range(1, args.max_repairs + 2):
        worker_stage = run_stage("worker", [sys.executable, "worker.py"])
        stages.append(worker_stage)
        if not worker_stage["ok"]:
            if attempt <= args.max_repairs:
                append_stage_feedback(worker_stage, attempt)
                continue
            break

        tester_stage = run_stage("tester", [sys.executable, "tester.py"])
        stages.append(tester_stage)
        if not tester_stage["ok"]:
            if attempt <= args.max_repairs:
                append_stage_feedback(tester_stage, attempt)
                continue
            break

        repair_command = [
            sys.executable,
            "repair.py",
            "--attempt",
            str(attempt),
            "--timeout",
            str(args.test_timeout),
        ]
        if attempt > args.max_repairs:
            repair_command.append("--no-append")

        repair_stage = run_stage("repair", repair_command)
        stages.append(repair_stage)
        if repair_stage["returncode"] == 0:
            report["passed"] = True
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_pipeline_result(report)
            print(f"Pipeline passed. See {PIPELINE_RESULT_PATH}")
            return 0
        if repair_stage["returncode"] != 2:
            break

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_pipeline_result(report)
    print(f"Pipeline failed. See {PIPELINE_RESULT_PATH}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
