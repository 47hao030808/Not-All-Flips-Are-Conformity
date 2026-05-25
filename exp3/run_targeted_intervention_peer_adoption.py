#!/usr/bin/env python3
"""Run top20 Round0 peer-adoption-risk targeted intervention tasks.

The intended design is targeted only: run intervention prompts for the top20
highest peer-adoption-risk samples and reuse the existing stance-only stance
baseline for the remaining 80 percent during evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api_client import APIClient, MODEL_ANSWER, MODEL_EXTRACT, calculate_implicit_confidence
from run_intervention_experiment import (
    append_log,
    append_result,
    append_text_log,
    call_answer_model,
    compact_error_message,
    completed_keys,
    extract_answer_with_model,
    log_section,
    validate_result_file,
)
from utils import extract_explicit_confidence


def experiment_dir() -> Path:
    return Path(__file__).resolve().parent


def load_tasks(path: Path) -> list[dict[str, object]]:
    tasks = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def filter_top20_tasks(tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        task
        for task in tasks
        if task.get("risk_group") == "top_20"
        and task.get("prompt_condition") == "targeted_intervention"
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        default=experiment_dir()
        / "results"
        / "round0_peer_adoption_targeted_intervention"
        / "top20_intervention_tasks.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=experiment_dir()
        / "results"
        / "round0_peer_adoption_targeted_intervention_runs"
        / "top20_intervention_results.csv",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="JSONL run log path. Defaults to run_log.jsonl next to --output.",
    )
    parser.add_argument(
        "--text-log",
        type=Path,
        default=None,
        help="Human-readable TXT run log path. Defaults to targeted_intervention_run_log.txt next to --output.",
    )
    args = parser.parse_args()

    log_path = args.log or (args.output.parent / "run_log.jsonl")
    text_log_path = args.text_log or (args.output.parent / "targeted_intervention_run_log.txt")
    validate_result_file(args.output, text_log_path)

    tasks = filter_top20_tasks(load_tasks(args.tasks))
    if args.limit is not None:
        tasks = tasks[: args.limit]

    done = set() if args.no_resume else completed_keys(args.output)
    pending = [
        task
        for task in tasks
        if (
            str(task.get("question_id", "")),
            str(task.get("agent_id", "")),
            str(task.get("risk_group", "")),
            str(task.get("prompt_condition", "")),
        )
        not in done
    ]

    print(
        f"Loaded {len(tasks)} top20 targeted tasks; {len(done)} already completed; "
        f"{len(pending)} pending."
    )
    append_log(
        log_path,
        "run_start",
        tasks_path=str(args.tasks),
        output_path=str(args.output),
        selected_tasks=len(tasks),
        completed=len(done),
        pending=len(pending),
        limit=args.limit,
        dry_run=args.dry_run,
    )
    log_section(text_log_path, "Round0 Top20 Peer-Adoption Targeted Intervention")
    append_text_log(text_log_path, "Configuration:")
    append_text_log(text_log_path, f"  - Tasks: {args.tasks}")
    append_text_log(text_log_path, f"  - Output CSV: {args.output}")
    append_text_log(text_log_path, f"  - JSONL log: {log_path}")
    append_text_log(text_log_path, f"  - Text log: {text_log_path}")
    append_text_log(text_log_path, f"  - Selected top20 tasks: {len(tasks)}")
    append_text_log(text_log_path, f"  - Completed valid tasks: {len(done)}")
    append_text_log(text_log_path, f"  - Pending tasks: {len(pending)}")
    append_text_log(text_log_path, f"  - Model: {MODEL_ANSWER}")
    append_text_log(text_log_path, f"  - Extract model: {MODEL_EXTRACT}")

    if args.dry_run:
        preview = pending[0] if pending else None
        print(json.dumps(preview, ensure_ascii=False, indent=2)[:4000])
        append_text_log(text_log_path, "\nDry-run preview:")
        append_text_log(text_log_path, json.dumps(preview, ensure_ascii=False, indent=2))
        append_log(log_path, "dry_run_preview", preview=preview)
        return

    client = APIClient()
    APIClient.set_fingerprint_log_path(str(args.output.parent))

    for index, task in enumerate(pending, start=1):
        messages = task["messages"]
        task_info = {
            "question_id": task.get("question_id", ""),
            "agent_id": task.get("agent_id", ""),
            "risk_group": task.get("risk_group", ""),
            "prompt_condition": task.get("prompt_condition", ""),
            "risk_score": task.get("risk_score", ""),
            "index": index,
            "total_pending": len(pending),
        }
        append_log(log_path, "task_start", **task_info)
        log_section(
            text_log_path,
            f"Task {index}/{len(pending)}: question_id={task_info['question_id']}, "
            f"agent_id={task_info['agent_id']}",
            "#",
        )
        append_text_log(text_log_path, f"Risk score: {task_info['risk_score']}")
        append_text_log(text_log_path, f"Initial answer: {task.get('initial_answer', '')}")
        append_text_log(text_log_path, f"Correct answer: {task.get('correct_answer', '')}")
        append_text_log(
            text_log_path,
            f"Baseline stance-only answer: {task.get('original_stance_only_answer', '')}",
        )
        append_text_log(text_log_path, "\n[Prompt]")
        for msg in messages:
            append_text_log(text_log_path, f"  {msg.get('role', '')}: {msg.get('content', '')}")

        started = time.time()
        response, tokens, fingerprint, logprobs_data, api_error = call_answer_model(
            client,
            messages,
            log_path,
            text_log_path,
            task_info,
        )
        if not str(response or "").strip():
            print(
                f"[{index}/{len(pending)}] q={task.get('question_id', '')} "
                f"agent={task.get('agent_id', '')} failed: empty model response."
            )
            append_log(
                log_path,
                "answer_model_empty_response",
                elapsed_seconds=round(time.time() - started, 3),
                tokens=tokens,
                system_fingerprint=fingerprint or "",
                error=api_error,
                **task_info,
            )
            append_text_log(
                text_log_path,
                f"[Answer model empty response] elapsed={round(time.time() - started, 3)}s "
                f"error={api_error}",
            )
            continue

        question_text = ""
        if messages and isinstance(messages[-1], dict):
            question_text = str(messages[-1].get("content", ""))
        append_log(
            log_path,
            "answer_model_success",
            elapsed_seconds=round(time.time() - started, 3),
            tokens=tokens,
            system_fingerprint=fingerprint or "",
            response_chars=len(response),
            response_preview=compact_error_message(response),
            **task_info,
        )
        append_text_log(text_log_path, "\n[Answer model response]")
        append_text_log(text_log_path, response)

        extracted = extract_answer_with_model(
            response, question_text, client, log_path, text_log_path, task_info
        )
        if not extracted:
            print(
                f"[{index}/{len(pending)}] q={task.get('question_id', '')} "
                f"agent={task.get('agent_id', '')} failed: no extracted answer."
            )
            append_log(log_path, "final_extract_empty", **task_info)
            append_text_log(text_log_path, "[Final extract empty; not marking as completed]")
            continue

        implicit_confidence = calculate_implicit_confidence(logprobs_data, extracted)
        explicit_confidence = extract_explicit_confidence(response)
        row = {
            "question_id": task.get("question_id", ""),
            "agent_id": task.get("agent_id", ""),
            "risk_group": task.get("risk_group", ""),
            "prompt_condition": task.get("prompt_condition", ""),
            "risk_score": task.get("risk_score", ""),
            "initial_answer": task.get("initial_answer", ""),
            "correct_answer": task.get("correct_answer", ""),
            "original_stance_only_answer": task.get("original_stance_only_answer", ""),
            "extracted_answer": extracted,
            "is_correct": int(extracted == task.get("correct_answer", "")),
            "changed_from_initial": int(bool(extracted) and extracted != task.get("initial_answer", "")),
            "tokens": tokens,
            "implicit_confidence": "" if implicit_confidence is None else f"{implicit_confidence:.6f}",
            "explicit_confidence": "" if explicit_confidence is None else explicit_confidence,
        }
        append_result(args.output, row)
        print(
            f"[{index}/{len(pending)}] q={row['question_id']} agent={row['agent_id']} "
            f"answer={extracted} correct={row['is_correct']}"
        )
        append_log(
            log_path,
            "task_written",
            extracted_answer=extracted,
            is_correct=row["is_correct"],
            changed_from_initial=row["changed_from_initial"],
            implicit_confidence=row["implicit_confidence"],
            explicit_confidence=row["explicit_confidence"],
            **task_info,
        )
        append_text_log(
            text_log_path,
            f"[Task written] extracted_answer={extracted}, is_correct={row['is_correct']}, "
            f"changed_from_initial={row['changed_from_initial']}",
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Wrote targeted intervention results to: {args.output}")
    append_log(log_path, "run_end", output_path=str(args.output))
    log_section(text_log_path, "Run End")


if __name__ == "__main__":
    main()
