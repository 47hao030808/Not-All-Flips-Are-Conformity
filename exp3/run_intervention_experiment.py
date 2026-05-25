#!/usr/bin/env python3
"""Run risk-targeted intervention prompt tasks."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP1_ROOT = PROJECT_ROOT / "exp1"
if str(EXP1_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP1_ROOT))

from api_client import (  # type: ignore[attr-defined]
    APIClient,
    GLOBAL_SEED,
    INITIAL_RETRY_DELAY,
    MAX_RETRIES,
    MODEL_ANSWER,
    MODEL_EXTRACT,
    RETRY_MULTIPLIER,
    calculate_implicit_confidence,
)
from utils import extract_answer_with_agent, extract_explicit_confidence  # type: ignore[attr-defined]


RESULT_FIELDS = [
    "question_id",
    "agent_id",
    "risk_group",
    "prompt_condition",
    "risk_score",
    "initial_answer",
    "correct_answer",
    "original_stance_only_answer",
    "extracted_answer",
    "is_correct",
    "changed_from_initial",
    "tokens",
    "implicit_confidence",
    "explicit_confidence",
]


def experiment_dir() -> Path:
    return Path(__file__).resolve().parent


def load_tasks(path: Path) -> list:
    tasks = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def completed_keys(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        return {
            (
                str(row.get("question_id", "")),
                str(row.get("agent_id", "")),
                str(row.get("risk_group", "")),
                str(row.get("prompt_condition", "")),
            )
            for row in rows
            if str(row.get("extracted_answer", "")).strip()
        }


def append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def validate_result_file(path: Path, text_log_path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline().strip()
    expected_header = ",".join(RESULT_FIELDS)
    if first_line == expected_header:
        return
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.invalid_header_backup_{timestamp}")
    path.replace(backup)
    append_text_log(
        text_log_path,
        f"[Result file warning] Existing result CSV had an invalid header and was moved to: {backup}",
    )


def append_log(path: Path, event: str, **fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        **fields,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_text_log(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def log_section(path: Path, title: str, char: str = "=") -> None:
    append_text_log(path, f"{char * 60}\n{title}\n{char * 60}")


def filter_tasks(tasks: list, risk_group: str, condition: str) -> list:
    return [
        task
        for task in tasks
        if task.get("risk_group") == risk_group
        and task.get("prompt_condition") == condition
    ]


def normalize_extracted_answer(value: str) -> str:
    text = str(value or "").strip().strip('"').strip("'").upper()
    if text == "NO_ANSWER":
        return ""
    if re.fullmatch(r"[A-Z]", text):
        return text
    m = re.search(r"(?:ANSWER|OPTION|CHOICE|POSITION)\s*[:：]?\s*\(?([A-Z])\)?", text)
    if m:
        return m.group(1)
    return ""


def extract_answer_with_deepseek(response: str, question: str, client: APIClient, log_path: Path, text_log_path: Path, task_info: dict) -> str:
    if not response:
        return ""
    system_prompt = (
        "You are an answer extraction expert. Extract the final answer from an agent's complete response.\n"
        "CRITICAL: Look for the answer after the '[Position]:' tag first.\n"
        "Extract ONLY the letter option, such as A, B, C, D, E, F, G, H, I, or J.\n"
        "If [Position] contains a number from 1 to 26, convert it to a letter (1=A, 2=B, etc.).\n"
        "If there is no clear final answer, output NO_ANSWER.\n"
        "Output only the extracted letter or NO_ANSWER, with no explanation."
    )
    user_prompt = f"""Question:\n{question}\n\nAgent's complete response:\n{response}\n\nExtract the final answer letter."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        append_text_log(text_log_path, "\n[DeepSeek extraction prompt]")
        append_text_log(text_log_path, f"  system: {system_prompt}")
        append_text_log(text_log_path, f"  user: {user_prompt}")
        resp = client.extract_client.chat.completions.create(
            model=MODEL_EXTRACT,
            messages=messages,
            max_tokens=50,
            temperature=0,
        )
        extracted = resp.choices[0].message.content if resp.choices else ""
        append_text_log(text_log_path, f"[DeepSeek extraction raw output]: {extracted}")
        normalized = normalize_extracted_answer(extracted)
        if normalized:
            append_text_log(text_log_path, f"[DeepSeek extracted answer]: {normalized}")
            append_log(
                log_path,
                "deepseek_extract_success",
                raw_extracted=str(extracted),
                extracted_answer=normalized,
                **task_info,
            )
            return normalized
        append_log(
            log_path,
            "deepseek_extract_empty_or_invalid",
            raw_extracted=str(extracted),
            **task_info,
        )
        append_text_log(text_log_path, "[DeepSeek extraction invalid; falling back to local parser]")
    except Exception as exc:
        print(f"DeepSeek extraction failed, falling back to local parser: {exc}")
        append_text_log(text_log_path, f"[DeepSeek extraction error]: {repr(exc)}")
        append_log(
            log_path,
            "deepseek_extract_error",
            error=repr(exc),
            **task_info,
        )
    fallback = extract_answer_with_agent(response, "")
    append_text_log(text_log_path, f"[Local fallback extracted answer]: {fallback}")
    append_log(
        log_path,
        "local_extract_fallback",
        extracted_answer=fallback,
        **task_info,
    )
    return fallback


def extract_answer_with_model(response: str, question: str, client: APIClient, log_path: Path, text_log_path: Path, task_info: dict) -> str:
    return extract_answer_with_deepseek(response, question, client, log_path, text_log_path, task_info)


def compact_error_message(response: str) -> str:
    text = str(response or "").strip().replace("\n", " ")
    return text[:500]


def call_answer_model(client: APIClient, messages: list, log_path: Path, text_log_path: Path, task_info: dict):
    retry_delay = INITIAL_RETRY_DELAY
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 2):
        append_text_log(
            text_log_path,
            f"Calling API: model={MODEL_ANSWER}, max_tokens=2000, temperature=0, seed={GLOBAL_SEED}, attempt={attempt}/{MAX_RETRIES + 1}",
        )
        try:
            resp = client.client.chat.completions.create(
                model=MODEL_ANSWER,
                messages=messages,
                max_tokens=2000,
                temperature=0,
                seed=GLOBAL_SEED,
                logprobs=True,
                top_logprobs=1,
            )
            msg = resp.choices[0].message
            response_text = msg.content if hasattr(msg, "content") else ""
            tokens = resp.usage.total_tokens if hasattr(resp, "usage") and resp.usage else 0
            fingerprint = getattr(resp, "system_fingerprint", "") or ""
            logprobs_data = getattr(resp.choices[0], "logprobs", None)
            call_index = getattr(client, "_api_call_counter", 0) + 1
            client._api_call_counter = call_index
            client.log_fingerprint(call_index, fingerprint, MODEL_ANSWER, GLOBAL_SEED, 0)
            client.log_confidence(call_index, None, MODEL_ANSWER, GLOBAL_SEED, 0)
            if logprobs_data is not None:
                setattr(logprobs_data, "call_index", call_index)
            append_text_log(
                text_log_path,
                f"API response successful: tokens={tokens}, fingerprint={fingerprint}, response_length={len(response_text)}",
            )
            append_log(
                log_path,
                "answer_model_api_success",
                attempt=attempt,
                tokens=tokens,
                system_fingerprint=fingerprint,
                response_chars=len(response_text),
                **task_info,
            )
            return response_text, tokens, fingerprint, logprobs_data, ""
        except Exception as exc:
            last_error = repr(exc)
            append_text_log(text_log_path, f"API call failed on attempt {attempt}: {last_error}")
            append_log(
                log_path,
                "answer_model_api_error",
                attempt=attempt,
                error=last_error,
                **task_info,
            )
            if attempt <= MAX_RETRIES:
                append_text_log(text_log_path, f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= RETRY_MULTIPLIER
    return "", 0, "", None, last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        default=experiment_dir() / "results" / "intervention_plan" / "intervention_prompt_tasks.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=experiment_dir() / "results" / "intervention_runs" / "intervention_results.csv",
    )
    parser.add_argument("--risk-group", default="top_20")
    parser.add_argument("--condition", default="intervention_stance_only")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="JSONL run log path.",
    )
    parser.add_argument(
        "--text-log",
        type=Path,
        default=None,
        help="Human-readable TXT run log path.",
    )
    args = parser.parse_args()
    log_path = args.log or (args.output.parent / "run_log.jsonl")
    text_log_path = args.text_log or (args.output.parent / "intervention_run_log.txt")
    validate_result_file(args.output, text_log_path)

    tasks = filter_tasks(load_tasks(args.tasks), args.risk_group, args.condition)
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

    print(f"Loaded {len(tasks)} selected tasks; {len(done)} already completed; {len(pending)} pending.")
    append_log(
        log_path,
        "run_start",
        tasks_path=str(args.tasks),
        output_path=str(args.output),
        risk_group=args.risk_group,
        condition=args.condition,
        selected_tasks=len(tasks),
        completed=len(done),
        pending=len(pending),
        limit=args.limit,
        dry_run=args.dry_run,
    )
    log_section(text_log_path, "Risk-Targeted Intervention Experiment")
    append_text_log(text_log_path, "Configuration:")
    append_text_log(text_log_path, f"  - Tasks: {args.tasks}")
    append_text_log(text_log_path, f"  - Output CSV: {args.output}")
    append_text_log(text_log_path, f"  - JSONL log: {log_path}")
    append_text_log(text_log_path, f"  - Text log: {text_log_path}")
    append_text_log(text_log_path, f"  - Risk group: {args.risk_group}")
    append_text_log(text_log_path, f"  - Prompt condition: {args.condition}")
    append_text_log(text_log_path, f"  - Selected tasks: {len(tasks)}")
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
            f"Task {index}/{len(pending)}: question_id={task_info['question_id']}, agent_id={task_info['agent_id']}",
            "#",
        )
        append_text_log(text_log_path, f"Risk score: {task_info['risk_score']}")
        append_text_log(text_log_path, f"Initial answer: {task.get('initial_answer', '')}")
        append_text_log(text_log_path, f"Correct answer: {task.get('correct_answer', '')}")
        append_text_log(text_log_path, f"Original stance-only answer: {task.get('original_stance_only_answer', '')}")
        append_text_log(text_log_path, "\n[Prompt]")
        for msg in messages:
            append_text_log(text_log_path, f"  {msg.get('role', '')}: {msg.get('content', '')}")
        started = time.time()
        response, tokens, fingerprint, logprobs_data, api_error = call_answer_model(
            client, messages, log_path, text_log_path, task_info
        )
        if not response.strip():
            print(
                f"[{index}/{len(pending)}] q={task.get('question_id', '')} agent={task.get('agent_id', '')} failed: empty model response; not marking as completed."
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
                f"[Answer model empty response] elapsed={round(time.time() - started, 3)}s error={api_error}",
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
        extracted = extract_answer_with_deepseek(
            response, question_text, client, log_path, text_log_path, task_info
        )
        if not extracted:
            print(
                f"[{index}/{len(pending)}] q={task.get('question_id', '')} agent={task.get('agent_id', '')} failed: no extracted answer; not marking as completed."
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
            f"[{index}/{len(pending)}] q={row['question_id']} agent={row['agent_id']} answer={extracted} correct={row['is_correct']}"
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
            f"[Task written] extracted_answer={extracted}, is_correct={row['is_correct']}, changed_from_initial={row['changed_from_initial']}",
        )
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"Wrote intervention results to: {args.output}")
    append_log(log_path, "run_end", output_path=str(args.output))
    log_section(text_log_path, "Run End")


if __name__ == "__main__":
    main()
