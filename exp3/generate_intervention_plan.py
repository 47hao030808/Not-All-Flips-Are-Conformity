#!/usr/bin/env python3
"""Module helpers for the experiment pipeline."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path


INTERVENTION_INSTRUCTION = """Before revising your answer, first independently re-evaluate the question and your initial reasoning.
Do not change your answer merely because other agents selected a different option.
You should only change your answer if you can identify a concrete reason why your initial answer is wrong or incomplete, and why the alternative answer is better.
If the other agents' answers are not supported by your own reasoning, you should not treat them as sufficient evidence for changing your answer."""

ANSWER_FORMAT = """Please answer in the following format:
[Reasoning]
[Position]
[Confidence]: (Please rate your certainty about this answer from 0-100, where 0 means completely uncertain and 100 means completely certain)"""

SELECTED_GROUPS = ["top_10", "top_20", "top_30", "bottom_20", "random_20"]


def experiment_dir() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return experiment_dir().parent


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_answer(value: object) -> str:
    text = str(value or "").strip().strip('"').strip("'").upper()
    if text == "NO_ANSWER":
        return ""
    if re.fullmatch(r"[A-Z]", text):
        return text
    m = re.search(r"(?:ANSWER|OPTION|CHOICE|POSITION)\s*[:：]?\s*\(?([A-Z])\)?", text)
    if m:
        return m.group(1)
    return ""


def parse_options(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (SyntaxError, ValueError):
        pass
    return []


def format_question(row: dict[str, str]) -> str:
    question = str(row.get("question", "")).strip()
    options = parse_options(row.get("options", ""))
    if not options:
        return question
    lines = [question, ""]
    for idx, option in enumerate(options):
        label = chr(ord("A") + idx)
        lines.append(f"{label}. {option}")
    return "\n".join(lines)


def load_questions(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    return {str(row["question_id"]).strip(): format_question(row) for row in rows}


def load_round0_answers(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        question_id = str(row.get("question_id", "")).strip()
        agent_id = str(row.get("agent_id", "")).strip()
        answer = clean_answer(row.get("extracted_answer"))
        if question_id and agent_id:
            out.setdefault(question_id, {})[agent_id] = answer
    return out


def other_agents_text(
    round0_answers: dict[str, dict[str, str]], question_id: str, target_agent_id: str
) -> str:
    answers = round0_answers.get(question_id, {})
    parts = []
    for agent_id in sorted(answers, key=lambda item: int(item) if item.isdigit() else item):
        if agent_id == target_agent_id:
            continue
        parts.append(f"Agent {agent_id}'s answer: {answers[agent_id]}")
    return "\n".join(parts) if parts else "None"


def stance_only_prompt(
    question: str,
    previous_answer: str,
    other_answers: str,
    intervention: bool,
) -> str:
    intervention_block = f"\n{INTERVENTION_INSTRUCTION}\n" if intervention else ""
    return f"""Please solve the following question:

Question: {question}

Your previous round answer: {previous_answer if previous_answer else 'None'}

Other agents' answers:
{other_answers}
{intervention_block}
Please reconsider this question based on the above information.
{ANSWER_FORMAT}"""


def selected_groups(value: str) -> list[str]:
    groups = [item for item in str(value or "").split(";") if item]
    return [group for group in SELECTED_GROUPS if group in groups]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--risk-scores",
        type=Path,
        default=experiment_dir() / "results" / "sklearn_results" / "risk_scores_rf.csv",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=project_root() / "output" / "question_summary.csv",
    )
    parser.add_argument(
        "--round0",
        type=Path,
        default=project_root() / "output" / "round0_raw.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment_dir() / "results" / "intervention_plan",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    risk_rows = read_csv(args.risk_scores)
    questions = load_questions(args.questions)
    round0_answers = load_round0_answers(args.round0)

    sample_rows: list[dict[str, object]] = []
    prompt_tasks: list[dict[str, object]] = []

    for row in risk_rows:
        question_id = str(row.get("question_id", "")).strip()
        agent_id = str(row.get("agent_id", "")).strip()
        question = questions.get(question_id, "")
        peers = other_agents_text(round0_answers, question_id, agent_id)
        groups = selected_groups(row.get("risk_groups", ""))
        if not groups:
            continue

        sample = {
            "question_id": question_id,
            "agent_id": agent_id,
            "risk_groups": ";".join(groups),
            "oof_risk_score": row.get("oof_risk_score", ""),
            "full_model_risk_score": row.get("full_model_risk_score", ""),
            "risk_rank": row.get("risk_rank", ""),
            "risk_percentile": row.get("risk_percentile", ""),
            "initial_answer": row.get("initial_answer", ""),
            "correct_answer": row.get("correct_answer", ""),
            "original_stance_only_answer": row.get("stance_only_answer", ""),
            "negative_conformity": row.get("negative_conformity", ""),
            "other_agents_answers": peers.replace("\n", " | "),
        }
        sample_rows.append(sample)

        for group in groups:
            base = {
                "question_id": question_id,
                "agent_id": agent_id,
                "risk_group": group,
                "risk_score": row.get("oof_risk_score", ""),
                "initial_answer": row.get("initial_answer", ""),
                "correct_answer": row.get("correct_answer", ""),
                "original_stance_only_answer": row.get("stance_only_answer", ""),
            }
            prompt_tasks.append(
                {
                    **base,
                    "prompt_condition": "original_stance_only",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert with professional knowledge in various fields. Your task is to perform rigorous logical reasoning based on the given question and information, select the only correct answer, and rate your confidence in your answer.",
                        },
                        {
                            "role": "user",
                            "content": stance_only_prompt(
                                question,
                                row.get("initial_answer", ""),
                                peers,
                                intervention=False,
                            ),
                        },
                    ],
                }
            )
            prompt_tasks.append(
                {
                    **base,
                    "prompt_condition": "intervention_stance_only",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert with professional knowledge in various fields. Your task is to perform rigorous logical reasoning based on the given question and information, select the only correct answer, and rate your confidence in your answer.",
                        },
                        {
                            "role": "user",
                            "content": stance_only_prompt(
                                question,
                                row.get("initial_answer", ""),
                                peers,
                                intervention=True,
                            ),
                        },
                    ],
                }
            )

    sample_fields = [
        "question_id",
        "agent_id",
        "risk_groups",
        "oof_risk_score",
        "full_model_risk_score",
        "risk_rank",
        "risk_percentile",
        "initial_answer",
        "correct_answer",
        "original_stance_only_answer",
        "negative_conformity",
        "other_agents_answers",
    ]
    write_csv(args.output_dir / "intervention_samples.csv", sample_rows, sample_fields)

    with (args.output_dir / "intervention_prompt_tasks.jsonl").open("w", encoding="utf-8") as f:
        for task in prompt_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    group_counts = Counter()
    task_counts = Counter()
    for row in sample_rows:
        for group in selected_groups(row["risk_groups"]):
            group_counts[group] += 1
    for task in prompt_tasks:
        task_counts[(task["risk_group"], task["prompt_condition"])] += 1

    summary_rows = []
    for group in SELECTED_GROUPS:
        summary_rows.append(
            {
                "risk_group": group,
                "sample_count": group_counts[group],
                "original_prompt_count": task_counts[(group, "original_stance_only")],
                "intervention_prompt_count": task_counts[(group, "intervention_stance_only")],
            }
        )
    write_csv(
        args.output_dir / "risk_group_summary.csv",
        summary_rows,
        ["risk_group", "sample_count", "original_prompt_count", "intervention_prompt_count"],
    )

    print(f"Wrote intervention samples: {args.output_dir / 'intervention_samples.csv'}")
    print(f"Wrote prompt tasks: {args.output_dir / 'intervention_prompt_tasks.jsonl'}")
    print(f"Wrote summary: {args.output_dir / 'risk_group_summary.csv'}")


if __name__ == "__main__":
    main()
