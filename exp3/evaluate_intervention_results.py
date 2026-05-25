#!/usr/bin/env python3
"""Evaluate prompt-intervention effects against the original stance-only run."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


METRIC_FIELDS = [
    "risk_group",
    "condition",
    "n",
    "flip_rate",
    "conformity_rate",
    "negative_conformity_rate",
    "beneficial_conformity_rate",
    "self_correction_rate",
    "accuracy",
]


def experiment_dir() -> Path:
    return Path(__file__).resolve().parent


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


def parse_peer_answers(value: str) -> set[str]:
    text = str(value or "")
    answers = set()
    for match in re.finditer(r"answer:\s*([A-Z])", text, flags=re.IGNORECASE):
        answers.add(match.group(1).upper())
    if answers:
        return answers
    for part in re.split(r"[|,;]", text):
        item = clean_answer(part)
        if re.fullmatch(r"[A-Z]", item):
            answers.add(item)
    return answers


def selected_groups(value: str) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


def outcome_flags(
    initial: str, final: str, correct: str, peer_answers: set[str]
) -> dict[str, int]:
    changed = final != "" and final != initial
    conformed = changed and final in peer_answers
    negative = conformed and final != correct
    beneficial = conformed and initial != correct and final == correct
    self_corrected = changed and initial != correct and final == correct
    return {
        "flip": int(changed),
        "conformity": int(conformed),
        "negative_conformity": int(negative),
        "beneficial_conformity": int(beneficial),
        "self_correction": int(self_corrected),
        "correct": int(final == correct),
    }


def summarize(rows: list[dict[str, object]], condition: str, group: str) -> dict[str, object]:
    n = len(rows)
    sums = defaultdict(int)
    for row in rows:
        flags = row["flags"]
        for key, value in flags.items():
            sums[key] += int(value)

    def rate(key: str) -> str:
        return f"{(sums[key] / n if n else 0.0):.6f}"

    return {
        "risk_group": group,
        "condition": condition,
        "n": n,
        "flip_rate": rate("flip"),
        "conformity_rate": rate("conformity"),
        "negative_conformity_rate": rate("negative_conformity"),
        "beneficial_conformity_rate": rate("beneficial_conformity"),
        "self_correction_rate": rate("self_correction"),
        "accuracy": rate("correct"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=Path,
        default=experiment_dir()
        / "results"
        / "intervention_plan"
        / "intervention_samples.csv",
    )
    parser.add_argument(
        "--intervention-results",
        type=Path,
        default=experiment_dir()
        / "results"
        / "intervention_runs"
        / "intervention_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment_dir() / "results" / "intervention_eval",
    )
    args = parser.parse_args()

    samples = read_csv(args.samples)
    result_rows = read_csv(args.intervention_results)
    result_lookup = {
        (
            str(row.get("question_id", "")),
            str(row.get("agent_id", "")),
            str(row.get("risk_group", "")),
        ): row
        for row in result_rows
    }

    original_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    intervention_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    paired_rows: list[dict[str, object]] = []

    for sample in samples:
        question_id = str(sample.get("question_id", ""))
        agent_id = str(sample.get("agent_id", ""))
        initial = clean_answer(sample.get("initial_answer", ""))
        correct = clean_answer(sample.get("correct_answer", ""))
        original_final = clean_answer(sample.get("original_stance_only_answer", ""))
        peer_answers = parse_peer_answers(sample.get("other_agents_answers", ""))

        for group in selected_groups(sample.get("risk_groups", "")):
            original_flags = outcome_flags(initial, original_final, correct, peer_answers)
            original_by_group[group].append({"flags": original_flags})

            result = result_lookup.get((question_id, agent_id, group))
            if not result:
                continue
            intervention_final = clean_answer(result.get("extracted_answer", ""))
            intervention_flags = outcome_flags(initial, intervention_final, correct, peer_answers)
            intervention_by_group[group].append({"flags": intervention_flags})
            paired_rows.append(
                {
                    "risk_group": group,
                    "question_id": question_id,
                    "agent_id": agent_id,
                    "risk_score": sample.get("oof_risk_score", ""),
                    "initial_answer": initial,
                    "correct_answer": correct,
                    "original_stance_only_answer": original_final,
                    "intervention_answer": intervention_final,
                    "original_negative_conformity": original_flags["negative_conformity"],
                    "intervention_negative_conformity": intervention_flags["negative_conformity"],
                    "original_correct": original_flags["correct"],
                    "intervention_correct": intervention_flags["correct"],
                }
            )

    groups = sorted(set(original_by_group) | set(intervention_by_group))
    metric_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    for group in groups:
        original_summary = summarize(original_by_group[group], "original_stance_only", group)
        intervention_summary = summarize(
            intervention_by_group[group], "intervention_stance_only", group
        )
        metric_rows.extend([original_summary, intervention_summary])
        if int(intervention_summary["n"]) > 0:
            delta_rows.append(
                {
                    "risk_group": group,
                    "paired_n": intervention_summary["n"],
                    "delta_negative_conformity_rate": f"{float(intervention_summary['negative_conformity_rate']) - float(original_summary['negative_conformity_rate']):.6f}",
                    "delta_accuracy": f"{float(intervention_summary['accuracy']) - float(original_summary['accuracy']):.6f}",
                    "delta_flip_rate": f"{float(intervention_summary['flip_rate']) - float(original_summary['flip_rate']):.6f}",
                    "delta_beneficial_conformity_rate": f"{float(intervention_summary['beneficial_conformity_rate']) - float(original_summary['beneficial_conformity_rate']):.6f}",
                }
            )

    write_csv(args.output_dir / "group_condition_metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(
        args.output_dir / "paired_delta_by_group.csv",
        delta_rows,
        [
            "risk_group",
            "paired_n",
            "delta_negative_conformity_rate",
            "delta_accuracy",
            "delta_flip_rate",
            "delta_beneficial_conformity_rate",
        ],
    )
    write_csv(
        args.output_dir / "paired_sample_outcomes.csv",
        paired_rows,
        [
            "risk_group",
            "question_id",
            "agent_id",
            "risk_score",
            "initial_answer",
            "correct_answer",
            "original_stance_only_answer",
            "intervention_answer",
            "original_negative_conformity",
            "intervention_negative_conformity",
            "original_correct",
            "intervention_correct",
        ],
    )

    print(f"Wrote metrics to: {args.output_dir / 'group_condition_metrics.csv'}")
    print(f"Wrote deltas to: {args.output_dir / 'paired_delta_by_group.csv'}")


if __name__ == "__main__":
    main()
