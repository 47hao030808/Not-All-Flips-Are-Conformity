#!/usr/bin/env python3
"""Evaluate Exp3b answer-agnostic peer-adoption targeted intervention.

This script compares:
  1) Original stance-only baseline for all Round-0 disagreement observations.
  2) Intervention results for the top-20% highest predicted peer-adoption risk samples.
  3) A targeted policy: use intervention answers for completed top20 samples and
     reuse the original stance-only baseline for the remaining non-top20 samples.

The risk model/selection is assumed to be produced by
train_round0_risk_model_peer_adoption.py, where top20 risk groups are assigned
using out-of-fold predicted peer-adoption probabilities.

Ground truth is used only for evaluation metrics, not for risk scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


METRIC_FIELDS = [
    "scope",
    "condition",
    "n",
    "invalid_final_n",
    "initial_accuracy",
    "accuracy",
    "flip_rate",
    "peer_adoption_rate",
    "negative_peer_adoption_rate",
    "harmful_peer_adoption_rate",
    "beneficial_peer_adoption_rate",
    "wrong_to_wrong_peer_adoption_rate",
    "self_correction_rate",
]

DELTA_FIELDS = [
    "scope",
    "comparison",
    "paired_n",
    "delta_accuracy",
    "delta_flip_rate",
    "delta_peer_adoption_rate",
    "delta_negative_peer_adoption_rate",
    "delta_harmful_peer_adoption_rate",
    "delta_beneficial_peer_adoption_rate",
]
# I/O helpers


def experiment_dir() -> Path:
    return Path(__file__).resolve().parent


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
# Answer parsing and outcome flags


def clean_answer(value: object) -> str:
    """Normalize an answer to a single uppercase option letter if possible."""
    text = str(value or "").strip().strip('"').strip("'").upper()
    if text == "NO_ANSWER":
        return ""
    if re.fullmatch(r"[A-Z]", text):
        return text
    m = re.search(r"(?:ANSWER|OPTION|CHOICE|POSITION)\s*[:：]?\s*\(?([A-Z])\)?", text)
    if m:
        return m.group(1)
    return ""


def parse_peer_answers(value: object) -> set[str]:
    """Parse peer answers from strings like "Agent 1's answer: B | Agent 2's answer: C"."""
    text = str(value or "")
    answers: set[str] = set()

    # Primary format used by the prompt-generation scripts.
    for match in re.finditer(r"answer:\s*([A-Z])", text, flags=re.IGNORECASE):
        answers.add(match.group(1).upper())
    if answers:
        return answers

    # Fallback for compact delimited formats.
    for part in re.split(r"[|,;\n]", text):
        item = clean_answer(part)
        if item:
            answers.add(item)
    return answers


def is_top20(row: dict[str, object]) -> bool:
    groups = str(row.get("risk_group", "") or "")
    return "top_20" in {item.strip() for item in groups.split(";") if item.strip()}


def outcome_flags(initial: str, final: str, correct: str, peer_answers: set[str]) -> dict[str, object]:
    """Compute sample-level evaluation flags.

    The answer-agnostic target is peer_adoption: changed from initial answer to
    any answer selected by at least one peer under stance-only exposure.

    Directional metrics use ground truth only for evaluation.
    """
    valid_final = bool(final)
    changed = valid_final and final != initial
    peer_adopted = changed and final in peer_answers
    initial_correct = bool(initial) and initial == correct
    final_correct = valid_final and final == correct

    return {
        "valid_final": int(valid_final),
        "initial_correct": int(initial_correct),
        "correct": int(final_correct),
        "flip": int(changed),
        "peer_adoption": int(peer_adopted),
        # Final adopted peer answer is wrong, regardless of initial correctness.
        "negative_peer_adoption": int(peer_adopted and final != correct),
        # Strict accuracy degradation: initially correct -> wrong peer answer.
        "harmful_peer_adoption": int(peer_adopted and initial == correct and final != correct),
        # Accuracy improvement through peer adoption: initially wrong -> correct peer answer.
        "beneficial_peer_adoption": int(peer_adopted and initial != correct and final == correct),
        # Peer adoption that remains wrong: initially wrong -> another wrong peer answer.
        "wrong_to_wrong_peer_adoption": int(peer_adopted and initial != correct and final != correct),
        # Any correction after changing, whether or not the new answer is a peer answer.
        "self_correction": int(changed and initial != correct and final == correct),
    }
# Metric aggregation


def summarize(flags_list: list[dict[str, object]], scope: str, condition: str) -> dict[str, object]:
    n_total = len(flags_list)
    valid_flags = [f for f in flags_list if int(f.get("valid_final", 0)) == 1]
    n = len(valid_flags)
    invalid_final_n = n_total - n

    def rate(key: str) -> str:
        return f"{(sum(int(f.get(key, 0)) for f in valid_flags) / n if n else 0.0):.6f}"

    return {
        "scope": scope,
        "condition": condition,
        "n": n,
        "invalid_final_n": invalid_final_n,
        "initial_accuracy": rate("initial_correct"),
        "accuracy": rate("correct"),
        "flip_rate": rate("flip"),
        "peer_adoption_rate": rate("peer_adoption"),
        "negative_peer_adoption_rate": rate("negative_peer_adoption"),
        "harmful_peer_adoption_rate": rate("harmful_peer_adoption"),
        "beneficial_peer_adoption_rate": rate("beneficial_peer_adoption"),
        "wrong_to_wrong_peer_adoption_rate": rate("wrong_to_wrong_peer_adoption"),
        "self_correction_rate": rate("self_correction"),
    }


def delta_row(scope: str, comparison: str, before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    def diff(metric: str) -> str:
        return f"{float(after[metric]) - float(before[metric]):.6f}"

    return {
        "scope": scope,
        "comparison": comparison,
        "paired_n": after["n"],
        "delta_accuracy": diff("accuracy"),
        "delta_flip_rate": diff("flip_rate"),
        "delta_peer_adoption_rate": diff("peer_adoption_rate"),
        "delta_negative_peer_adoption_rate": diff("negative_peer_adoption_rate"),
        "delta_harmful_peer_adoption_rate": diff("harmful_peer_adoption_rate"),
        "delta_beneficial_peer_adoption_rate": diff("beneficial_peer_adoption_rate"),
    }


def metric_value(flags: dict[str, object], metric: str) -> float:
    if metric == "accuracy":
        return float(flags["correct"])
    if metric == "flip_rate":
        return float(flags["flip"])
    if metric == "peer_adoption_rate":
        return float(flags["peer_adoption"])
    if metric == "negative_peer_adoption_rate":
        return float(flags["negative_peer_adoption"])
    if metric == "harmful_peer_adoption_rate":
        return float(flags["harmful_peer_adoption"])
    if metric == "beneficial_peer_adoption_rate":
        return float(flags["beneficial_peer_adoption"])
    raise KeyError(metric)


def bootstrap_delta_ci(
    pairs: list[tuple[dict[str, object], dict[str, object]]],
    metrics: Iterable[str],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    """Bootstrap CIs for after-before metric deltas over paired samples."""
    if n_bootstrap <= 0 or not pairs:
        return []

    rng = np.random.default_rng(seed)
    n = len(pairs)
    rows: list[dict[str, object]] = []
    for metric in metrics:
        deltas = np.empty(n_bootstrap, dtype=float)
        before_values = np.array([metric_value(b, metric) for b, _ in pairs], dtype=float)
        after_values = np.array([metric_value(a, metric) for _, a in pairs], dtype=float)
        sample_delta = float(after_values.mean() - before_values.mean())
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            deltas[b] = float(after_values[idx].mean() - before_values[idx].mean())
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        rows.append(
            {
                "metric": metric,
                "paired_n": n,
                "delta": f"{sample_delta:.6f}",
                "ci_low": f"{lo:.6f}",
                "ci_high": f"{hi:.6f}",
                "n_bootstrap": n_bootstrap,
            }
        )
    return rows
# Main evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--risk-scores",
        type=Path,
        default=experiment_dir()
        / "results"
        / "round0_peer_adoption_targeted_intervention"
        / "round0_risk_scores_rf.csv",
    )
    parser.add_argument(
        "--intervention-results",
        type=Path,
        default=experiment_dir()
        / "results"
        / "round0_peer_adoption_targeted_intervention_runs"
        / "top20_intervention_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment_dir() / "results" / "round0_peer_adoption_eval",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    risk_rows = read_csv(args.risk_scores)
    intervention_rows = read_csv(args.intervention_results)
    if not risk_rows:
        raise ValueError(f"No risk-score rows found at {args.risk_scores}")

    intervention_lookup = {
        (str(row.get("question_id", "")), str(row.get("agent_id", ""))): row
        for row in intervention_rows
        if str(row.get("extracted_answer", "")).strip()
    }

    sample_rows: list[dict[str, object]] = []
    baseline_all_flags: list[dict[str, object]] = []
    baseline_non_top20_flags: list[dict[str, object]] = []
    baseline_top20_all_flags: list[dict[str, object]] = []
    baseline_top20_completed_flags: list[dict[str, object]] = []
    intervention_top20_completed_flags: list[dict[str, object]] = []
    policy_baseline_matched_flags: list[dict[str, object]] = []
    policy_flags: list[dict[str, object]] = []
    top20_pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    policy_pairs: list[tuple[dict[str, object], dict[str, object]]] = []

    top20_total = 0
    top20_completed = 0
    top20_missing = 0
    top20_non_oof = 0

    for row in risk_rows:
        question_id = str(row.get("question_id", ""))
        agent_id = str(row.get("agent_id", ""))
        key = (question_id, agent_id)
        top20 = is_top20(row)

        initial = clean_answer(row.get("initial_answer", ""))
        correct = clean_answer(row.get("correct_answer", ""))
        baseline_final = clean_answer(row.get("baseline_stance_only_answer", ""))
        peer_answers = parse_peer_answers(row.get("other_agents_answers", ""))
        baseline_flags = outcome_flags(initial, baseline_final, correct, peer_answers)
        baseline_all_flags.append(baseline_flags)

        intervention_row = intervention_lookup.get(key)
        intervention_final = ""
        intervention_flags: dict[str, object] | None = None
        policy_final = baseline_final
        policy_condition = "baseline_stance_only"
        policy_included = True

        if top20:
            top20_total += 1
            if str(row.get("score_source", "")) != "oof":
                top20_non_oof += 1
            baseline_top20_all_flags.append(baseline_flags)
            if intervention_row:
                top20_completed += 1
                intervention_final = clean_answer(intervention_row.get("extracted_answer", ""))
                intervention_flags = outcome_flags(initial, intervention_final, correct, peer_answers)
                baseline_top20_completed_flags.append(baseline_flags)
                intervention_top20_completed_flags.append(intervention_flags)
                top20_pairs.append((baseline_flags, intervention_flags))
                policy_final = intervention_final
                policy_condition = "targeted_intervention"
            else:
                top20_missing += 1
                # Exclude missing top20 rows from matched policy comparison rather than
                # silently treating them as non-intervened.
                policy_included = False
        else:
            baseline_non_top20_flags.append(baseline_flags)

        if policy_included:
            policy_eval_flags = outcome_flags(initial, policy_final, correct, peer_answers)
            policy_baseline_matched_flags.append(baseline_flags)
            policy_flags.append(policy_eval_flags)
            policy_pairs.append((baseline_flags, policy_eval_flags))
        else:
            policy_eval_flags = None

        sample_rows.append(
            {
                "question_id": question_id,
                "agent_id": agent_id,
                "is_top20": int(top20),
                "score_source": row.get("score_source", ""),
                "risk_score": row.get("selection_risk_score", row.get("oof_risk_score", "")),
                "risk_rank": row.get("risk_rank", ""),
                "risk_percentile": row.get("risk_percentile", ""),
                "initial_answer": initial,
                "correct_answer": correct,
                "peer_answers": ";".join(sorted(peer_answers)),
                "baseline_answer": baseline_final,
                "intervention_answer": intervention_final,
                "policy_answer": policy_final if policy_included else "",
                "policy_condition": policy_condition if policy_included else "missing_top20_intervention",
                "baseline_correct": baseline_flags["correct"],
                "baseline_flip": baseline_flags["flip"],
                "baseline_peer_adoption": baseline_flags["peer_adoption"],
                "baseline_negative_peer_adoption": baseline_flags["negative_peer_adoption"],
                "baseline_harmful_peer_adoption": baseline_flags["harmful_peer_adoption"],
                "baseline_beneficial_peer_adoption": baseline_flags["beneficial_peer_adoption"],
                "intervention_correct": "" if intervention_flags is None else intervention_flags["correct"],
                "intervention_flip": "" if intervention_flags is None else intervention_flags["flip"],
                "intervention_peer_adoption": "" if intervention_flags is None else intervention_flags["peer_adoption"],
                "intervention_negative_peer_adoption": "" if intervention_flags is None else intervention_flags["negative_peer_adoption"],
                "intervention_harmful_peer_adoption": "" if intervention_flags is None else intervention_flags["harmful_peer_adoption"],
                "intervention_beneficial_peer_adoption": "" if intervention_flags is None else intervention_flags["beneficial_peer_adoption"],
                "policy_correct": "" if policy_eval_flags is None else policy_eval_flags["correct"],
                "policy_peer_adoption": "" if policy_eval_flags is None else policy_eval_flags["peer_adoption"],
            }
        )

    metric_rows: list[dict[str, object]] = []
    metric_rows.append(summarize(baseline_all_flags, "all_samples", "baseline_stance_only"))
    metric_rows.append(summarize(baseline_non_top20_flags, "non_top20", "baseline_stance_only"))
    metric_rows.append(summarize(baseline_top20_all_flags, "top20_all", "baseline_stance_only"))
    metric_rows.append(summarize(baseline_top20_completed_flags, "top20_completed", "baseline_stance_only"))
    metric_rows.append(summarize(intervention_top20_completed_flags, "top20_completed", "targeted_intervention"))
    metric_rows.append(summarize(policy_baseline_matched_flags, "policy_matched_set", "baseline_stance_only"))
    metric_rows.append(summarize(policy_flags, "policy_matched_set", "targeted_policy"))

    summary_by_key = {(row["scope"], row["condition"]): row for row in metric_rows}
    delta_rows: list[dict[str, object]] = []
    if baseline_top20_completed_flags and intervention_top20_completed_flags:
        delta_rows.append(
            delta_row(
                "top20_completed",
                "targeted_intervention - baseline_stance_only",
                summary_by_key[("top20_completed", "baseline_stance_only")],
                summary_by_key[("top20_completed", "targeted_intervention")],
            )
        )
    if policy_flags:
        delta_rows.append(
            delta_row(
                "policy_matched_set",
                "targeted_policy - baseline_stance_only",
                summary_by_key[("policy_matched_set", "baseline_stance_only")],
                summary_by_key[("policy_matched_set", "targeted_policy")],
            )
        )

    bootstrap_metrics = [
        "accuracy",
        "flip_rate",
        "peer_adoption_rate",
        "negative_peer_adoption_rate",
        "harmful_peer_adoption_rate",
        "beneficial_peer_adoption_rate",
    ]
    bootstrap_rows: list[dict[str, object]] = []
    for scope, pairs in [
        ("top20_completed", top20_pairs),
        ("policy_matched_set", policy_pairs),
    ]:
        for row in bootstrap_delta_ci(pairs, bootstrap_metrics, args.bootstrap, args.seed):
            row["scope"] = scope
            bootstrap_rows.append(row)

    sample_fieldnames = [
        "question_id",
        "agent_id",
        "is_top20",
        "score_source",
        "risk_score",
        "risk_rank",
        "risk_percentile",
        "initial_answer",
        "correct_answer",
        "peer_answers",
        "baseline_answer",
        "intervention_answer",
        "policy_answer",
        "policy_condition",
        "baseline_correct",
        "baseline_flip",
        "baseline_peer_adoption",
        "baseline_negative_peer_adoption",
        "baseline_harmful_peer_adoption",
        "baseline_beneficial_peer_adoption",
        "intervention_correct",
        "intervention_flip",
        "intervention_peer_adoption",
        "intervention_negative_peer_adoption",
        "intervention_harmful_peer_adoption",
        "intervention_beneficial_peer_adoption",
        "policy_correct",
        "policy_peer_adoption",
    ]

    write_csv(args.output_dir / "group_condition_metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(args.output_dir / "paired_delta_metrics.csv", delta_rows, DELTA_FIELDS)
    write_csv(
        args.output_dir / "bootstrap_delta_ci.csv",
        bootstrap_rows,
        ["scope", "metric", "paired_n", "delta", "ci_low", "ci_high", "n_bootstrap"],
    )
    write_csv(args.output_dir / "sample_level_outcomes.csv", sample_rows, sample_fieldnames)

    summary = {
        "risk_scores_path": str(args.risk_scores),
        "intervention_results_path": str(args.intervention_results),
        "all_samples": len(risk_rows),
        "top20_total": top20_total,
        "top20_completed": top20_completed,
        "top20_missing_intervention": top20_missing,
        "top20_non_oof_score_source": top20_non_oof,
        "policy_matched_n": len(policy_flags),
        "notes": [
            "policy_matched_set excludes top20 samples whose intervention result is missing.",
            "non-top20 samples reuse the original stance-only baseline under the targeted policy.",
            "ground truth is used only for evaluation metrics.",
        ],
        "outputs": {
            "group_condition_metrics": "group_condition_metrics.csv",
            "paired_delta_metrics": "paired_delta_metrics.csv",
            "bootstrap_delta_ci": "bootstrap_delta_ci.csv",
            "sample_level_outcomes": "sample_level_outcomes.csv",
        },
    }
    write_json(args.output_dir / "evaluation_summary.json", summary)

    print("Evaluation complete.")
    print(f"All samples: {len(risk_rows)}")
    print(f"Top20 total: {top20_total}")
    print(f"Top20 completed: {top20_completed}")
    print(f"Top20 missing intervention: {top20_missing}")
    print(f"Top20 with non-OOF score source: {top20_non_oof}")
    print(f"Policy matched n: {len(policy_flags)}")
    print(f"Wrote metrics to: {args.output_dir / 'group_condition_metrics.csv'}")
    print(f"Wrote deltas to: {args.output_dir / 'paired_delta_metrics.csv'}")
    print(f"Wrote bootstrap CIs to: {args.output_dir / 'bootstrap_delta_ci.csv'}")


if __name__ == "__main__":
    main()
