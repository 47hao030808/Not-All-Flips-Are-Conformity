#!/usr/bin/env python3
"""Sklearn LR/RF risk-ranking models for negative conformity prediction.

This is the main prediction model for experiment 3.  It keeps the compact
8-feature set and treats predicted probabilities as risk scores for selecting
intervention samples, not as hard labels at threshold 0.5.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


FEATURES = [
    "round0_implicit_confidence",
    "relative_implicit_confidence",
    "implicit_confidence_rank_in_group",
    "round0_reasoning_length",
    "num_peers_same_as_target",
    "target_is_alone",
    "peer_wrong_majority_size",
    "answer_entropy",
]

LABEL = "negative_conformity"
ID_COLUMNS = [
    "question_id",
    "agent_id",
    "initial_answer",
    "correct_answer",
    "stance_only_answer",
]
RISK_GROUPS = {
    "top_10": 0.10,
    "top_20": 0.20,
    "top_30": 0.30,
    "bottom_20": 0.20,
}


def experiment_dir() -> Path:
    return Path(__file__).resolve().parent


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_data(path: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    rows = load_rows(path)
    missing = [col for col in [*FEATURES, LABEL] if rows and col not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    x = np.array([[float(row[col]) for col in FEATURES] for row in rows])
    y = np.array([int(row[LABEL]) for row in rows])
    return rows, x, y


def save_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_cv(
    model_name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, object]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "model": model_name,
        "threshold": f"{threshold:.2f}",
        "roc_auc": f"{roc_auc_score(y_true, y_prob):.4f}",
        "accuracy": f"{accuracy_score(y_true, y_pred):.4f}",
        "precision": f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
        "recall": f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
        "f1": f"{f1_score(y_true, y_pred, zero_division=0):.4f}",
        "n_positive": int(y_true.sum()),
        "n_total": len(y_true),
    }


def threshold_sweep(
    model_name: str, y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray
) -> list[dict[str, object]]:
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "model": model_name,
                "threshold": f"{threshold:.2f}",
                "precision": f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
                "recall": f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
                "f1": f"{f1_score(y_true, y_pred, zero_division=0):.4f}",
            }
        )
    return rows


def optimal_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1s[:-1]))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def lr_grid(cv: StratifiedKFold) -> GridSearchCV:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42),
            ),
        ]
    )
    return GridSearchCV(
        pipe,
        {
            "clf__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            "clf__class_weight": [None, "balanced", {0: 1, 1: 2}, {0: 1, 1: 3}],
        },
        cv=cv,
        scoring="roc_auc",
        refit=True,
        n_jobs=-1,
        return_train_score=True,
    )


def rf_grid(cv: StratifiedKFold) -> GridSearchCV:
    return GridSearchCV(
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [100, 200, 500],
            "max_depth": [3, 5, 8, None],
            "min_samples_leaf": [5, 10, 20],
            "class_weight": [None, "balanced", "balanced_subsample"],
        },
        cv=cv,
        scoring="roc_auc",
        refit=True,
        n_jobs=-1,
        return_train_score=True,
    )


def oof_probabilities(model, x: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> np.ndarray:
    y_prob = np.zeros(len(y))
    for train_idx, test_idx in cv.split(x, y):
        model.fit(x[train_idx], y[train_idx])
        y_prob[test_idx] = model.predict_proba(x[test_idx])[:, 1]
    return y_prob


def plain_params(params: dict[str, object]) -> dict[str, object]:
    return {key.replace("clf__", ""): value for key, value in params.items()}


def assign_risk_groups(scores: np.ndarray, seed: int = 42) -> dict[int, list[str]]:
    rng = np.random.default_rng(seed)
    order_desc = np.argsort(-scores, kind="mergesort")
    order_asc = np.argsort(scores, kind="mergesort")
    groups: dict[int, list[str]] = {idx: [] for idx in range(len(scores))}

    for name, frac in RISK_GROUPS.items():
        n = max(1, int(np.ceil(len(scores) * frac)))
        selected = order_asc[:n] if name == "bottom_20" else order_desc[:n]
        for idx in selected:
            groups[int(idx)].append(name)

    n_random = max(1, int(np.ceil(len(scores) * RISK_GROUPS["top_20"])))
    random_selected = rng.choice(len(scores), size=n_random, replace=False)
    for idx in random_selected:
        groups[int(idx)].append("random_20")

    return groups


def save_risk_scores(
    output_path: Path,
    rows: list[dict[str, str]],
    y: np.ndarray,
    oof_scores: np.ndarray,
    full_scores: np.ndarray,
    seed: int,
) -> None:
    group_lookup = assign_risk_groups(oof_scores, seed=seed)
    sorted_indices = np.argsort(-oof_scores, kind="mergesort")
    ranks = np.empty(len(rows), dtype=int)
    for rank, idx in enumerate(sorted_indices, start=1):
        ranks[idx] = rank

    out_rows: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        risk_percentile = 1 - ((ranks[idx] - 1) / max(1, len(rows) - 1))
        out_rows.append(
            {
                **{col: row.get(col, "") for col in ID_COLUMNS},
                "negative_conformity": int(y[idx]),
                "oof_risk_score": f"{oof_scores[idx]:.6f}",
                "full_model_risk_score": f"{full_scores[idx]:.6f}",
                "risk_rank": int(ranks[idx]),
                "risk_percentile": f"{risk_percentile:.6f}",
                "risk_groups": ";".join(group_lookup[idx]),
                **{col: row.get(col, "") for col in FEATURES},
            }
        )

    fieldnames = [
        *ID_COLUMNS,
        "negative_conformity",
        "oof_risk_score",
        "full_model_risk_score",
        "risk_rank",
        "risk_percentile",
        "risk_groups",
        *FEATURES,
    ]
    save_csv(output_path, out_rows, fieldnames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=experiment_dir() / "data" / "negative_conformity_feature_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment_dir() / "results" / "sklearn_results",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, x, y = load_data(args.data)
    print(f"Loaded {len(y)} samples, {int(y.sum())} positive ({y.mean() * 100:.1f}%)")
    print(f"Features ({len(FEATURES)}): {FEATURES}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    lr_search = lr_grid(cv)
    lr_search.fit(x, y)
    print(f"LR best params: {plain_params(lr_search.best_params_)}")
    print(f"LR best CV ROC-AUC: {lr_search.best_score_:.4f}")

    lr_best = lr_search.best_estimator_
    lr_oof = oof_probabilities(lr_best, x, y, cv)
    lr_full_scores = lr_best.fit(x, y).predict_proba(x)[:, 1]
    lr_opt_thresh, lr_opt_f1 = optimal_f1_threshold(y, lr_oof)

    lr_coefs = []
    best_lr_clf = lr_best.named_steps["clf"]
    for feat, coef in sorted(zip(FEATURES, best_lr_clf.coef_[0]), key=lambda item: -abs(item[1])):
        lr_coefs.append({"feature": feat, "coefficient": f"{coef:.4f}", "abs_coef": f"{abs(coef):.4f}"})
    save_csv(args.output_dir / "lr_coefficients.csv", lr_coefs, ["feature", "coefficient", "abs_coef"])

    rf_search = rf_grid(cv)
    rf_search.fit(x, y)
    print(f"RF best params: {rf_search.best_params_}")
    print(f"RF best CV ROC-AUC: {rf_search.best_score_:.4f}")

    rf_best = rf_search.best_estimator_
    rf_oof = oof_probabilities(rf_best, x, y, cv)
    rf_full_scores = rf_best.fit(x, y).predict_proba(x)[:, 1]
    rf_opt_thresh, rf_opt_f1 = optimal_f1_threshold(y, rf_oof)

    rf_importances = []
    for feat, imp in sorted(zip(FEATURES, rf_best.feature_importances_), key=lambda item: -item[1]):
        rf_importances.append({"feature": feat, "importance": f"{imp:.4f}"})
    save_csv(args.output_dir / "rf_feature_importance.csv", rf_importances, ["feature", "importance"])

    metrics_rows = [
        evaluate_cv("LR", y, lr_oof, 0.5),
        evaluate_cv("LR (optimal threshold)", y, lr_oof, lr_opt_thresh),
        evaluate_cv("RF", y, rf_oof, 0.5),
        evaluate_cv("RF (optimal threshold)", y, rf_oof, rf_opt_thresh),
    ]
    save_csv(
        args.output_dir / "model_comparison.csv",
        metrics_rows,
        ["model", "threshold", "roc_auc", "accuracy", "precision", "recall", "f1", "n_positive", "n_total"],
    )

    thresholds = np.arange(0.10, 0.85, 0.05)
    save_csv(
        args.output_dir / "threshold_sweep.csv",
        threshold_sweep("LR", y, lr_oof, thresholds) + threshold_sweep("RF", y, rf_oof, thresholds),
        ["model", "threshold", "precision", "recall", "f1"],
    )

    save_risk_scores(
        args.output_dir / "risk_scores_lr.csv",
        rows,
        y,
        lr_oof,
        lr_full_scores,
        args.seed,
    )
    save_risk_scores(
        args.output_dir / "risk_scores_rf.csv",
        rows,
        y,
        rf_oof,
        rf_full_scores,
        args.seed,
    )

    best_params = {
        "main_model": "sklearn",
        "recommended_risk_score_file": "risk_scores_rf.csv",
        "lr_best_params": plain_params(lr_search.best_params_),
        "lr_best_cv_roc_auc": lr_search.best_score_,
        "lr_optimal_threshold": lr_opt_thresh,
        "lr_optimal_f1": lr_opt_f1,
        "rf_best_params": {key: str(value) for key, value in rf_search.best_params_.items()},
        "rf_best_cv_roc_auc": rf_search.best_score_,
        "rf_optimal_threshold": rf_opt_thresh,
        "rf_optimal_f1": rf_opt_f1,
        "features_used": FEATURES,
    }
    with (args.output_dir / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, default=str)

    print(f"LR OOF ROC-AUC={roc_auc_score(y, lr_oof):.4f}, F1@opt={lr_opt_f1:.4f}")
    print(f"RF OOF ROC-AUC={roc_auc_score(y, rf_oof):.4f}, F1@opt={rf_opt_f1:.4f}")
    print(f"Wrote sklearn model outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
