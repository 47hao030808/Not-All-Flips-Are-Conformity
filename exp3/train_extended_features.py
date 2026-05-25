#!/usr/bin/env python3
"""Module helpers for the experiment pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent

HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "likely", "unlikely",
    "probably", "suggest", "suggests", "seem", "seems", "appear", "appears",
    "approximately", "roughly", "somewhat", "fairly", "rather",
    "may", "believe", "think", "assume", "guess",
}

CERTAINTY_WORDS = {
    "clearly", "obviously", "certainly", "definitely", "undoubtedly",
    "must", "always", "never", "absolute", "absolutely", "sure", "confident",
    "without doubt", "no doubt", "evident", "indisputable",
}

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "cannot", "can't", "don't", "doesn't", "won't", "isn't", "aren't", "wasn't", "weren't"}


def extract_text_features(reasoning: str) -> dict[str, float]:
    """Extract linguistic features from Round 0 reasoning text."""
    text = reasoning.strip()
    words = re.findall(r'\b\w+\b', text.lower())
    n_words = max(len(words), 1)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sentences = max(len(sentences), 1)

    word_set = set(words)
    # bigrams for multi-word phrases
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]

    hedge_count = sum(1 for w in words if w in HEDGE_WORDS)
    certainty_count = sum(1 for w in words if w in CERTAINTY_WORDS)
    negation_count = sum(1 for w in words if w in NEGATION_WORDS)

    # Question marks in reasoning (sign of self-questioning)
    question_marks = text.count("?")

    # "However", "but", "although" - signs of weighing alternatives
    contrast_words = {"however", "but", "although", "though", "nevertheless",
                      "nonetheless", "yet", "whereas", "while", "despite"}
    contrast_count = sum(1 for w in words if w in contrast_words)

    # First-person markers ("I think", "I believe")
    first_person = sum(1 for w in words if w == "i")

    # Enumeration/structure markers ("first", "second", "step")
    structure_words = {"first", "second", "third", "finally", "step",
                       "therefore", "thus", "hence", "consequently"}
    structure_count = sum(1 for w in words if w in structure_words)

    # Average sentence length (proxy for reasoning complexity)
    avg_sent_len = n_words / n_sentences

    # Type-token ratio (vocabulary richness)
    ttr = len(set(words)) / n_words

    return {
        "hedge_ratio": hedge_count / n_words,
        "certainty_ratio": certainty_count / n_words,
        "negation_ratio": negation_count / n_words,
        "question_mark_count": question_marks,
        "contrast_ratio": contrast_count / n_words,
        "first_person_ratio": first_person / n_words,
        "structure_ratio": structure_count / n_words,
        "n_sentences": n_sentences,
        "avg_sentence_length": avg_sent_len,
        "type_token_ratio": ttr,
    }

def load_base_features(data_path: Path) -> tuple[list[dict], np.ndarray]:
    """Load the existing feature table."""
    with data_path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    y = np.array([int(r["negative_conformity"]) for r in rows])
    return rows, y


def load_reasoning_texts(root: Path) -> dict[tuple[str, str], str]:
    """Load round0 reasoning texts keyed by (question_id, agent_id)."""
    reasoning_path = root / "output" / "round0_reasoning.jsonl"
    texts = {}
    with reasoning_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            key = (str(obj["question_id"]).strip(), str(obj["agent_id"]).strip())
            texts[key] = obj.get("reasoning", "")
    return texts


def load_question_meta(root: Path) -> tuple[dict[str, str], dict[str, float]]:
    """Load question category and compute per-question accuracy."""
    # Category
    q_cat = {}
    with (root / "output" / "question_summary.csv").open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            q_cat[str(row["question_id"]).strip()] = row["type"].strip()

    # Accuracy from round0
    q_correct = defaultdict(list)
    with (root / "output" / "round0_raw.csv").open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            is_correct = row["is_correct"].strip().lower() == "true"
            q_correct[str(row["question_id"]).strip()].append(is_correct)

    q_acc = {qid: sum(v) / len(v) for qid, v in q_correct.items()}
    return q_cat, q_acc

BASELINE_FEATURES = [
    "round0_implicit_confidence",
    "relative_implicit_confidence",
    "implicit_confidence_rank_in_group",
    "round0_reasoning_length",
    "num_peers_same_as_target",
    "target_is_alone",
    "peer_wrong_majority_size",
    "answer_entropy",
]

TEXT_FEATURES = [
    "hedge_ratio",
    "certainty_ratio",
    "negation_ratio",
    "question_mark_count",
    "contrast_ratio",
    "first_person_ratio",
    "structure_ratio",
    "n_sentences",
    "avg_sentence_length",
    "type_token_ratio",
]

QUESTION_FEATURES = [
    "question_accuracy",
    # category will be target-encoded as "category_nc_rate"
    "category_nc_rate",
]

INTERACTION_FEATURES = [
    "confidence_x_peers_same",       # low confidence + few supporters = high risk
    "confidence_x_wrong_majority",   # low confidence + large wrong majority = high risk
    "alone_x_wrong_majority",        # alone + large wrong majority
]


def build_extended_table(
    base_rows: list[dict],
    y: np.ndarray,
    reasoning_texts: dict[tuple[str, str], str],
    q_cat: dict[str, str],
    q_acc: dict[str, float],
) -> dict[str, np.ndarray]:
    """Build feature matrices for each feature set variant."""

    n = len(base_rows)

    # A) Baseline
    X_base = np.array([
        [float(row[col]) for col in BASELINE_FEATURES]
        for row in base_rows
    ])

    # B) Question features
    # Target-encode category using global NC rate per category
    cat_nc = defaultdict(list)
    for row, label in zip(base_rows, y):
        cat_nc[q_cat.get(row["question_id"], "unknown")].append(label)
    cat_nc_rate = {cat: sum(v) / len(v) for cat, v in cat_nc.items()}
    global_nc_rate = y.mean()

    X_question = np.zeros((n, 2))
    for i, row in enumerate(base_rows):
        qid = row["question_id"]
        X_question[i, 0] = q_acc.get(qid, 0.75)
        cat = q_cat.get(qid, "unknown")
        X_question[i, 1] = cat_nc_rate.get(cat, global_nc_rate)

    # C) Text features
    X_text = np.zeros((n, len(TEXT_FEATURES)))
    missing_text = 0
    for i, row in enumerate(base_rows):
        key = (row["question_id"], row["agent_id"])
        reasoning = reasoning_texts.get(key, "")
        if not reasoning:
            missing_text += 1
        feats = extract_text_features(reasoning)
        for j, col in enumerate(TEXT_FEATURES):
            X_text[i, j] = feats[col]
    print(f"Text features: {missing_text}/{n} samples missing reasoning text")

    # D) Interaction features
    X_interact = np.zeros((n, 3))
    for i, row in enumerate(base_rows):
        conf = float(row["round0_implicit_confidence"])
        peers_same = float(row["num_peers_same_as_target"])
        wrong_maj = float(row["peer_wrong_majority_size"])
        alone = float(row["target_is_alone"])
        X_interact[i, 0] = conf * peers_same
        X_interact[i, 1] = conf * wrong_maj
        X_interact[i, 2] = alone * wrong_maj

    return {
        "A_baseline": X_base,
        "B_+question": np.hstack([X_base, X_question]),
        "C_+text": np.hstack([X_base, X_text]),
        "D_+interaction": np.hstack([X_base, X_interact]),
        "E_all": np.hstack([X_base, X_question, X_text, X_interact]),
    }


def get_feature_names(variant: str) -> list[str]:
    mapping = {
        "A_baseline": BASELINE_FEATURES,
        "B_+question": BASELINE_FEATURES + QUESTION_FEATURES,
        "C_+text": BASELINE_FEATURES + TEXT_FEATURES,
        "D_+interaction": BASELINE_FEATURES + INTERACTION_FEATURES,
        "E_all": BASELINE_FEATURES + QUESTION_FEATURES + TEXT_FEATURES + INTERACTION_FEATURES,
    }
    return mapping[variant]

def run_cv_experiment(
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    variant_name: str,
    feature_names: list[str],
) -> dict[str, object]:
    """Run LR and RF with grid search, return summary metrics."""

    results = {}

    for model_name, Model, param_grid, needs_scaling in [
        ("LR", LogisticRegression, {
            "C": [0.01, 0.1, 1.0],
            "class_weight": [None, "balanced"],
        }, True),
        ("RF", RandomForestClassifier, {
            "n_estimators": [200],
            "max_depth": [5, 8],
            "min_samples_leaf": [10, 20],
            "class_weight": [None, "balanced"],
        }, False),
    ]:
        X_use = X.copy()
        if needs_scaling:
            scaler = StandardScaler()
            X_use = scaler.fit_transform(X_use)

        extra = {"max_iter": 2000, "solver": "lbfgs"} if model_name == "LR" else {}

        grid = GridSearchCV(
            Model(random_state=42, **extra),
            param_grid,
            cv=cv,
            scoring="roc_auc",
            refit=True,
            n_jobs=-1,
        )
        grid.fit(X_use, y)

        # Out-of-fold predictions
        y_prob = np.zeros(len(y))
        for train_idx, test_idx in cv.split(X_use, y):
            if needs_scaling:
                sc = StandardScaler()
                X_train = sc.fit_transform(X_use[train_idx])
                X_test = sc.transform(X_use[test_idx])
            else:
                X_train, X_test = X_use[train_idx], X_use[test_idx]

            model = Model(random_state=42, **extra, **grid.best_params_)
            model.fit(X_train, y[train_idx])
            y_prob[test_idx] = model.predict_proba(X_test)[:, 1]

        # Optimal F1 threshold
        best_f1, best_thresh = 0, 0.5
        for t in np.arange(0.10, 0.80, 0.01):
            f1 = f1_score(y, (y_prob >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t

        y_pred_default = (y_prob >= 0.5).astype(int)
        y_pred_optimal = (y_prob >= best_thresh).astype(int)

        roc = roc_auc_score(y, y_prob)

        results[model_name] = {
            "roc_auc": roc,
            "f1_default": f1_score(y, y_pred_default, zero_division=0),
            "f1_optimal": best_f1,
            "optimal_threshold": best_thresh,
            "precision_opt": precision_score(y, y_pred_optimal, zero_division=0),
            "recall_opt": recall_score(y, y_pred_optimal, zero_division=0),
            "best_params": grid.best_params_,
        }

        # Feature importance
        if model_name == "RF":
            imp = grid.best_estimator_.feature_importances_
            results["rf_importance"] = sorted(
                zip(feature_names, imp), key=lambda x: -x[1]
            )
        elif model_name == "LR":
            coefs = grid.best_estimator_.coef_[0]
            results["lr_coefs"] = sorted(
                zip(feature_names, coefs), key=lambda x: -abs(x[1])
            )

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root())
    args = parser.parse_args()

    root = args.root
    exp_dir = root / "experiments" / "exp03_confidence_prediction"
    data_path = exp_dir / "data" / "negative_conformity_feature_table.csv"
    output_dir = exp_dir / "results" / "extended_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data sources
    print("Loading data...")
    base_rows, y = load_base_features(data_path)
    reasoning_texts = load_reasoning_texts(root)
    q_cat, q_acc = load_question_meta(root)
    print(f"Samples: {len(y)}, Positive: {y.sum()} ({y.mean()*100:.1f}%)")

    # Build feature matrices
    feature_sets = build_extended_table(base_rows, y, reasoning_texts, q_cat, q_acc)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Run experiments for each variant
    summary_rows = []

    for variant, X in feature_sets.items():
        feat_names = get_feature_names(variant)
        print(f"\n{'='*60}")
        print(f"Variant: {variant} ({X.shape[1]} features)")
        print(f"{'='*60}")

        results = run_cv_experiment(X, y, cv, variant, feat_names)

        for model_name in ["LR", "RF"]:
            r = results[model_name]
            row = {
                "variant": variant,
                "model": model_name,
                "n_features": X.shape[1],
                "roc_auc": f"{r['roc_auc']:.4f}",
                "f1_default": f"{r['f1_default']:.4f}",
                "f1_optimal": f"{r['f1_optimal']:.4f}",
                "optimal_threshold": f"{r['optimal_threshold']:.2f}",
                "precision_opt": f"{r['precision_opt']:.4f}",
                "recall_opt": f"{r['recall_opt']:.4f}",
                "best_params": str(r["best_params"]),
            }
            summary_rows.append(row)
            print(f"  {model_name}: ROC-AUC={r['roc_auc']:.4f}  "
                  f"F1@0.5={r['f1_default']:.4f}  "
                  f"F1@opt={r['f1_optimal']:.4f} (t={r['optimal_threshold']:.2f})")

        # Print feature importance for 'all' variant
        if variant == "E_all":
            print("\n  LR Coefficients (|coef| sorted):")
            for feat, coef in results["lr_coefs"][:10]:
                print(f"    {feat:40s} {coef:+.4f}")
            print("\n  RF Feature Importance:")
            for feat, imp in results["rf_importance"][:10]:
                print(f"    {feat:40s} {imp:.4f}")

    # Save summary
    fieldnames = ["variant", "model", "n_features", "roc_auc",
                  "f1_default", "f1_optimal", "optimal_threshold",
                  "precision_opt", "recall_opt", "best_params"]
    with (output_dir / "ablation_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Print comparison table
    print(f"\n{'='*60}")
    print("ABLATION COMPARISON")
    print(f"{'='*60}")
    print(f"{'Variant':<20} {'Model':<5} {'#Feat':>5} {'ROC-AUC':>8} {'F1@0.5':>7} {'F1@opt':>7}")
    print("-" * 60)
    for row in summary_rows:
        print(f"{row['variant']:<20} {row['model']:<5} {row['n_features']:>5} "
              f"{row['roc_auc']:>8} {row['f1_default']:>7} {row['f1_optimal']:>7}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
