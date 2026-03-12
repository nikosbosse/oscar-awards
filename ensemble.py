"""
ensemble.py — Ensemble methods for Oscar prediction
====================================================

Combines predictions from all base models (precursor + ML) into ensemble
forecasts using three strategies:

  A. Weighted Ensemble  — weight each model's votes by its confidence
  B. Rank Ensemble      — average candidate ranks across models
  C. Equal Ensemble     — sum confidence scores with equal model weights

Also provides CSV-based backtesting that reuses saved backtest_results.csv
to evaluate ensemble performance without re-running all base models.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from collections import defaultdict

from helpers import (
    OSCAR_CATEGORIES,
    PREDICTION_YEAR,
    SCRIPT_DIR,
    load_data,
    build_category_mapping,
    compute_historical_accuracy,
    names_match,
)
from predictions import (
    Prediction,
    ALL_MODELS,
    model_weighted_precursor,
    model_consensus_momentum,
    model_logistic_regression,
    model_random_forest,
    model_gradient_boosting,
)

# ---------------------------------------------------------------------------
# Base models only (exclude ensemble / enhanced models to avoid recursion)
# ---------------------------------------------------------------------------

BASE_MODELS = {
    "weighted": model_weighted_precursor,
    "momentum": model_consensus_momentum,
    "logreg": model_logistic_regression,
    "rf": model_random_forest,
    "gbm": model_gradient_boosting,
}


# ---------------------------------------------------------------------------
# Helpers: cluster candidates across models
# ---------------------------------------------------------------------------

def _cluster_candidate_scores(
    candidate_scores: list[tuple[str, float]],
) -> dict[str, float]:
    """
    Merge candidate entries that refer to the same entity (via names_match).
    Returns {canonical_name: total_score}.
    """
    clusters: dict[str, float] = {}

    for name, score in candidate_scores:
        matched_key = None
        for existing in clusters:
            if names_match(name, existing):
                matched_key = existing
                break
        if matched_key is not None:
            clusters[matched_key] += score
            # Keep longer name as canonical
            if len(name) > len(matched_key):
                merged_score = clusters.pop(matched_key)
                clusters[name] = merged_score
        else:
            clusters[name] = score

    return clusters


def _cluster_candidate_ranks(
    candidate_ranks: list[tuple[str, list[int]]],
) -> dict[str, list[int]]:
    """
    Merge candidate rank entries that refer to the same entity.
    Returns {canonical_name: [list of ranks across models]}.
    """
    clusters: dict[str, list[int]] = {}

    for name, ranks in candidate_ranks:
        matched_key = None
        for existing in clusters:
            if names_match(name, existing):
                matched_key = existing
                break
        if matched_key is not None:
            clusters[matched_key].extend(ranks)
            if len(name) > len(matched_key):
                old_ranks = clusters.pop(matched_key)
                old_ranks.extend(ranks)
                clusters[name] = old_ranks
        else:
            clusters[name] = list(ranks)

    return clusters


# ============================================================================
# Strategy A: Accuracy-Weighted Ensemble
# ============================================================================

# Map from backtest model names to BASE_MODELS keys
_BT_NAME_TO_KEY = {
    "Weighted": "weighted",
    "Momentum": "momentum",
    "LogReg": "logreg",
    "RandomForest": "rf",
    "GBM": "gbm",
}


def _load_model_weights() -> dict[str, float]:
    """Load per-model backtest accuracy as ensemble weights.

    Falls back to equal weights if backtest_results.csv is unavailable.
    """
    import os
    bt_path = SCRIPT_DIR / "backtest_results.csv"
    if not os.path.exists(bt_path):
        return {k: 1.0 for k in BASE_MODELS}

    bt = pd.read_csv(bt_path)
    bt["Correct"] = bt["Correct"].map(
        lambda v: v if isinstance(v, bool) else str(v).strip().lower() == "true"
    )

    weights = {}
    for bt_name, key in _BT_NAME_TO_KEY.items():
        if key in BASE_MODELS:
            rows = bt[bt["Model"] == bt_name]
            weights[key] = rows["Correct"].mean() if len(rows) > 0 else 0.5
    # Any base model not in the CSV gets default weight
    for key in BASE_MODELS:
        if key not in weights:
            weights[key] = 0.5
    return weights


def model_weighted_ensemble(
    df: pd.DataFrame,
    category_mapping: dict,
    accuracy: dict,
    prediction_year: int = PREDICTION_YEAR,
) -> list[Prediction]:
    """
    Run all base models, then for each category combine candidate scores
    weighted by each model's historical backtest accuracy.
    """
    model_weights = _load_model_weights()

    # Collect predictions from all base models
    all_preds: dict[str, list[Prediction]] = {}
    for key, model_fn in BASE_MODELS.items():
        all_preds[key] = model_fn(
            df, category_mapping, accuracy, prediction_year=prediction_year,
        )

    predictions = []
    for oscar_cat in OSCAR_CATEGORIES:
        candidate_scores: list[tuple[str, float]] = []

        for key, preds in all_preds.items():
            p = next((x for x in preds if x.oscar_category == oscar_cat), None)
            if p is None or p.confidence <= 0:
                continue

            w = model_weights.get(key, 0.5)
            for cand_name, cand_pct in p.all_candidates:
                candidate_scores.append((cand_name, (cand_pct / 100.0) * w))

        clustered = _cluster_candidate_scores(candidate_scores)

        if not clustered:
            predictions.append(Prediction(
                oscar_category=oscar_cat,
                predicted_winner="N/A — No model data",
                confidence=0.0,
                all_candidates=[],
                model_name="Weighted Ensemble",
            ))
            continue

        ranked = sorted(clustered.items(), key=lambda x: -x[1])
        total = sum(s for _, s in ranked)
        if total == 0:
            total = 1.0

        top_name, top_score = ranked[0]
        top_conf = top_score / total

        runner_up = ranked[1][0] if len(ranked) > 1 else ""
        runner_up_conf = ranked[1][1] / total if len(ranked) > 1 else 0.0

        all_candidates = [
            (name, round(score / total * 100, 1))
            for name, score in ranked
        ]

        predictions.append(Prediction(
            oscar_category=oscar_cat,
            predicted_winner=top_name,
            confidence=top_conf,
            all_candidates=all_candidates,
            model_name="Weighted Ensemble",
            runner_up=runner_up,
            runner_up_confidence=runner_up_conf,
        ))

    return predictions


# ============================================================================
# Strategy B: Rank-Based Ensemble
# ============================================================================

def model_rank_ensemble(
    df: pd.DataFrame,
    category_mapping: dict,
    accuracy: dict,
    prediction_year: int = PREDICTION_YEAR,
) -> list[Prediction]:
    """
    Run all base models, rank candidates within each model, then average
    ranks across models. Lowest average rank wins.
    """
    all_preds: dict[str, list[Prediction]] = {}
    for key, model_fn in BASE_MODELS.items():
        all_preds[key] = model_fn(
            df, category_mapping, accuracy, prediction_year=prediction_year,
        )

    predictions = []
    for oscar_cat in OSCAR_CATEGORIES:
        # Collect per-model rankings and track which models contributed
        candidate_ranks_raw: list[tuple[str, list[int]]] = []
        n_models_contributing = 0

        for key, preds in all_preds.items():
            p = next((x for x in preds if x.oscar_category == oscar_cat), None)
            if p is None or p.confidence <= 0:
                continue

            n_models_contributing += 1
            # all_candidates is sorted best-first; assign rank 1, 2, ...
            for rank_pos, (cand_name, _) in enumerate(p.all_candidates, start=1):
                candidate_ranks_raw.append((cand_name, [rank_pos]))

        clustered = _cluster_candidate_ranks(candidate_ranks_raw)

        if not clustered:
            predictions.append(Prediction(
                oscar_category=oscar_cat,
                predicted_winner="N/A — No model data",
                confidence=0.0,
                all_candidates=[],
                model_name="Rank Ensemble",
            ))
            continue

        # Penalize candidates missing from some models: impute missing
        # ranks as (max_observed_rank + 1) for each model that didn't
        # include them
        max_rank = max(r for ranks in clustered.values() for r in ranks)
        penalty_rank = max_rank + 1
        for name, ranks in clustered.items():
            n_missing = n_models_contributing - len(ranks)
            if n_missing > 0:
                ranks.extend([penalty_rank] * n_missing)

        # Average rank per candidate (lower is better)
        avg_ranks = {
            name: np.mean(ranks) for name, ranks in clustered.items()
        }
        # Sort ascending (best rank first)
        ranked = sorted(avg_ranks.items(), key=lambda x: x[1])

        # Convert average rank to a confidence-like score
        # Use inverse rank normalised across candidates
        inv_scores = [(name, 1.0 / avg_rank) for name, avg_rank in ranked]
        total_inv = sum(s for _, s in inv_scores)
        if total_inv == 0:
            total_inv = 1.0

        top_name = ranked[0][0]
        top_conf = inv_scores[0][1] / total_inv

        runner_up = ranked[1][0] if len(ranked) > 1 else ""
        runner_up_conf = inv_scores[1][1] / total_inv if len(ranked) > 1 else 0.0

        all_candidates = [
            (name, round(inv / total_inv * 100, 1))
            for name, inv in inv_scores
        ]

        predictions.append(Prediction(
            oscar_category=oscar_cat,
            predicted_winner=top_name,
            confidence=top_conf,
            all_candidates=all_candidates,
            model_name="Rank Ensemble",
            runner_up=runner_up,
            runner_up_confidence=runner_up_conf,
        ))

    return predictions


# ============================================================================
# Strategy C: Equal-Weights Ensemble
# ============================================================================

def model_equal_ensemble(
    df: pd.DataFrame,
    category_mapping: dict,
    accuracy: dict,
    prediction_year: int = PREDICTION_YEAR,
) -> list[Prediction]:
    """
    Run all base models with equal weight (1.0 each). For each category,
    sum up confidence scores across models for each candidate.
    """
    all_preds: dict[str, list[Prediction]] = {}
    for key, model_fn in BASE_MODELS.items():
        all_preds[key] = model_fn(
            df, category_mapping, accuracy, prediction_year=prediction_year,
        )

    predictions = []
    for oscar_cat in OSCAR_CATEGORIES:
        candidate_scores: list[tuple[str, float]] = []

        for key, preds in all_preds.items():
            p = next((x for x in preds if x.oscar_category == oscar_cat), None)
            if p is None or p.confidence <= 0:
                continue

            # Each model contributes 1.0 to its top pick's score,
            # scaled by candidate confidence within that model
            for cand_name, cand_pct in p.all_candidates:
                candidate_scores.append((cand_name, cand_pct / 100.0))

        clustered = _cluster_candidate_scores(candidate_scores)

        if not clustered:
            predictions.append(Prediction(
                oscar_category=oscar_cat,
                predicted_winner="N/A — No model data",
                confidence=0.0,
                all_candidates=[],
                model_name="Equal Ensemble",
            ))
            continue

        ranked = sorted(clustered.items(), key=lambda x: -x[1])
        total = sum(s for _, s in ranked)
        if total == 0:
            total = 1.0

        top_name, top_score = ranked[0]
        top_conf = top_score / total

        runner_up = ranked[1][0] if len(ranked) > 1 else ""
        runner_up_conf = ranked[1][1] / total if len(ranked) > 1 else 0.0

        all_candidates = [
            (name, round(score / total * 100, 1))
            for name, score in ranked
        ]

        predictions.append(Prediction(
            oscar_category=oscar_cat,
            predicted_winner=top_name,
            confidence=top_conf,
            all_candidates=all_candidates,
            model_name="Equal Ensemble",
            runner_up=runner_up,
            runner_up_confidence=runner_up_conf,
        ))

    return predictions


# ============================================================================
# Backtesting from CSV
# ============================================================================

def backtest_ensemble_from_csv(
    bt_path: str | None = None,
) -> pd.DataFrame:
    """
    Load backtest_results.csv and compute ensemble performance.

    For each held-out year: compute model weights from accuracy on all
    other years, then combine each model's Predicted + Confidence using
    weighted voting. Pick the candidate with highest combined score.

    Returns DataFrame with same schema as backtest results:
      Year, Category, Predicted, Actual, Correct, Confidence, Model
    """
    if bt_path is None:
        bt_path = str(SCRIPT_DIR / "backtest_results.csv")

    bt = pd.read_csv(bt_path)
    bt["Correct"] = bt["Correct"].astype(bool)

    years = sorted(bt["Year"].unique())
    models = sorted(bt["Model"].unique())
    categories = sorted(bt["Category"].unique())

    results = []

    for year in years:
        # Compute model weights from accuracy on all OTHER years
        other_years = bt[bt["Year"] != year]
        model_accuracy = {}
        for model in models:
            model_rows = other_years[other_years["Model"] == model]
            if len(model_rows) > 0:
                model_accuracy[model] = model_rows["Correct"].mean()
            else:
                model_accuracy[model] = 0.0

        # For this year, combine predictions across models per category
        year_bt = bt[bt["Year"] == year]

        for cat in categories:
            cat_rows = year_bt[year_bt["Category"] == cat]
            if len(cat_rows) == 0:
                continue

            actual = cat_rows.iloc[0]["Actual"]

            # Weighted voting: accumulate scores per candidate
            candidate_scores: list[tuple[str, float]] = []
            for _, row in cat_rows.iterrows():
                model = row["Model"]
                predicted = row["Predicted"]
                confidence = row["Confidence"]
                weight = model_accuracy.get(model, 0.0)
                candidate_scores.append((predicted, confidence * weight))

            # Cluster candidates
            clustered = _cluster_candidate_scores(candidate_scores)

            if not clustered:
                continue

            ranked = sorted(clustered.items(), key=lambda x: -x[1])
            top_name, top_score = ranked[0]
            total = sum(s for _, s in ranked)
            conf = top_score / total if total > 0 else 0.0

            correct = names_match(top_name, actual) if top_name else False

            results.append({
                "Year": year,
                "Category": cat,
                "Predicted": top_name[:50],
                "Actual": actual[:50] if isinstance(actual, str) else actual,
                "Correct": correct,
                "Confidence": conf,
                "Model": "Weighted Ensemble (BT)",
            })

    return pd.DataFrame(results)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    import os

    # --- Part 1: Backtest from CSV ---
    bt_path = SCRIPT_DIR / "backtest_results.csv"
    if os.path.exists(bt_path):
        print("=" * 100)
        print("  ENSEMBLE BACKTEST — from backtest_results.csv")
        print("=" * 100)

        bt_ensemble = backtest_ensemble_from_csv(str(bt_path))

        if len(bt_ensemble) > 0:
            # Load original backtest for comparison
            bt_orig = pd.read_csv(bt_path)
            bt_orig["Correct"] = bt_orig["Correct"].astype(bool)

            # Per-model accuracy from original
            orig_acc = bt_orig.groupby("Model")["Correct"].mean().sort_values(ascending=False)

            print("\n  BASE MODEL ACCURACY (backtest):")
            print(f"  {'Model':<25} {'Accuracy':>10}")
            print(f"  {'─'*24} {'─'*10}")
            for model, acc in orig_acc.items():
                print(f"  {model:<25} {acc:>9.1%}")

            # Ensemble accuracy
            ens_acc = bt_ensemble["Correct"].mean()
            ens_correct = bt_ensemble["Correct"].sum()
            ens_total = len(bt_ensemble)
            print(f"\n  {'Weighted Ensemble (BT)':<25} {ens_acc:>9.1%}  ({ens_correct}/{ens_total})")

            # Per-category breakdown
            print("\n  PER-CATEGORY ENSEMBLE ACCURACY:")
            cat_acc = bt_ensemble.groupby("Category")["Correct"].agg(["mean", "sum", "count"])
            cat_acc.columns = ["Accuracy", "Correct", "Total"]
            cat_acc = cat_acc.sort_values("Accuracy", ascending=False)
            print(f"  {'Category':<35} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
            print(f"  {'─'*34} {'─'*10} {'─'*10} {'─'*8}")
            for cat, row in cat_acc.iterrows():
                print(f"  {cat:<35} {row['Accuracy']:>9.1%} {int(row['Correct']):>10} {int(row['Total']):>8}")
        else:
            print("  No ensemble backtest results produced.")
    else:
        print(f"  Backtest CSV not found at {bt_path}; skipping backtest.")

    # --- Part 2: Forward predictions for 2026 ---
    print()
    print("=" * 100)
    print(f"  ENSEMBLE PREDICTIONS — {PREDICTION_YEAR}")
    print("=" * 100)

    print("\nLoading data...")
    df = load_data()
    category_mapping = build_category_mapping()

    print("Computing historical accuracy...")
    accuracy = compute_historical_accuracy(df, category_mapping)

    ensemble_strategies = [
        ("Weighted Ensemble", model_weighted_ensemble),
        ("Rank Ensemble", model_rank_ensemble),
        ("Equal Ensemble", model_equal_ensemble),
    ]

    all_results: dict[str, list[Prediction]] = {}
    for name, fn in ensemble_strategies:
        print(f"\n  Running: {name}...")
        preds = fn(df, category_mapping, accuracy, prediction_year=PREDICTION_YEAR)
        all_results[name] = preds

    # Print comparison across ensemble strategies
    key_categories = [
        "Best Picture", "Best Director", "Best Actor", "Best Actress",
        "Best Supporting Actor", "Best Supporting Actress",
    ]

    for cat in key_categories:
        print(f"\n{'─' * 100}")
        print(f"  {cat}")
        print(f"{'─' * 100}")
        print(f"  {'Strategy':<25} {'Prediction':<45} {'Conf':>6}  {'Runner-up':<30}")
        print(f"  {'─'*24} {'─'*44} {'─'*6}  {'─'*30}")

        for name, preds in all_results.items():
            p = next((x for x in preds if x.oscar_category == cat), None)
            if p is None:
                continue
            pred_str = p.predicted_winner[:43]
            ru_str = (
                f"{p.runner_up[:25]} ({round(p.runner_up_confidence * 100)}%)"
                if p.runner_up else "—"
            )
            print(
                f"  {name:<25} {pred_str:<45} "
                f"{p.confidence_pct:>5.1f}%  {ru_str:<30}"
            )

    # Agreement summary across ensemble strategies
    print(f"\n{'=' * 100}")
    print("  ENSEMBLE AGREEMENT SUMMARY")
    print(f"{'=' * 100}")
    print(f"  {'Category':<35} {'All agree?':<12} {'Consensus pick':<45}")
    print(f"  {'─'*34} {'─'*11} {'─'*44}")

    for cat in OSCAR_CATEGORIES:
        picks = []
        for name, preds in all_results.items():
            p = next((x for x in preds if x.oscar_category == cat), None)
            if p and p.confidence > 0:
                picks.append(p.predicted_winner)

        if not picks:
            continue

        clustered_counts: dict[str, int] = defaultdict(int)
        for pick in picks:
            matched = False
            for existing in list(clustered_counts.keys()):
                if names_match(pick, existing):
                    clustered_counts[existing] += 1
                    matched = True
                    break
            if not matched:
                clustered_counts[pick] += 1

        top_pick = max(clustered_counts, key=clustered_counts.get)
        top_count = clustered_counts[top_pick]
        unanimous = top_count == len(picks)
        agreement_str = "YES" if unanimous else f"{top_count}/{len(picks)}"

        print(f"  {cat:<35} {agreement_str:<12} {top_pick[:43]:<45}")


if __name__ == "__main__":
    main()
