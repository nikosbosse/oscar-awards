"""
ensemble.py — Ensemble probability aggregation
================================================

Combines P(frontrunner wins) estimates from individual ML models using:

  1. Mean Ensemble      — arithmetic mean of probabilities
  2. Log-Odds Ensemble  — geometric mean of log odds (log-linear opinion pool)

All base models predict the same frontrunner (identified by the weighted
precursor model), so the ensemble only affects confidence, not the pick.
"""

from __future__ import annotations

import numpy as np

from helpers import (
    OSCAR_CATEGORIES,
    PREDICTION_YEAR,
    SCRIPT_DIR,
    load_data,
    build_category_mapping,
    compute_historical_accuracy,
)
from predictions import (
    Prediction,
    model_logistic_regression,
    model_gradient_boosting,
)


# ---------------------------------------------------------------------------
# Base models for ensembling (ML models only — they output calibrated P(win))
# ---------------------------------------------------------------------------

BASE_MODELS = {
    "logreg": model_logistic_regression,
    "gbm": model_gradient_boosting,
}


# ---------------------------------------------------------------------------
# Probability aggregation
# ---------------------------------------------------------------------------

def _aggregate_probabilities(probs: list[float], method: str = "mean") -> float:
    """Aggregate probability estimates from multiple models."""
    if not probs:
        return 0.0

    if method == "mean":
        return sum(probs) / len(probs)

    if method == "logodds":
        # Log-linear opinion pool: average log-odds, convert back
        eps = 1e-6
        clamped = [max(eps, min(1 - eps, p)) for p in probs]
        log_odds = [np.log(p / (1 - p)) for p in clamped]
        mean_log_odds = sum(log_odds) / len(log_odds)
        return float(1.0 / (1.0 + np.exp(-mean_log_odds)))

    raise ValueError(f"Unknown aggregation method: {method}")


# ---------------------------------------------------------------------------
# Ensemble model functions (same interface as all other models)
# ---------------------------------------------------------------------------

def _build_ensemble(
    df, category_mapping, accuracy, method: str, model_name: str, **kw,
) -> list[Prediction]:
    """Generic ensemble: run base models, aggregate P(frontrunner wins)."""
    base_results = {
        key: fn(df, category_mapping, accuracy, **kw)
        for key, fn in BASE_MODELS.items()
    }

    predictions = []
    for oscar_cat in OSCAR_CATEGORIES:
        probs = []
        frontrunner = None

        for key, preds in base_results.items():
            p = next((x for x in preds if x.oscar_category == oscar_cat), None)
            if p and p.confidence > 0:
                probs.append(p.confidence)
                if frontrunner is None:
                    frontrunner = p.predicted_winner

        if not probs or not frontrunner:
            predictions.append(Prediction(
                oscar_category=oscar_cat,
                predicted_winner="N/A",
                confidence=0.0,
                all_candidates=[],
                model_name=model_name,
            ))
            continue

        agg_prob = _aggregate_probabilities(probs, method)

        predictions.append(Prediction(
            oscar_category=oscar_cat,
            predicted_winner=frontrunner,
            confidence=agg_prob,
            all_candidates=[(frontrunner, round(agg_prob * 100, 1))],
            model_name=model_name,
            details={
                "base_probs": {
                    key: next(
                        (p.confidence for p in v if p.oscar_category == oscar_cat),
                        0,
                    )
                    for key, v in base_results.items()
                },
                "method": method,
            },
        ))

    return predictions


def model_mean_ensemble(df, category_mapping, accuracy, **kw):
    """Arithmetic mean of P(frontrunner wins) across ML models."""
    return _build_ensemble(
        df, category_mapping, accuracy, "mean", "Mean Ensemble", **kw,
    )


def model_logodds_ensemble(df, category_mapping, accuracy, **kw):
    """Geometric mean of log odds of P(frontrunner wins)."""
    return _build_ensemble(
        df, category_mapping, accuracy, "logodds", "Log-Odds Ensemble", **kw,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data...")
    df = load_data()
    category_mapping = build_category_mapping()

    print("Computing historical accuracy...")
    accuracy = compute_historical_accuracy(df, category_mapping)

    print(f"\nRunning ensemble predictions for {PREDICTION_YEAR}...")

    key_categories = [
        "Best Picture", "Best Director", "Best Actor", "Best Actress",
        "Best Supporting Actor", "Best Supporting Actress",
    ]

    for name, fn in [("Mean Ensemble", model_mean_ensemble),
                     ("Log-Odds Ensemble", model_logodds_ensemble)]:
        print(f"\n{'=' * 80}")
        print(f"  {name}")
        print(f"{'=' * 80}")

        preds = fn(df, category_mapping, accuracy, prediction_year=PREDICTION_YEAR)

        for p in preds:
            if p.oscar_category not in key_categories:
                continue
            base = p.details.get("base_probs", {})
            base_str = ", ".join(f"{k}={v:.1%}" for k, v in base.items())
            print(f"  {p.oscar_category:<30} {p.predicted_winner:<40} "
                  f"{p.confidence_pct:>5.1f}%  [{base_str}]")


if __name__ == "__main__":
    main()
