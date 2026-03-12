"""
enhanced_features.py — Enhanced ML features for Oscar prediction models
=======================================================================

Extends the base ML models from predictions.py with engineered features
derived from precursor accuracy statistics:

  - n_precursors_won:       total precursor wins per candidate
  - won_top_precursor:      1 if candidate won the single most accurate precursor
  - weighted_precursor_sum: dot product of wins with accuracy weights
  - n_top3_precursors_won:  wins among the top-3 most accurate precursors

These features give the ML models richer signal beyond the raw binary
precursor-win indicators.
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from predictions import (
    Prediction,
    build_candidate_features,
    _make_sklearn_model,
    _SklearnAdapter,
    backtest_ml_model,
    _ml_model_predict,
)
from helpers import (
    OSCAR_CATEGORIES,
    PREDICTION_YEAR,
    HISTORICAL_START,
    HISTORICAL_END,
    SCRIPT_DIR,
    load_data,
    build_category_mapping,
    build_winner_lookup,
    compute_historical_accuracy,
    names_match,
    PrecursorAccuracy,
)


# ---------------------------------------------------------------------------
# Enhanced feature construction
# ---------------------------------------------------------------------------

def build_enhanced_candidate_features(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    accuracy: dict[str, list[PrecursorAccuracy]],
    oscar_cat: str,
    years: list[int],
    require_target: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[int, str]]]:
    """
    Build candidate-level feature matrix with enhanced features.

    Starts from the base binary features (from build_candidate_features),
    then appends four engineered columns derived from precursor accuracy.

    Returns:
      X_enhanced: (n_candidates, n_base_features + 4)
      y:          (n_candidates,) binary target
      enhanced_feature_names: list of feature name strings
      meta:       list of (year, candidate_name)
    """
    X, y, feature_names, meta = build_candidate_features(
        df, category_mapping, oscar_cat, years, require_target=require_target,
    )

    if X.shape[0] == 0:
        enhanced_names = feature_names + [
            "n_precursors_won", "won_top_precursor",
            "weighted_precursor_sum", "n_top3_precursors_won",
        ]
        n_cols = len(feature_names) + 4
        return np.empty((0, n_cols)), y, enhanced_names, meta

    precursors = category_mapping.get(oscar_cat, [])
    pa_list = accuracy.get(oscar_cat, [])

    # Build accuracy lookup: (award, precursor_category) -> accuracy float
    acc_by_key = {}
    for pa in pa_list:
        acc_by_key[(pa.award, pa.precursor_category)] = pa.accuracy

    # Build accuracy weight vector aligned with feature columns
    n_precursors = len(precursors)
    acc_weights = np.zeros(n_precursors)
    for i, (award_name, prec_cat) in enumerate(precursors):
        acc_weights[i] = acc_by_key.get((award_name, prec_cat), 0.0)

    # --- Feature 1: n_precursors_won (row sum of binary features) ---
    n_precursors_won = X.sum(axis=1).reshape(-1, 1)

    # --- Feature 2: won_top_precursor ---
    # Find the precursor index with highest accuracy
    if n_precursors > 0 and acc_weights.max() > 0:
        top_idx = int(np.argmax(acc_weights))
        won_top = X[:, top_idx].reshape(-1, 1)
    else:
        won_top = np.zeros((X.shape[0], 1))

    # --- Feature 3: weighted_precursor_sum (dot product with accuracy) ---
    weighted_sum = (X * acc_weights).sum(axis=1).reshape(-1, 1)

    # --- Feature 4: n_top3_precursors_won ---
    if n_precursors >= 3:
        top3_indices = np.argsort(acc_weights)[-3:]
        n_top3 = X[:, top3_indices].sum(axis=1).reshape(-1, 1)
    elif n_precursors > 0:
        # Fewer than 3 precursors — use all of them
        n_top3 = X.sum(axis=1).reshape(-1, 1)
    else:
        n_top3 = np.zeros((X.shape[0], 1))

    X_enhanced = np.hstack([X, n_precursors_won, won_top, weighted_sum, n_top3])

    enhanced_feature_names = feature_names + [
        "n_precursors_won", "won_top_precursor",
        "weighted_precursor_sum", "n_top3_precursors_won",
    ]

    return X_enhanced, y, enhanced_feature_names, meta


# ---------------------------------------------------------------------------
# Enhanced ML model prediction wrapper
# ---------------------------------------------------------------------------

def _enhanced_ml_model_predict(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    accuracy: dict[str, list[PrecursorAccuracy]],
    model_class,
    model_kwargs: dict,
    model_name: str,
    prediction_year: int = PREDICTION_YEAR,
) -> list[Prediction]:
    """
    Generic wrapper for enhanced ML models. Mirrors _ml_model_predict but
    uses build_enhanced_candidate_features for richer feature vectors.
    """
    predictions = []

    for oscar_cat in OSCAR_CATEGORIES:
        train_years = list(range(HISTORICAL_START, prediction_year))

        X_train, y_train, feat_names, meta_train = build_enhanced_candidate_features(
            df, category_mapping, accuracy, oscar_cat, train_years,
        )

        X_pred, _, _, meta_pred = build_enhanced_candidate_features(
            df, category_mapping, accuracy, oscar_cat, [prediction_year],
            require_target=False,
        )

        if len(X_train) < 10 or len(X_pred) == 0:
            predictions.append(Prediction(
                oscar_category=oscar_cat,
                predicted_winner="N/A — Insufficient data",
                confidence=0.0,
                all_candidates=[],
                model_name=model_name,
            ))
            continue

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_pred)

        scored = sorted(
            zip(meta_pred, probs),
            key=lambda x: -x[1],
        )

        total = sum(p for _, p in scored)
        if total == 0:
            total = 1.0

        all_candidates = [
            (name, round(prob / total * 100, 1))
            for (year, name), prob in scored
        ]

        top_name = scored[0][0][1]
        top_conf = scored[0][1] / total
        runner_up = scored[1][0][1] if len(scored) > 1 else ""
        runner_up_conf = scored[1][1] / total if len(scored) > 1 else 0.0

        predictions.append(Prediction(
            oscar_category=oscar_cat,
            predicted_winner=top_name,
            confidence=top_conf,
            all_candidates=all_candidates,
            model_name=model_name,
            runner_up=runner_up,
            runner_up_confidence=runner_up_conf,
            details={"n_train": len(X_train), "n_features": X_train.shape[1]},
        ))

    return predictions


# ---------------------------------------------------------------------------
# Enhanced model functions (same hyperparameters as originals)
# ---------------------------------------------------------------------------

def model_enhanced_logistic_regression(df, category_mapping, accuracy, **kw):
    return _enhanced_ml_model_predict(
        df, category_mapping, accuracy,
        model_class=_make_sklearn_model(
            LogisticRegression(C=2.0, max_iter=1000, random_state=42)
        ),
        model_kwargs={},
        model_name="Enhanced Logistic Regression",
        **kw,
    )


def model_enhanced_random_forest(df, category_mapping, accuracy, **kw):
    return _enhanced_ml_model_predict(
        df, category_mapping, accuracy,
        model_class=_make_sklearn_model(RandomForestClassifier(
            n_estimators=80, max_depth=3, min_samples_leaf=3, random_state=42,
        )),
        model_kwargs={},
        model_name="Enhanced Random Forest",
        **kw,
    )


def model_enhanced_gradient_boosting(df, category_mapping, accuracy, **kw):
    return _enhanced_ml_model_predict(
        df, category_mapping, accuracy,
        model_class=_make_sklearn_model(GradientBoostingClassifier(
            n_estimators=50, max_depth=2, learning_rate=0.1,
            min_samples_leaf=5, random_state=42,
        )),
        model_kwargs={},
        model_name="Enhanced Gradient Boosting",
        **kw,
    )


# ---------------------------------------------------------------------------
# Enhanced backtesting (with data-leakage prevention)
# ---------------------------------------------------------------------------

def backtest_enhanced_ml_model(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    model_class,
    model_kwargs: dict,
    model_key: str,
    categories: list[str] | None = None,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Leave-one-year-out backtesting for enhanced ML models.

    CRITICAL: For each held-out year, accuracy is recomputed excluding that
    year to prevent data leakage (the enhanced features depend on accuracy).
    """
    if categories is None:
        categories = [
            "Best Picture", "Best Director", "Best Actor", "Best Actress",
            "Best Supporting Actor", "Best Supporting Actress",
        ]
    if years is None:
        years = list(range(max(HISTORICAL_START + 5, 2005), HISTORICAL_END + 1))

    oscar_lookup = build_winner_lookup(df, award_filter="Oscars")
    prec_lookup = build_winner_lookup(df, exclude_award="Oscars")
    full_accuracy = compute_historical_accuracy(
        df, category_mapping,
        start_year=HISTORICAL_START,
        end_year=HISTORICAL_END,
    )

    results = []

    for year in years:
        # Deep-copy accuracy and subtract the held-out year
        train_accuracy = copy.deepcopy(full_accuracy)
        for cat, pa_list in train_accuracy.items():
            for pa in pa_list:
                if year in pa.match_years:
                    pa.matches -= 1
                    pa.match_years.remove(year)
                oscar_key = ("Oscars", cat, year)
                prec_key = (pa.award, pa.precursor_category, year)
                if oscar_lookup.get(oscar_key) and prec_lookup.get(prec_key):
                    pa.total -= 1

        train_years = [y for y in range(HISTORICAL_START, HISTORICAL_END + 1) if y != year]

        for oscar_cat in categories:
            actual = oscar_lookup.get(("Oscars", oscar_cat, year), "")
            if not actual:
                continue

            X_train, y_train, _, _ = build_enhanced_candidate_features(
                df, category_mapping, train_accuracy, oscar_cat, train_years,
            )
            X_test, _, _, meta_test = build_enhanced_candidate_features(
                df, category_mapping, train_accuracy, oscar_cat, [year],
            )

            if len(X_train) < 10 or len(X_test) == 0:
                continue

            model = model_class(**model_kwargs)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)

            best_idx = int(np.argmax(probs))
            predicted = meta_test[best_idx][1]
            confidence = probs[best_idx] / probs.sum() if probs.sum() > 0 else 0

            correct = names_match(predicted, actual)
            results.append({
                "Year": year,
                "Category": oscar_cat,
                "Predicted": predicted[:50],
                "Actual": actual[:50],
                "Correct": correct,
                "Confidence": confidence,
                "Model": model_key,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main — standalone comparison
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data...")
    df = load_data()
    category_mapping = build_category_mapping()

    print("Computing historical accuracy...")
    accuracy = compute_historical_accuracy(df, category_mapping)

    # --- Enhanced ML backtests ---
    enhanced_ml_models = {
        "Enh_LogReg": (
            _make_sklearn_model(LogisticRegression(C=2.0, max_iter=1000, random_state=42)),
            {},
        ),
        "Enh_RF": (
            _make_sklearn_model(RandomForestClassifier(
                n_estimators=80, max_depth=3, min_samples_leaf=3, random_state=42,
            )),
            {},
        ),
        "Enh_GBM": (
            _make_sklearn_model(GradientBoostingClassifier(
                n_estimators=50, max_depth=2, learning_rate=0.1,
                min_samples_leaf=5, random_state=42,
            )),
            {},
        ),
    }

    enhanced_results = []
    for key, (cls, kwargs) in enhanced_ml_models.items():
        print(f"  Backtesting enhanced: {key}...")
        res = backtest_enhanced_ml_model(df, category_mapping, cls, kwargs, key)
        enhanced_results.append(res)

    # --- Original ML backtests for comparison ---
    original_ml_models = {
        "LogReg": (
            _make_sklearn_model(LogisticRegression(C=2.0, max_iter=1000, random_state=42)),
            {},
        ),
        "RandomForest": (
            _make_sklearn_model(RandomForestClassifier(
                n_estimators=80, max_depth=3, min_samples_leaf=3, random_state=42,
            )),
            {},
        ),
        "GBM": (
            _make_sklearn_model(GradientBoostingClassifier(
                n_estimators=50, max_depth=2, learning_rate=0.1,
                min_samples_leaf=5, random_state=42,
            )),
            {},
        ),
    }

    original_results = []
    for key, (cls, kwargs) in original_ml_models.items():
        print(f"  Backtesting original: {key}...")
        res = backtest_ml_model(df, category_mapping, cls, kwargs, key)
        original_results.append(res)

    # --- Combine and print ---
    all_results = pd.concat(enhanced_results + original_results, ignore_index=True)

    print()
    print("=" * 100)
    print("  ENHANCED vs ORIGINAL ML MODELS — Backtest Comparison")
    print("=" * 100)

    overall = all_results.groupby("Model")["Correct"].agg(["mean", "sum", "count"])
    overall.columns = ["Accuracy", "Correct", "Total"]
    overall = overall.sort_values("Accuracy", ascending=False)

    print(f"\n  {'Model':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    print(f"  {'─'*19} {'─'*10} {'─'*10} {'─'*8}")
    for model, row in overall.iterrows():
        print(f"  {model:<20} {row['Accuracy']:>9.1%} {int(row['Correct']):>10} {int(row['Total']):>8}")

    # Per-category breakdown
    print("\n  PER-CATEGORY ACCURACY:")
    cats = all_results["Category"].unique()
    models = overall.index.tolist()

    header = f"  {'Category':<28}" + "".join(f"{m:>14}" for m in models)
    print(header)
    print(f"  {'─'*27}" + "─" * 14 * len(models))

    for cat in sorted(cats):
        row_str = f"  {cat:<28}"
        for model in models:
            subset = all_results[(all_results["Model"] == model) & (all_results["Category"] == cat)]
            if len(subset) > 0:
                acc = subset["Correct"].mean()
                row_str += f"{acc:>13.0%} "
            else:
                row_str += f"{'—':>14}"
        print(row_str)

    # Save
    out_path = SCRIPT_DIR / "enhanced_backtest_results.csv"
    all_results.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
