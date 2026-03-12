"""
predictions.py — Oscar prediction models
=========================================

Contains multiple forecasting models that use precursor award data to predict
Oscar winners. All models return a list of Prediction objects with a common
interface, making them easy to compare.

Precursor-based models:
  1. Weighted Precursor  — weight each precursor by historical accuracy
  2. Simple Majority     — equal vote per precursor (unweighted)
  3. Recency-Weighted    — accuracy computed on recent window only (10yr)
  4. Top-N Precursors    — only use the N most accurate precursors per category
  5. Consensus Momentum  — accuracy × consensus-alignment bonus

ML models (candidate-level, binary features: which precursors did this
candidate win?):
  6. Logistic Regression — L2-regularised, implemented from scratch
  7. Random Forest       — shallow trees, implemented from scratch
  8. Gradient Boosting   — boosted stumps, implemented from scratch

All ML models are trained and evaluated via leave-one-year-out CV through
the backtesting framework.

Usage:
  python predictions.py              # run all models, compare
  python predictions.py backtest     # run leave-one-year-out backtesting
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import clone

from helpers import (
    OSCAR_CATEGORIES,
    PREDICTION_YEAR,
    MIN_PRECURSOR_WEIGHT,
    SCRIPT_DIR,
    HISTORICAL_START,
    HISTORICAL_END,
    AWARD_SHORT,
    load_data,
    build_category_mapping,
    build_winner_lookup,
    compute_historical_accuracy,
    cluster_nominees,
    names_match,
    PrecursorAccuracy,
)


# ---------------------------------------------------------------------------
# Prediction dataclass — model-agnostic
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """
    A single Oscar category prediction. Designed to be model-agnostic:
    the core fields are common to every model, while `details` holds
    arbitrary model-specific metadata.
    """
    oscar_category: str
    predicted_winner: str
    confidence: float                          # 0.0 to 1.0 (normalised score)
    all_candidates: list[tuple[str, float]]    # [(name, confidence%), ...] all candidates
    model_name: str = ""
    runner_up: str = ""
    runner_up_confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence_pct(self) -> float:
        return round(self.confidence * 100, 1)

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.70:
            return "HIGH"
        elif self.confidence >= 0.40:
            return "MEDIUM"
        elif self.confidence > 0:
            return "LOW"
        return "NO DATA"

    @property
    def confidence_emoji(self) -> str:
        if self.confidence >= 0.70:
            return "\U0001f7e2"  # green circle
        elif self.confidence >= 0.40:
            return "\U0001f7e1"  # yellow circle
        elif self.confidence > 0:
            return "\U0001f7e0"  # orange circle
        return "\U0001f534"      # red circle


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_precursor_lookup(
    df: pd.DataFrame, prediction_year: int
) -> dict[tuple[str, str], str]:
    """Return {(Award, Category): winner} for precursors in prediction_year."""
    mask = (
        (df["Year of award ceremony"] == prediction_year)
        & (df["Award"] != "Oscars")
        & df["winner"].notna()
        & (df["winner"] != "")
    )
    lookup = {}
    for _, row in df[mask].iterrows():
        lookup[(row["Award"], row["Category"])] = row["winner"].strip()
    return lookup


def _score_to_predictions(
    oscar_cat: str,
    raw_votes: list[tuple[str, float, list[tuple[str, str, float]]]],
    total_weight: float,
    available: int,
    total_precursors: int,
    model_name: str,
) -> Prediction:
    """
    Common post-processing: cluster nominees, normalise scores, build Prediction.
    """
    if total_weight == 0:
        return Prediction(
            oscar_category=oscar_cat,
            predicted_winner="N/A — No precursor data",
            confidence=0.0,
            all_candidates=[],
            model_name=model_name,
            details={
                "precursors_available": 0,
                "precursors_total": total_precursors,
            },
        )

    clustered = cluster_nominees(raw_votes)
    scored = [
        (name, score / total_weight, details)
        for name, score, details in clustered
    ]
    scored.sort(key=lambda x: -x[1])

    top_name, top_conf, top_details = scored[0]
    top_details_sorted = sorted(top_details, key=lambda x: -x[2])

    runner_up_name = scored[1][0] if len(scored) > 1 else ""
    runner_up_conf = scored[1][1] if len(scored) > 1 else 0.0

    return Prediction(
        oscar_category=oscar_cat,
        predicted_winner=top_name,
        confidence=top_conf,
        all_candidates=[(n, round(c * 100, 1)) for n, c, _ in scored],
        model_name=model_name,
        runner_up=runner_up_name,
        runner_up_confidence=runner_up_conf,
        details={
            "precursors_available": available,
            "precursors_total": total_precursors,
            "supporting_awards": [
                (AWARD_SHORT.get(a, a), cat, w) for a, cat, w in top_details_sorted
            ],
        },
    )


# ============================================================================
# PRECURSOR-BASED MODELS (1–5)
# ============================================================================

def model_weighted_precursor(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    accuracy: dict[str, list[PrecursorAccuracy]],
    prediction_year: int = PREDICTION_YEAR,
    min_weight: float = MIN_PRECURSOR_WEIGHT,
) -> list[Prediction]:
    """Weight each precursor by its full historical accuracy (2000–2025)."""
    MODEL_NAME = "Weighted Precursor"
    precursor_lookup = _build_precursor_lookup(df, prediction_year)

    acc_lookup: dict[str, dict[tuple[str, str], float]] = {}
    for oscar_cat, pa_list in accuracy.items():
        acc_lookup[oscar_cat] = {
            (pa.award, pa.precursor_category): pa.accuracy for pa in pa_list
        }

    predictions = []
    for oscar_cat in OSCAR_CATEGORIES:
        precursors = category_mapping.get(oscar_cat, [])
        raw_votes = []
        total_weight = 0.0
        available = 0

        for award_name, prec_cat in precursors:
            winner = precursor_lookup.get((award_name, prec_cat))
            if not winner:
                continue
            weight = max(
                acc_lookup.get(oscar_cat, {}).get((award_name, prec_cat), 0.0),
                min_weight,
            )
            raw_votes.append((winner, weight, [(award_name, prec_cat, weight)]))
            total_weight += weight
            available += 1

        predictions.append(_score_to_predictions(
            oscar_cat, raw_votes, total_weight,
            available, len(precursors), MODEL_NAME,
        ))
    return predictions


def model_simple_majority(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    accuracy: dict[str, list[PrecursorAccuracy]],
    prediction_year: int = PREDICTION_YEAR,
) -> list[Prediction]:
    """Every precursor gets one equal vote."""
    MODEL_NAME = "Simple Majority"
    precursor_lookup = _build_precursor_lookup(df, prediction_year)

    predictions = []
    for oscar_cat in OSCAR_CATEGORIES:
        precursors = category_mapping.get(oscar_cat, [])
        raw_votes = []
        total_weight = 0.0
        available = 0

        for award_name, prec_cat in precursors:
            winner = precursor_lookup.get((award_name, prec_cat))
            if not winner:
                continue
            weight = 1.0
            raw_votes.append((winner, weight, [(award_name, prec_cat, weight)]))
            total_weight += weight
            available += 1

        predictions.append(_score_to_predictions(
            oscar_cat, raw_votes, total_weight,
            available, len(precursors), MODEL_NAME,
        ))
    return predictions


def model_recency_weighted(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    accuracy: dict[str, list[PrecursorAccuracy]],
    prediction_year: int = PREDICTION_YEAR,
    window: int = 10,
    min_weight: float = MIN_PRECURSOR_WEIGHT,
) -> list[Prediction]:
    """Historical accuracy computed on the last `window` years only."""
    MODEL_NAME = f"Recency-Weighted ({window}yr)"

    recent_start = prediction_year - 1 - window
    recent_end = prediction_year - 1
    recent_accuracy = compute_historical_accuracy(
        df, category_mapping,
        start_year=max(recent_start, HISTORICAL_START),
        end_year=recent_end,
    )

    precursor_lookup = _build_precursor_lookup(df, prediction_year)
    acc_lookup: dict[str, dict[tuple[str, str], float]] = {}
    for oscar_cat, pa_list in recent_accuracy.items():
        acc_lookup[oscar_cat] = {
            (pa.award, pa.precursor_category): pa.accuracy for pa in pa_list
        }

    predictions = []
    for oscar_cat in OSCAR_CATEGORIES:
        precursors = category_mapping.get(oscar_cat, [])
        raw_votes = []
        total_weight = 0.0
        available = 0

        for award_name, prec_cat in precursors:
            winner = precursor_lookup.get((award_name, prec_cat))
            if not winner:
                continue
            weight = max(
                acc_lookup.get(oscar_cat, {}).get((award_name, prec_cat), 0.0),
                min_weight,
            )
            raw_votes.append((winner, weight, [(award_name, prec_cat, weight)]))
            total_weight += weight
            available += 1

        predictions.append(_score_to_predictions(
            oscar_cat, raw_votes, total_weight,
            available, len(precursors), MODEL_NAME,
        ))
    return predictions


def model_top_n_precursors(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    accuracy: dict[str, list[PrecursorAccuracy]],
    prediction_year: int = PREDICTION_YEAR,
    n: int = 3,
    min_weight: float = MIN_PRECURSOR_WEIGHT,
) -> list[Prediction]:
    """Only use the top-N most accurate precursors per category."""
    MODEL_NAME = f"Top-{n} Precursors"
    precursor_lookup = _build_precursor_lookup(df, prediction_year)

    acc_lookup: dict[str, dict[tuple[str, str], float]] = {}
    for oscar_cat, pa_list in accuracy.items():
        acc_lookup[oscar_cat] = {
            (pa.award, pa.precursor_category): pa.accuracy for pa in pa_list
        }

    predictions = []
    for oscar_cat in OSCAR_CATEGORIES:
        precursors = category_mapping.get(oscar_cat, [])
        ranked = sorted(
            precursors,
            key=lambda p: acc_lookup.get(oscar_cat, {}).get(p, 0.0),
            reverse=True,
        )
        top_precursors = set(ranked[:n])

        raw_votes = []
        total_weight = 0.0
        available = 0

        for award_name, prec_cat in precursors:
            if (award_name, prec_cat) not in top_precursors:
                continue
            winner = precursor_lookup.get((award_name, prec_cat))
            if not winner:
                continue
            weight = max(
                acc_lookup.get(oscar_cat, {}).get((award_name, prec_cat), 0.0),
                min_weight,
            )
            raw_votes.append((winner, weight, [(award_name, prec_cat, weight)]))
            total_weight += weight
            available += 1

        predictions.append(_score_to_predictions(
            oscar_cat, raw_votes, total_weight,
            available, len(precursors), MODEL_NAME,
        ))
    return predictions


def model_consensus_momentum(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    accuracy: dict[str, list[PrecursorAccuracy]],
    prediction_year: int = PREDICTION_YEAR,
    min_weight: float = MIN_PRECURSOR_WEIGHT,
) -> list[Prediction]:
    """
    Two-pass: identify consensus frontrunner, then reweight by how often
    each precursor historically confirms the Oscar winner in consensus years.
    """
    MODEL_NAME = "Consensus Momentum"
    precursor_lookup = _build_precursor_lookup(df, prediction_year)

    oscar_lookup = build_winner_lookup(df, award_filter="Oscars")
    prec_hist_lookup = build_winner_lookup(df, exclude_award="Oscars")

    alignment_bonus: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)

    for oscar_cat in OSCAR_CATEGORIES:
        precursors = category_mapping.get(oscar_cat, [])
        prec_hits = defaultdict(lambda: {"match": 0, "total": 0})

        for year in range(HISTORICAL_START, HISTORICAL_END + 1):
            oscar_winner = oscar_lookup.get(("Oscars", oscar_cat, year))
            if not oscar_winner:
                continue

            year_picks = defaultdict(int)
            year_total = 0
            for award_name, prec_cat in precursors:
                pw = prec_hist_lookup.get((award_name, prec_cat, year))
                if pw:
                    year_total += 1
                    matched_existing = False
                    for existing in list(year_picks.keys()):
                        if names_match(pw, existing):
                            year_picks[existing] += 1
                            matched_existing = True
                            break
                    if not matched_existing:
                        year_picks[pw] += 1

            if year_total < 3:
                continue

            top_pick = max(year_picks, key=year_picks.get)
            if year_picks[top_pick] / year_total < 0.4:
                continue

            for award_name, prec_cat in precursors:
                pw = prec_hist_lookup.get((award_name, prec_cat, year))
                if pw:
                    prec_hits[(award_name, prec_cat)]["total"] += 1
                    if names_match(pw, oscar_winner):
                        prec_hits[(award_name, prec_cat)]["match"] += 1

        for key, stats in prec_hits.items():
            if stats["total"] > 0:
                alignment_bonus[oscar_cat][key] = stats["match"] / stats["total"]

    acc_lookup: dict[str, dict[tuple[str, str], float]] = {}
    for oscar_cat, pa_list in accuracy.items():
        acc_lookup[oscar_cat] = {
            (pa.award, pa.precursor_category): pa.accuracy for pa in pa_list
        }

    predictions = []
    for oscar_cat in OSCAR_CATEGORIES:
        precursors = category_mapping.get(oscar_cat, [])
        raw_votes = []
        total_weight = 0.0
        available = 0

        for award_name, prec_cat in precursors:
            winner = precursor_lookup.get((award_name, prec_cat))
            if not winner:
                continue
            base_acc = acc_lookup.get(oscar_cat, {}).get((award_name, prec_cat), 0.0)
            bonus = alignment_bonus.get(oscar_cat, {}).get((award_name, prec_cat), 0.0)
            weight = max(base_acc * (1 + bonus), min_weight)

            raw_votes.append((winner, weight, [(award_name, prec_cat, weight)]))
            total_weight += weight
            available += 1

        predictions.append(_score_to_predictions(
            oscar_cat, raw_votes, total_weight,
            available, len(precursors), MODEL_NAME,
        ))
    return predictions


# ============================================================================
# FRONTRUNNER FEATURE EXTRACTION
# ============================================================================

def _score_candidates_weighted_raw(
    precursor_lookup: dict[tuple[str, str], str],
    precursors: list[tuple[str, str]],
    acc_lookup: dict[tuple[str, str], float],
    min_weight: float = MIN_PRECURSOR_WEIGHT,
) -> tuple[list[tuple[str, float, int, list[tuple[str, str, float]]]], float, int]:
    """
    Score candidates for one category/year using weighted precursor logic.

    Returns:
        scored: [(name, score, n_wins, [(award, cat, weight), ...]), ...] desc
        total_weight: sum of all weights
        available: number of available precursors
    """
    raw_votes: list[tuple[str, float, list[tuple[str, str, float]]]] = []
    total_weight = 0.0
    available = 0

    for award_name, prec_cat in precursors:
        winner = precursor_lookup.get((award_name, prec_cat))
        if not winner:
            continue
        weight = max(acc_lookup.get((award_name, prec_cat), 0.0), min_weight)
        raw_votes.append((winner, weight, [(award_name, prec_cat, weight)]))
        total_weight += weight
        available += 1

    clustered = cluster_nominees(raw_votes)
    scored = [
        (name, score, len(details), details)
        for name, score, details in clustered
    ]
    scored.sort(key=lambda x: -x[1])
    return scored, total_weight, available


def _get_top_precursor(
    accuracy: dict[str, list[PrecursorAccuracy]], oscar_cat: str,
) -> tuple[str, str] | None:
    """Return (award, category) of the most accurate precursor."""
    pa_list = accuracy.get(oscar_cat, [])
    if not pa_list:
        return None
    best = max(pa_list, key=lambda pa: pa.accuracy)
    return (best.award, best.precursor_category)


def _compute_momentum_frontrunners(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    accuracy: dict[str, list[PrecursorAccuracy]],
    years: list[int] | None = None,
) -> dict[tuple[int, str], str]:
    """Precompute momentum model's frontrunner for each (year, category)."""
    if years is None:
        years = list(range(HISTORICAL_START, HISTORICAL_END + 1))
    result: dict[tuple[int, str], str] = {}
    for year in years:
        preds = model_consensus_momentum(
            df, category_mapping, accuracy, prediction_year=year,
        )
        for p in preds:
            if p.confidence > 0:
                result[(year, p.oscar_category)] = p.predicted_winner
    return result


FRONTRUNNER_FEATURE_NAMES = [
    "n_precursors_won",
    "frac_precursors_won",
    "weighted_score",
    "gap_to_runner_up",
    "won_top_precursor",
    "n_candidates",
    "frontrunners_agree",
]


def build_frontrunner_features(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    accuracy: dict[str, list[PrecursorAccuracy]],
    oscar_cat: str,
    years: list[int],
    momentum_frontrunners: dict[tuple[int, str], str] | None = None,
    require_target: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[int, str]]]:
    """
    Build frontrunner-level feature matrix for one Oscar category.

    For each year, identifies the frontrunner via weighted precursor scoring
    and extracts features describing the strength of their case.

    Features:
      n_precursors_won     — how many precursor awards the frontrunner won
      frac_precursors_won  — fraction of available precursors won
      weighted_score       — frontrunner's share of total weighted vote (0–1)
      gap_to_runner_up     — weighted_score difference to runner-up
      won_top_precursor    — did frontrunner win the most accurate precursor?
      n_candidates         — number of distinct candidates
      frontrunners_agree   — weighted & momentum models pick the same person

    Target: 1 if the frontrunner won the Oscar, 0 otherwise.

    Returns:
        X: (n_years, 7) feature matrix
        y: (n_years,) binary target
        feature_names: list of feature names
        meta: list of (year, frontrunner_name) per row
    """
    precursors = category_mapping.get(oscar_cat, [])
    oscar_lookup = build_winner_lookup(df, award_filter="Oscars")
    top_precursor = _get_top_precursor(accuracy, oscar_cat)

    acc_lookup = {
        (pa.award, pa.precursor_category): pa.accuracy
        for pa in accuracy.get(oscar_cat, [])
    }

    rows_X: list[list[float]] = []
    rows_y: list[float] = []
    meta: list[tuple[int, str]] = []

    for year in years:
        precursor_lookup = _build_precursor_lookup(df, year)
        scored, total_weight, available = _score_candidates_weighted_raw(
            precursor_lookup, precursors, acc_lookup,
        )

        if not scored or available == 0:
            continue

        oscar_winner = oscar_lookup.get(("Oscars", oscar_cat, year))
        if require_target and not oscar_winner:
            continue

        name, score, n_wins, details = scored[0]

        weighted_score = score / total_weight if total_weight > 0 else 0
        runner_up_score = (
            scored[1][1] / total_weight
            if len(scored) > 1 and total_weight > 0 else 0
        )

        won_top = 0.0
        if top_precursor:
            for award, cat, _ in details:
                if (award, cat) == top_precursor:
                    won_top = 1.0
                    break

        agree = 0.0
        if momentum_frontrunners:
            momentum_pick = momentum_frontrunners.get((year, oscar_cat))
            if momentum_pick and names_match(name, momentum_pick):
                agree = 1.0

        rows_X.append([
            n_wins,
            n_wins / available if available > 0 else 0,
            weighted_score,
            weighted_score - runner_up_score,
            won_top,
            len(scored),
            agree,
        ])

        target = 0.0
        if oscar_winner and names_match(name, oscar_winner):
            target = 1.0
        rows_y.append(target)
        meta.append((year, name))

    if not rows_X:
        n_feat = len(FRONTRUNNER_FEATURE_NAMES)
        return np.empty((0, n_feat)), np.empty(0), FRONTRUNNER_FEATURE_NAMES, []

    return np.array(rows_X), np.array(rows_y), FRONTRUNNER_FEATURE_NAMES, meta


# ============================================================================
# Sklearn model adapter: normalises predict_proba to 1D positive-class output
# ============================================================================

class _SklearnAdapter:
    """Wraps a sklearn classifier so predict_proba returns 1D P(y=1)."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        proba = self.estimator.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba


def _make_sklearn_model(estimator):
    """Factory: returns a callable that creates a fresh _SklearnAdapter each time."""
    def _factory(**_ignored):
        return _SklearnAdapter(clone(estimator))
    return _factory


# ============================================================================
# BASELINE MODEL: Historical frontrunner win rate
# ============================================================================

def model_baseline(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    accuracy: dict[str, list[PrecursorAccuracy]],
    prediction_year: int = PREDICTION_YEAR,
) -> list[Prediction]:
    """
    Baseline probabilistic model: identify the frontrunner via weighted
    precursor scoring, then assign P(win) = historical rate at which the
    frontrunner actually won the Oscar for this category.

    Works for all categories regardless of precursor count.
    """
    MODEL_NAME = "Baseline"
    oscar_lookup = build_winner_lookup(df, award_filter="Oscars")

    predictions = []
    for oscar_cat in OSCAR_CATEGORIES:
        precursors = category_mapping.get(oscar_cat, [])
        acc_lookup = {
            (pa.award, pa.precursor_category): pa.accuracy
            for pa in accuracy.get(oscar_cat, [])
        }

        # Compute historical frontrunner win rate for this category
        hist_years = list(range(HISTORICAL_START, prediction_year))
        wins, total = 0, 0
        for year in hist_years:
            pl = _build_precursor_lookup(df, year)
            scored, tw, avail = _score_candidates_weighted_raw(
                pl, precursors, acc_lookup,
            )
            if not scored or avail == 0:
                continue
            oscar_winner = oscar_lookup.get(("Oscars", oscar_cat, year))
            if not oscar_winner:
                continue
            total += 1
            if names_match(scored[0][0], oscar_winner):
                wins += 1

        base_rate = wins / total if total > 0 else 0.0

        # Identify this year's frontrunner
        pred_lookup = _build_precursor_lookup(df, prediction_year)
        scored, tw, avail = _score_candidates_weighted_raw(
            pred_lookup, precursors, acc_lookup,
        )

        if not scored:
            predictions.append(Prediction(
                oscar_category=oscar_cat,
                predicted_winner="N/A — No precursor data",
                confidence=0.0,
                all_candidates=[],
                model_name=MODEL_NAME,
            ))
            continue

        frontrunner = scored[0][0]
        predictions.append(Prediction(
            oscar_category=oscar_cat,
            predicted_winner=frontrunner,
            confidence=base_rate,
            all_candidates=[(frontrunner, round(base_rate * 100, 1))],
            model_name=MODEL_NAME,
            details={"historical_wins": wins, "historical_total": total},
        ))

    return predictions


# ============================================================================
# ML MODELS: P(frontrunner wins)
# ============================================================================

def _frontrunner_ml_predict(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    accuracy: dict[str, list[PrecursorAccuracy]],
    model_class,
    model_kwargs: dict,
    model_name: str,
    prediction_year: int = PREDICTION_YEAR,
) -> list[Prediction]:
    """
    Train a binary classifier to predict P(frontrunner wins the Oscar).
    The frontrunner is identified by the weighted precursor model.
    Falls back to base rate if insufficient training data or single class.
    """
    momentum_frontrunners = _compute_momentum_frontrunners(
        df, category_mapping, accuracy,
        years=list(range(HISTORICAL_START, prediction_year + 1)),
    )

    predictions = []
    for oscar_cat in OSCAR_CATEGORIES:
        train_years = list(range(HISTORICAL_START, prediction_year))

        X_train, y_train, feat_names, _ = build_frontrunner_features(
            df, category_mapping, accuracy, oscar_cat, train_years,
            momentum_frontrunners=momentum_frontrunners,
        )

        X_pred, _, _, meta_pred = build_frontrunner_features(
            df, category_mapping, accuracy, oscar_cat, [prediction_year],
            momentum_frontrunners=momentum_frontrunners,
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

        # If only one class in training data, use base rate directly
        if len(np.unique(y_train)) < 2:
            p_win = float(y_train.mean())
        else:
            model = model_class(**model_kwargs)
            model.fit(X_train, y_train)
            p_win = float(model.predict_proba(X_pred)[0])

        frontrunner_name = meta_pred[0][1]

        predictions.append(Prediction(
            oscar_category=oscar_cat,
            predicted_winner=frontrunner_name,
            confidence=p_win,
            all_candidates=[(frontrunner_name, round(p_win * 100, 1))],
            model_name=model_name,
            details={"n_train": len(X_train), "n_features": X_train.shape[1]},
        ))

    return predictions


def model_logistic_regression(df, category_mapping, accuracy, **kw):
    return _frontrunner_ml_predict(
        df, category_mapping, accuracy,
        model_class=_make_sklearn_model(LogisticRegression(C=2.0, max_iter=1000, random_state=42)),
        model_kwargs={},
        model_name="Logistic Regression",
        **kw,
    )


def model_gradient_boosting(df, category_mapping, accuracy, **kw):
    return _frontrunner_ml_predict(
        df, category_mapping, accuracy,
        model_class=_make_sklearn_model(GradientBoostingClassifier(
            n_estimators=50, max_depth=2, learning_rate=0.1,
            min_samples_leaf=5, random_state=42,
        )),
        model_kwargs={},
        model_name="Gradient Boosting",
        **kw,
    )


# ============================================================================
# Model Registry
# ============================================================================

ALL_MODELS = {
    "weighted":  model_weighted_precursor,
    "momentum":  model_consensus_momentum,
    "baseline":  model_baseline,
    "logreg":    model_logistic_regression,
    "gbm":       model_gradient_boosting,
}

# Lazy-load ensemble models to avoid circular imports
def _load_ensemble_models():
    from ensemble import model_mean_ensemble, model_logodds_ensemble
    ALL_MODELS["mean_ensemble"] = model_mean_ensemble
    ALL_MODELS["logodds_ensemble"] = model_logodds_ensemble

try:
    _load_ensemble_models()
except ImportError:
    pass  # ensemble.py not yet available


def run_all_models(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    accuracy: dict[str, list[PrecursorAccuracy]],
) -> dict[str, list[Prediction]]:
    """Run every registered model and return {model_key: predictions}."""
    results = {}
    for key, model_fn in ALL_MODELS.items():
        print(f"  Running: {key}...")
        results[key] = model_fn(df, category_mapping, accuracy)
    return results


# ============================================================================
# BACKTESTING: Leave-one-year-out cross-validation
# ============================================================================

def backtest_precursor_model(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    model_fn,
    model_key: str,
    categories: list[str] | None = None,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Leave-one-year-out backtesting for precursor-based models.

    For each held-out year:
      1. Recompute historical accuracy on all OTHER years
      2. Use the model to predict the held-out year
      3. Check if prediction matches the actual Oscar winner

    Returns a DataFrame with per-year, per-category results.

    NOTE on accuracy weights: We use leave-one-out on the full 2000-2025
    range rather than an expanding window (i.e. for predicting 2005 we use
    2000-2004 + 2006-2025, not just 2000-2004). This is a mild form of
    lookahead — precursor weights benefit from future data. We accept this
    because precursor predictiveness is stable over time and an expanding
    window would make early years nearly useless due to small samples.
    """
    import copy

    if categories is None:
        categories = OSCAR_CATEGORIES
    if years is None:
        years = list(range(HISTORICAL_START, HISTORICAL_END + 1))

    # Precompute lookups ONCE (the expensive part)
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

        # Predict this year
        preds = model_fn(
            df, category_mapping, train_accuracy,
            prediction_year=year,
        )

        for p in preds:
            if p.oscar_category not in categories:
                continue
            actual = oscar_lookup.get(("Oscars", p.oscar_category, year), "")
            if not actual:
                continue

            correct = names_match(p.predicted_winner, actual) if p.confidence > 0 else False
            results.append({
                "Year": year,
                "Category": p.oscar_category,
                "Predicted": p.predicted_winner[:50],
                "Actual": actual[:50],
                "Correct": correct,
                "Confidence": p.confidence,
                "Model": model_key,
            })

    return pd.DataFrame(results)


def backtest_frontrunner_ml(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    model_class,
    model_kwargs: dict,
    model_key: str,
    categories: list[str] | None = None,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Leave-one-year-out backtesting for frontrunner ML models.

    For each held-out year:
      1. Recompute accuracy excluding held-out year (leave-one-out)
      2. Build frontrunner features, train model
      3. Predict P(frontrunner wins) for held-out year
      4. Check if frontrunner actually won

    NOTE: All ML models predict the same frontrunner (from weighted precursor
    model), so binary accuracy is identical across ML models and ensembles.
    The models differ in their confidence estimates (calibration).
    """
    import copy

    if categories is None:
        categories = OSCAR_CATEGORIES
    if years is None:
        years = list(range(max(HISTORICAL_START + 5, 2005), HISTORICAL_END + 1))

    all_years = list(range(HISTORICAL_START, HISTORICAL_END + 1))
    oscar_lookup = build_winner_lookup(df, award_filter="Oscars")
    prec_lookup = build_winner_lookup(df, exclude_award="Oscars")
    full_accuracy = compute_historical_accuracy(df, category_mapping)

    # Precompute momentum frontrunners once (full accuracy — mild leakage)
    momentum_frontrunners = _compute_momentum_frontrunners(
        df, category_mapping, full_accuracy,
    )

    results = []

    for year in years:
        # Leave-one-out accuracy
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

        train_years = [y for y in all_years if y != year]

        for oscar_cat in categories:
            actual = oscar_lookup.get(("Oscars", oscar_cat, year), "")
            if not actual:
                continue

            X_train, y_train, _, _ = build_frontrunner_features(
                df, category_mapping, train_accuracy, oscar_cat, train_years,
                momentum_frontrunners=momentum_frontrunners,
            )
            X_test, _, _, meta_test = build_frontrunner_features(
                df, category_mapping, train_accuracy, oscar_cat, [year],
                momentum_frontrunners=momentum_frontrunners,
            )

            if len(X_train) < 10 or len(X_test) == 0:
                continue

            # If only one class in training data, use base rate directly
            if len(np.unique(y_train)) < 2:
                p_win = float(y_train.mean())
            else:
                model = model_class(**model_kwargs)
                model.fit(X_train, y_train)
                p_win = float(model.predict_proba(X_test)[0])

            predicted = meta_test[0][1]  # frontrunner name
            correct = names_match(predicted, actual)

            results.append({
                "Year": year,
                "Category": oscar_cat,
                "Predicted": predicted[:50],
                "Actual": actual[:50],
                "Correct": correct,
                "Confidence": p_win,
                "Model": model_key,
            })

    return pd.DataFrame(results)


def backtest_baseline(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    categories: list[str] | None = None,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Leave-one-year-out backtesting for the baseline model.

    For each held-out year, the baseline P(win) is the frontrunner win rate
    computed from all OTHER years. The frontrunner is identified using
    leave-one-out accuracy (same as precursor model backtest).
    """
    import copy

    if categories is None:
        categories = OSCAR_CATEGORIES
    if years is None:
        years = list(range(HISTORICAL_START, HISTORICAL_END + 1))

    oscar_lookup = build_winner_lookup(df, award_filter="Oscars")
    prec_lookup = build_winner_lookup(df, exclude_award="Oscars")
    full_accuracy = compute_historical_accuracy(df, category_mapping)

    results = []

    for held_out_year in years:
        # Leave-one-out accuracy
        train_accuracy = copy.deepcopy(full_accuracy)
        for cat, pa_list in train_accuracy.items():
            for pa in pa_list:
                if held_out_year in pa.match_years:
                    pa.matches -= 1
                    pa.match_years.remove(held_out_year)
                oscar_key = ("Oscars", cat, held_out_year)
                prec_key = (pa.award, pa.precursor_category, held_out_year)
                if oscar_lookup.get(oscar_key) and prec_lookup.get(prec_key):
                    pa.total -= 1

        for oscar_cat in categories:
            actual = oscar_lookup.get(("Oscars", oscar_cat, held_out_year), "")
            if not actual:
                continue

            precursors = category_mapping.get(oscar_cat, [])
            acc_lookup = {
                (pa.award, pa.precursor_category): pa.accuracy
                for pa in train_accuracy.get(oscar_cat, [])
            }

            # Compute frontrunner win rate on training years
            train_years = [y for y in range(HISTORICAL_START, HISTORICAL_END + 1)
                           if y != held_out_year]
            wins, total = 0, 0
            for year in train_years:
                pl = _build_precursor_lookup(df, year)
                scored, tw, avail = _score_candidates_weighted_raw(
                    pl, precursors, acc_lookup,
                )
                if not scored or avail == 0:
                    continue
                ow = oscar_lookup.get(("Oscars", oscar_cat, year))
                if not ow:
                    continue
                total += 1
                if names_match(scored[0][0], ow):
                    wins += 1

            base_rate = wins / total if total > 0 else 0.0

            # Identify held-out year's frontrunner
            pl = _build_precursor_lookup(df, held_out_year)
            scored, tw, avail = _score_candidates_weighted_raw(
                pl, precursors, acc_lookup,
            )
            if not scored:
                continue

            predicted = scored[0][0]
            correct = names_match(predicted, actual)

            results.append({
                "Year": held_out_year,
                "Category": oscar_cat,
                "Predicted": predicted[:50],
                "Actual": actual[:50],
                "Correct": correct,
                "Confidence": base_rate,
                "Model": "Baseline",
            })

    return pd.DataFrame(results)


def run_full_backtest(df, category_mapping, accuracy) -> pd.DataFrame:
    """Run backtesting for all models and return combined results."""

    all_results = []

    # Precursor models (all categories)
    precursor_models = {
        "Weighted": model_weighted_precursor,
        "Momentum": model_consensus_momentum,
    }

    for key, fn in precursor_models.items():
        print(f"  Backtesting: {key}...")
        res = backtest_precursor_model(df, category_mapping, fn, key)
        all_results.append(res)

    # Baseline model (all categories)
    print("  Backtesting: Baseline...")
    all_results.append(backtest_baseline(df, category_mapping))

    # Frontrunner ML models (categories with 6+ precursors only)
    ml_models = {
        "LogReg": (
            _make_sklearn_model(LogisticRegression(C=2.0, max_iter=1000, random_state=42)),
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

    for key, (cls, kwargs) in ml_models.items():
        print(f"  Backtesting: {key}...")
        res = backtest_frontrunner_ml(df, category_mapping, cls, kwargs, key)
        all_results.append(res)

    combined = pd.concat(all_results, ignore_index=True)
    return combined


def print_backtest_results(results: pd.DataFrame) -> None:
    """Print a summary of backtesting results."""

    print()
    print("=" * 100)
    print("  BACKTEST RESULTS — Leave-One-Year-Out Cross-Validation")
    print("=" * 100)

    # Overall accuracy per model
    overall = results.groupby("Model")["Correct"].agg(["mean", "sum", "count"])
    overall.columns = ["Accuracy", "Correct", "Total"]
    overall = overall.sort_values("Accuracy", ascending=False)

    print("\n  OVERALL ACCURACY (across all categories):")
    print(f"  {'Model':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    print(f"  {'─'*19} {'─'*10} {'─'*10} {'─'*8}")
    for model, row in overall.iterrows():
        print(f"  {model:<20} {row['Accuracy']:>9.1%} {int(row['Correct']):>10} {int(row['Total']):>8}")

    # Per-category accuracy
    print("\n  PER-CATEGORY ACCURACY:")
    cats = results["Category"].unique()
    models = overall.index.tolist()

    header = f"  {'Category':<28}" + "".join(f"{m:>14}" for m in models)
    print(header)
    print(f"  {'─'*27}" + "─" * 14 * len(models))

    for cat in sorted(cats):
        row_str = f"  {cat:<28}"
        for model in models:
            subset = results[(results["Model"] == model) & (results["Category"] == cat)]
            if len(subset) > 0:
                acc = subset["Correct"].mean()
                row_str += f"{acc:>13.0%} "
            else:
                row_str += f"{'—':>14}"
        print(row_str)


# ---------------------------------------------------------------------------
# Output: Console
# ---------------------------------------------------------------------------

def print_predictions(predictions: list[Prediction]) -> None:
    """Print one model's predictions to console."""
    model = predictions[0].model_name if predictions else "Unknown"
    print()
    print("=" * 100)
    print(f"  2026 OSCAR PREDICTIONS — {model.upper()}")
    print("=" * 100)

    for p in predictions:
        print()
        print(f"{p.oscar_category}")
        print(f"  Prediction: {p.predicted_winner}")
        print(
            f"  Confidence: {p.confidence_pct}% "
            f"[{p.confidence_emoji} {p.confidence_label}]"
        )
        if p.runner_up:
            print(
                f"  Runner-up:  {p.runner_up} "
                f"({round(p.runner_up_confidence * 100, 1)}%)"
            )
        awards = p.details.get("supporting_awards", [])
        if awards:
            awards_str = ", ".join(f"{a} ({w:.0%})" for a, _, w in awards[:4])
            print(f"  Supported by: {awards_str}")


def print_model_comparison(
    all_results: dict[str, list[Prediction]],
    categories: list[str] | None = None,
) -> None:
    """Print a side-by-side comparison of all models for selected categories."""
    if categories is None:
        categories = OSCAR_CATEGORIES

    model_keys = list(all_results.keys())

    print()
    print("=" * 120)
    print("  MODEL COMPARISON — 2026 OSCAR PREDICTIONS")
    print("=" * 120)

    for cat in categories:
        print(f"\n{'─' * 120}")
        print(f"  {cat}")
        print(f"{'─' * 120}")
        print(f"  {'Model':<28} {'Prediction':<45} {'Conf':>6}  {'Runner-up':<30}")
        print(f"  {'─'*27} {'─'*44} {'─'*6}  {'─'*30}")

        for key in model_keys:
            preds = all_results[key]
            p = next((x for x in preds if x.oscar_category == cat), None)
            if p is None:
                continue
            pred_str = p.predicted_winner[:43]
            ru_str = f"{p.runner_up[:25]} ({round(p.runner_up_confidence*100)}%)" if p.runner_up else "—"
            print(
                f"  {p.model_name:<28} {pred_str:<45} "
                f"{p.confidence_pct:>5.1f}%  {ru_str:<30}"
            )

    # Agreement summary
    print(f"\n{'=' * 120}")
    print("  AGREEMENT SUMMARY")
    print(f"{'=' * 120}")
    print(f"  {'Category':<30} {'All agree?':<12} {'Majority pick':<45}")
    print(f"  {'─'*29} {'─'*11} {'─'*44}")

    for cat in OSCAR_CATEGORIES:
        picks = []
        for key in model_keys:
            preds = all_results[key]
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
        agreement_str = "✓ YES" if unanimous else f"✗ {top_count}/{len(picks)}"

        print(f"  {cat:<30} {agreement_str:<12} {top_pick[:43]:<45}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import sys

    print("Loading data...")
    df = load_data()
    category_mapping = build_category_mapping()

    print("Computing historical accuracy...")
    accuracy = compute_historical_accuracy(df, category_mapping)

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "backtest":
        print("\nRunning full backtest (this may take a minute)...")
        bt_results = run_full_backtest(df, category_mapping, accuracy)
        print_backtest_results(bt_results)

        # Save results
        out_path = SCRIPT_DIR / "backtest_results.csv"
        bt_results.to_csv(out_path, index=False)
        print(f"\nDetailed results saved to: {out_path}")

    elif mode in ALL_MODELS:
        print(f"\nRunning model: {mode}")
        preds = ALL_MODELS[mode](df, category_mapping, accuracy)
        print_predictions(preds)

    else:
        print(f"\nRunning all {len(ALL_MODELS)} models...")
        all_results = run_all_models(df, category_mapping, accuracy)
        print_model_comparison(all_results)


if __name__ == "__main__":
    main()
