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
    all_candidates: list[tuple[str, float]]    # [(name, confidence%), ...] top 5
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
        all_candidates=[(n, round(c * 100, 1)) for n, c, _ in scored[:5]],
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
# ML INFRASTRUCTURE: Feature construction
# ============================================================================

def build_candidate_features(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    oscar_cat: str,
    years: list[int],
    require_target: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[int, str]]]:
    """
    Build candidate-level feature matrix for one Oscar category.

    For each year, every person/film that won at least one mapped precursor
    becomes a candidate row. Features are binary: did this candidate win
    precursor_i? Target: did they win the Oscar?

    Args:
      require_target: If True (default), skip years without an Oscar winner.
                      Set to False for prediction years where no winner exists yet.

    Returns:
      X: (n_candidates, n_precursors) binary feature matrix
      y: (n_candidates,) binary target (1 = won Oscar; all zeros if no winner)
      feature_names: list of precursor short names
      meta: list of (year, candidate_name) for each row
    """
    precursors = category_mapping.get(oscar_cat, [])
    oscar_lookup = build_winner_lookup(df, award_filter="Oscars")
    prec_lookup = build_winner_lookup(df, exclude_award="Oscars")

    feature_names = [
        AWARD_SHORT.get(a, a) + ": " + c[:20] for a, c in precursors
    ]

    rows_X = []
    rows_y = []
    meta = []

    for year in years:
        oscar_winner = oscar_lookup.get(("Oscars", oscar_cat, year))
        if require_target and not oscar_winner:
            continue

        # Collect all precursor winners this year → build candidate set
        candidates: dict[str, np.ndarray] = {}  # canonical_name -> feature vector
        candidate_names: dict[str, str] = {}     # canonical_name -> original name

        for pi, (award_name, prec_cat) in enumerate(precursors):
            pw = prec_lookup.get((award_name, prec_cat, year))
            if not pw:
                continue

            # Find or create candidate
            matched_key = None
            for existing_key in candidates:
                if names_match(pw, existing_key):
                    matched_key = existing_key
                    break

            if matched_key is None:
                matched_key = pw
                candidates[matched_key] = np.zeros(len(precursors))
                candidate_names[matched_key] = pw

            candidates[matched_key][pi] = 1.0

        # Build rows
        for cand_key, feat_vec in candidates.items():
            rows_X.append(feat_vec)
            if oscar_winner:
                is_winner = 1.0 if names_match(cand_key, oscar_winner) else 0.0
            else:
                is_winner = 0.0  # unknown — prediction year
            rows_y.append(is_winner)
            meta.append((year, candidate_names[cand_key]))

    if not rows_X:
        return np.empty((0, len(precursors))), np.empty(0), feature_names, []

    return np.array(rows_X), np.array(rows_y), feature_names, meta


# ============================================================================
# ML MODEL: Logistic Regression (from scratch, L2-regularised)
# ============================================================================

class LogisticRegressionFromScratch:
    """
    L2-regularised logistic regression via gradient descent.
    Minimal implementation — no dependencies beyond numpy.
    """

    def __init__(self, C: float = 1.0, lr: float = 0.1, max_iter: int = 500):
        self.C = C          # regularisation strength (higher = more regularisation)
        self.lr = lr
        self.max_iter = max_iter
        self.w = None
        self.b = 0.0

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.max_iter):
            z = X @ self.w + self.b
            p = self._sigmoid(z)
            error = p - y

            grad_w = (X.T @ error) / n + self.C * self.w
            grad_b = error.mean()

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

        return self

    def predict_proba(self, X):
        z = X @ self.w + self.b
        return self._sigmoid(z)


# ============================================================================
# ML MODEL: Decision Tree (from scratch, shallow)
# ============================================================================

class DecisionTreeFromScratch:
    """
    Minimal binary classification tree. Splits on the binary feature that
    maximises information gain. Constrained by max_depth and min_samples_leaf.
    """

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 3):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        p = y.mean()
        return 2 * p * (1 - p)

    def _best_split(self, X, y):
        best_gain = -1
        best_feat = None
        parent_gini = self._gini(y)
        n = len(y)

        for j in range(X.shape[1]):
            left_mask = X[:, j] == 1
            right_mask = ~left_mask

            if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                continue

            gain = parent_gini - (
                left_mask.sum() / n * self._gini(y[left_mask])
                + right_mask.sum() / n * self._gini(y[right_mask])
            )
            if gain > best_gain:
                best_gain = gain
                best_feat = j

        return best_feat, best_gain

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return {"leaf": True, "prob": y.mean() if len(y) > 0 else 0.5}

        feat, gain = self._best_split(X, y)
        if feat is None or gain <= 0:
            return {"leaf": True, "prob": y.mean() if len(y) > 0 else 0.5}

        left_mask = X[:, feat] == 1
        return {
            "leaf": False,
            "feature": feat,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)
        return self

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["prob"]
        if x[node["feature"]] == 1:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict_proba(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


# ============================================================================
# ML MODEL: Random Forest (from scratch)
# ============================================================================

class RandomForestFromScratch:
    """
    Ensemble of shallow decision trees with feature bagging.
    """

    def __init__(
        self, n_estimators: int = 80, max_depth: int = 3,
        min_samples_leaf: int = 3, max_features: float = 0.7,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_subsets = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n, d = X.shape
        n_features = max(1, int(d * self.max_features))

        self.trees = []
        self.feature_subsets = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = rng.choice(n, size=n, replace=True)
            feat_idx = rng.choice(d, size=n_features, replace=False)

            tree = DecisionTreeFromScratch(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X[np.ix_(indices, feat_idx)], y[indices])
            self.trees.append(tree)
            self.feature_subsets.append(feat_idx)

        return self

    def predict_proba(self, X):
        preds = np.zeros(X.shape[0])
        for tree, feat_idx in zip(self.trees, self.feature_subsets):
            preds += tree.predict_proba(X[:, feat_idx])
        return preds / len(self.trees)


# ============================================================================
# ML MODEL: Gradient Boosting (from scratch, boosted stumps)
# ============================================================================

class GradientBoostingFromScratch:
    """
    Gradient boosted decision stumps (depth=1 or 2) for binary classification.
    Uses log-loss gradient. Heavy regularisation via learning rate and depth.
    """

    def __init__(
        self, n_estimators: int = 50, max_depth: int = 2,
        learning_rate: float = 0.1, min_samples_leaf: int = 5,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []
        self.init_pred = 0.0

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        # Initialise with log-odds
        p = y.mean()
        p = np.clip(p, 0.01, 0.99)
        self.init_pred = np.log(p / (1 - p))

        F = np.full(len(y), self.init_pred)
        self.trees = []

        for _ in range(self.n_estimators):
            # Negative gradient (pseudo-residuals)
            p = self._sigmoid(F)
            residuals = y - p

            # Fit a shallow tree to residuals
            tree = DecisionTreeFromScratch(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X, residuals)  # Note: tree predicts mean of residuals in leaf

            # Update
            update = tree.predict_proba(X)  # actually predicts residual means
            F += self.learning_rate * update
            self.trees.append(tree)

        return self

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            F += self.learning_rate * tree.predict_proba(X)
        return self._sigmoid(F)


# ============================================================================
# ML model wrappers (same interface as precursor models)
# ============================================================================

def _ml_model_predict(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    accuracy: dict[str, list[PrecursorAccuracy]],
    model_class,
    model_kwargs: dict,
    model_name: str,
    prediction_year: int = PREDICTION_YEAR,
) -> list[Prediction]:
    """
    Generic wrapper: train an ML model on historical data, predict the
    given year. For each Oscar category independently.
    """
    predictions = []

    for oscar_cat in OSCAR_CATEGORIES:
        precursors = category_mapping.get(oscar_cat, [])
        train_years = list(range(HISTORICAL_START, prediction_year))

        X_train, y_train, feat_names, meta_train = build_candidate_features(
            df, category_mapping, oscar_cat, train_years,
        )

        # Build prediction-year features (no Oscar winner exists yet)
        X_pred, _, _, meta_pred = build_candidate_features(
            df, category_mapping, oscar_cat, [prediction_year],
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

        # Train
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)

        # Predict
        probs = model.predict_proba(X_pred)

        # Build prediction
        scored = sorted(
            zip(meta_pred, probs),
            key=lambda x: -x[1],
        )

        total = sum(p for _, p in scored)
        if total == 0:
            total = 1.0

        all_candidates = [
            (name, round(prob / total * 100, 1))
            for (year, name), prob in scored[:5]
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


def model_logistic_regression(df, category_mapping, accuracy, **kw):
    return _ml_model_predict(
        df, category_mapping, accuracy,
        model_class=LogisticRegressionFromScratch,
        model_kwargs={"C": 0.5, "lr": 0.05, "max_iter": 1000},
        model_name="Logistic Regression",
        **kw,
    )


def model_random_forest(df, category_mapping, accuracy, **kw):
    return _ml_model_predict(
        df, category_mapping, accuracy,
        model_class=RandomForestFromScratch,
        model_kwargs={"n_estimators": 80, "max_depth": 3, "min_samples_leaf": 3},
        model_name="Random Forest",
        **kw,
    )


def model_gradient_boosting(df, category_mapping, accuracy, **kw):
    return _ml_model_predict(
        df, category_mapping, accuracy,
        model_class=GradientBoostingFromScratch,
        model_kwargs={
            "n_estimators": 50, "max_depth": 2,
            "learning_rate": 0.1, "min_samples_leaf": 5,
        },
        model_name="Gradient Boosting",
        **kw,
    )


# ============================================================================
# Model Registry
# ============================================================================

ALL_MODELS = {
    "weighted":  model_weighted_precursor,
    "majority":  model_simple_majority,
    "recency":   model_recency_weighted,
    "top3":      lambda df, cm, acc, **kw: model_top_n_precursors(df, cm, acc, n=3, **kw),
    "top5":      lambda df, cm, acc, **kw: model_top_n_precursors(df, cm, acc, n=5, **kw),
    "momentum":  model_consensus_momentum,
    "logreg":    model_logistic_regression,
    "rf":        model_random_forest,
    "gbm":       model_gradient_boosting,
}


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
    """
    import copy

    if categories is None:
        categories = [
            "Best Picture", "Best Director", "Best Actor", "Best Actress",
            "Best Supporting Actor", "Best Supporting Actress",
        ]
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


def backtest_ml_model(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    model_class,
    model_kwargs: dict,
    model_key: str,
    categories: list[str] | None = None,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Leave-one-year-out backtesting for ML models.

    For each held-out year:
      1. Train on all OTHER years
      2. Predict the held-out year's candidates
      3. Check if the top-predicted candidate is the actual Oscar winner
    """
    if categories is None:
        categories = [
            "Best Picture", "Best Director", "Best Actor", "Best Actress",
            "Best Supporting Actor", "Best Supporting Actress",
        ]
    if years is None:
        years = list(range(max(HISTORICAL_START + 5, 2005), HISTORICAL_END + 1))
        # Start from 2005+ to have enough training data

    oscar_lookup = build_winner_lookup(df, award_filter="Oscars")
    results = []

    for year in years:
        train_years = [y for y in range(HISTORICAL_START, HISTORICAL_END + 1) if y != year]

        for oscar_cat in categories:
            actual = oscar_lookup.get(("Oscars", oscar_cat, year), "")
            if not actual:
                continue

            X_train, y_train, _, _ = build_candidate_features(
                df, category_mapping, oscar_cat, train_years,
            )
            X_test, _, _, meta_test = build_candidate_features(
                df, category_mapping, oscar_cat, [year],
            )

            if len(X_train) < 10 or len(X_test) == 0:
                continue

            model = model_class(**model_kwargs)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)

            # Pick candidate with highest predicted probability
            best_idx = np.argmax(probs)
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


def run_full_backtest(df, category_mapping, accuracy) -> pd.DataFrame:
    """Run backtesting for all models and return combined results."""

    all_results = []

    # Precursor models
    precursor_models = {
        "Weighted": model_weighted_precursor,
        "Majority": model_simple_majority,
        "Recency": model_recency_weighted,
        "Top-3": lambda df, cm, acc, **kw: model_top_n_precursors(df, cm, acc, n=3, **kw),
        "Top-5": lambda df, cm, acc, **kw: model_top_n_precursors(df, cm, acc, n=5, **kw),
        "Momentum": model_consensus_momentum,
    }

    for key, fn in precursor_models.items():
        print(f"  Backtesting: {key}...")
        res = backtest_precursor_model(df, category_mapping, fn, key)
        all_results.append(res)

    # ML models
    ml_models = {
        "LogReg": (LogisticRegressionFromScratch, {"C": 0.5, "lr": 0.05, "max_iter": 1000}),
        "RandomForest": (RandomForestFromScratch, {"n_estimators": 80, "max_depth": 3, "min_samples_leaf": 3}),
        "GBM": (GradientBoostingFromScratch, {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.1, "min_samples_leaf": 5}),
    }

    for key, (cls, kwargs) in ml_models.items():
        print(f"  Backtesting: {key}...")
        res = backtest_ml_model(df, category_mapping, cls, kwargs, key)
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
        categories = [
            "Best Picture", "Best Director", "Best Actor", "Best Actress",
            "Best Supporting Actor", "Best Supporting Actress",
        ]

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
