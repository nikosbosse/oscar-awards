"""
build_data.py — Export all analysis data as JSON for the interactive website
=============================================================================

Computes all existing EDA analyses plus new ones (streaks, complementary pairs,
decade trends, bellwether analysis, biggest upsets ranked by consensus, etc.)
and exports everything as website_data.json.

Usage:
  .venv/bin/python build_data.py
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

from helpers import (
    SCRIPT_DIR,
    DATA_FILES,
    HISTORICAL_START,
    HISTORICAL_END,
    PREDICTION_YEAR,
    AWARD_SHORT,
    OSCAR_CATEGORIES,
    load_data,
    build_category_mapping,
    build_winner_lookup,
    compute_historical_accuracy,
    names_match,
    PrecursorAccuracy,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_PATH = SCRIPT_DIR / "website_data.json"

AWARD_TYPES = {
    "Industry Guilds": ["DGA", "PGA", "SAG", "WGA", "ACE Eddie"],
    "Major Shows": ["BAFTA", "Golden Globes", "Critics Choice"],
    "Critics Circles": ["NYFCC", "LAFCA", "NSFC", "NBR"],
    "Festivals": ["Venice", "Cannes", "TIFF"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short(award_name: str) -> str:
    return AWARD_SHORT.get(award_name, award_name)


def _build_lookups(df: pd.DataFrame):
    """Build oscar and precursor winner lookups."""
    oscar_lookup = {}
    for _, row in df[df["Award"] == "Oscars"].iterrows():
        if pd.notna(row["winner"]) and row["winner"] != "":
            oscar_lookup[(row["Year of award ceremony"], row["Category"])] = row["winner"].strip()

    prec_lookup = {}
    for _, row in df[df["Award"] != "Oscars"].iterrows():
        if pd.notna(row["winner"]) and row["winner"] != "":
            prec_lookup[(row["Award"], row["Category"], row["Year of award ceremony"])] = row["winner"].strip()

    return oscar_lookup, prec_lookup


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def compute_metadata(df: pd.DataFrame, category_mapping: dict) -> dict:
    awards = sorted(df[df["Award"] != "Oscars"]["Award"].unique().tolist())
    return {
        "total_records": len(df),
        "year_range": [int(HISTORICAL_START), int(HISTORICAL_END)],
        "prediction_year": int(PREDICTION_YEAR),
        "award_shows": awards,
        "award_short_names": {k: v for k, v in AWARD_SHORT.items() if k != "Oscars"},
        "oscar_categories": OSCAR_CATEGORIES,
    }


# ---------------------------------------------------------------------------
# Overall accuracy
# ---------------------------------------------------------------------------

def compute_overall_accuracy(accuracy: dict) -> dict:
    award_stats = defaultdict(lambda: {"hits": 0, "total": 0, "categories": 0})
    for oscar_cat, pa_list in accuracy.items():
        for pa in pa_list:
            if pa.total > 0:
                short = _short(pa.award)
                award_stats[short]["hits"] += pa.matches
                award_stats[short]["total"] += pa.total
                award_stats[short]["categories"] += 1

    result = {}
    for award, stats in award_stats.items():
        if stats["total"] > 0:
            result[award] = {
                "accuracy": round(stats["hits"] / stats["total"], 4),
                "matches": stats["hits"],
                "total": stats["total"],
                "categories_covered": stats["categories"],
            }
    return result


# ---------------------------------------------------------------------------
# Accuracy by category
# ---------------------------------------------------------------------------

def compute_accuracy_by_category(accuracy: dict) -> dict:
    result = {}
    for oscar_cat, pa_list in accuracy.items():
        cat_data = {}
        for pa in pa_list:
            if pa.total >= 3:
                short = _short(pa.award)
                cat_data[short] = {
                    "accuracy": round(pa.accuracy, 4),
                    "matches": pa.matches,
                    "total": pa.total,
                }
        if cat_data:
            result[oscar_cat] = cat_data
    return result


# ---------------------------------------------------------------------------
# Accuracy over time (year-by-year per precursor per category)
# ---------------------------------------------------------------------------

def compute_accuracy_over_time(
    df: pd.DataFrame,
    category_mapping: dict,
    oscar_lookup: dict,
    prec_lookup: dict,
) -> dict:
    """Year-by-year match (1/0) for each precursor in each Oscar category."""
    result = {}
    for oscar_cat, precursors in category_mapping.items():
        cat_data = {}
        for award_name, prec_cat in precursors:
            short = _short(award_name)
            yearly = {}
            for year in range(HISTORICAL_START, HISTORICAL_END + 1):
                oscar_w = oscar_lookup.get((year, oscar_cat))
                prec_w = prec_lookup.get((award_name, prec_cat, year))
                if oscar_w and prec_w:
                    yearly[str(year)] = 1 if names_match(oscar_w, prec_w) else 0
            if yearly:
                # If multiple precursor categories map to the same short name,
                # store under a disambiguated key
                key = short
                if key in cat_data:
                    key = f"{short} ({prec_cat[:40]})"
                cat_data[key] = yearly
        if cat_data:
            result[oscar_cat] = cat_data
    return result


# ---------------------------------------------------------------------------
# Award agreement (pairwise)
# ---------------------------------------------------------------------------

def compute_award_agreement(
    category_mapping: dict,
    prec_lookup: dict,
) -> dict:
    """For each Oscar category, compute pairwise agreement between precursors."""
    result = {}
    for oscar_cat, precursors in category_mapping.items():
        # Filter to precursors with enough data
        valid_precs = []
        for award_name, prec_cat in precursors:
            count = sum(
                1 for y in range(HISTORICAL_START, HISTORICAL_END + 1)
                if prec_lookup.get((award_name, prec_cat, y))
            )
            if count >= 5:
                valid_precs.append((award_name, prec_cat))

        if len(valid_precs) < 2:
            continue

        pairs = {}
        for i, (a1, c1) in enumerate(valid_precs):
            for j, (a2, c2) in enumerate(valid_precs):
                if i >= j:
                    continue
                matches = 0
                total = 0
                for year in range(HISTORICAL_START, HISTORICAL_END + 1):
                    w1 = prec_lookup.get((a1, c1, year))
                    w2 = prec_lookup.get((a2, c2, year))
                    if w1 and w2:
                        total += 1
                        if names_match(w1, w2):
                            matches += 1
                if total >= 3:
                    s1 = _short(a1)
                    s2 = _short(a2)
                    pair_key = f"{s1}-{s2}"
                    pairs[pair_key] = round(matches / total, 4)

        if pairs:
            result[oscar_cat] = pairs

    return result


# ---------------------------------------------------------------------------
# Upsets — ranked by consensus strength
# ---------------------------------------------------------------------------

def compute_upsets(
    category_mapping: dict,
    oscar_lookup: dict,
    prec_lookup: dict,
) -> list[dict]:
    """Find upsets where Oscar winner disagreed with precursor consensus.
    Ranked by how strong the consensus was for a different pick."""
    upsets = []

    for oscar_cat, precursors in category_mapping.items():
        for year in range(HISTORICAL_START, HISTORICAL_END + 1):
            oscar_w = oscar_lookup.get((year, oscar_cat))
            if not oscar_w:
                continue

            # Tally precursor picks
            nominee_counts = defaultdict(int)
            precursor_picks = {}
            total_reporting = 0

            for award_name, prec_cat in precursors:
                pw = prec_lookup.get((award_name, prec_cat, year))
                if pw:
                    total_reporting += 1
                    precursor_picks[_short(award_name)] = pw
                    # Cluster nominees
                    matched_existing = False
                    for existing in list(nominee_counts.keys()):
                        if names_match(pw, existing):
                            nominee_counts[existing] += 1
                            matched_existing = True
                            break
                    if not matched_existing:
                        nominee_counts[pw] += 1

            if total_reporting < 3:
                continue

            # Find the frontrunner
            frontrunner = max(nominee_counts, key=nominee_counts.get)
            consensus_pct = nominee_counts[frontrunner] / total_reporting

            # Did the frontrunner win?
            frontrunner_won = names_match(frontrunner, oscar_w)

            if not frontrunner_won:
                # This is an upset
                # How many precursors picked the actual winner?
                winner_picks = 0
                for pw_name, pw_count in nominee_counts.items():
                    if names_match(pw_name, oscar_w):
                        winner_picks = pw_count
                        break

                upsets.append({
                    "year": int(year),
                    "category": oscar_cat,
                    "winner": oscar_w,
                    "frontrunner": frontrunner,
                    "consensus_pct": round(consensus_pct, 4),
                    "winner_precursor_pct": round(winner_picks / total_reporting, 4),
                    "total_precursors": total_reporting,
                    "precursor_picks": precursor_picks,
                })

    # Sort by consensus_pct descending (biggest upsets first = strongest consensus wrong)
    upsets.sort(key=lambda x: -x["consensus_pct"])
    return upsets


# ---------------------------------------------------------------------------
# Sweep stats
# ---------------------------------------------------------------------------

def compute_sweep_stats(
    category_mapping: dict,
    oscar_lookup: dict,
    prec_lookup: dict,
) -> dict:
    result = {}
    for oscar_cat, precursors in category_mapping.items():
        years_data = {}
        for year in range(HISTORICAL_START, HISTORICAL_END + 1):
            oscar_w = oscar_lookup.get((year, oscar_cat))
            if not oscar_w:
                continue
            prec_wins = 0
            prec_available = 0
            for award_name, prec_cat in precursors:
                pw = prec_lookup.get((award_name, prec_cat, year))
                if pw:
                    prec_available += 1
                    if names_match(oscar_w, pw):
                        prec_wins += 1
            if prec_available > 0:
                years_data[str(year)] = {
                    "pct": round(prec_wins / prec_available, 4),
                    "wins": prec_wins,
                    "available": prec_available,
                    "oscar_winner": oscar_w,
                }
        if years_data:
            result[oscar_cat] = {"years": years_data}
    return result


# ---------------------------------------------------------------------------
# Category predictability
# ---------------------------------------------------------------------------

def compute_category_predictability(accuracy: dict) -> dict:
    result = {}
    for oscar_cat, pa_list in accuracy.items():
        valid = [pa for pa in pa_list if pa.total >= 5]
        if not valid:
            continue
        best = max(valid, key=lambda pa: pa.accuracy)
        avg = float(np.mean([pa.accuracy for pa in valid]))
        result[oscar_cat] = {
            "best_precursor": _short(best.award),
            "best_accuracy": round(best.accuracy, 4),
            "avg_accuracy": round(avg, 4),
            "num_precursors": len(valid),
        }
    return result


# ---------------------------------------------------------------------------
# Award type comparison
# ---------------------------------------------------------------------------

def compute_award_type_comparison(accuracy: dict) -> dict:
    result = {}
    for type_name, awards in AWARD_TYPES.items():
        hits = 0
        total = 0
        for oscar_cat, pa_list in accuracy.items():
            for pa in pa_list:
                short = _short(pa.award)
                if short in awards and pa.total > 0:
                    hits += pa.matches
                    total += pa.total
        if total > 0:
            result[type_name] = {
                "avg_accuracy": round(hits / total, 4),
                "matches": hits,
                "total": total,
                "awards": awards,
            }
    return result


# ---------------------------------------------------------------------------
# Consensus vs outcome
# ---------------------------------------------------------------------------

def compute_consensus_vs_outcome(
    category_mapping: dict,
    oscar_lookup: dict,
    prec_lookup: dict,
) -> list[dict]:
    """Bin consensus strength and compute win rates."""
    data = []
    for oscar_cat, precursors in category_mapping.items():
        for year in range(HISTORICAL_START, HISTORICAL_END + 1):
            oscar_w = oscar_lookup.get((year, oscar_cat))
            if not oscar_w:
                continue

            nominee_counts = defaultdict(int)
            total_reporting = 0
            for award_name, prec_cat in precursors:
                pw = prec_lookup.get((award_name, prec_cat, year))
                if pw:
                    total_reporting += 1
                    matched_existing = False
                    for existing in list(nominee_counts.keys()):
                        if names_match(pw, existing):
                            nominee_counts[existing] += 1
                            matched_existing = True
                            break
                    if not matched_existing:
                        nominee_counts[pw] += 1

            if total_reporting < 3:
                continue

            frontrunner = max(nominee_counts, key=nominee_counts.get)
            consensus = nominee_counts[frontrunner] / total_reporting
            frontrunner_won = names_match(frontrunner, oscar_w)

            data.append({
                "year": int(year),
                "category": oscar_cat,
                "consensus": round(consensus, 4),
                "frontrunner_won": frontrunner_won,
                "frontrunner": frontrunner,
                "oscar_winner": oscar_w,
            })

    # Bin by consensus strength
    bins = [(0, 0.3, "0-30%"), (0.3, 0.5, "30-50%"), (0.5, 0.7, "50-70%"),
            (0.7, 0.9, "70-90%"), (0.9, 1.01, "90-100%")]

    binned = []
    for lo, hi, label in bins:
        matching = [d for d in data if lo <= d["consensus"] < hi]
        if matching:
            win_rate = sum(1 for d in matching if d["frontrunner_won"]) / len(matching)
            binned.append({
                "consensus_bin": label,
                "win_rate": round(win_rate, 4),
                "count": len(matching),
            })
        else:
            binned.append({"consensus_bin": label, "win_rate": 0.0, "count": 0})

    return binned


# ---------------------------------------------------------------------------
# Streaks — longest consecutive correct/incorrect per precursor per category
# ---------------------------------------------------------------------------

def compute_streaks(
    category_mapping: dict,
    oscar_lookup: dict,
    prec_lookup: dict,
) -> dict:
    result = {}
    for oscar_cat, precursors in category_mapping.items():
        cat_streaks = {}
        for award_name, prec_cat in precursors:
            short = _short(award_name)
            # Build sequence of results
            seq = []
            for year in range(HISTORICAL_START, HISTORICAL_END + 1):
                oscar_w = oscar_lookup.get((year, oscar_cat))
                prec_w = prec_lookup.get((award_name, prec_cat, year))
                if oscar_w and prec_w:
                    seq.append((year, names_match(oscar_w, prec_w)))

            if not seq:
                continue

            # Compute streaks
            longest_correct = 0
            longest_incorrect = 0
            current_correct = 0
            current_incorrect = 0

            for _, matched in seq:
                if matched:
                    current_correct += 1
                    current_incorrect = 0
                else:
                    current_incorrect += 1
                    current_correct = 0
                longest_correct = max(longest_correct, current_correct)
                longest_incorrect = max(longest_incorrect, current_incorrect)

            # Current streak (from most recent)
            current_val = seq[-1][1]
            current_streak = 0
            for _, matched in reversed(seq):
                if matched == current_val:
                    current_streak += 1
                else:
                    break

            key = short
            if key in cat_streaks:
                key = f"{short} ({prec_cat[:30]})"

            cat_streaks[key] = {
                "longest_correct": longest_correct,
                "longest_incorrect": longest_incorrect,
                "current_streak": current_streak,
                "current_streak_type": "correct" if current_val else "incorrect",
                "total_comparisons": len(seq),
            }

        if cat_streaks:
            result[oscar_cat] = cat_streaks

    return result


# ---------------------------------------------------------------------------
# Complementary pairs
# ---------------------------------------------------------------------------

def compute_complementary_pairs(
    accuracy: dict,
    category_mapping: dict,
    oscar_lookup: dict,
    prec_lookup: dict,
) -> list[dict]:
    """Find pairs of precursors that are most complementary:
    high joint accuracy but low individual correlation."""
    pairs = []

    for oscar_cat, precursors in category_mapping.items():
        # Need per-year results for each precursor
        prec_results = {}
        for award_name, prec_cat in precursors:
            short = _short(award_name)
            yearly = {}
            for year in range(HISTORICAL_START, HISTORICAL_END + 1):
                oscar_w = oscar_lookup.get((year, oscar_cat))
                prec_w = prec_lookup.get((award_name, prec_cat, year))
                if oscar_w and prec_w:
                    yearly[year] = 1 if names_match(oscar_w, prec_w) else 0
            if len(yearly) >= 5:
                key = (short, award_name, prec_cat)
                prec_results[key] = yearly

        keys = list(prec_results.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                r1, r2 = prec_results[k1], prec_results[k2]

                # Find common years
                common_years = sorted(set(r1.keys()) & set(r2.keys()))
                if len(common_years) < 5:
                    continue

                v1 = [r1[y] for y in common_years]
                v2 = [r2[y] for y in common_years]

                # Joint accuracy: at least one is correct
                joint_correct = sum(1 for a, b in zip(v1, v2) if a or b)
                joint_accuracy = joint_correct / len(common_years)

                # Individual correlation
                if np.std(v1) > 0 and np.std(v2) > 0:
                    corr = float(np.corrcoef(v1, v2)[0, 1])
                else:
                    corr = 1.0  # Treat zero-variance as fully correlated

                # Complementary = high joint accuracy + low correlation
                complementary_score = joint_accuracy * (1 - abs(corr))

                pairs.append({
                    "category": oscar_cat,
                    "pair": [k1[0], k2[0]],
                    "joint_accuracy": round(joint_accuracy, 4),
                    "individual_corr": round(corr, 4),
                    "complementary_score": round(complementary_score, 4),
                    "common_years": len(common_years),
                    "individual_accuracy_1": round(sum(v1) / len(v1), 4),
                    "individual_accuracy_2": round(sum(v2) / len(v2), 4),
                })

    # Sort by complementary score descending
    pairs.sort(key=lambda x: -x["complementary_score"])
    return pairs[:100]  # Top 100


# ---------------------------------------------------------------------------
# Decade trends
# ---------------------------------------------------------------------------

def compute_decade_trends(
    accuracy: dict,
    category_mapping: dict,
    oscar_lookup: dict,
    prec_lookup: dict,
) -> dict:
    decades = {
        "2000s": (2000, 2009),
        "2010s": (2010, 2019),
        "2020s": (2020, 2025),
    }

    result = {}
    for decade_name, (start, end) in decades.items():
        total_correct = 0
        total_comparisons = 0
        per_category = {}

        for oscar_cat, precursors in category_mapping.items():
            cat_correct = 0
            cat_total = 0
            for award_name, prec_cat in precursors:
                for year in range(start, end + 1):
                    oscar_w = oscar_lookup.get((year, oscar_cat))
                    prec_w = prec_lookup.get((award_name, prec_cat, year))
                    if oscar_w and prec_w:
                        cat_total += 1
                        total_comparisons += 1
                        if names_match(oscar_w, prec_w):
                            cat_correct += 1
                            total_correct += 1
            if cat_total > 0:
                per_category[oscar_cat] = round(cat_correct / cat_total, 4)

        result[decade_name] = {
            "avg_predictability": round(total_correct / total_comparisons, 4) if total_comparisons > 0 else 0,
            "total_comparisons": total_comparisons,
            "per_category": per_category,
        }

    return result


# ---------------------------------------------------------------------------
# Bellwether analysis — which single precursor most often breaks ties
# ---------------------------------------------------------------------------

def compute_bellwether(
    category_mapping: dict,
    oscar_lookup: dict,
    prec_lookup: dict,
) -> dict:
    """For contested years (consensus < 60%), which precursor's pick
    most often matches the eventual Oscar winner?"""
    # Count tiebreaker successes per precursor (across all categories)
    tiebreaker_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    per_category = {}

    for oscar_cat, precursors in category_mapping.items():
        cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for year in range(HISTORICAL_START, HISTORICAL_END + 1):
            oscar_w = oscar_lookup.get((year, oscar_cat))
            if not oscar_w:
                continue

            # Compute consensus
            nominee_counts = defaultdict(int)
            total_reporting = 0
            prec_results_year = {}  # short -> winner

            for award_name, prec_cat in precursors:
                pw = prec_lookup.get((award_name, prec_cat, year))
                if pw:
                    total_reporting += 1
                    short = _short(award_name)
                    prec_results_year[(short, award_name, prec_cat)] = pw
                    matched_existing = False
                    for existing in list(nominee_counts.keys()):
                        if names_match(pw, existing):
                            nominee_counts[existing] += 1
                            matched_existing = True
                            break
                    if not matched_existing:
                        nominee_counts[pw] += 1

            if total_reporting < 3:
                continue

            frontrunner = max(nominee_counts, key=nominee_counts.get)
            consensus = nominee_counts[frontrunner] / total_reporting

            # Only look at contested races
            if consensus >= 0.6:
                continue

            # For each precursor that reported, did it pick the winner?
            for (short, award_name, prec_cat), pw in prec_results_year.items():
                tiebreaker_stats[short]["total"] += 1
                cat_stats[short]["total"] += 1
                if names_match(pw, oscar_w):
                    tiebreaker_stats[short]["correct"] += 1
                    cat_stats[short]["correct"] += 1

        cat_result = {}
        for short, stats in cat_stats.items():
            if stats["total"] >= 2:
                cat_result[short] = {
                    "accuracy": round(stats["correct"] / stats["total"], 4),
                    "correct": stats["correct"],
                    "total": stats["total"],
                }
        if cat_result:
            per_category[oscar_cat] = cat_result

    overall = {}
    for short, stats in tiebreaker_stats.items():
        if stats["total"] >= 5:
            overall[short] = {
                "accuracy": round(stats["correct"] / stats["total"], 4),
                "correct": stats["correct"],
                "total": stats["total"],
            }

    return {
        "overall": overall,
        "per_category": per_category,
    }


# ---------------------------------------------------------------------------
# Backtest results
# ---------------------------------------------------------------------------

def compute_backtest_results() -> dict:
    bt_path = SCRIPT_DIR / "backtest_results.csv"
    if not bt_path.exists():
        return {}

    bt_df = pd.read_csv(bt_path)

    # Convert Correct column
    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() == "true"
        return bool(val)

    bt_df["Correct"] = bt_df["Correct"].apply(to_bool).astype(int)

    models = sorted(bt_df["Model"].unique().tolist())
    by_model = {}

    for model in models:
        mdf = bt_df[bt_df["Model"] == model]
        overall_acc = float(mdf["Correct"].mean())
        total = len(mdf)
        correct = int(mdf["Correct"].sum())

        # By year
        by_year = {}
        for year, ydf in mdf.groupby("Year"):
            by_year[str(int(year))] = {
                "accuracy": round(float(ydf["Correct"].mean()), 4),
                "correct": int(ydf["Correct"].sum()),
                "total": len(ydf),
            }

        # By category
        by_category = {}
        for cat, cdf in mdf.groupby("Category"):
            by_category[cat] = {
                "accuracy": round(float(cdf["Correct"].mean()), 4),
                "correct": int(cdf["Correct"].sum()),
                "total": len(cdf),
            }

        by_model[model] = {
            "accuracy": round(overall_acc, 4),
            "correct": correct,
            "total": total,
            "by_year": by_year,
            "by_category": by_category,
        }

    # Calibration
    from calibration import compute_calibration, _compute_ece, BIN_LABELS

    calibration = {}
    for model in models:
        cal_df = compute_calibration(bt_df, model=model)
        total = cal_df["count"].sum()
        ece = float(_compute_ece(cal_df, total)) if total > 0 else 0.0

        mdf = bt_df[bt_df["Model"] == model]
        brier = float(((mdf["Confidence"] - mdf["Correct"]) ** 2).mean())

        bins_data = []
        for _, row in cal_df.iterrows():
            bins_data.append({
                "bin_label": str(row["bin_label"]),
                "mean_confidence": round(float(row["mean_confidence"]), 4),
                "actual_accuracy": round(float(row["actual_accuracy"]), 4),
                "count": int(row["count"]),
            })

        calibration[model] = {
            "ece": round(ece, 4),
            "brier": round(brier, 4),
            "bins": bins_data,
        }

    return {
        "models": models,
        "by_model": by_model,
        "calibration": calibration,
    }


# ---------------------------------------------------------------------------
# 2026 Predictions
# ---------------------------------------------------------------------------

def compute_predictions_2026(df, category_mapping, accuracy) -> list[dict]:
    from predictions import run_all_models

    all_results = run_all_models(df, category_mapping, accuracy)

    predictions = []
    for oscar_cat in OSCAR_CATEGORIES:
        cat_preds = {}
        for model_key, preds in all_results.items():
            p = next((x for x in preds if x.oscar_category == oscar_cat), None)
            if p and p.confidence > 0:
                cat_preds[p.model_name] = {
                    "winner": p.predicted_winner,
                    "confidence": round(p.confidence, 4),
                    "runner_up": p.runner_up if p.runner_up else None,
                    "runner_up_confidence": round(p.runner_up_confidence, 4) if p.runner_up else None,
                    "all_candidates": [
                        {"name": name, "confidence_pct": conf}
                        for name, conf in p.all_candidates[:5]
                    ],
                }
        if cat_preds:
            predictions.append({
                "category": oscar_cat,
                "predictions": cat_preds,
            })

    return predictions


# ---------------------------------------------------------------------------
# Precursor value-add analysis
# ---------------------------------------------------------------------------

def compute_precursor_value_add(
    accuracy: dict,
    category_mapping: dict,
    oscar_lookup: dict,
    prec_lookup: dict,
) -> dict:
    """For each precursor, compute how often it correctly predicts the Oscar
    winner when the majority of other precursors are wrong."""
    result = {}

    for oscar_cat, precursors in category_mapping.items():
        cat_data = {}
        for idx, (award_name, prec_cat) in enumerate(precursors):
            short = _short(award_name)
            unique_correct = 0  # times THIS precursor was right but majority was wrong
            total_contested = 0  # times majority was wrong
            total_years = 0

            for year in range(HISTORICAL_START, HISTORICAL_END + 1):
                oscar_w = oscar_lookup.get((year, oscar_cat))
                prec_w = prec_lookup.get((award_name, prec_cat, year))
                if not oscar_w or not prec_w:
                    continue

                total_years += 1
                this_correct = names_match(prec_w, oscar_w)

                # Check majority of OTHER precursors
                other_correct = 0
                other_total = 0
                for j, (other_award, other_cat) in enumerate(precursors):
                    if j == idx:
                        continue
                    other_w = prec_lookup.get((other_award, other_cat, year))
                    if other_w:
                        other_total += 1
                        if names_match(other_w, oscar_w):
                            other_correct += 1

                if other_total >= 2:
                    majority_correct = other_correct > other_total / 2
                    if not majority_correct:
                        total_contested += 1
                        if this_correct:
                            unique_correct += 1

            if total_years >= 5:
                key = short
                if key in cat_data:
                    key = f"{short} ({prec_cat[:30]})"
                cat_data[key] = {
                    "unique_correct": unique_correct,
                    "contested_races": total_contested,
                    "value_add_rate": round(unique_correct / total_contested, 4) if total_contested > 0 else 0.0,
                    "total_years": total_years,
                }

        if cat_data:
            result[oscar_cat] = cat_data

    return result


# ---------------------------------------------------------------------------
# Year-by-year detailed data (for the timeline)
# ---------------------------------------------------------------------------

def compute_yearly_detail(
    category_mapping: dict,
    oscar_lookup: dict,
    prec_lookup: dict,
) -> dict:
    """Per-year, per-category: Oscar winner, each precursor's pick, match status."""
    result = {}
    for oscar_cat, precursors in category_mapping.items():
        years_data = {}
        for year in range(HISTORICAL_START, HISTORICAL_END + 1):
            oscar_w = oscar_lookup.get((year, oscar_cat))
            if not oscar_w:
                continue

            prec_data = {}
            for award_name, prec_cat in precursors:
                pw = prec_lookup.get((award_name, prec_cat, year))
                if pw:
                    prec_data[_short(award_name)] = {
                        "pick": pw,
                        "correct": names_match(pw, oscar_w),
                    }

            if prec_data:
                years_data[str(year)] = {
                    "oscar_winner": oscar_w,
                    "precursors": prec_data,
                    "num_correct": sum(1 for v in prec_data.values() if v["correct"]),
                    "num_reporting": len(prec_data),
                }

        if years_data:
            result[oscar_cat] = years_data

    return result


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("  BUILD WEBSITE DATA")
    print("=" * 60)

    print("\nLoading data...")
    df = load_data()
    category_mapping = build_category_mapping()

    print("Computing historical accuracy...")
    accuracy = compute_historical_accuracy(df, category_mapping)

    print("Building lookups...")
    oscar_lookup, prec_lookup = _build_lookups(df)

    data = {}

    print("Computing metadata...")
    data["metadata"] = compute_metadata(df, category_mapping)

    print("Computing overall accuracy...")
    data["overall_accuracy"] = compute_overall_accuracy(accuracy)

    print("Computing accuracy by category...")
    data["accuracy_by_category"] = compute_accuracy_by_category(accuracy)

    print("Computing accuracy over time...")
    data["accuracy_over_time"] = compute_accuracy_over_time(
        df, category_mapping, oscar_lookup, prec_lookup
    )

    print("Computing award agreement...")
    data["award_agreement"] = compute_award_agreement(category_mapping, prec_lookup)

    print("Computing upsets...")
    data["upsets"] = compute_upsets(category_mapping, oscar_lookup, prec_lookup)

    print("Computing sweep stats...")
    data["sweep_stats"] = compute_sweep_stats(category_mapping, oscar_lookup, prec_lookup)

    print("Computing category predictability...")
    data["category_predictability"] = compute_category_predictability(accuracy)

    print("Computing award type comparison...")
    data["award_type_comparison"] = compute_award_type_comparison(accuracy)

    print("Computing consensus vs outcome...")
    data["consensus_vs_outcome"] = compute_consensus_vs_outcome(
        category_mapping, oscar_lookup, prec_lookup
    )

    print("Computing streaks...")
    data["streaks"] = compute_streaks(category_mapping, oscar_lookup, prec_lookup)

    print("Computing complementary pairs...")
    data["complementary_pairs"] = compute_complementary_pairs(
        accuracy, category_mapping, oscar_lookup, prec_lookup
    )

    print("Computing decade trends...")
    data["decade_trends"] = compute_decade_trends(
        accuracy, category_mapping, oscar_lookup, prec_lookup
    )

    print("Computing bellwether analysis...")
    data["bellwether"] = compute_bellwether(category_mapping, oscar_lookup, prec_lookup)

    print("Computing precursor value-add...")
    data["precursor_value_add"] = compute_precursor_value_add(
        accuracy, category_mapping, oscar_lookup, prec_lookup
    )

    print("Computing yearly detail...")
    data["yearly_detail"] = compute_yearly_detail(
        category_mapping, oscar_lookup, prec_lookup
    )

    print("Loading backtest results...")
    data["backtest_results"] = compute_backtest_results()

    print("Computing 2026 predictions...")
    data["predictions_2026"] = compute_predictions_2026(df, category_mapping, accuracy)

    # Write JSON
    print(f"\nWriting {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)

    # Stats
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Top-level keys: {list(data.keys())}")

    print("\n" + "=" * 60)
    print("  DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
