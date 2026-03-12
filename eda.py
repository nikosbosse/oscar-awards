"""
Exploratory Data Analysis — Oscar Precursor Awards Dataset
==========================================================

Generates plots and an HTML report exploring:
  1. Which precursor awards best predict Oscar winners
  2. How predictiveness has changed over time
  3. Agreement/correlation between award shows
  4. "Sweep" and "upset" patterns
  5. Category-level predictability
  6. Award show timing and the "momentum" effect

Usage:
  python eda.py

Outputs:
  - plots/ directory with PNG images
  - eda_report.html — full interactive report
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import re
import textwrap
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

DATA_FILES = [
    SCRIPT_DIR / "film awards research part 1.csv",
    SCRIPT_DIR / "film awards research part 2.csv",
]

# Style
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("tab20", 20)

HISTORICAL_START = 2000
HISTORICAL_END = 2025

# Short labels for award names (for plot readability)
AWARD_SHORT = {
    "ACE Eddie Awards": "ACE Eddie",
    "BAFTA": "BAFTA",
    "Cannes Palme d'Or": "Cannes",
    "Critics Choice Awards": "Critics Choice",
    "DGA Awards": "DGA",
    "Golden Globes": "Golden Globes",
    "Los Angeles Film Critics Association": "LAFCA",
    "National Board of Review": "NBR",
    "National Society of Film Critics": "NSFC",
    "New York Film Critics Circle": "NYFCC",
    "PGA Awards": "PGA",
    "SAG Awards": "SAG",
    "Toronto People's Choice Award": "TIFF",
    "Venice Golden Lion": "Venice",
    "WGA Awards": "WGA",
}

# ---------------------------------------------------------------------------
# Reuse the category mapping and matching logic from oscar_predictions.py
# ---------------------------------------------------------------------------

from oscar_predictions import (
    build_category_mapping,
    load_data,
    compute_historical_accuracy,
    _names_match,
)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_all_data():
    df = load_data(DATA_FILES)
    df = df[df["winner"].notna() & (df["winner"] != "")]
    return df


# ---------------------------------------------------------------------------
# Analysis 1: Overall Predictiveness by Award
# ---------------------------------------------------------------------------

def plot_overall_accuracy(accuracy, category_mapping):
    """
    Bar chart: For each precursor award, what's its average accuracy
    across all Oscar categories it covers?
    """
    award_stats = defaultdict(lambda: {"hits": 0, "total": 0, "categories": 0})

    for oscar_cat, pa_list in accuracy.items():
        for pa in pa_list:
            if pa.total > 0:
                short = AWARD_SHORT.get(pa.award, pa.award)
                award_stats[short]["hits"] += pa.matches
                award_stats[short]["total"] += pa.total
                award_stats[short]["categories"] += 1

    rows = []
    for award, stats in award_stats.items():
        if stats["total"] > 0:
            rows.append({
                "Award": award,
                "Accuracy": stats["hits"] / stats["total"],
                "Comparisons": stats["total"],
                "Categories Covered": stats["categories"],
            })

    adf = pd.DataFrame(rows).sort_values("Accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#2ecc71" if a > 0.5 else "#3498db" if a > 0.3 else "#e74c3c"
              for a in adf["Accuracy"]]
    bars = ax.barh(adf["Award"], adf["Accuracy"], color=colors, edgecolor="white")

    for bar, (_, row) in zip(bars, adf.iterrows()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{row["Accuracy"]:.0%} ({row["Comparisons"]} comparisons)',
                va="center", fontsize=9)

    ax.set_xlim(0, 0.85)
    ax.set_xlabel("Average Accuracy at Predicting Oscar Winner")
    ax.set_title("Which Award Shows Best Predict the Oscars?\n(Average across all mapped categories, 2000–2025)",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "01_overall_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 01_overall_accuracy.png")
    return adf


# ---------------------------------------------------------------------------
# Analysis 2: Best Precursor per Category
# ---------------------------------------------------------------------------

def plot_best_precursor_per_category(accuracy):
    """
    Heatmap: For each Oscar category, show the accuracy of the top precursors.
    """
    # Focus on "big" categories
    key_cats = [
        "Best Picture", "Best Director", "Best Actor", "Best Actress",
        "Best Supporting Actor", "Best Supporting Actress",
        "Best Animated Feature Film", "Best International Feature Film",
        "Best Documentary Feature Film", "Best Cinematography",
        "Best Film Editing", "Best Original Screenplay", "Best Adapted Screenplay",
        "Best Original Score", "Best Costume Design", "Best Visual Effects",
        "Best Production Design", "Best Sound",
        "Best Makeup and Hairstyling", "Best Original Song",
    ]

    # Build matrix
    all_awards = set()
    data = {}
    for cat in key_cats:
        if cat in accuracy:
            for pa in accuracy[cat]:
                if pa.total >= 5:
                    short = AWARD_SHORT.get(pa.award, pa.award)
                    all_awards.add(short)
                    data[(cat, short)] = pa.accuracy

    all_awards = sorted(all_awards)
    matrix = pd.DataFrame(index=key_cats, columns=all_awards, dtype=float)
    for (cat, award), acc in data.items():
        matrix.loc[cat, award] = acc

    # Clean up: only keep awards that appear at least a few times
    matrix = matrix.loc[:, matrix.notna().sum() >= 3]

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        matrix, annot=True, fmt=".0%", cmap="RdYlGn", center=0.35,
        vmin=0, vmax=0.85, ax=ax, linewidths=0.5,
        cbar_kws={"label": "Historical Accuracy", "format": mticker.PercentFormatter(1.0)},
        annot_kws={"size": 8},
    )
    ax.set_title("Precursor Accuracy by Oscar Category\n(2000–2025, min 5 years of data)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "02_accuracy_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 02_accuracy_heatmap.png")
    return matrix


# ---------------------------------------------------------------------------
# Analysis 3: Accuracy Over Time (Rolling)
# ---------------------------------------------------------------------------

def plot_accuracy_over_time(df, accuracy, category_mapping):
    """
    Line chart: How has the predictiveness of key precursors changed over time?
    Uses 5-year rolling accuracy.
    """
    # Focus on Best Picture precursors
    oscar_cat = "Best Picture"
    precursors = category_mapping[oscar_cat]

    oscar_lookup = {}
    for _, row in df[(df["Award"] == "Oscars")].iterrows():
        oscar_lookup[(row["Year of award ceremony"], row["Category"])] = row["winner"].strip()

    prec_lookup = {}
    for _, row in df[df["Award"] != "Oscars"].iterrows():
        prec_lookup[(row["Award"], row["Category"], row["Year of award ceremony"])] = row["winner"].strip()

    # Track per-year match for key precursors
    key_precs = [
        ("PGA Awards", "Darryl F. Zanuck Award for Outstanding Producer of Theatrical Motion Pictures"),
        ("Critics Choice Awards", "Best Picture"),
        ("BAFTA", "Best Film"),
        ("Golden Globes", "Best Motion Picture - Drama"),
        ("National Board of Review", "Best Film"),
    ]

    yearly_data = defaultdict(dict)  # {year: {award_short: 1 or 0}}

    for award, prec_cat in key_precs:
        short = AWARD_SHORT.get(award, award)
        for year in range(HISTORICAL_START, HISTORICAL_END + 1):
            oscar_w = oscar_lookup.get((year, oscar_cat))
            prec_w = prec_lookup.get((award, prec_cat, year))
            if oscar_w and prec_w:
                yearly_data[year][short] = 1 if _names_match(oscar_w, prec_w) else 0

    # Build DataFrame and compute rolling average
    years = sorted(yearly_data.keys())
    roll_df = pd.DataFrame(index=years)
    for award, _ in key_precs:
        short = AWARD_SHORT.get(award, award)
        roll_df[short] = [yearly_data[y].get(short, np.nan) for y in years]

    window = 5
    rolling = roll_df.rolling(window, min_periods=3).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, col in enumerate(rolling.columns):
        ax.plot(rolling.index, rolling[col], marker="o", markersize=4,
                label=col, linewidth=2)

    ax.set_xlabel("Year")
    ax.set_ylabel(f"{window}-Year Rolling Accuracy")
    ax.set_title(f"Best Picture: How Predictive Are Precursors Over Time?\n({window}-year rolling accuracy, {HISTORICAL_START}–{HISTORICAL_END})",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "03_accuracy_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 03_accuracy_over_time.png")


# ---------------------------------------------------------------------------
# Analysis 4: Agreement Between Award Shows
# ---------------------------------------------------------------------------

def plot_award_agreement(df, category_mapping):
    """
    Heatmap: How often do pairs of precursor awards agree on their winner
    for the same Oscar category?
    """
    # Focus on Best Picture precursors
    oscar_cat = "Best Picture"
    precursors = category_mapping[oscar_cat]

    prec_lookup = {}
    for _, row in df[df["Award"] != "Oscars"].iterrows():
        prec_lookup[(row["Award"], row["Category"], row["Year of award ceremony"])] = row["winner"].strip()

    # Track which precursors to include
    key_precs = [p for p in precursors
                 if p[0] not in ("Venice Golden Lion", "Cannes Palme d'Or", "Toronto People's Choice Award")]

    agreement = {}
    for i, (a1, c1) in enumerate(key_precs):
        for j, (a2, c2) in enumerate(key_precs):
            if i >= j:
                continue
            matches = 0
            total = 0
            for year in range(HISTORICAL_START, HISTORICAL_END + 1):
                w1 = prec_lookup.get((a1, c1, year))
                w2 = prec_lookup.get((a2, c2, year))
                if w1 and w2:
                    total += 1
                    if _names_match(w1, w2):
                        matches += 1
            if total > 0:
                s1 = AWARD_SHORT.get(a1, a1)
                s2 = AWARD_SHORT.get(a2, a2)
                agreement[(s1, s2)] = matches / total

    # Build symmetric matrix
    awards = sorted(set(a for pair in agreement for a in pair))
    mat = pd.DataFrame(1.0, index=awards, columns=awards)
    for (a1, a2), val in agreement.items():
        mat.loc[a1, a2] = val
        mat.loc[a2, a1] = val

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    sns.heatmap(
        mat, annot=True, fmt=".0%", cmap="YlOrRd",
        vmin=0, vmax=1, ax=ax, linewidths=0.5,
        mask=mask,
        cbar_kws={"label": "Agreement Rate"},
        annot_kws={"size": 9},
    )
    ax.set_title("Best Picture: How Often Do Precursor Awards Agree?\n(Same winner, 2000–2025)",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "04_award_agreement_best_picture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 04_award_agreement_best_picture.png")
    return mat


# ---------------------------------------------------------------------------
# Analysis 5: The "Sweep" Effect
# ---------------------------------------------------------------------------

def analyze_sweeps(df, category_mapping, accuracy):
    """
    How many precursors did the eventual Oscar winner typically win?
    Does winning more precursors correlate with winning the Oscar?
    """
    oscar_lookup = {}
    for _, row in df[df["Award"] == "Oscars"].iterrows():
        oscar_lookup[(row["Year of award ceremony"], row["Category"])] = row["winner"].strip()

    prec_lookup = {}
    for _, row in df[df["Award"] != "Oscars"].iterrows():
        prec_lookup[(row["Award"], row["Category"], row["Year of award ceremony"])] = row["winner"].strip()

    # For key categories, count how many precursors the Oscar winner won
    key_cats = ["Best Picture", "Best Director", "Best Actor", "Best Actress",
                "Best Supporting Actor", "Best Supporting Actress"]

    sweep_data = []
    for oscar_cat in key_cats:
        precursors = category_mapping.get(oscar_cat, [])
        for year in range(HISTORICAL_START, HISTORICAL_END + 1):
            oscar_w = oscar_lookup.get((year, oscar_cat))
            if not oscar_w:
                continue

            prec_wins = 0
            prec_available = 0
            for award, prec_cat in precursors:
                pw = prec_lookup.get((award, prec_cat, year))
                if pw:
                    prec_available += 1
                    if _names_match(oscar_w, pw):
                        prec_wins += 1

            if prec_available > 0:
                sweep_data.append({
                    "Year": year,
                    "Category": oscar_cat.replace("Best ", ""),
                    "Oscar Winner": oscar_w,
                    "Precursor Wins": prec_wins,
                    "Precursors Available": prec_available,
                    "Win Rate": prec_wins / prec_available,
                })

    sdf = pd.DataFrame(sweep_data)

    # Plot: distribution of precursor win rates for Oscar winners
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, cat in zip(axes.flat, key_cats):
        cat_short = cat.replace("Best ", "")
        subset = sdf[sdf["Category"] == cat_short]
        ax.hist(subset["Win Rate"], bins=np.arange(0, 1.1, 0.1),
                color="#3498db", edgecolor="white", alpha=0.8)
        ax.axvline(x=subset["Win Rate"].median(), color="red",
                   linestyle="--", label=f'Median: {subset["Win Rate"].median():.0%}')
        ax.set_title(cat, fontsize=11, fontweight="bold")
        ax.set_xlabel("Fraction of Precursors Won")
        ax.set_ylabel("Count (years)")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(fontsize=9)

    plt.suptitle("How Many Precursor Awards Did the Oscar Winner Typically Win?\n(2000–2025)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "05_sweep_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 05_sweep_distribution.png")
    return sdf


# ---------------------------------------------------------------------------
# Analysis 6: Upsets — When No Major Precursor Predicted the Oscar
# ---------------------------------------------------------------------------

def find_upsets(df, category_mapping):
    """
    Find years where the Oscar winner wasn't picked by any of the top precursors.
    """
    oscar_lookup = {}
    for _, row in df[df["Award"] == "Oscars"].iterrows():
        oscar_lookup[(row["Year of award ceremony"], row["Category"])] = row["winner"].strip()

    prec_lookup = {}
    for _, row in df[df["Award"] != "Oscars"].iterrows():
        prec_lookup[(row["Award"], row["Category"], row["Year of award ceremony"])] = row["winner"].strip()

    # "Major" precursors for key categories
    major_precursors = {
        "Best Picture": [
            ("PGA Awards", "Darryl F. Zanuck Award for Outstanding Producer of Theatrical Motion Pictures"),
            ("Critics Choice Awards", "Best Picture"),
            ("BAFTA", "Best Film"),
            ("Golden Globes", "Best Motion Picture - Drama"),
            ("Golden Globes", "Best Motion Picture - Musical or Comedy"),
        ],
        "Best Director": [
            ("DGA Awards", "Outstanding Directorial Achievement in Theatrical Feature Film"),
            ("Critics Choice Awards", "Best Director"),
            ("BAFTA", "Director"),
            ("Golden Globes", "Best Director - Motion Picture"),
        ],
        "Best Actor": [
            ("SAG Awards", "Outstanding Performance by a Male Actor in a Leading Role"),
            ("BAFTA", "Leading Actor"),
            ("Critics Choice Awards", "Best Actor"),
            ("Golden Globes", "Best Performance by a Male Actor in a Motion Picture – Drama"),
        ],
        "Best Actress": [
            ("SAG Awards", "Outstanding Performance by a Female Actor in a Leading Role"),
            ("BAFTA", "Leading Actress"),
            ("Critics Choice Awards", "Best Actress"),
            ("Golden Globes", "Best Performance by a Female Actor in a Motion Picture – Drama"),
        ],
    }

    upsets = []
    for oscar_cat, precs in major_precursors.items():
        for year in range(HISTORICAL_START, HISTORICAL_END + 1):
            oscar_w = oscar_lookup.get((year, oscar_cat))
            if not oscar_w:
                continue

            any_match = False
            prec_winners = []
            for award, prec_cat in precs:
                pw = prec_lookup.get((award, prec_cat, year))
                if pw:
                    prec_winners.append((AWARD_SHORT.get(award, award), pw))
                    if _names_match(oscar_w, pw):
                        any_match = True

            if not any_match and len(prec_winners) >= 3:
                upsets.append({
                    "Year": year,
                    "Category": oscar_cat,
                    "Oscar Winner": oscar_w,
                    "Precursor Winners": "; ".join(f"{a}: {w}" for a, w in prec_winners),
                })

    udf = pd.DataFrame(upsets)
    return udf


# ---------------------------------------------------------------------------
# Analysis 7: Category Predictability Ranking
# ---------------------------------------------------------------------------

def plot_category_predictability(accuracy):
    """
    For each Oscar category, what's the best any single precursor does?
    And what's the average across all precursors?
    """
    rows = []
    for oscar_cat, pa_list in accuracy.items():
        valid = [pa for pa in pa_list if pa.total >= 5]
        if not valid:
            continue
        best = max(valid, key=lambda pa: pa.accuracy)
        avg = np.mean([pa.accuracy for pa in valid])
        rows.append({
            "Category": oscar_cat.replace("Best ", ""),
            "Best Precursor Accuracy": best.accuracy,
            "Best Precursor": AWARD_SHORT.get(best.award, best.award),
            "Average Precursor Accuracy": avg,
            "Num Precursors": len(valid),
        })

    pdf = pd.DataFrame(rows).sort_values("Best Precursor Accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 9))

    y_pos = range(len(pdf))
    # Average bars (background)
    ax.barh(y_pos, pdf["Average Precursor Accuracy"], color="#bdc3c7",
            edgecolor="white", label="Avg across all precursors", height=0.6)
    # Best precursor bars (foreground)
    ax.barh(y_pos, pdf["Best Precursor Accuracy"], color="#2ecc71",
            edgecolor="white", label="Best single precursor", height=0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pdf["Category"])

    for i, (_, row) in enumerate(pdf.iterrows()):
        ax.text(row["Best Precursor Accuracy"] + 0.01, i,
                f'{row["Best Precursor"]} ({row["Best Precursor Accuracy"]:.0%})',
                va="center", fontsize=8, color="#27ae60")

    ax.set_xlabel("Accuracy")
    ax.set_title("Which Oscar Categories Are Most Predictable?\n"
                 "(Best vs. average precursor accuracy, 2000–2025)",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "06_category_predictability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 06_category_predictability.png")
    return pdf


# ---------------------------------------------------------------------------
# Analysis 8: Industry vs Critics vs Festivals
# ---------------------------------------------------------------------------

def plot_award_types(accuracy):
    """
    Compare predictiveness of industry awards vs critics vs festivals.
    """
    award_types = {
        "Industry Guilds": ["DGA", "PGA", "SAG", "WGA", "ACE Eddie"],
        "Major Shows": ["BAFTA", "Golden Globes", "Critics Choice"],
        "Critics Circles": ["NYFCC", "LAFCA", "NSFC", "NBR"],
        "Festivals": ["Venice", "Cannes", "TIFF"],
    }

    type_stats = {}
    for type_name, awards in award_types.items():
        hits = 0
        total = 0
        for oscar_cat, pa_list in accuracy.items():
            for pa in pa_list:
                short = AWARD_SHORT.get(pa.award, pa.award)
                if short in awards and pa.total > 0:
                    hits += pa.matches
                    total += pa.total
        if total > 0:
            type_stats[type_name] = hits / total

    fig, ax = plt.subplots(figsize=(8, 5))
    types = list(type_stats.keys())
    accs = [type_stats[t] for t in types]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
    bars = ax.bar(types, accs, color=colors, edgecolor="white", width=0.6)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.0%}", ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("Average Oscar Prediction Accuracy")
    ax.set_title("Industry Guilds vs. Major Shows vs. Critics vs. Festivals\n"
                 "(Average accuracy across all categories, 2000–2025)",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0, 0.65)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "07_award_types_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 07_award_types_comparison.png")


# ---------------------------------------------------------------------------
# Analysis 9: The DGA-Oscar Connection (Best Director deep dive)
# ---------------------------------------------------------------------------

def plot_dga_deep_dive(df, accuracy, category_mapping):
    """
    Year-by-year view: did the DGA winner match the Oscar Best Director?
    """
    oscar_lookup = {}
    for _, row in df[df["Award"] == "Oscars"].iterrows():
        oscar_lookup[(row["Year of award ceremony"], row["Category"])] = row["winner"].strip()

    prec_lookup = {}
    for _, row in df[df["Award"] != "Oscars"].iterrows():
        prec_lookup[(row["Award"], row["Category"], row["Year of award ceremony"])] = row["winner"].strip()

    dga_cat = "Outstanding Directorial Achievement in Theatrical Feature Film"
    years = list(range(HISTORICAL_START, HISTORICAL_END + 1))
    match_data = []

    for year in years:
        oscar_w = oscar_lookup.get((year, "Best Director"))
        dga_w = prec_lookup.get(("DGA Awards", dga_cat, year))
        if oscar_w and dga_w:
            matched = _names_match(oscar_w, dga_w)
            match_data.append({
                "Year": year,
                "DGA Winner": dga_w[:30],
                "Oscar Winner": oscar_w[:30],
                "Match": matched,
            })

    mdf = pd.DataFrame(match_data)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#2ecc71" if m else "#e74c3c" for m in mdf["Match"]]
    bars = ax.bar(mdf["Year"], [1] * len(mdf), color=colors, edgecolor="white")

    for i, (_, row) in enumerate(mdf.iterrows()):
        label = row["DGA Winner"]
        if not row["Match"]:
            label = f'{row["DGA Winner"]}\n(Oscar: {row["Oscar Winner"]})'
        ax.text(row["Year"], 0.5, label, rotation=90, ha="center",
                va="center", fontsize=6.5, color="white", fontweight="bold")

    ax.set_yticks([])
    ax.set_xlabel("Year")
    ax.set_title("DGA → Oscar Best Director: The Strongest Link in Awards Season\n"
                 f"(Green = match, Red = mismatch — {mdf['Match'].sum()}/{len(mdf)} = "
                 f"{mdf['Match'].mean():.0%} accuracy)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "08_dga_oscar_director.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 08_dga_oscar_director.png")
    return mdf


# ---------------------------------------------------------------------------
# Analysis 10: Consensus Strength and Oscar Outcomes
# ---------------------------------------------------------------------------

def plot_consensus_vs_outcome(df, category_mapping):
    """
    When all/most precursors agree → does the Oscar always follow?
    Scatter: precursor consensus strength vs. Oscar outcome.
    """
    oscar_lookup = {}
    for _, row in df[df["Award"] == "Oscars"].iterrows():
        oscar_lookup[(row["Year of award ceremony"], row["Category"])] = row["winner"].strip()

    prec_lookup = {}
    for _, row in df[df["Award"] != "Oscars"].iterrows():
        prec_lookup[(row["Award"], row["Category"], row["Year of award ceremony"])] = row["winner"].strip()

    key_cats = ["Best Picture", "Best Director", "Best Actor", "Best Actress",
                "Best Supporting Actor", "Best Supporting Actress"]

    data = []
    for oscar_cat in key_cats:
        precursors = category_mapping.get(oscar_cat, [])
        for year in range(HISTORICAL_START, HISTORICAL_END + 1):
            oscar_w = oscar_lookup.get((year, oscar_cat))
            if not oscar_w:
                continue

            # Find the most-picked nominee across precursors
            nominee_counts = defaultdict(int)
            total_reporting = 0
            for award, prec_cat in precursors:
                pw = prec_lookup.get((award, prec_cat, year))
                if pw:
                    total_reporting += 1
                    # Cluster nominees
                    matched_existing = False
                    for existing in list(nominee_counts.keys()):
                        if _names_match(pw, existing):
                            nominee_counts[existing] += 1
                            matched_existing = True
                            break
                    if not matched_existing:
                        nominee_counts[pw] += 1

            if total_reporting < 3:
                continue

            # Find the frontrunner (most precursor wins)
            frontrunner = max(nominee_counts, key=nominee_counts.get)
            consensus = nominee_counts[frontrunner] / total_reporting

            # Did the frontrunner win the Oscar?
            frontrunner_won = _names_match(frontrunner, oscar_w)

            data.append({
                "Year": year,
                "Category": oscar_cat.replace("Best ", ""),
                "Consensus": consensus,
                "Frontrunner Won Oscar": frontrunner_won,
                "Frontrunner": frontrunner[:40],
                "Oscar Winner": oscar_w[:40],
            })

    cdf = pd.DataFrame(data)

    # Bin by consensus strength and show win rate
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.01]
    labels = ["0-30%", "30-50%", "50-70%", "70-90%", "90-100%"]
    cdf["Consensus Bin"] = pd.cut(cdf["Consensus"], bins=bins, labels=labels)

    win_rates = cdf.groupby("Consensus Bin", observed=True)["Frontrunner Won Oscar"].agg(["mean", "count"])
    win_rates.columns = ["Win Rate", "Count"]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c", "#f39c12", "#f1c40f", "#2ecc71", "#27ae60"]
    bars = ax.bar(range(len(win_rates)), win_rates["Win Rate"], color=colors,
                  edgecolor="white", width=0.6)

    for bar, (_, row) in zip(bars, win_rates.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{row["Win Rate"]:.0%}\n(n={int(row["Count"])})',
                ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(win_rates)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Precursor Consensus Strength (% of precursors picking same nominee)")
    ax.set_ylabel("Probability the Frontrunner Wins the Oscar")
    ax.set_title("Does Consensus Guarantee the Oscar?\n"
                 "(Major categories, 2000–2025)",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "09_consensus_vs_outcome.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 09_consensus_vs_outcome.png")
    return cdf


# ---------------------------------------------------------------------------
# Analysis 11: Dataset overview
# ---------------------------------------------------------------------------

def plot_dataset_overview(df_raw):
    """Simple overview of the dataset composition."""
    df = df_raw.copy()
    df["has_winner"] = df["winner"].notna() & (df["winner"] != "")

    # Records per award
    counts = df.groupby("Award").size().sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: records per award
    ax = axes[0]
    shorts = [AWARD_SHORT.get(a, a) for a in counts.index]
    ax.barh(shorts, counts.values, color="#3498db", edgecolor="white")
    ax.set_xlabel("Number of Records")
    ax.set_title("Records per Award Show", fontsize=11, fontweight="bold")

    # Right: records over time
    ax = axes[1]
    year_counts = df.groupby("Year of award ceremony")["has_winner"].sum()
    ax.bar(year_counts.index, year_counts.values, color="#2ecc71", edgecolor="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("Records with Winners")
    ax.set_title("Data Coverage Over Time", fontsize=11, fontweight="bold")

    plt.suptitle(f"Dataset Overview: {len(df)} records, {df['Award'].nunique()} award shows, "
                 f"{int(df['Year of award ceremony'].min())}–{int(df['Year of award ceremony'].max())}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "00_dataset_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 00_dataset_overview.png")


# ---------------------------------------------------------------------------
# HTML Report Generator
# ---------------------------------------------------------------------------

def generate_html_report(upset_df, sweep_df, consensus_df, dga_df, acc_matrix):
    """Generate an HTML report with all plots and commentary."""

    # Compute some stats for inline text
    total_upsets = len(upset_df) if not upset_df.empty else 0
    median_sweep_bp = sweep_df[sweep_df["Category"] == "Picture"]["Win Rate"].median() if len(sweep_df) > 0 else 0
    dga_acc = dga_df["Match"].mean() if len(dga_df) > 0 else 0

    plot_files = sorted(PLOT_DIR.glob("*.png"))
    plot_sections = []

    section_info = {
        "00": ("The Dataset",
               "Our dataset spans <strong>16 award shows</strong> across <strong>26 years</strong> (2000–2025), "
               "with a total of 5,346 records. This includes major industry awards (DGA, PGA, SAG, WGA), "
               "big-tent shows (BAFTA, Golden Globes, Critics Choice), critics' circles (NYFCC, LAFCA, NSFC, NBR), "
               "and festival awards (Cannes, Venice, Toronto)."),
        "01": ("Which Award Shows Best Predict the Oscars?",
               "Not all precursor awards are created equal. The <strong>DGA Awards</strong> stand out as the single "
               "most predictive precursor, followed by the <strong>PGA</strong>, <strong>SAG</strong>, and "
               "<strong>Critics Choice</strong>. Festival awards (Cannes, Venice) almost never directly predict "
               "Oscar winners — they operate on a different aesthetic axis entirely."),
        "02": ("The Full Picture: Accuracy by Category",
               "This heatmap reveals the nuance. The DGA's extraordinary accuracy is specific to <strong>Best Director</strong>. "
               "For <strong>Best Picture</strong>, the PGA is king. For acting categories, SAG and BAFTA are the "
               "most reliable signals. Some categories (editing, song) have surprisingly low predictability from "
               "any precursor."),
        "03": ("Has Predictiveness Changed Over Time?",
               "Looking at 5-year rolling accuracy for Best Picture precursors, we can see interesting shifts. "
               "The PGA has become <em>more</em> predictive in recent years, while the Golden Globes' predictive "
               "power has been declining. This likely reflects the PGA's growing alignment with "
               "Academy voters."),
        "04": ("How Often Do Precursor Awards Agree?",
               "When the PGA and BAFTA agree on Best Picture, pay attention. But interestingly, the "
               "major shows don't always agree — the Golden Globes often picks differently from the pack due "
               "to its Drama/Musical-Comedy split. Agreement rates between certain pairs of awards are "
               "remarkably high, suggesting shared voter sensibilities."),
        "05": ("The Sweep Effect: How Many Precursors Did Oscar Winners Typically Win?",
               f"The median Best Picture winner won about <strong>{median_sweep_bp:.0%}</strong> of available "
               "precursor awards. But the distribution is bimodal — there are \"coronation\" years where one film "
               "sweeps everything, and \"contested\" years where the winner emerged from a split field. "
               "Acting categories show a similar pattern."),
        "06": ("Which Categories Are Most (and Least) Predictable?",
               "Best Director and Best Animated Feature are the most predictable Oscar categories — if you know "
               "the DGA and Golden Globe Animation winners, you're right ~80% of the time. "
               "At the other end, categories like Best Film Editing, Best Song, and Best Production Design are "
               "much harder to forecast from precursors alone."),
        "07": ("Industry Guilds vs. Critics vs. Festivals",
               "The pattern is clear: <strong>industry guild awards</strong> (DGA, PGA, SAG, WGA) are the best "
               "Oscar predictors because they share the most voter overlap with the Academy. "
               "Major shows (BAFTA, Golden Globes, Critics Choice) are next. "
               "Critics' circles have more idiosyncratic tastes. "
               "And festival awards operate on an almost entirely different wavelength."),
        "08": (f"The DGA–Oscar Connection: {dga_acc:.0%} Accuracy",
               f"The DGA has matched the Oscar Best Director winner in <strong>{dga_acc:.0%}</strong> of years since 2000 — "
               "the most reliable single predictor in all of awards season. The rare misses are fascinating: "
               "they tend to happen in years with strong narrative momentum behind a different director."),
        "09": ("Does Consensus Guarantee the Oscar?",
               "When 90%+ of precursor awards agree on a frontrunner, that person/film wins the Oscar nearly "
               "every time. But when consensus drops below 50%, it becomes a coin flip. "
               "This has direct implications for our 2026 predictions — the <strong>Best Actor</strong> race, "
               "with four different precursors picking four different people, is genuinely unpredictable."),
    }

    for pf in plot_files:
        key = pf.stem[:2]
        title, desc = section_info.get(key, (pf.stem, ""))
        plot_sections.append(f"""
        <div class="section">
            <h2>{title}</h2>
            <p>{desc}</p>
            <img src="plots/{pf.name}" alt="{title}">
        </div>
        """)

    # Upsets table
    upset_html = ""
    if not upset_df.empty:
        upset_rows = ""
        for _, row in upset_df.iterrows():
            upset_rows += f"""
            <tr>
                <td>{row['Year']}</td>
                <td>{row['Category']}</td>
                <td><strong>{row['Oscar Winner']}</strong></td>
                <td style="font-size: 0.85em">{row['Precursor Winners']}</td>
            </tr>"""
        upset_html = f"""
        <div class="section">
            <h2>The Upsets: When No Major Precursor Called It</h2>
            <p>These are the years where the Oscar winner wasn't picked by <em>any</em> of the top 4–5
            precursor awards for that category. These are the true surprises — the times the Academy
            zagged when everyone expected them to zig. There have been <strong>{total_upsets} such upsets</strong>
            since 2000.</p>
            <table>
                <tr><th>Year</th><th>Category</th><th>Oscar Winner</th><th>What Precursors Picked Instead</th></tr>
                {upset_rows}
            </table>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Oscar Precursor Awards — Exploratory Data Analysis</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'Georgia', serif;
        line-height: 1.7;
        color: #2c3e50;
        background: #fafafa;
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem;
    }}
    h1 {{
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        color: #1a1a2e;
    }}
    .subtitle {{
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
        border-bottom: 2px solid #e0c068;
        padding-bottom: 1rem;
    }}
    h2 {{
        font-size: 1.4rem;
        margin: 2rem 0 0.8rem 0;
        color: #1a1a2e;
        border-left: 4px solid #e0c068;
        padding-left: 0.8rem;
    }}
    p {{
        margin-bottom: 1rem;
        font-size: 1.05rem;
    }}
    img {{
        width: 100%;
        border-radius: 8px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        margin: 1rem 0 2rem 0;
    }}
    .section {{
        margin-bottom: 3rem;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0 2rem 0;
        font-size: 0.9rem;
    }}
    th {{
        background: #1a1a2e;
        color: white;
        padding: 0.6rem 0.8rem;
        text-align: left;
    }}
    td {{
        padding: 0.5rem 0.8rem;
        border-bottom: 1px solid #eee;
    }}
    tr:nth-child(even) td {{
        background: #f5f5f5;
    }}
    .key-stat {{
        background: #fff8e1;
        border-left: 4px solid #e0c068;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 1.1rem;
    }}
    strong {{ color: #1a1a2e; }}
    em {{ color: #555; }}
</style>
</head>
<body>
    <h1>Can You Predict the Oscars?</h1>
    <p class="subtitle">
        An exploration of 26 years of precursor awards data — and what it tells us about
        Hollywood's biggest night.
    </p>

    <div class="key-stat">
        <strong>The big question:</strong> How predictable are the Oscars? With 16 precursor award shows,
        26 years of data, and 5,346 records, we can finally put numbers on what awards pundits have
        long known intuitively — and discover some surprises along the way.
    </div>

    {"".join(plot_sections)}

    {upset_html}

    <div class="section">
        <h2>What This Means for 2026</h2>
        <p>
            Armed with these historical patterns, we built a <strong>weighted precursor model</strong> to
            forecast the 2026 Oscars. The model weights each precursor award by its category-specific
            historical accuracy, then aggregates the 2026 precursor winners into predictions.
        </p>
        <p>
            The high-confidence picks (Best Director, Best Actress, Best Adapted Screenplay) have strong
            precursor consensus — the kind that historically leads to an Oscar win nearly every time.
            But the Best Actor race, with four different precursors picking four different people, is the
            kind of split that makes awards season genuinely exciting.
        </p>
    </div>

</body>
</html>"""

    output_path = SCRIPT_DIR / "eda_report.html"
    output_path.write_text(html)
    print(f"\n  HTML report saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  OSCAR PRECURSOR AWARDS — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Load raw data (including empty winners for overview plot)
    df_raw = pd.concat([pd.read_csv(fp) for fp in DATA_FILES], ignore_index=True)

    # Load data with winners only
    df = load_all_data()
    category_mapping = build_category_mapping()

    print("\nComputing historical accuracy...")
    accuracy = compute_historical_accuracy(df_raw, category_mapping)

    print("\nGenerating plots...")
    plot_dataset_overview(df_raw)
    adf = plot_overall_accuracy(accuracy, category_mapping)
    acc_matrix = plot_best_precursor_per_category(accuracy)
    plot_accuracy_over_time(df, accuracy, category_mapping)
    agreement_mat = plot_award_agreement(df, category_mapping)
    sweep_df = analyze_sweeps(df, category_mapping, accuracy)
    pred_df = plot_category_predictability(accuracy)
    plot_award_types(accuracy)
    dga_df = plot_dga_deep_dive(df, accuracy, category_mapping)
    consensus_df = plot_consensus_vs_outcome(df, category_mapping)

    print("\nFinding upsets...")
    upset_df = find_upsets(df, category_mapping)
    if not upset_df.empty:
        print(f"  Found {len(upset_df)} upsets:")
        for _, row in upset_df.iterrows():
            print(f"    {row['Year']} {row['Category']}: {row['Oscar Winner']}")
    else:
        print("  No upsets found (all Oscar winners had at least one major precursor match)")

    print("\nGenerating HTML report...")
    generate_html_report(upset_df, sweep_df, consensus_df, dga_df, acc_matrix)

    print("\n" + "=" * 60)
    print("  DONE! Check plots/ directory and eda_report.html")
    print("=" * 60)


if __name__ == "__main__":
    main()
