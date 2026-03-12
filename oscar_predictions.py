"""
2026 Oscar Predictions — Weighted Precursor Model
==================================================

Predicts Oscar winners using historically-weighted precursor award signals.

Methodology:
  For each Oscar category, precursor awards (BAFTA, Golden Globes, SAG, etc.)
  are mapped to the analogous Oscar category. Each precursor is weighted by its
  historical accuracy at predicting the Oscar winner (2000–2025). The 2026
  precursor winners are then aggregated using these weights to produce a
  prediction with a confidence score.

Usage:
  python oscar_predictions.py

  By default, reads from:
    - "film awards research part 1.csv"
    - "film awards research part 2.csv"
  in the same directory as this script.

  Outputs:
    - Prints predictions to console
    - Saves "2026_oscar_predictions.md" in the same directory
"""

import pandas as pd
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILES = [
    SCRIPT_DIR / "film awards research part 1.csv",
    SCRIPT_DIR / "film awards research part 2.csv",
]
OUTPUT_FILE = SCRIPT_DIR / "2026_oscar_predictions.md"

# Year range for computing historical accuracy
HISTORICAL_START = 2000
HISTORICAL_END = 2025

# The year we're predicting
PREDICTION_YEAR = 2026

# Minimum weight given to any precursor (prevents zero-weight precursors
# from being entirely ignored — they still carry some signal)
MIN_PRECURSOR_WEIGHT = 0.01

# Oscar categories to predict
OSCAR_CATEGORIES = [
    "Best Picture",
    "Best Director",
    "Best Actor",
    "Best Actress",
    "Best Supporting Actor",
    "Best Supporting Actress",
    "Best Cinematography",
    "Best Film Editing",
    "Best Adapted Screenplay",
    "Best Original Screenplay",
    "Best Original Song",
    "Best Original Score",
    "Best Costume Design",
    "Best Makeup and Hairstyling",
    "Best Production Design",
    "Best Sound",
    "Best Animated Feature Film",
    "Best International Feature Film",
    "Best Documentary Feature Film",
    "Best Animated Short Film",
    "Best Live Action Short Film",
    "Best Visual Effects",
    "Best Documentary Short Film",
    "Best Casting",
]


# ---------------------------------------------------------------------------
# Category Mapping: Precursor Award → Oscar Category
# ---------------------------------------------------------------------------

def build_category_mapping() -> dict[str, list[tuple[str, str]]]:
    """
    Returns a dict mapping each Oscar category to a list of
    (precursor_award_name, precursor_category_name) tuples.

    These mappings reflect which precursor categories are analogous to
    each Oscar category. For example, the BAFTA "Best Film" maps to
    the Oscar "Best Picture".
    """
    # Shared screenplay precursors: critics circles that give a single
    # "Best Screenplay" award (not split into adapted/original).
    # We include these for BOTH adapted and original screenplay predictions,
    # since the winner could be either type.
    shared_screenplay_precursors = [
        ("National Society of Film Critics", "Best Screenplay"),
        ("New York Film Critics Circle", "Best Screenplay"),
        ("Los Angeles Film Critics Association", "Best Screenplay"),
        ("Venice Golden Lion", "Award for Best Screenplay"),
        ("Cannes Palme d'Or", "Prix du scénario"),
    ]

    mapping = {
        "Best Picture": [
            ("BAFTA", "Best Film"),
            ("Critics Choice Awards", "Best Picture"),
            ("Golden Globes", "Best Motion Picture - Drama"),
            ("Golden Globes", "Best Motion Picture - Musical or Comedy"),
            ("National Board of Review", "Best Film"),
            ("National Society of Film Critics", "Best Picture"),
            ("New York Film Critics Circle", "Best Film"),
            ("Los Angeles Film Critics Association", "Best Picture"),
            ("PGA Awards", "Darryl F. Zanuck Award for Outstanding Producer of Theatrical Motion Pictures"),
            ("Toronto People's Choice Award", "People's Choice Award"),
            ("Venice Golden Lion", "Golden Lion for Best Film"),
            ("Cannes Palme d'Or", "Palme d'Or"),
        ],
        "Best Director": [
            ("BAFTA", "Director"),
            ("Critics Choice Awards", "Best Director"),
            ("Golden Globes", "Best Director - Motion Picture"),
            ("National Board of Review", "Best Director"),
            ("National Society of Film Critics", "Best Director"),
            ("New York Film Critics Circle", "Best Director"),
            ("Los Angeles Film Critics Association", "Best Director"),
            ("DGA Awards", "Outstanding Directorial Achievement in Theatrical Feature Film"),
            ("Venice Golden Lion", "Silver Lion – Award for Best Director"),
            ("Cannes Palme d'Or", "Prix de la mise en scène"),
        ],
        "Best Actor": [
            ("BAFTA", "Leading Actor"),
            ("Critics Choice Awards", "Best Actor"),
            ("Golden Globes", "Best Performance by a Male Actor in a Motion Picture – Drama"),
            ("Golden Globes", "Best Performance by a Male Actor in a Motion Picture – Musical or Comedy"),
            ("National Board of Review", "Best Actor"),
            ("National Society of Film Critics", "Best Actor"),
            ("New York Film Critics Circle", "Best Actor"),
            ("SAG Awards", "Outstanding Performance by a Male Actor in a Leading Role"),
            ("Venice Golden Lion", "Coppa Volpi for Best Actor"),
            ("Cannes Palme d'Or", "Prix d'interprétation masculine"),
        ],
        "Best Actress": [
            ("BAFTA", "Leading Actress"),
            ("Critics Choice Awards", "Best Actress"),
            ("Golden Globes", "Best Performance by a Female Actor in a Motion Picture – Drama"),
            ("Golden Globes", "Best Performance by a Female Actor in a Motion Picture – Musical or Comedy"),
            ("National Board of Review", "Best Actress"),
            ("National Society of Film Critics", "Best Actress"),
            ("New York Film Critics Circle", "Best Actress"),
            ("SAG Awards", "Outstanding Performance by a Female Actor in a Leading Role"),
            ("Venice Golden Lion", "Coppa Volpi for Best Actress"),
            ("Cannes Palme d'Or", "Prix d'interprétation féminine"),
        ],
        "Best Supporting Actor": [
            ("BAFTA", "Supporting Actor"),
            ("Critics Choice Awards", "Best Supporting Actor"),
            ("Golden Globes", "Best Performance by a Male Actor in a Supporting Role in any Motion Picture"),
            ("National Board of Review", "Best Supporting Actor"),
            ("National Society of Film Critics", "Best Supporting Actor"),
            ("New York Film Critics Circle", "Best Supporting Actor"),
            ("SAG Awards", "Outstanding Performance by a Male Actor in a Supporting Role"),
        ],
        "Best Supporting Actress": [
            ("BAFTA", "Supporting Actress"),
            ("Critics Choice Awards", "Best Supporting Actress"),
            ("Golden Globes", "Best Performance by a Female Actor in a Supporting Role in any Motion Picture"),
            ("National Board of Review", "Best Supporting Actress"),
            ("National Society of Film Critics", "Best Supporting Actress"),
            ("New York Film Critics Circle", "Best Supporting Actress"),
            ("SAG Awards", "Outstanding Performance by a Female Actor in a Supporting Role"),
        ],
        "Best Cinematography": [
            ("BAFTA", "Cinematography"),
            ("Critics Choice Awards", "Best Cinematography"),
            ("National Society of Film Critics", "Best Cinematography"),
            ("New York Film Critics Circle", "Best Cinematography"),
            ("Los Angeles Film Critics Association", "Best Cinematography"),
            ("National Board of Review", "Outstanding Achievement in Cinematography"),
        ],
        "Best Film Editing": [
            ("BAFTA", "Editing"),
            ("Critics Choice Awards", "Best Editing"),
            ("ACE Eddie Awards", "Best Edited Feature Film (Drama, Theatrical)"),
            ("ACE Eddie Awards", "Best Edited Feature Film (Comedy, Theatrical)"),
            ("Los Angeles Film Critics Association", "Best Editing"),
            ("National Board of Review", "Best Film Editing"),
        ],
        "Best Adapted Screenplay": [
            ("BAFTA", "Adapted Screenplay"),
            ("Critics Choice Awards", "Best Adapted Screenplay"),
            ("National Board of Review", "Best Adapted Screenplay"),
            ("WGA Awards", "Adapted Screenplay"),
        ] + shared_screenplay_precursors,
        "Best Original Screenplay": [
            ("BAFTA", "Original Screenplay"),
            ("Critics Choice Awards", "Best Original Screenplay"),
            ("National Board of Review", "Best Original Screenplay"),
            ("WGA Awards", "Original Screenplay"),
        ] + shared_screenplay_precursors,
        "Best Original Song": [
            ("Critics Choice Awards", "Best Song"),
            ("Golden Globes", "Best Original Song - Motion Picture"),
            ("National Board of Review", "Best Original Song"),
        ],
        "Best Original Score": [
            ("BAFTA", "Original Score"),
            ("Critics Choice Awards", "Best Score"),
            ("Golden Globes", "Best Original Score - Motion Picture"),
            ("Los Angeles Film Critics Association", "Best Music Score"),
            ("National Board of Review", "Outstanding Film Music Composition"),
        ],
        "Best Costume Design": [
            ("BAFTA", "Costume Design"),
            ("Critics Choice Awards", "Best Costume Design"),
            ("National Board of Review", "Best Costume Design"),
        ],
        "Best Makeup and Hairstyling": [
            ("BAFTA", "Make Up & Hair"),
            ("Critics Choice Awards", "Best Hair and Makeup"),
            ("National Board of Review", "Best Makeup and Hairstyling"),
        ],
        "Best Production Design": [
            ("BAFTA", "Production Design"),
            ("Critics Choice Awards", "Best Production Design"),
            ("Los Angeles Film Critics Association", "Best Production Design"),
            ("National Board of Review", "Production Design Award"),
        ],
        "Best Sound": [
            ("BAFTA", "Sound"),
            ("Critics Choice Awards", "Best Sound"),
            ("National Board of Review", "Best Sound"),
        ],
        "Best Animated Feature Film": [
            ("BAFTA", "Animated Film"),
            ("Critics Choice Awards", "Best Animated Feature"),
            ("Golden Globes", "Best Motion Picture - Animated"),
            ("National Board of Review", "Best Animated Feature"),
            ("New York Film Critics Circle", "Best Animated Film"),
            ("Los Angeles Film Critics Association", "Best Animation"),
            ("PGA Awards", "Award for Outstanding Producer of Animated Theatrical Motion Pictures"),
        ],
        "Best International Feature Film": [
            ("BAFTA", "Film Not in the English Language"),
            ("Critics Choice Awards", "Best Foreign Language Film"),
            ("Golden Globes", "Best Motion Picture – Non-English Language"),
            ("National Board of Review", "Best International Film"),
            ("National Society of Film Critics", "Best Film Not in the English Language"),
            ("New York Film Critics Circle", "Best International Film"),
            ("Los Angeles Film Critics Association", "Best Film Not In The English Language"),
        ],
        "Best Documentary Feature Film": [
            ("BAFTA", "Documentary"),
            ("Critics Choice Awards", "Best Documentary Feature"),
            ("National Board of Review", "Best Documentary"),
            ("National Society of Film Critics", "Best Non-fiction Film"),
            ("New York Film Critics Circle", "Best Non-Fiction Film"),
            ("Los Angeles Film Critics Association", "Best Documentary/Non-Fiction Film"),
            ("PGA Awards", "Award for Outstanding Producer of Documentary Motion Pictures"),
            ("DGA Awards", "Outstanding Directorial Achievement in Documentary Film"),
            ("Toronto People's Choice Award", "People's Choice Documentary Award"),
        ],
        "Best Animated Short Film": [
            ("BAFTA", "British Short Animation"),
            ("National Board of Review", "Best Animated Short Film"),
        ],
        "Best Live Action Short Film": [
            ("BAFTA", "British Short Film"),
            ("National Board of Review", "Best Live Action Short Film"),
        ],
        "Best Visual Effects": [
            ("BAFTA", "Special Visual Effects"),
            ("Critics Choice Awards", "Best Visual Effects"),
            ("National Board of Review", "Best Visual Effects"),
        ],
        "Best Documentary Short Film": [
            ("National Board of Review", "Best Documentary Short Film"),
        ],
        "Best Casting": [
            ("BAFTA", "Casting"),
            ("Critics Choice Awards", "Best Casting and Ensemble"),
            ("SAG Awards", "Outstanding Performance by a Cast in a Motion Picture"),
            ("National Board of Review", "Best Cast (Acting Ensemble)"),
        ],
    }

    return mapping


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data(file_paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate all CSV data files."""
    frames = [pd.read_csv(fp) for fp in file_paths]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} records from {len(file_paths)} files")
    return df


def build_winner_lookup(
    df: pd.DataFrame,
    award_filter: str | None = None,
    exclude_award: str | None = None,
) -> dict[tuple, str]:
    """
    Build a lookup dict from the dataframe.

    Returns:
      {(Award, Category, Year): winner_name} for all rows with non-empty winners.

    Args:
      award_filter: If set, only include rows where Award == award_filter.
      exclude_award: If set, exclude rows where Award == exclude_award.
    """
    mask = df["winner"].notna() & (df["winner"] != "")
    if award_filter:
        mask &= df["Award"] == award_filter
    if exclude_award:
        mask &= df["Award"] != exclude_award

    lookup = {}
    for _, row in df[mask].iterrows():
        key = (row["Award"], row["Category"], row["Year of award ceremony"])
        lookup[key] = row["winner"].strip()
    return lookup


# ---------------------------------------------------------------------------
# Historical Accuracy Computation
# ---------------------------------------------------------------------------

@dataclass
class PrecursorAccuracy:
    """Stores the historical accuracy of one precursor for one Oscar category."""
    award: str
    precursor_category: str
    matches: int = 0
    total: int = 0
    match_years: list[int] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.matches / self.total if self.total > 0 else 0.0


def compute_historical_accuracy(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    start_year: int = HISTORICAL_START,
    end_year: int = HISTORICAL_END,
) -> dict[str, list[PrecursorAccuracy]]:
    """
    For each Oscar category and each mapped precursor, compute how often
    the precursor winner matched the Oscar winner over the historical period.

    Returns:
      {oscar_category: [PrecursorAccuracy, ...]}
    """
    # Build lookups
    oscar_lookup = build_winner_lookup(df, award_filter="Oscars")
    precursor_lookup = build_winner_lookup(df, exclude_award="Oscars")

    results: dict[str, list[PrecursorAccuracy]] = {}

    for oscar_cat, precursors in category_mapping.items():
        cat_results = []

        for award_name, prec_cat in precursors:
            pa = PrecursorAccuracy(award=award_name, precursor_category=prec_cat)

            for year in range(start_year, end_year + 1):
                oscar_key = ("Oscars", oscar_cat, year)
                prec_key = (award_name, prec_cat, year)

                oscar_winner = oscar_lookup.get(oscar_key)
                prec_winner = precursor_lookup.get(prec_key)

                if oscar_winner and prec_winner:
                    pa.total += 1
                    if _names_match(oscar_winner, prec_winner):
                        pa.matches += 1
                        pa.match_years.append(year)

            cat_results.append(pa)

        results[oscar_cat] = cat_results

    return results


# ---------------------------------------------------------------------------
# Name Matching
# ---------------------------------------------------------------------------

def _names_match(a: str, b: str) -> bool:
    """
    Check if two winner strings refer to the same entity.

    Handles cases like:
      - "Brady Corbet" vs "Brady Corbet (The Brutalist)"
      - "The Brutalist" vs "Brady Corbet (The Brutalist)"
      - "One Battle After Another" vs "One Battle After Another (Paul Thomas Anderson)"
    """
    a_low = a.strip().lower()
    b_low = b.strip().lower()

    # Exact match
    if a_low == b_low:
        return True

    # Substring match (one contains the other)
    if a_low in b_low or b_low in a_low:
        return True

    # Compare core text (outside parentheses) and parenthetical text
    a_core = re.sub(r"\([^)]*\)", "", a).strip().lower()
    b_core = re.sub(r"\([^)]*\)", "", b).strip().lower()

    if a_core and b_core and (a_core in b_core or b_core in a_core):
        return True

    # Check if core of one matches inside parens of the other
    a_parens = " ".join(re.findall(r"\(([^)]*)\)", a)).lower()
    b_parens = " ".join(re.findall(r"\(([^)]*)\)", b)).lower()

    if (a_core and b_parens and a_core in b_parens) or \
       (b_core and a_parens and b_core in a_parens):
        return True

    # Check overlap of capitalized proper nouns (at least 2 shared)
    a_words = set(re.findall(r"[A-Z][a-z]+", a))
    b_words = set(re.findall(r"[A-Z][a-z]+", b))
    if len(a_words) >= 2 and len(b_words) >= 2 and len(a_words & b_words) >= 2:
        return True

    return False


def _cluster_nominees(
    raw_votes: list[tuple[str, float, list[tuple[str, str, float]]]],
) -> list[tuple[str, float, list[tuple[str, str, float]]]]:
    """
    Cluster nominee names that refer to the same entity.

    Input:  list of (nominee_name, weight, [(award, category, weight), ...])
    Output: same format, but with similar names merged (keeping the longest
            name as the canonical version).
    """
    clusters: list[list[int]] = []  # indices into raw_votes
    canonical: list[str] = []

    for i, (name, _, _) in enumerate(raw_votes):
        matched_cluster = None
        for ci, canon in enumerate(canonical):
            if _names_match(name, canon):
                matched_cluster = ci
                break

        if matched_cluster is not None:
            clusters[matched_cluster].append(i)
            # Keep the longer name as canonical
            if len(name) > len(canonical[matched_cluster]):
                canonical[matched_cluster] = name
        else:
            clusters.append([i])
            canonical.append(name)

    # Merge clusters
    merged = []
    for ci, indices in enumerate(clusters):
        total_weight = sum(raw_votes[i][1] for i in indices)
        all_details = []
        for i in indices:
            all_details.extend(raw_votes[i][2])
        merged.append((canonical[ci], total_weight, all_details))

    return merged


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """A single Oscar category prediction."""
    oscar_category: str
    predicted_winner: str
    confidence: float  # 0.0 to 1.0
    precursors_available: int
    precursors_total: int
    supporting_awards: list[tuple[str, str, float]]  # (award, category, weight)
    all_candidates: list[tuple[str, float]]  # (name, confidence) — top 5
    runner_up: str = ""
    runner_up_confidence: float = 0.0

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
            return "\U0001f7e2"  # green
        elif self.confidence >= 0.40:
            return "\U0001f7e1"  # yellow
        elif self.confidence > 0:
            return "\U0001f7e0"  # orange
        return "\U0001f534"  # red


def generate_predictions(
    df: pd.DataFrame,
    category_mapping: dict[str, list[tuple[str, str]]],
    accuracy: dict[str, list[PrecursorAccuracy]],
    prediction_year: int = PREDICTION_YEAR,
) -> list[Prediction]:
    """
    Generate Oscar predictions for the given year using weighted precursor votes.
    """
    # Build 2026 precursor winner lookup: (Award, Category) -> winner
    mask = (
        (df["Year of award ceremony"] == prediction_year)
        & (df["Award"] != "Oscars")
        & df["winner"].notna()
        & (df["winner"] != "")
    )
    precursor_2026 = {}
    for _, row in df[mask].iterrows():
        precursor_2026[(row["Award"], row["Category"])] = row["winner"].strip()

    # Build accuracy lookup for quick access
    acc_lookup: dict[str, dict[tuple[str, str], float]] = {}
    for oscar_cat, pa_list in accuracy.items():
        acc_lookup[oscar_cat] = {
            (pa.award, pa.precursor_category): pa.accuracy for pa in pa_list
        }

    predictions = []

    for oscar_cat in OSCAR_CATEGORIES:
        precursors = category_mapping.get(oscar_cat, [])
        raw_votes: list[tuple[str, float, list[tuple[str, str, float]]]] = []
        total_weight = 0.0
        available = 0

        for award_name, prec_cat in precursors:
            winner = precursor_2026.get((award_name, prec_cat))
            if not winner:
                continue

            weight = acc_lookup.get(oscar_cat, {}).get(
                (award_name, prec_cat), 0.0
            )
            weight = max(weight, MIN_PRECURSOR_WEIGHT)

            raw_votes.append((winner, weight, [(award_name, prec_cat, weight)]))
            total_weight += weight
            available += 1

        # Handle no-data case
        if total_weight == 0:
            predictions.append(Prediction(
                oscar_category=oscar_cat,
                predicted_winner="N/A — No precursor data",
                confidence=0.0,
                precursors_available=0,
                precursors_total=len(precursors),
                supporting_awards=[],
                all_candidates=[],
            ))
            continue

        # Cluster similar nominee names
        clustered = _cluster_nominees(raw_votes)

        # Normalize to confidence scores and sort
        scored = [
            (name, score / total_weight, details)
            for name, score, details in clustered
        ]
        scored.sort(key=lambda x: -x[1])

        top_name, top_conf, top_details = scored[0]
        top_details_sorted = sorted(top_details, key=lambda x: -x[2])

        runner_up_name = scored[1][0] if len(scored) > 1 else ""
        runner_up_conf = scored[1][1] if len(scored) > 1 else 0.0

        predictions.append(Prediction(
            oscar_category=oscar_cat,
            predicted_winner=top_name,
            confidence=top_conf,
            precursors_available=available,
            precursors_total=len(precursors),
            supporting_awards=top_details_sorted,
            all_candidates=[(n, round(c * 100, 1)) for n, c, _ in scored[:5]],
            runner_up=runner_up_name,
            runner_up_confidence=runner_up_conf,
        ))

    return predictions


# ---------------------------------------------------------------------------
# Output: Console
# ---------------------------------------------------------------------------

def print_predictions(predictions: list[Prediction]) -> None:
    """Print predictions to console in a readable format."""
    print()
    print("=" * 100)
    print("  2026 OSCAR PREDICTIONS — WEIGHTED PRECURSOR MODEL")
    print("=" * 100)

    for p in predictions:
        print()
        print(f"{p.oscar_category}")
        print(f"  Prediction: {p.predicted_winner}")
        print(
            f"  Confidence: {p.confidence_pct}% "
            f"[{p.confidence_emoji} {p.confidence_label}] "
            f"(precursors: {p.precursors_available}/{p.precursors_total})"
        )
        if p.runner_up:
            print(
                f"  Runner-up:  {p.runner_up} "
                f"({round(p.runner_up_confidence * 100, 1)}%)"
            )
        if p.supporting_awards:
            awards_str = ", ".join(
                f"{a} ({w:.0%})" for a, _, w in p.supporting_awards[:4]
            )
            print(f"  Supported by: {awards_str}")


# ---------------------------------------------------------------------------
# Output: Markdown Report
# ---------------------------------------------------------------------------

def generate_markdown_report(
    predictions: list[Prediction],
    accuracy: dict[str, list[PrecursorAccuracy]],
) -> str:
    """Generate a full markdown report of predictions."""
    lines = [
        "# 2026 Oscar Predictions — Weighted Precursor Model",
        "",
        "## Methodology",
        "",
        "This forecast uses a **weighted precursor model** built on "
        f"{HISTORICAL_END - HISTORICAL_START + 1} years of historical data "
        f"({HISTORICAL_START}–{HISTORICAL_END}).",
        "For each Oscar category, precursor awards (BAFTA, Golden Globes, SAG, "
        "Critics Choice, DGA, PGA, WGA, etc.) are weighted by their historical "
        "accuracy at predicting the Oscar winner in that specific category. "
        f"The {PREDICTION_YEAR} precursor winners are then aggregated using "
        "these weights to produce a prediction with a confidence score.",
        "",
        "**Confidence interpretation:**",
        "",
        "- \U0001f7e2 HIGH (70%+): Strong precursor consensus",
        "- \U0001f7e1 MEDIUM (40–69%): Moderate consensus; leading but not locked in",
        "- \U0001f7e0 LOW (20–39%): Weak consensus; competitive race",
        "- \U0001f534 VERY LOW (<20%): No meaningful signal from precursors",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "| Category | Predicted Winner | Confidence | Runner-Up |",
        "|----------|-----------------|------------|-----------|",
    ]

    for p in predictions:
        pred = p.predicted_winner[:60]
        ru = p.runner_up[:45] if p.runner_up else "—"
        ru_conf = f" ({round(p.runner_up_confidence * 100, 1)}%)" if p.runner_up else ""
        lines.append(
            f"| {p.oscar_category} | {pred} "
            f"| {p.confidence_emoji} {p.confidence_pct}% "
            f"| {ru}{ru_conf} |"
        )

    lines += ["", "---", "", "## Detailed Predictions", ""]

    for p in predictions:
        lines.append(f"### {p.oscar_category}")
        lines.append("")
        lines.append(f"**Prediction:** {p.predicted_winner}")
        lines.append(
            f"**Confidence:** {p.confidence_pct}% — "
            f"{p.confidence_emoji} {p.confidence_label}"
        )
        lines.append(
            f"**Precursors reporting:** "
            f"{p.precursors_available}/{p.precursors_total}"
        )

        if p.supporting_awards:
            awards_str = ", ".join(
                f"{a} ({w:.0%})" for a, _, w in p.supporting_awards[:4]
            )
            lines.append(f"**Basis:** {awards_str}")

        if len(p.all_candidates) > 1:
            lines.append("")
            lines.append("Full rankings:")
            lines.append("")
            for i, (name, c) in enumerate(p.all_candidates):
                marker = "→" if i == 0 else " "
                lines.append(f"  {marker} {name} — {c}%")

        if p.runner_up:
            lines.append("")
            lines.append(
                f"**Runner-up:** {p.runner_up} "
                f"({round(p.runner_up_confidence * 100, 1)}%)"
            )

        lines += ["", "---", ""]

    # Historical accuracy section for key categories
    lines += ["## Historical Accuracy of Top Precursors", ""]
    key_cats = [
        "Best Picture", "Best Director", "Best Actor", "Best Actress",
        "Best Supporting Actor", "Best Supporting Actress",
        "Best Animated Feature Film",
    ]
    for cat in key_cats:
        if cat in accuracy:
            lines.append(f"**{cat}:**")
            lines.append("")
            sorted_precs = sorted(
                accuracy[cat], key=lambda pa: -pa.accuracy
            )
            for pa in sorted_precs[:5]:
                if pa.total > 0 and pa.accuracy > 0:
                    lines.append(
                        f"  - {pa.award}: {pa.accuracy:.0%} "
                        f"({pa.matches}/{pa.total} years matched)"
                    )
            lines.append("")

    # Caveats
    lines += [
        "---",
        "",
        "## Caveats",
        "",
        "1. **Short films and Documentary Short**: Very few precursor awards "
        "cover these categories. Predictions here are low-confidence.",
        "2. **Best Casting**: A new Oscar category with no historical baseline. "
        "All precursors get minimum weight.",
        "3. **Best Actor race**: Unusually competitive — four different precursors "
        "picked four different winners.",
        "4. **Festival awards** (Cannes, Venice, Toronto): Not yet held for the "
        "current season, so their signal is missing.",
        "5. **Name matching**: Some precursors list winners by person name, "
        "others by film title. Fuzzy matching handles most cases but some "
        "signal may be lost.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load data
    print("Loading data...")
    df = load_data(DATA_FILES)

    # 2. Build category mapping
    print("Building category mapping...")
    category_mapping = build_category_mapping()
    print(f"  Mapped {len(category_mapping)} Oscar categories")

    # 3. Compute historical accuracy
    print("Computing historical accuracy...")
    accuracy = compute_historical_accuracy(df, category_mapping)

    # Print top precursors for key categories
    for cat in ["Best Picture", "Best Director", "Best Actor", "Best Actress"]:
        top = sorted(accuracy[cat], key=lambda pa: -pa.accuracy)[:3]
        top_str = ", ".join(
            f"{pa.award} ({pa.accuracy:.0%})" for pa in top if pa.accuracy > 0
        )
        print(f"  {cat}: {top_str}")

    # 4. Generate predictions
    print(f"\nGenerating {PREDICTION_YEAR} predictions...")
    predictions = generate_predictions(df, category_mapping, accuracy)

    # 5. Print to console
    print_predictions(predictions)

    # 6. Save markdown report
    report = generate_markdown_report(predictions, accuracy)
    OUTPUT_FILE.write_text(report)
    print(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
