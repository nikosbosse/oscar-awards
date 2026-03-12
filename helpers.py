"""
helpers.py — Shared utilities for Oscar prediction project
==========================================================

Contains:
  - Configuration constants (file paths, year ranges, category lists)
  - Data loading functions
  - Category mapping (precursor award → Oscar category)
  - Name matching and nominee clustering logic
  - Historical accuracy computation
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

# Short labels for award names (for plot readability / display)
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
    "Oscars": "Oscars",
    "PGA Awards": "PGA",
    "SAG Awards": "SAG",
    "Toronto People's Choice Award": "TIFF",
    "Venice Golden Lion": "Venice",
    "WGA Awards": "WGA",
}


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

def load_data(file_paths: list[Path] | None = None) -> pd.DataFrame:
    """Load and concatenate all CSV data files."""
    if file_paths is None:
        file_paths = DATA_FILES
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
# Name Matching
# ---------------------------------------------------------------------------

def names_match(a: str, b: str) -> bool:
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


def cluster_nominees(
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
            if names_match(name, canon):
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
# Historical Accuracy
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
                    if names_match(oscar_winner, prec_winner):
                        pa.matches += 1
                        pa.match_years.append(year)

            cat_results.append(pa)

        results[oscar_cat] = cat_results

    return results
