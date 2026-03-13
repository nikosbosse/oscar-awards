"""
Build data.json for the Oscar Awards prediction website.

Reads CSV data from two files and computes:
1. Accuracy heatmap: historical accuracy of precursor awards predicting Oscar winners
2. Year-by-year match grid: per-year, per-category, per-precursor match details
3. Agreement matrix: pairwise agreement between precursor awards
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILES = [
    SCRIPT_DIR / "film awards research part 1.csv",
    SCRIPT_DIR / "film awards research part 2.csv",
]
OUTPUT_FILE = SCRIPT_DIR / "data.json"

YEAR_RANGE = range(2000, 2026)  # 2000-2025 inclusive

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

# Awards whose "Year of award ceremony" in the CSV needs +1 to align with the
# Oscar ceremony year.  These awards happen before the Oscar eligibility window
# closes (or use the film-release year convention), so their year N corresponds
# to Oscar year N+1.
#
# - TIFF / Venice (September) and Cannes (May): festival year N -> Oscar year N+1
# - LAFCA (December): consistently year N -> Oscar year N+1
# - NBR, NSFC, NYFCC: announce Dec/Jan with inconsistent year column in the CSV;
#   a date-based heuristic is applied in load_data() to fix them.
YEAR_OFFSET_AWARDS = {
    "Toronto People's Choice Award",
    "Venice Golden Lion",
    "Cannes Palme d'Or",
    "Los Angeles Film Critics Association",
}

shared_screenplay_precursors = [
    ("National Society of Film Critics", "Best Screenplay"),
    ("New York Film Critics Circle", "Best Screenplay"),
    ("Los Angeles Film Critics Association", "Best Screenplay"),
    ("Venice Golden Lion", "Award for Best Screenplay"),
    ("Cannes Palme d'Or", "Prix du scénario"),
]

CATEGORY_MAPPING = {
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

MIN_YEARS = 5  # Minimum years of data for accuracy/agreement cells

# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------

_paren_re = re.compile(r"\(([^)]*)\)")
_cap_word_re = re.compile(r"\b([A-Z][a-zA-Z''-]+)\b")


def _core_text(name: str) -> str:
    """Return name with parenthetical text removed, stripped & lowered."""
    return _paren_re.sub("", name).strip().lower()


def _paren_text(name: str) -> str:
    """Return the parenthetical text content, stripped & lowered."""
    matches = _paren_re.findall(name)
    return " ".join(matches).strip().lower()


def _proper_nouns(name: str) -> set[str]:
    """Extract capitalized words (proper nouns) from a name."""
    return set(_cap_word_re.findall(name))


def names_match(a: str, b: str) -> bool:
    """Fuzzy match two winner names."""
    if not a or not b:
        return False

    a_stripped = a.strip()
    b_stripped = b.strip()

    # Exact match (case-insensitive)
    if a_stripped.lower() == b_stripped.lower():
        return True

    # Substring match (one contains the other, case-insensitive)
    a_low = a_stripped.lower()
    b_low = b_stripped.lower()
    if a_low in b_low or b_low in a_low:
        return True

    # Core text match (ignoring parentheticals)
    a_core = _core_text(a_stripped)
    b_core = _core_text(b_stripped)
    if a_core and b_core and a_core == b_core:
        return True

    # Cross-match: core text of one matches parenthetical of other
    a_paren = _paren_text(a_stripped)
    b_paren = _paren_text(b_stripped)
    if a_core and b_paren and (a_core in b_paren or b_paren in a_core):
        return True
    if b_core and a_paren and (b_core in a_paren or a_paren in b_core):
        return True

    # Proper noun overlap (at least 2 shared capitalized words)
    a_nouns = _proper_nouns(a_stripped)
    b_nouns = _proper_nouns(b_stripped)
    if len(a_nouns & b_nouns) >= 2:
        return True

    return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _normalize_quotes(s: str) -> str:
    """Replace smart apostrophes with regular ASCII apostrophes."""
    if not isinstance(s, str):
        return s
    return (
        s.replace("\u2018", "'")  # left single quotation mark
        .replace("\u2019", "'")   # right single quotation mark
    )


# Awards with inconsistent year columns where we use date_awarded to fix alignment.
# If date_awarded falls in months 7-12, the precursor is for the *next* Oscar year.
# If date_awarded falls in months 1-6, it matches the Oscar year already.
DATE_BASED_OFFSET_AWARDS = {
    "National Board of Review",
    "National Society of Film Critics",
    "New York Film Critics Circle",
}


def load_data() -> pd.DataFrame:
    """Load and concatenate the two CSV files."""
    frames = [pd.read_csv(f) for f in CSV_FILES]
    df = pd.concat(frames, ignore_index=True)

    # Normalize smart quotes in text columns
    for col in ["Award", "Category", "winner"]:
        df[col] = df[col].apply(
            lambda x: _normalize_quotes(x) if isinstance(x, str) else x
        )

    # Convert year to int, drop rows with invalid years
    df["Year of award ceremony"] = pd.to_numeric(
        df["Year of award ceremony"], errors="coerce"
    )
    df = df.dropna(subset=["Year of award ceremony"])
    df["Year of award ceremony"] = df["Year of award ceremony"].astype(int)

    # --- Align precursor years to Oscar ceremony year ---

    # Fixed-offset awards: their CSV year N always maps to Oscar year N+1
    mask_offset = df["Award"].isin(YEAR_OFFSET_AWARDS)
    df.loc[mask_offset, "Year of award ceremony"] += 1

    # Date-based offset awards: use date_awarded month to decide
    mask_date = df["Award"].isin(DATE_BASED_OFFSET_AWARDS)
    if mask_date.any():
        dates = pd.to_datetime(df.loc[mask_date, "date_awarded"], errors="coerce")
        # Where date is in Jul-Dec, the entry is for the next Oscar year
        late_year = dates.dt.month.ge(7) & dates.notna()
        df.loc[mask_date & late_year.reindex(df.index, fill_value=False),
               "Year of award ceremony"] += 1

    # Filter to year range (after adjustments, some may fall outside)
    df = df[df["Year of award ceremony"].isin(YEAR_RANGE)]

    return df


# ---------------------------------------------------------------------------
# Build lookup: (award, category, year) -> winner
# ---------------------------------------------------------------------------

def build_winner_lookup(df: pd.DataFrame) -> dict[tuple[str, str, int], str]:
    """Build a dict mapping (Award, Category, Year) -> winner string."""
    lookup = {}
    for _, row in df.iterrows():
        award = row["Award"]
        cat = row["Category"]
        year = int(row["Year of award ceremony"])
        winner = row["winner"]
        if pd.isna(winner) or str(winner).strip() == "":
            continue
        lookup[(award, cat, year)] = str(winner).strip()
    return lookup


# ---------------------------------------------------------------------------
# Build precursor data per Oscar category
# ---------------------------------------------------------------------------

def get_precursor_pick(
    lookup: dict[tuple[str, str, int], str],
    oscar_cat: str,
    precursor_award: str,
    year: int,
    oscar_winner: str | None,
    precursor_entries: list[tuple[str, str]],
) -> tuple[str | None, bool]:
    """
    For a given Oscar category, precursor award, and year, find the precursor's pick.

    When multiple categories from the same award map to the same Oscar category,
    aggregate: if ANY matched the Oscar winner, use that one. Otherwise pick any.

    Returns (pick_name, matched_oscar).
    """
    # Collect all (award, cat) entries for this precursor award
    cats_for_award = [
        (aw, cat) for aw, cat in precursor_entries if aw == precursor_award
    ]

    picks = []
    for aw, cat in cats_for_award:
        winner = lookup.get((aw, cat, year))
        if winner:
            matched = oscar_winner is not None and names_match(winner, oscar_winner)
            picks.append((winner, matched))

    if not picks:
        return None, False

    # If any matched, prefer the matched one
    for pick, matched in picks:
        if matched:
            return pick, True

    # Otherwise return the first available pick
    return picks[0][0], False


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def build_datasets(df: pd.DataFrame) -> dict:
    lookup = build_winner_lookup(df)

    # Oscar winners: (oscar_category, year) -> winner
    oscar_winners = {}
    for (aw, cat, yr), winner in lookup.items():
        if aw == "Oscars":
            oscar_winners[(cat, yr)] = winner

    # Collect all precursor award short names used across all categories
    all_precursor_short_names = set()
    for entries in CATEGORY_MAPPING.values():
        for aw, _ in entries:
            all_precursor_short_names.add(AWARD_SHORT[aw])

    oscar_categories = list(CATEGORY_MAPPING.keys())
    precursor_awards_sorted = sorted(all_precursor_short_names)
    years = list(YEAR_RANGE)

    # Reverse lookup: short name -> full award name(s)
    short_to_full = defaultdict(set)
    for full, short in AWARD_SHORT.items():
        short_to_full[short].add(full)

    # -----------------------------------------------------------------------
    # 1. Accuracy heatmap & 2. Yearly grid (computed together)
    # -----------------------------------------------------------------------
    accuracy_heatmap = {}
    yearly_grid = {str(y): {} for y in years}

    for oscar_cat, precursor_entries in CATEGORY_MAPPING.items():
        # Identify unique precursor awards (by full name) for this category
        precursor_full_names = sorted(set(aw for aw, _ in precursor_entries))

        # Per-award tracking for accuracy
        award_hits = defaultdict(int)   # full_award_name -> hit count
        award_years = defaultdict(int)  # full_award_name -> years with data

        for year in years:
            oscar_winner = oscar_winners.get((oscar_cat, year))

            year_data = yearly_grid[str(year)].setdefault(oscar_cat, {
                "oscar_winner": oscar_winner,
                "precursors": {},
            })

            for precursor_full in precursor_full_names:
                pick, matched = get_precursor_pick(
                    lookup, oscar_cat, precursor_full, year,
                    oscar_winner, precursor_entries,
                )
                short = AWARD_SHORT[precursor_full]

                if pick is not None:
                    year_data["precursors"][short] = {
                        "pick": pick,
                        "matched": matched,
                    }

                    # For accuracy, only count if we have both oscar winner and pick
                    if oscar_winner is not None:
                        award_years[precursor_full] += 1
                        if matched:
                            award_hits[precursor_full] += 1

        # Compute accuracy
        cat_accuracy = {}
        for precursor_full in precursor_full_names:
            short = AWARD_SHORT[precursor_full]
            n = award_years[precursor_full]
            if n >= MIN_YEARS:
                cat_accuracy[short] = round(
                    100 * award_hits[precursor_full] / n
                )
        accuracy_heatmap[oscar_cat] = cat_accuracy

    # -----------------------------------------------------------------------
    # 3. Agreement matrix
    # -----------------------------------------------------------------------
    agreement_matrix = {}

    for oscar_cat, precursor_entries in CATEGORY_MAPPING.items():
        precursor_full_names = sorted(set(aw for aw, _ in precursor_entries))

        # Collect picks per award per year
        award_picks: dict[str, dict[int, str]] = defaultdict(dict)
        for year in years:
            oscar_winner = oscar_winners.get((oscar_cat, year))
            for precursor_full in precursor_full_names:
                pick, _ = get_precursor_pick(
                    lookup, oscar_cat, precursor_full, year,
                    oscar_winner, precursor_entries,
                )
                if pick is not None:
                    award_picks[precursor_full][year] = pick

        # Pairwise agreement
        cat_agreement = {}
        for i, a1 in enumerate(precursor_full_names):
            s1 = AWARD_SHORT[a1]
            for a2 in precursor_full_names[i + 1:]:
                s2 = AWARD_SHORT[a2]
                # Find overlapping years
                overlap_years = set(award_picks[a1].keys()) & set(award_picks[a2].keys())
                if len(overlap_years) < MIN_YEARS:
                    continue
                agree_count = sum(
                    1 for y in overlap_years
                    if names_match(award_picks[a1][y], award_picks[a2][y])
                )
                pct = round(100 * agree_count / len(overlap_years))

                cat_agreement.setdefault(s1, {})[s2] = pct
                cat_agreement.setdefault(s2, {})[s1] = pct

        if cat_agreement:
            agreement_matrix[oscar_cat] = cat_agreement

    # -----------------------------------------------------------------------
    # 4. Precursor count vs Oscar win probability
    # -----------------------------------------------------------------------
    # For each category and year, find all distinct people/films who won at
    # least one precursor, count how many they won, and note if they won Oscar.
    precursor_count_data = {}

    for oscar_cat, precursor_entries in CATEGORY_MAPPING.items():
        precursor_full_names = sorted(set(aw for aw, _ in precursor_entries))
        # Collect observations: list of (count, won_oscar) across all years
        observations_by_count: dict[int, dict] = defaultdict(
            lambda: {"wins": 0, "total": 0, "instances": []}
        )

        for year in years:
            oscar_winner = oscar_winners.get((oscar_cat, year))
            if oscar_winner is None:
                continue

            # Build dict: canonical_name -> {precursors_won: set, is_oscar_winner: bool}
            # We need to cluster names that refer to the same person/film
            name_clusters: list[dict] = []  # [{name, precursors: set, is_oscar_winner: bool}]

            for precursor_full in precursor_full_names:
                pick, matched = get_precursor_pick(
                    lookup, oscar_cat, precursor_full, year,
                    oscar_winner, precursor_entries,
                )
                if pick is None:
                    continue
                short = AWARD_SHORT[precursor_full]

                # Try to merge into existing cluster
                merged = False
                for cluster in name_clusters:
                    if names_match(pick, cluster["name"]):
                        cluster["precursors"].add(short)
                        if matched:
                            cluster["is_oscar_winner"] = True
                        # Keep longer name as canonical
                        if len(pick) > len(cluster["name"]):
                            cluster["name"] = pick
                        merged = True
                        break
                if not merged:
                    name_clusters.append({
                        "name": pick,
                        "precursors": {short},
                        "is_oscar_winner": matched,
                    })

            for cluster in name_clusters:
                count = len(cluster["precursors"])
                observations_by_count[count]["total"] += 1
                if cluster["is_oscar_winner"]:
                    observations_by_count[count]["wins"] += 1
                observations_by_count[count]["instances"].append({
                    "year": year,
                    "name": cluster["name"],
                    "won_oscar": cluster["is_oscar_winner"],
                })

        # Convert to serializable format
        cat_count_data = {}
        for count, stats in sorted(observations_by_count.items()):
            if stats["total"] >= 2:  # Need at least 2 observations
                instances = sorted(stats["instances"], key=lambda x: x["year"])
                cat_count_data[str(count)] = {
                    "win_pct": round(100 * stats["wins"] / stats["total"]),
                    "wins": stats["wins"],
                    "total": stats["total"],
                    "instances": instances,
                }
        if cat_count_data:
            precursor_count_data[oscar_cat] = cat_count_data

    # -----------------------------------------------------------------------
    # 5. Winning precursor combinations
    # -----------------------------------------------------------------------
    # For each category, look at each year's Oscar winner and record which
    # specific precursors they won. Aggregate common combinations.
    combination_data = {}

    for oscar_cat, precursor_entries in CATEGORY_MAPPING.items():
        precursor_full_names = sorted(set(aw for aw, _ in precursor_entries))
        # combo_key (sorted tuple) -> {wins, total, instances}
        combo_stats: dict[tuple, dict] = defaultdict(
            lambda: {"wins": 0, "total": 0, "instances": []}
        )

        for year in years:
            oscar_winner = oscar_winners.get((oscar_cat, year))
            if oscar_winner is None:
                continue

            # Build clusters like above
            name_clusters: list[dict] = []
            for precursor_full in precursor_full_names:
                pick, matched = get_precursor_pick(
                    lookup, oscar_cat, precursor_full, year,
                    oscar_winner, precursor_entries,
                )
                if pick is None:
                    continue
                short = AWARD_SHORT[precursor_full]
                merged = False
                for cluster in name_clusters:
                    if names_match(pick, cluster["name"]):
                        cluster["precursors"].add(short)
                        if matched:
                            cluster["is_oscar_winner"] = True
                        if len(pick) > len(cluster["name"]):
                            cluster["name"] = pick
                        merged = True
                        break
                if not merged:
                    name_clusters.append({
                        "name": pick,
                        "precursors": {short},
                        "is_oscar_winner": matched,
                    })

            for cluster in name_clusters:
                combo_key = tuple(sorted(cluster["precursors"]))
                combo_stats[combo_key]["total"] += 1
                if cluster["is_oscar_winner"]:
                    combo_stats[combo_key]["wins"] += 1
                combo_stats[combo_key]["instances"].append({
                    "year": year,
                    "name": cluster["name"],
                    "won_oscar": cluster["is_oscar_winner"],
                })

        # Convert to list, filter to combos seen at least 2 times
        combos = []
        for combo_key, stats in combo_stats.items():
            if stats["total"] >= 2:
                # Sort instances by year
                instances = sorted(stats["instances"], key=lambda x: x["year"])
                combos.append({
                    "precursors": list(combo_key),
                    "win_pct": round(100 * stats["wins"] / stats["total"]),
                    "wins": stats["wins"],
                    "total": stats["total"],
                    "instances": instances,
                })
        # Sort by total occurrences descending, then win_pct descending
        combos.sort(key=lambda x: (-x["total"], -x["win_pct"]))
        if combos:
            combination_data[oscar_cat] = combos

    return {
        "oscar_categories": oscar_categories,
        "precursor_awards": precursor_awards_sorted,
        "years": years,
        "accuracy_heatmap": accuracy_heatmap,
        "yearly_grid": yearly_grid,
        "agreement_matrix": agreement_matrix,
        "precursor_count": precursor_count_data,
        "combinations": combination_data,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Loading CSV data...")
    df = load_data()
    print(f"  Loaded {len(df)} rows, years {df['Year of award ceremony'].min()}-{df['Year of award ceremony'].max()}")

    print("Building datasets...")
    data = build_datasets(df)

    print(f"Writing {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"  Oscar categories: {len(data['oscar_categories'])}")
    print(f"  Precursor awards: {len(data['precursor_awards'])}")
    print(f"  Accuracy heatmap cells: {sum(len(v) for v in data['accuracy_heatmap'].values())}")
    yearly_cells = sum(
        len(cat_data.get('precursors', {}))
        for year_data in data['yearly_grid'].values()
        for cat_data in year_data.values()
    )
    print(f"  Yearly grid cells: {yearly_cells}")
    print(f"  Agreement matrix categories: {len(data['agreement_matrix'])}")
    print("Done.")


if __name__ == "__main__":
    main()
