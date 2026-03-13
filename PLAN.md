# Oscar Awards Website — Redesign Plan

## Dataset Overview

This project analyzes **Oscar prediction data** — specifically, how well other film industry awards predict Oscar winners. The dataset contains **5,346 records** across two CSV files, spanning **26 years** of ceremony data from **2000 to 2025**.

Each record represents a winner in a specific award category for a given year, with fields including the award show, category, winner name, date awarded, and country.

### Oscar Categories

The dataset tracks all **24 Oscar categories**: Best Picture, Best Director, the four acting categories (Actor, Actress, Supporting Actor, Supporting Actress), both screenplay categories, and technical/craft categories like Cinematography, Film Editing, Sound, Visual Effects, Costume Design, Makeup and Hairstyling, Production Design, Original Score, Original Song, and the short film and international/documentary/animated feature categories. Best Casting is also included as a newly added category.

### Precursor Awards

The core concept behind this project is **precursor awards** — industry and critical awards that take place *before* the Oscars each year and have historically shown predictive power for Oscar outcomes. The idea is that by tracking which films and artists win these earlier awards, we can estimate who is likely to win at the Oscars.

The dataset tracks **15 precursor award shows**, organized into four types:

- **Industry Guilds** — DGA (Directors Guild), PGA (Producers Guild), SAG (Screen Actors Guild), WGA (Writers Guild), ACE Eddie Awards. These are voted on by professionals in the relevant craft, making them strong predictors for their corresponding Oscar categories.
- **Major Award Shows** — BAFTA, Golden Globes, Critics Choice Awards. High-profile ceremonies with broad category coverage.
- **Critics Circles** — New York Film Critics Circle (NYFCC), Los Angeles Film Critics Association (LAFCA), National Society of Film Critics (NSFC), National Board of Review (NBR). Smaller bodies of film critics whose picks sometimes diverge significantly from Oscar outcomes.
- **Festivals** — Venice (Golden Lion), Cannes (Palme d'Or), Toronto (People's Choice Award). These happen months before the Oscars and offer early signals, though with weaker predictive power overall.

Predictive accuracy varies widely — guild awards like the DGA (85% for Best Director) and SAG tend to be the strongest predictors, while critics circles and festivals are less reliable but still informative as part of the broader consensus picture.

## Visualization: Accuracy Heatmap

The centerpiece visualization is a **heatmap of precursor accuracy by Oscar category**.

- **Y-axis (rows):** Oscar categories — Best Picture, Best Director, the acting categories, screenplay, and technical/craft awards.
- **X-axis (columns):** Precursor award shows — BAFTA, Cannes, Critics Choice, Golden Globes, LAFCA, NBR, NSFC, NYFCC, PGA, SAG, Venice, etc.
- **Cell values:** The historical accuracy (%) of that precursor for that Oscar category — i.e., how often the precursor's winner went on to win the Oscar, over the 2000–2025 period.
- **Color scale:** A red-to-yellow-to-green gradient. Deep red means the precursor almost never predicts the Oscar winner for that category; deep green means it almost always does.
- **Empty cells:** Not every precursor covers every Oscar category. Where a precursor has no corresponding category (or fewer than 5 years of data), the cell is left blank. This sparsity is itself informative — it shows which categories have rich precursor coverage and which are essentially unpredictable from prior awards.

### What the heatmap reveals

The heatmap makes several patterns immediately visible:

1. **Precursor strength is category-specific.** The DGA's extraordinary 85% accuracy is specific to Best Director — it tells you nothing about Best Picture. For Best Picture, the PGA is king (73%). For acting categories, SAG and BAFTA are the most reliable signals.
2. **Coverage varies dramatically.** Prestige categories (Picture, Director, Acting) have a full row of data across many precursors. Technical categories (Sound, Makeup, Visual Effects) may only have one or two precursors — typically BAFTA and Critics Choice — leaving large gaps.
3. **Some categories are hard to predict from any precursor.** Film Editing and Original Song show low accuracy across the board, suggesting Oscar voters in those branches diverge from other award bodies.
4. **Festivals and critics circles are weak but not useless.** Cannes and Venice frequently show 0% cells, but they still contribute signal for Best Picture and Best International Feature Film.
5. **BAFTA and Critics Choice have the broadest coverage** — they appear in almost every row, making them useful "generalist" precursors even when they aren't the most accurate for a given category.

## Visualization: Year-by-Year Match Grid

A per-year "scorecard" view using the same heatmap structure as the accuracy chart, but showing hit/miss results for a single awards season.

### Layout

Same axes as the accuracy heatmap, with a **year selector** (dropdown, slider, or arrow buttons) to flip between years:

- **Rows:** Oscar categories (same order as the accuracy heatmap).
- **Columns:** Precursor award shows (same columns as the accuracy heatmap), plus one additional **Oscars column** that displays the actual Oscar winner's name. This column is the "answer key."
- **Cell color:** Binary — **green** if the precursor's pick matched the Oscar winner for that category, **red** if it didn't, **grey/empty** if no data.
- **Hover tooltip:** Reveals the film or person name the precursor picked. For green cells this confirms the match; for red cells it shows who they picked instead.

### What this reveals

Each year's grid is a complete scorecard of the awards season:

- **Dominant years** show columns of near-solid green — a precursor that nailed almost every category.
- **Upset categories** stand out as rows of mostly red — the Oscar winner surprised everyone.
- **Consensus rows** (all green) show categories where the outcome was a foregone conclusion across all precursors.
- **Fragmented rows** (mixed green/red) reveal genuinely competitive races where precursors disagreed.
- Flipping between years tells a story: some seasons are predictable (2020), others are chaotic (2017).

### Design notes

- The **Oscars column** should be visually distinct — e.g., a wider column, different background, or bold text — since it's the reference point, not a precursor.
- Cells are color-only (no text) to keep the grid compact. All detail lives in hover tooltips.
- Consider a summary row at the bottom showing each precursor's hit rate for that year (e.g., "14/21").

## Visualization: Precursor Agreement Matrix

A pairwise agreement heatmap showing how often any two precursor awards pick the same winner — filtered by Oscar category.

### Layout

- **Both axes (rows and columns):** Precursor award shows (only those relevant to the selected category). The matrix is symmetric, so only the lower triangle needs to be filled (diagonal is always 100%).
- **Cell value:** Agreement rate (%) — how often the two precursors picked the same winner over the 2000–2025 period.
- **Color scale:** Light/pale (low agreement) to dark (high agreement). The screenshot uses a cream-to-dark-burgundy scale.
- **Category navigation:** The user can switch between Oscar categories using left/right arrows, a dropdown, or horizontal tabs. The title updates to reflect the selected category (e.g., "Best Picture: How Often Do Precursor Awards Agree?").

### What this reveals

- **Clusters of aligned awards.** For Best Picture, PGA and Critics Choice agree 62% of the time, while critics circles (LAFCA, NBR, NSFC) agree strongly with each other but diverge from industry guilds. This shows the "industry vs. critics" split.
- **Independent voices.** Low agreement between two precursors means they bring distinct signal — useful for ensemble predictions. High agreement means they're partially redundant.
- **Category-specific alliances.** The agreement structure shifts when you switch categories. Guilds dominate acting categories (SAG aligns with BAFTA), while critics circles matter more for screenplay and international film.
- **The diagonal anchors the reading** — 100% is always dark, giving a consistent reference for how to interpret the color scale.

### Design notes

- Only show the lower triangle (or upper) to avoid redundancy. The diagonal (100%) can be shown or omitted.
- The category switcher should feel lightweight — arrow keys or swipe, not a full page reload. The grid animates or transitions smoothly between categories.
- Since different categories have different sets of relevant precursors, the matrix dimensions change per category. The grid should resize gracefully.
