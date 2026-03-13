(function () {
  "use strict";

  let DATA = null;
  let currentYearIndex = 0;
  let heatmapMode = "accuracy"; // "accuracy" or "yearly"
  let currentCatIndex = 0;
  let currentCountCatIndex = 0;
  let currentCombosCatIndex = 0;
  let predActiveAwards = new Set();
  let countModeAll = false;
  let combosModeAll = false;
  let countThreshold = 2;
  let combosThreshold = 2;

  const tooltip = document.getElementById("tooltip");

  // ── Helpers ──────────────────────────────────────────────

  /**
   * Accuracy color: HSL interpolation from red (0%) through yellow (50%) to green (80%+).
   * 0% → hue 0 (red), 50% → hue 60 (yellow), 80%+ → hue 120 (green).
   */
  function accuracyColor(pct) {
    let hue;
    if (pct <= 50) {
      hue = (pct / 50) * 60; // 0→60
    } else {
      hue = 60 + ((pct - 50) / 30) * 60; // 60→120, capped
      if (hue > 120) hue = 120;
    }
    const sat = 70;
    const light = 35 + (pct / 100) * 15; // slightly brighter for higher values
    return `hsl(${hue}, ${sat}%, ${light}%)`;
  }

  /**
   * Agreement color: cream (#FFF8E1) to dark burgundy (#4A0000).
   */
  function agreementColor(pct) {
    const t = pct / 100;
    const r = Math.round(255 - t * (255 - 74));
    const g = Math.round(248 - t * 248);
    const b = Math.round(225 - t * 225);
    return `rgb(${r}, ${g}, ${b})`;
  }

  function agreementTextColor(pct) {
    return pct > 50 ? "#f0d0c0" : "#333";
  }

  function showTooltip(e, html) {
    tooltip.innerHTML = html;
    tooltip.classList.add("visible");
    positionTooltip(e);
  }

  function hideTooltip() {
    tooltip.classList.remove("visible");
  }

  function positionTooltip(e) {
    const pad = 12;
    let x = e.clientX + pad;
    let y = e.clientY + pad;
    const rect = tooltip.getBoundingClientRect();
    if (x + rect.width > window.innerWidth - pad) {
      x = e.clientX - rect.width - pad;
    }
    if (y + rect.height > window.innerHeight - pad) {
      y = e.clientY - rect.height - pad;
    }
    tooltip.style.left = x + "px";
    tooltip.style.top = y + "px";
  }

  // ── Category Chip Selectors ────────────────────────────

  function initCatSelector(containerId, categories, activeIndex, onSelect) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";
    for (let i = 0; i < categories.length; i++) {
      const chip = document.createElement("span");
      chip.className = "cat-chip" + (i === activeIndex ? " active" : "");
      chip.textContent = categories[i].replace("Best ", "");
      chip.dataset.index = i;
      chip.addEventListener("click", () => {
        container.querySelector(".cat-chip.active")?.classList.remove("active");
        chip.classList.add("active");
        onSelect(i);
      });
      container.appendChild(chip);
    }
  }

  function updateCatSelector(containerId, activeIndex) {
    const container = document.getElementById(containerId);
    container.querySelector(".cat-chip.active")?.classList.remove("active");
    const chip = container.querySelector(`[data-index="${activeIndex}"]`);
    if (chip) chip.classList.add("active");
  }

  // ── Predictability Bar Chart ────────────────────────────

  function renderPredictabilityChart() {
    const container = document.getElementById("predictability-chart");
    const categories = DATA.oscar_categories;
    const heatmap = DATA.accuracy_heatmap;

    // Compute best and second-best accuracy per category
    const rows = [];
    for (const cat of categories) {
      const catData = heatmap[cat] || {};
      const values = Object.entries(catData).filter(([, v]) => v !== null && v !== undefined);
      if (values.length === 0) continue;

      // Sort by value descending
      values.sort((a, b) => b[1] - a[1]);
      const bestVal = values[0][1];
      const bestName = values[0][0];
      const secondVal = values.length > 1 ? values[1][1] : null;
      const secondName = values.length > 1 ? values[1][0] : null;
      rows.push({ cat, bestVal, bestName, secondVal, secondName });
    }

    // Sort by best accuracy descending
    rows.sort((a, b) => b.bestVal - a.bestVal);

    // Define tiers
    const tiers = [
      { label: "Highly predictable", min: 70 },
      { label: "Moderately predictable", min: 50 },
      { label: "Hard to predict", min: 0 },
    ];

    container.innerHTML = "";

    const maxVal = 100;

    let rowIndex = 0;
    let currentTier = -1;
    for (const row of rows) {
      // Determine tier
      const tierIdx = tiers.findIndex(t => row.bestVal >= t.min);
      if (tierIdx !== currentTier) {
        currentTier = tierIdx;
        const tierLabel = document.createElement("div");
        tierLabel.className = "predictability-tier-label";
        tierLabel.textContent = tiers[tierIdx].label;
        container.appendChild(tierLabel);
      }

      const rowEl = document.createElement("div");
      rowEl.className = "predictability-row";

      const label = document.createElement("div");
      label.className = "predictability-label";
      label.textContent = row.cat.replace("Best ", "");
      rowEl.appendChild(label);

      const bars = document.createElement("div");
      bars.className = "predictability-bars";

      // Best precursor bar, colored by award
      const bestBar = document.createElement("div");
      bestBar.className = "predictability-bar-best";
      bestBar.style.width = (row.bestVal / maxVal * 100) + "%";
      bestBar.style.background = AWARD_COLORS[row.bestName] || "#4CAF50";
      bestBar.style.animationDelay = (rowIndex * 40) + "ms";
      bars.appendChild(bestBar);

      // Annotation with colored award name
      const annotation = document.createElement("span");
      annotation.className = "predictability-annotation";
      annotation.style.left = (row.bestVal / maxVal * 100) + "%";
      annotation.style.animationDelay = (rowIndex * 40 + 300) + "ms";

      const nameSpan = document.createElement("span");
      nameSpan.className = "award-name";
      nameSpan.textContent = row.bestName;
      nameSpan.style.color = AWARD_COLORS[row.bestName] || "#4CAF50";
      annotation.appendChild(nameSpan);

      const pctSpan = document.createElement("span");
      pctSpan.className = "award-pct";
      pctSpan.textContent = ` ${row.bestVal}%`;
      annotation.appendChild(pctSpan);

      bars.appendChild(annotation);

      rowEl.appendChild(bars);

      // Richer tooltip: show top 2 precursors
      rowEl.addEventListener("mouseenter", (e) => {
        let html = `<strong>${row.cat}</strong><br>` +
          `Best: ${row.bestName} (${row.bestVal}%)`;
        if (row.secondVal !== null) {
          html += `<br>2nd: ${row.secondName} (${row.secondVal}%)`;
        }
        showTooltip(e, html);
      });
      rowEl.addEventListener("mousemove", positionTooltip);
      rowEl.addEventListener("mouseleave", hideTooltip);

      container.appendChild(rowEl);
      rowIndex++;
    }

    // Add 50% reference line to the bars area
    // We need to measure after render, so defer
    requestAnimationFrame(() => {
      const firstRow = container.querySelector(".predictability-bars");
      if (!firstRow) return;
      const barsRect = firstRow.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();
      const barsLeft = barsRect.left - containerRect.left;
      const barsWidth = barsRect.width;

      const refLine = document.createElement("div");
      refLine.style.cssText = `
        position: absolute;
        left: ${barsLeft + barsWidth * 0.5}px;
        top: 0;
        bottom: 0;
        width: 1px;
        background: repeating-linear-gradient(
          to bottom,
          transparent,
          transparent 3px,
          rgba(255,255,255,0.08) 3px,
          rgba(255,255,255,0.08) 6px
        );
        pointer-events: none;
        z-index: 0;
      `;
      container.style.position = "relative";
      container.appendChild(refLine);

      const refLabel = document.createElement("span");
      refLabel.textContent = "50%";
      refLabel.style.cssText = `
        position: absolute;
        left: ${barsLeft + barsWidth * 0.5}px;
        top: -2px;
        transform: translateX(-50%);
        font-size: 0.6rem;
        color: rgba(255,255,255,0.2);
        pointer-events: none;
        z-index: 0;
      `;
      container.appendChild(refLabel);
    });
  }

  // ── 2026 Oscar Forecast ────────────────────────────────

  // Default 8 forecast categories
  const FORECAST_DEFAULT_CATS = new Set([
    "Best Picture", "Best Director", "Best Actor", "Best Actress",
    "Best Original Screenplay",
  ]);
  let forecastActiveCats = new Set(FORECAST_DEFAULT_CATS);

  function initForecastCatSelector() {
    const predictions = DATA.oscar_predictions_2026 || {};
    const selector = document.getElementById("forecast-cat-selector");
    selector.innerHTML = "";

    // Ordered: defaults first, then the rest
    const defaultList = [...FORECAST_DEFAULT_CATS].filter(c => predictions[c]);
    const otherList = DATA.oscar_categories.filter(
      c => !FORECAST_DEFAULT_CATS.has(c) && predictions[c]
    );
    const allCats = [...defaultList, ...otherList];

    for (const cat of allCats) {
      const chip = document.createElement("span");
      const shortName = cat.replace("Best ", "").replace("Feature Film", "Feature");
      chip.className = "award-chip" + (forecastActiveCats.has(cat) ? "" : " inactive");
      chip.textContent = shortName;
      chip.dataset.cat = cat;
      chip.addEventListener("click", () => {
        if (forecastActiveCats.has(cat)) {
          forecastActiveCats.delete(cat);
          chip.classList.add("inactive");
        } else {
          forecastActiveCats.add(cat);
          chip.classList.remove("inactive");
        }
        renderForecast();
      });
      selector.appendChild(chip);
    }
  }

  function renderForecast() {
    const container = document.getElementById("forecast-grid");
    const predictions = DATA.oscar_predictions_2026 || {};
    container.innerHTML = "";

    // Ordered: defaults first, then the rest
    const defaultList = [...FORECAST_DEFAULT_CATS].filter(c => predictions[c] && forecastActiveCats.has(c));
    const otherList = DATA.oscar_categories.filter(
      c => !FORECAST_DEFAULT_CATS.has(c) && predictions[c] && forecastActiveCats.has(c)
    );
    const allCats = [...defaultList, ...otherList];

    for (const cat of allCats) {
      const nominees = predictions[cat];
      if (!nominees || nominees.length === 0) continue;

      const section = document.createElement("div");
      section.className = "forecast-category";

      const title = document.createElement("h3");
      title.textContent = cat;
      section.appendChild(title);

      const maxProb = nominees[0].probability;

      for (const nominee of nominees) {

        const row = document.createElement("div");
        row.className = "forecast-nominee";

        const name = document.createElement("div");
        name.className = "forecast-name";
        name.textContent = nominee.name;
        name.title = nominee.name;
        row.appendChild(name);

        const barWrap = document.createElement("div");
        barWrap.className = "forecast-bar-wrap";

        const bar = document.createElement("div");
        bar.className = "forecast-bar " + (nominee === nominees[0] ? "frontrunner" : "other");
        bar.style.width = (nominee.probability / maxProb * 100) + "%";
        barWrap.appendChild(bar);

        const pct = document.createElement("span");
        pct.className = "forecast-pct";
        pct.textContent = nominee.probability.toFixed(1) + "%";
        barWrap.appendChild(pct);

        const precursors = document.createElement("span");
        precursors.className = "forecast-precursors";
        precursors.textContent = nominee.precursors.join(", ");
        barWrap.appendChild(precursors);

        row.appendChild(barWrap);

        // Tooltip with accuracy breakdown
        const detailParts = nominee.details
          .filter(d => d.accuracy > 0)
          .map(d => `${d.precursor}: ${d.accuracy}%`);
        if (detailParts.length > 0) {
          row.addEventListener("mouseenter", (e) => {
            showTooltip(e, `<strong>${nominee.name}</strong><br>Precursor accuracy: ${detailParts.join(", ")}`);
          });
          row.addEventListener("mousemove", positionTooltip);
          row.addEventListener("mouseleave", hideTooltip);
        }

        section.appendChild(row);
      }

      container.appendChild(section);
    }
  }

  // ── 2026 Predictions Heatmap ────────────────────────────

  // Fixed color per precursor award
  const AWARD_COLORS = {
    "ACE Eddie":      "hsl(15, 60%, 38%)",
    "BAFTA":          "hsl(210, 55%, 40%)",
    "Cannes":         "hsl(50, 65%, 38%)",
    "Critics Choice": "hsl(140, 50%, 35%)",
    "DGA":            "hsl(270, 45%, 42%)",
    "Golden Globes":  "hsl(35, 70%, 40%)",
    "LAFCA":          "hsl(180, 50%, 35%)",
    "NBR":            "hsl(330, 50%, 38%)",
    "NSFC":           "hsl(100, 45%, 35%)",
    "NYFCC":          "hsl(0, 50%, 40%)",
    "PGA":            "hsl(200, 60%, 38%)",
    "SAG":            "hsl(160, 55%, 35%)",
    "TIFF":           "hsl(285, 40%, 40%)",
    "Venice":         "hsl(60, 50%, 35%)",
    "WGA":            "hsl(22, 65%, 42%)",
  };

  function initPredAwardSelector() {
    // Find all precursors that have any 2026 data
    const allPrecursors = new Set();
    for (const catData of Object.values(DATA.predictions_2026 || {})) {
      for (const p of Object.keys(catData)) {
        allPrecursors.add(p);
      }
    }
    // Default: top 8 most predictive precursors
    const defaultOn = new Set(["BAFTA", "Critics Choice", "DGA", "Golden Globes", "PGA", "SAG", "WGA", "ACE Eddie"]);
    predActiveAwards = new Set(DATA.precursor_awards.filter(p => allPrecursors.has(p) && defaultOn.has(p)));

    const selector = document.getElementById("pred-award-selector");
    selector.innerHTML = "";

    for (const p of DATA.precursor_awards) {
      if (!allPrecursors.has(p)) continue;
      const chip = document.createElement("span");
      chip.className = "award-chip" + (predActiveAwards.has(p) ? "" : " inactive");
      chip.textContent = p;
      chip.dataset.award = p;

      chip.addEventListener("click", () => {
        if (predActiveAwards.has(p)) {
          predActiveAwards.delete(p);
          chip.classList.add("inactive");
        } else {
          predActiveAwards.add(p);
          chip.classList.remove("inactive");
        }
        renderPredictionsGrid();
      });

      selector.appendChild(chip);
    }
  }

  function renderPredictionsGrid() {
    const container = document.getElementById("predictions-grid");
    const predictions = DATA.predictions_2026 || {};
    const categories = DATA.oscar_categories;
    const precursors = DATA.precursor_awards.filter(p => predActiveAwards.has(p));

    container.style.gridTemplateColumns = `180px repeat(${precursors.length}, minmax(0, 1fr))`;
    container.innerHTML = "";

    // Corner
    const corner = document.createElement("div");
    corner.className = "corner";
    container.appendChild(corner);

    // Column headers
    for (const p of precursors) {
      const hdr = document.createElement("div");
      hdr.className = "col-header";
      const span = document.createElement("span");
      span.textContent = p;
      hdr.appendChild(span);
      container.appendChild(hdr);
    }

    // Rows
    for (const cat of categories) {
      const catData = predictions[cat] || {};

      // Row label
      const label = document.createElement("div");
      label.className = "row-label";
      label.textContent = cat;
      container.appendChild(label);

      for (const p of precursors) {
        const cell = document.createElement("div");
        cell.className = "pred-cell";

        const winner = catData[p];
        if (!winner) {
          cell.classList.add("no-data");
        } else {
          cell.style.background = AWARD_COLORS[p] || "hsl(0, 0%, 35%)";
          cell.textContent = winner;

          cell.addEventListener("mouseenter", (e) => {
            showTooltip(e, `<strong>${p}</strong><br>${winner}`);
          });
          cell.addEventListener("mousemove", positionTooltip);
          cell.addEventListener("mouseleave", hideTooltip);
        }

        container.appendChild(cell);
      }
    }
  }

  // Simple JS-side name matching for clustering nominees in the predictions grid
  // ── Heatmap mode switching ─────────────────────────────

  function setHeatmapMode(mode) {
    heatmapMode = mode;
    const accBtn = document.getElementById("heatmap-mode-accuracy");
    const yearBtn = document.getElementById("heatmap-mode-yearly");
    const yearNav = document.getElementById("yearly-nav");
    const accGrid = document.getElementById("accuracy-heatmap");
    const yearGrid = document.getElementById("yearly-grid");
    const title = document.getElementById("heatmap-title");
    const subtitle = document.getElementById("heatmap-subtitle");

    if (mode === "accuracy") {
      accBtn.classList.add("active");
      yearBtn.classList.remove("active");
      yearNav.style.display = "none";
      accGrid.style.display = "";
      yearGrid.style.display = "none";
      title.textContent = "Which Precursors Actually Matter?";
      subtitle.textContent = "For each precursor, the percentage of years where its winner went on to win the Oscar in the same category (2000\u20132025, minimum 5 years of data)";
    } else {
      yearBtn.classList.add("active");
      accBtn.classList.remove("active");
      yearNav.style.display = "";
      accGrid.style.display = "none";
      yearGrid.style.display = "";
      title.textContent = "Awards Season Scorecard";
      subtitle.textContent = "Compare precursor picks against Oscar winners for each year. Green = matched, Red = missed.";
      renderYearlyGrid();
    }
  }

  // ── Section 1: Accuracy Heatmap ─────────────────────────

  function renderAccuracyHeatmap() {
    const container = document.getElementById("accuracy-heatmap");
    const categories = DATA.oscar_categories;
    const precursors = DATA.precursor_awards;
    const numCols = precursors.length + 1; // +1 for row label

    container.style.gridTemplateColumns = `180px repeat(${precursors.length}, 1fr)`;
    container.innerHTML = "";

    // Corner
    const corner = document.createElement("div");
    corner.className = "corner";
    container.appendChild(corner);

    // Column headers
    for (const p of precursors) {
      const hdr = document.createElement("div");
      hdr.className = "col-header";
      const span = document.createElement("span");
      span.textContent = p;
      hdr.appendChild(span);
      container.appendChild(hdr);
    }

    // Rows
    for (const cat of categories) {
      // Row label
      const label = document.createElement("div");
      label.className = "row-label";
      label.textContent = cat;
      container.appendChild(label);

      const catData = DATA.accuracy_heatmap[cat] || {};

      for (const p of precursors) {
        const cell = document.createElement("div");
        cell.className = "cell";

        const val = catData[p];
        if (val === undefined || val === null) {
          cell.classList.add("empty");
        } else {
          cell.style.background = accuracyColor(val);
          cell.textContent = val + "%";
        }

        cell.addEventListener("mouseenter", (e) => {
          if (val !== undefined && val !== null) {
            showTooltip(e, `<strong>${p}</strong> → ${cat}<br>Accuracy: ${val}%`);
          }
        });
        cell.addEventListener("mousemove", positionTooltip);
        cell.addEventListener("mouseleave", hideTooltip);

        container.appendChild(cell);
      }
    }
  }

  // ── Section 2: Yearly Grid ──────────────────────────────

  function renderYearlyGrid() {
    const container = document.getElementById("yearly-grid");
    const year = DATA.years[currentYearIndex];
    const categories = DATA.oscar_categories;
    const precursors = DATA.precursor_awards;
    const yearData = DATA.yearly_grid[year] || {};

    document.getElementById("year-display").textContent = year;

    // Fade transition
    container.classList.add("fading");

    setTimeout(() => {
      container.style.gridTemplateColumns = `180px 140px repeat(${precursors.length}, 1fr)`;
      container.innerHTML = "";

      // Corner
      const corner = document.createElement("div");
      corner.className = "corner";
      container.appendChild(corner);

      // Oscar column header
      const oscarHdr = document.createElement("div");
      oscarHdr.className = "col-header";
      const oscarSpan = document.createElement("span");
      oscarSpan.textContent = "Oscar Winner";
      oscarSpan.style.color = "#d4a843";
      oscarSpan.style.fontWeight = "600";
      oscarHdr.appendChild(oscarSpan);
      container.appendChild(oscarHdr);

      // Precursor column headers
      for (const p of precursors) {
        const hdr = document.createElement("div");
        hdr.className = "col-header";
        const span = document.createElement("span");
        span.textContent = p;
        hdr.appendChild(span);
        container.appendChild(hdr);
      }

      // Track hits per precursor
      const hits = {};
      const totals = {};
      for (const p of precursors) {
        hits[p] = 0;
        totals[p] = 0;
      }

      // Rows
      for (const cat of categories) {
        const catData = yearData[cat] || {};
        const oscarWinner = catData.oscar_winner || "";
        const precursorData = catData.precursors || {};

        // Row label
        const label = document.createElement("div");
        label.className = "row-label";
        label.textContent = cat;
        container.appendChild(label);

        // Oscar winner cell
        const oscarCell = document.createElement("div");
        oscarCell.className = "oscar-cell";
        oscarCell.textContent = oscarWinner;
        container.appendChild(oscarCell);

        // Precursor cells
        for (const p of precursors) {
          const cell = document.createElement("div");
          cell.className = "cell";

          const pData = precursorData[p];
          if (!pData) {
            cell.classList.add("no-data");
          } else {
            totals[p]++;
            if (pData.matched) {
              cell.classList.add("match");
              hits[p]++;
            } else {
              cell.classList.add("miss");
            }

            cell.addEventListener("mouseenter", (e) => {
              showTooltip(
                e,
                `<strong>${p}</strong> picked: ${pData.pick}<br>Oscar winner: ${oscarWinner}`
              );
            });
            cell.addEventListener("mousemove", positionTooltip);
            cell.addEventListener("mouseleave", hideTooltip);
          }

          container.appendChild(cell);
        }
      }

      // Summary row
      const summaryLabel = document.createElement("div");
      summaryLabel.className = "summary-label";
      summaryLabel.textContent = "Hits";
      container.appendChild(summaryLabel);

      const summaryOscar = document.createElement("div");
      summaryOscar.className = "summary-oscar";
      container.appendChild(summaryOscar);

      for (const p of precursors) {
        const sc = document.createElement("div");
        sc.className = "summary-cell";
        sc.textContent = totals[p] > 0 ? `${hits[p]}/${totals[p]}` : "—";
        container.appendChild(sc);
      }

      container.classList.remove("fading");
    }, 150);
  }

  // ── Section 3: Agreement Matrix ─────────────────────────

  function renderAgreementMatrix() {
    const container = document.getElementById("agreement-matrix");
    const cat = DATA.oscar_categories[currentCatIndex];
    const catData = DATA.agreement_matrix[cat] || {};

    document.getElementById("agreement-cat-title").textContent = cat;

    // Determine which precursors have data for this category
    const precursorsInCat = Object.keys(catData);
    // Also collect any precursors mentioned in the inner objects
    const allKeys = new Set(precursorsInCat);
    for (const p of precursorsInCat) {
      for (const q of Object.keys(catData[p] || {})) {
        allKeys.add(q);
      }
    }
    const precursors = DATA.precursor_awards.filter((p) => allKeys.has(p));

    container.classList.add("fading");

    setTimeout(() => {
      container.style.gridTemplateColumns = `180px repeat(${precursors.length}, 1fr)`;
      container.innerHTML = "";

      if (precursors.length === 0) {
        const msg = document.createElement("div");
        msg.style.gridColumn = "1 / -1";
        msg.style.textAlign = "center";
        msg.style.padding = "40px";
        msg.style.color = "#666";
        msg.textContent = "No agreement data available for this category.";
        container.appendChild(msg);
        container.classList.remove("fading");
        return;
      }

      // Corner
      const corner = document.createElement("div");
      corner.className = "corner";
      container.appendChild(corner);

      // Column headers
      for (const p of precursors) {
        const hdr = document.createElement("div");
        hdr.className = "col-header";
        const span = document.createElement("span");
        span.textContent = p;
        hdr.appendChild(span);
        container.appendChild(hdr);
      }

      // Rows
      for (let i = 0; i < precursors.length; i++) {
        const rowP = precursors[i];

        const label = document.createElement("div");
        label.className = "row-label";
        label.textContent = rowP;
        container.appendChild(label);

        for (let j = 0; j < precursors.length; j++) {
          const colP = precursors[j];
          const cell = document.createElement("div");
          cell.className = "cell";

          if (j > i) {
            // Upper triangle — blank
            cell.classList.add("upper-tri");
          } else if (j === i) {
            // Diagonal = 100%
            const val = 100;
            cell.style.background = agreementColor(val);
            cell.style.color = agreementTextColor(val);
            cell.textContent = "100%";
            cell.style.textShadow = "none";
          } else {
            // Lower triangle
            const rowData = catData[rowP] || {};
            const val = rowData[colP];
            if (val === undefined || val === null) {
              cell.classList.add("empty");
            } else {
              cell.style.background = agreementColor(val);
              cell.style.color = agreementTextColor(val);
              cell.style.textShadow = "none";
              cell.textContent = val + "%";

              cell.addEventListener("mouseenter", (e) => {
                showTooltip(
                  e,
                  `<strong>${rowP}</strong> vs <strong>${colP}</strong><br>Agreement: ${val}%`
                );
              });
              cell.addEventListener("mousemove", positionTooltip);
              cell.addEventListener("mouseleave", hideTooltip);
            }
          }

          container.appendChild(cell);
        }
      }

      container.classList.remove("fading");
    }, 150);
  }

  // ── Section 4: Precursor Count Chart ────────────────────

  function renderCountChart() {
    const container = document.getElementById("count-chart");
    const cat = DATA.oscar_categories[currentCountCatIndex];
    const source = countModeAll ? DATA.precursor_count_all : DATA.precursor_count;
    const catData = (source || {})[cat];

    document.getElementById("count-cat-title").textContent = cat;

    container.classList.add("fading");

    setTimeout(() => {
      container.innerHTML = "";

      if (!catData || Object.keys(catData).length === 0) {
        container.innerHTML = '<div class="chart-no-data">No data available for this category.</div>';
        container.classList.remove("fading");
        return;
      }

      const counts = Object.keys(catData)
        .map(Number)
        .filter(c => catData[String(c)].total >= countThreshold)
        .sort((a, b) => a - b);
      const maxPct = 100;

      const chart = document.createElement("div");
      chart.className = "bar-chart";

      const detailPanel = document.createElement("div");
      detailPanel.className = "count-detail-panel";
      let activeBar = null;

      for (const count of counts) {
        const d = catData[String(count)];
        const group = document.createElement("div");
        group.className = "bar-group";

        const wrapper = document.createElement("div");
        wrapper.className = "bar-wrapper";

        const labelTop = document.createElement("div");
        labelTop.className = "bar-label-top";
        labelTop.textContent = d.win_pct + "%";

        const bar = document.createElement("div");
        bar.className = "bar";
        const heightPct = Math.max((d.win_pct / maxPct) * 100, 1);
        bar.style.height = heightPct + "%";
        bar.style.background = accuracyColor(d.win_pct);
        bar.style.cursor = "pointer";

        bar.addEventListener("mouseenter", (e) => {
          showTooltip(e,
            `<strong>${count} precursor${count > 1 ? "s" : ""} won</strong><br>` +
            `Oscar win rate: ${d.win_pct}%<br>` +
            `${d.wins} wins out of ${d.total} cases<br>` +
            `<em>Click to see details</em>`
          );
        });
        bar.addEventListener("mousemove", positionTooltip);
        bar.addEventListener("mouseleave", hideTooltip);

        bar.addEventListener("click", () => {
          // Toggle
          if (activeBar === bar) {
            activeBar.classList.remove("bar-active");
            activeBar = null;
            detailPanel.classList.remove("open");
            return;
          }
          if (activeBar) activeBar.classList.remove("bar-active");
          activeBar = bar;
          bar.classList.add("bar-active");

          // Build detail table
          const instances = d.instances || [];
          let html = `<table class="combo-detail-table"><tbody>`;
          for (const inst of instances) {
            const color = inst.won_oscar ? "#4CAF50" : "#f44336";
            const label = inst.won_oscar ? "Won Oscar" : "Did not win";
            const tags = (inst.precursors || [])
              .map(p => `<span class="combo-tag">${p}</span>`)
              .join(" ");
            html += `<tr>
              <td style="color:#d4a843;font-weight:600;width:60px">${inst.year}</td>
              <td>${inst.name}</td>
              <td><div class="combo-tags">${tags}</div></td>
              <td style="text-align:right;font-weight:600;color:${color};white-space:nowrap">${label}</td>
            </tr>`;
          }
          html += `</tbody></table>`;
          detailPanel.innerHTML = html;
          detailPanel.classList.add("open");
        });

        const labelN = document.createElement("div");
        labelN.className = "bar-label-n";
        labelN.textContent = `n=${d.total}`;

        wrapper.appendChild(labelTop);
        wrapper.appendChild(bar);
        wrapper.appendChild(labelN);
        group.appendChild(wrapper);

        const xLabel = document.createElement("div");
        xLabel.className = "bar-x-label";
        xLabel.textContent = count;
        group.appendChild(xLabel);

        chart.appendChild(group);
      }

      container.appendChild(chart);

      const xTitle = document.createElement("div");
      xTitle.className = "chart-x-title";
      xTitle.textContent = "Number of precursor awards won";
      container.appendChild(xTitle);

      const hint = document.createElement("div");
      hint.className = "click-hint";
      hint.textContent = "Click a bar to see individual instances";
      container.appendChild(hint);

      container.appendChild(detailPanel);

      container.classList.remove("fading");
    }, 150);
  }

  // ── Section 5: Combinations Table ─────────────────────

  function renderCombosTable() {
    const container = document.getElementById("combos-table");
    const cat = DATA.oscar_categories[currentCombosCatIndex];
    const source = combosModeAll ? DATA.combinations_all : DATA.combinations;
    const combos = (source || {})[cat];

    document.getElementById("combos-cat-title").textContent = cat;

    container.classList.add("fading");

    setTimeout(() => {
      container.innerHTML = "";

      if (!combos || combos.length === 0) {
        container.innerHTML = '<div class="combos-no-data">No combination data available for this category.</div>';
        container.classList.remove("fading");
        return;
      }

      const table = document.createElement("table");
      table.className = "combos-table";

      const thead = document.createElement("thead");
      thead.innerHTML = `<tr>
        <th style="width:35%">Precursor Combination</th>
        <th style="width:10%;text-align:center">Times Seen</th>
        <th style="width:55%">Oscar Win Rate</th>
      </tr>`;
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      // Filter by threshold
      const shown = combos.filter(c => c.total >= combosThreshold);

      for (const combo of shown) {
        const tr = document.createElement("tr");
        tr.className = "combo-row";
        tr.style.cursor = "pointer";

        // Combo tags
        const tdCombo = document.createElement("td");
        const tagsDiv = document.createElement("div");
        tagsDiv.className = "combo-tags";
        for (const p of combo.precursors) {
          const tag = document.createElement("span");
          tag.className = "combo-tag";
          tag.textContent = p;
          tagsDiv.appendChild(tag);
        }
        tdCombo.appendChild(tagsDiv);
        tr.appendChild(tdCombo);

        // Times seen
        const tdCount = document.createElement("td");
        tdCount.textContent = combo.total;
        tdCount.style.textAlign = "center";
        tr.appendChild(tdCount);

        // Win rate bar
        const tdRate = document.createElement("td");
        const bWrapper = document.createElement("div");
        bWrapper.className = "combo-bar-wrapper";
        const barBg = document.createElement("div");
        barBg.className = "combo-bar-bg";
        const barFill = document.createElement("div");
        barFill.className = "combo-bar-fill";
        barFill.style.width = combo.win_pct + "%";
        barFill.style.background = accuracyColor(combo.win_pct);
        barBg.appendChild(barFill);
        const barText = document.createElement("span");
        barText.className = "combo-bar-text";
        barText.textContent = `${combo.win_pct}% (${combo.wins}/${combo.total})`;
        bWrapper.appendChild(barBg);
        bWrapper.appendChild(barText);
        tdRate.appendChild(bWrapper);
        tr.appendChild(tdRate);

        tbody.appendChild(tr);

        // Expandable detail row
        const detailTr = document.createElement("tr");
        detailTr.className = "combo-detail-row";
        const detailTd = document.createElement("td");
        detailTd.colSpan = 3;

        const detailTable = document.createElement("table");
        detailTable.className = "combo-detail-table";
        const instances = combo.instances || [];
        for (const inst of instances) {
          const dtr = document.createElement("tr");
          const tdYear = document.createElement("td");
          tdYear.textContent = inst.year;
          const tdName = document.createElement("td");
          tdName.textContent = inst.name;
          const tdResult = document.createElement("td");
          tdResult.textContent = inst.won_oscar ? "Won Oscar" : "Did not win";
          tdResult.style.color = inst.won_oscar ? "#4CAF50" : "#f44336";
          dtr.appendChild(tdYear);
          dtr.appendChild(tdName);
          dtr.appendChild(tdResult);
          detailTable.appendChild(dtr);
        }
        detailTd.appendChild(detailTable);
        detailTr.appendChild(detailTd);
        tbody.appendChild(detailTr);

        // Toggle on click
        tr.addEventListener("click", () => {
          const isOpen = detailTr.classList.contains("open");
          // Close all other open rows in this table
          tbody.querySelectorAll(".combo-detail-row.open").forEach(r => {
            r.classList.remove("open");
            r.previousElementSibling.classList.remove("expanded");
          });
          if (!isOpen) {
            detailTr.classList.add("open");
            tr.classList.add("expanded");
          }
        });
      }

      table.appendChild(tbody);

      container.appendChild(table);

      // Place hint outside the scrollable container
      let hint = container.parentElement.querySelector(".combos-click-hint");
      if (!hint) {
        hint = document.createElement("div");
        hint.className = "click-hint combos-click-hint";
        container.parentElement.appendChild(hint);
      }
      hint.textContent = "Click a row to see individual instances";

      container.classList.remove("fading");
    }, 150);
  }

  // ── Navigation ──────────────────────────────────────────

  function setupNavigation() {
    // Heatmap mode toggle
    document.getElementById("heatmap-mode-accuracy").addEventListener("click", () => {
      setHeatmapMode("accuracy");
    });
    document.getElementById("heatmap-mode-yearly").addEventListener("click", () => {
      setHeatmapMode("yearly");
    });

    // Year nav
    document.getElementById("year-prev").addEventListener("click", () => {
      currentYearIndex =
        (currentYearIndex - 1 + DATA.years.length) % DATA.years.length;
      renderYearlyGrid();
    });
    document.getElementById("year-next").addEventListener("click", () => {
      currentYearIndex = (currentYearIndex + 1) % DATA.years.length;
      renderYearlyGrid();
    });

    // (Category nav now handled by chip selectors)

    // Count mode toggle
    document.getElementById("count-mode-category").addEventListener("click", () => {
      countModeAll = false;
      document.getElementById("count-mode-category").classList.add("active");
      document.getElementById("count-mode-all").classList.remove("active");
      renderCountChart();
    });
    document.getElementById("count-mode-all").addEventListener("click", () => {
      countModeAll = true;
      document.getElementById("count-mode-all").classList.add("active");
      document.getElementById("count-mode-category").classList.remove("active");
      renderCountChart();
    });

    // Count threshold slider
    document.getElementById("count-threshold").addEventListener("input", (e) => {
      countThreshold = Number(e.target.value);
      document.getElementById("count-threshold-display").textContent = countThreshold;
      renderCountChart();
    });

    // Combos threshold slider
    document.getElementById("combos-threshold").addEventListener("input", (e) => {
      combosThreshold = Number(e.target.value);
      document.getElementById("combos-threshold-display").textContent = combosThreshold;
      renderCombosTable();
    });

    // Combos mode toggle
    document.getElementById("combos-mode-category").addEventListener("click", () => {
      combosModeAll = false;
      document.getElementById("combos-mode-category").classList.add("active");
      document.getElementById("combos-mode-all").classList.remove("active");
      renderCombosTable();
    });
    document.getElementById("combos-mode-all").addEventListener("click", () => {
      combosModeAll = true;
      document.getElementById("combos-mode-all").classList.add("active");
      document.getElementById("combos-mode-category").classList.remove("active");
      renderCombosTable();
    });

    // (Combos category nav now handled by chip selectors)

    // Keyboard nav
    document.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        // Determine which section is most visible
        const sections = [
          { el: document.getElementById("accuracy-section"), type: "year" },
          { el: document.getElementById("agreement-section"), type: "cat" },
          { el: document.getElementById("count-section"), type: "countCat" },
          { el: document.getElementById("combos-section"), type: "combosCat" },
        ];

        const viewMid = window.innerHeight / 2;
        let closest = null;
        let closestDist = Infinity;

        for (const s of sections) {
          const rect = s.el.getBoundingClientRect();
          const mid = (rect.top + rect.bottom) / 2;
          const dist = Math.abs(mid - viewMid);
          if (dist < closestDist) {
            closestDist = dist;
            closest = s;
          }
        }

        if (closest) {
          if (closest.type === "year" && heatmapMode === "yearly") {
            if (e.key === "ArrowLeft") {
              currentYearIndex =
                (currentYearIndex - 1 + DATA.years.length) % DATA.years.length;
            } else {
              currentYearIndex =
                (currentYearIndex + 1) % DATA.years.length;
            }
            renderYearlyGrid();
          } else if (closest.type === "cat") {
            if (e.key === "ArrowLeft") {
              currentCatIndex =
                (currentCatIndex - 1 + DATA.oscar_categories.length) %
                DATA.oscar_categories.length;
            } else {
              currentCatIndex =
                (currentCatIndex + 1) % DATA.oscar_categories.length;
            }
            updateCatSelector("agreement-cat-selector", currentCatIndex);
            renderAgreementMatrix();
          } else if (closest.type === "countCat") {
            if (e.key === "ArrowLeft") {
              currentCountCatIndex =
                (currentCountCatIndex - 1 + DATA.oscar_categories.length) %
                DATA.oscar_categories.length;
            } else {
              currentCountCatIndex =
                (currentCountCatIndex + 1) % DATA.oscar_categories.length;
            }
            updateCatSelector("count-cat-selector", currentCountCatIndex);
            renderCountChart();
          } else if (closest.type === "combosCat") {
            if (e.key === "ArrowLeft") {
              currentCombosCatIndex =
                (currentCombosCatIndex - 1 + DATA.oscar_categories.length) %
                DATA.oscar_categories.length;
            } else {
              currentCombosCatIndex =
                (currentCombosCatIndex + 1) % DATA.oscar_categories.length;
            }
            updateCatSelector("combos-cat-selector", currentCombosCatIndex);
            renderCombosTable();
          }
        }
      }
    });
  }

  // ── Init ────────────────────────────────────────────────

  async function init() {
    try {
      const res = await fetch("data.json");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      DATA = await res.json();
    } catch (err) {
      document.querySelector("main").innerHTML =
        `<div class="card" style="text-align:center;padding:60px;color:#f44336;">
          <h2 style="color:#f44336;">Failed to load data</h2>
          <p>${err.message}</p>
          <p style="color:#666;margin-top:12px;">Make sure <code>data.json</code> is in the same directory and served over HTTP.</p>
        </div>`;
      return;
    }

    // Default to most recent year
    currentYearIndex = DATA.years.length - 1;
    currentCatIndex = 0;

    renderPredictabilityChart();
    renderAccuracyHeatmap();
    renderYearlyGrid();

    // Init category chip selectors
    initCatSelector("count-cat-selector", DATA.oscar_categories, currentCountCatIndex, (i) => {
      currentCountCatIndex = i;
      renderCountChart();
    });
    initCatSelector("combos-cat-selector", DATA.oscar_categories, currentCombosCatIndex, (i) => {
      currentCombosCatIndex = i;
      renderCombosTable();
    });
    initCatSelector("agreement-cat-selector", DATA.oscar_categories, currentCatIndex, (i) => {
      currentCatIndex = i;
      renderAgreementMatrix();
    });

    renderCountChart();
    renderCombosTable();
    renderAgreementMatrix();
    initPredAwardSelector();
    renderPredictionsGrid();
    initForecastCatSelector();
    renderForecast();
    setupNavigation();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
