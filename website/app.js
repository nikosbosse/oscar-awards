(function () {
  "use strict";

  let DATA = null;
  let currentYearIndex = 0;
  let currentCatIndex = 0;
  let currentCountCatIndex = 0;
  let currentCombosCatIndex = 0;

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

    document.getElementById("cat-display").textContent = cat;

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
    const catData = (DATA.precursor_count || {})[cat];

    document.getElementById("count-cat-display").textContent = cat;

    container.classList.add("fading");

    setTimeout(() => {
      container.innerHTML = "";

      if (!catData || Object.keys(catData).length === 0) {
        container.innerHTML = '<div class="chart-no-data">No data available for this category.</div>';
        container.classList.remove("fading");
        return;
      }

      const counts = Object.keys(catData).map(Number).sort((a, b) => a - b);
      const maxPct = 100;

      const chart = document.createElement("div");
      chart.className = "bar-chart";

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
        bar.style.cursor = "default";

        bar.addEventListener("mouseenter", (e) => {
          showTooltip(e,
            `<strong>${count} precursor${count > 1 ? "s" : ""} won</strong><br>` +
            `Oscar win rate: ${d.win_pct}%<br>` +
            `${d.wins} wins out of ${d.total} cases`
          );
        });
        bar.addEventListener("mousemove", positionTooltip);
        bar.addEventListener("mouseleave", hideTooltip);

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

      container.classList.remove("fading");
    }, 150);
  }

  // ── Section 5: Combinations Table ─────────────────────

  function renderCombosTable() {
    const container = document.getElementById("combos-table");
    const cat = DATA.oscar_categories[currentCombosCatIndex];
    const combos = (DATA.combinations || {})[cat];

    document.getElementById("combos-cat-display").textContent = cat;

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
      // Show top 15 combos
      const shown = combos.slice(0, 15);

      for (const combo of shown) {
        const tr = document.createElement("tr");

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
        const wrapper = document.createElement("div");
        wrapper.className = "combo-bar-wrapper";
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
        wrapper.appendChild(barBg);
        wrapper.appendChild(barText);
        tdRate.appendChild(wrapper);
        tr.appendChild(tdRate);

        tbody.appendChild(tr);
      }

      table.appendChild(tbody);
      container.appendChild(table);
      container.classList.remove("fading");
    }, 150);
  }

  // ── Navigation ──────────────────────────────────────────

  function setupNavigation() {
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

    // Category nav
    document.getElementById("cat-prev").addEventListener("click", () => {
      currentCatIndex =
        (currentCatIndex - 1 + DATA.oscar_categories.length) %
        DATA.oscar_categories.length;
      renderAgreementMatrix();
    });
    document.getElementById("cat-next").addEventListener("click", () => {
      currentCatIndex =
        (currentCatIndex + 1) % DATA.oscar_categories.length;
      renderAgreementMatrix();
    });

    // Count chart category nav
    document.getElementById("count-cat-prev").addEventListener("click", () => {
      currentCountCatIndex =
        (currentCountCatIndex - 1 + DATA.oscar_categories.length) %
        DATA.oscar_categories.length;
      renderCountChart();
    });
    document.getElementById("count-cat-next").addEventListener("click", () => {
      currentCountCatIndex =
        (currentCountCatIndex + 1) % DATA.oscar_categories.length;
      renderCountChart();
    });

    // Combos category nav
    document.getElementById("combos-cat-prev").addEventListener("click", () => {
      currentCombosCatIndex =
        (currentCombosCatIndex - 1 + DATA.oscar_categories.length) %
        DATA.oscar_categories.length;
      renderCombosTable();
    });
    document.getElementById("combos-cat-next").addEventListener("click", () => {
      currentCombosCatIndex =
        (currentCombosCatIndex + 1) % DATA.oscar_categories.length;
      renderCombosTable();
    });

    // Keyboard nav
    document.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        // Determine which section is most visible
        const sections = [
          { el: document.getElementById("yearly-section"), type: "year" },
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
          if (closest.type === "year") {
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

    renderAccuracyHeatmap();
    renderYearlyGrid();
    renderAgreementMatrix();
    renderCountChart();
    renderCombosTable();
    setupNavigation();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
