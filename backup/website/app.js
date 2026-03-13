/* ==========================================================================
   Oscar Awards Prediction Website — Application Logic
   ========================================================================== */

(function () {
  "use strict";

  // -----------------------------------------------------------------------
  // Plotly layout defaults
  // -----------------------------------------------------------------------
  const COLORS = {
    navy: "#1a1a2e",
    navyLight: "#22223a",
    navyMid: "#2a2a45",
    navySurface: "#30304d",
    gold: "#e0c068",
    goldDim: "#b89a4a",
    goldBright: "#f0d888",
    text: "#e8e6e3",
    textDim: "#9a97a0",
    textMuted: "#6b6875",
    red: "#e06868",
    green: "#68c088",
    blue: "#6888e0",
    purple: "#a068e0",
    orange: "#e09848",
  };

  // Award type color mapping
  const AWARD_TYPE_COLORS = {
    Industry: COLORS.gold,
    Critics: COLORS.blue,
    Festival: COLORS.purple,
    Major: COLORS.green,
    Other: COLORS.textDim,
  };

  // Known award type classifications
  const AWARD_TYPES = {
    "DGA Awards": "Industry",
    DGA: "Industry",
    "PGA Awards": "Industry",
    PGA: "Industry",
    "SAG Awards": "Industry",
    SAG: "Industry",
    "ACE Eddie Awards": "Industry",
    "ACE Eddie": "Industry",
    "ASC Awards": "Industry",
    ASC: "Industry",
    "WGA Awards": "Industry",
    WGA: "Industry",
    "CDG Awards": "Industry",
    CDG: "Industry",
    "CAS Awards": "Industry",
    CAS: "Industry",
    "ADG Awards": "Industry",
    ADG: "Industry",
    "MUAHS Awards": "Industry",
    MUAHS: "Industry",
    BAFTA: "Major",
    "Golden Globes": "Major",
    "Critics Choice Awards": "Major",
    "Critics Choice": "Major",
    "National Board of Review": "Critics",
    NBR: "Critics",
    "New York Film Critics Circle": "Critics",
    NYFCC: "Critics",
    "Los Angeles Film Critics Association": "Critics",
    LAFCA: "Critics",
    "National Society of Film Critics": "Critics",
    NSFC: "Critics",
    "Cannes Palme d'Or": "Festival",
    Cannes: "Festival",
    "Venice Golden Lion": "Festival",
    Venice: "Festival",
    "Independent Spirit Awards": "Other",
    "Gotham Awards": "Other",
    "Toronto Audience Award": "Festival",
    Toronto: "Festival",
  };

  function getAwardType(name) {
    return AWARD_TYPES[name] || "Other";
  }

  function getAwardColor(name) {
    return AWARD_TYPE_COLORS[getAwardType(name)] || COLORS.textDim;
  }

  const PLOTLY_LAYOUT_BASE = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Source Sans 3, sans-serif", color: COLORS.text, size: 13 },
    margin: { l: 60, r: 30, t: 40, b: 50 },
    hoverlabel: {
      bgcolor: COLORS.navySurface,
      bordercolor: COLORS.gold,
      font: { color: COLORS.text, family: "Source Sans 3, sans-serif", size: 13 },
    },
    xaxis: {
      gridcolor: "rgba(224,192,104,0.06)",
      zerolinecolor: "rgba(224,192,104,0.1)",
      tickfont: { size: 11, color: COLORS.textDim },
    },
    yaxis: {
      gridcolor: "rgba(224,192,104,0.06)",
      zerolinecolor: "rgba(224,192,104,0.1)",
      tickfont: { size: 11, color: COLORS.textDim },
    },
  };

  const PLOTLY_CONFIG = {
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
    responsive: true,
  };

  function mergeLayout(overrides) {
    const base = JSON.parse(JSON.stringify(PLOTLY_LAYOUT_BASE));
    return deepMerge(base, overrides);
  }

  function deepMerge(target, source) {
    for (const key of Object.keys(source)) {
      if (
        source[key] &&
        typeof source[key] === "object" &&
        !Array.isArray(source[key]) &&
        target[key] &&
        typeof target[key] === "object"
      ) {
        deepMerge(target[key], source[key]);
      } else {
        target[key] = source[key];
      }
    }
    return target;
  }

  // -----------------------------------------------------------------------
  // Category grouping
  // -----------------------------------------------------------------------
  const CATEGORY_GROUPS = {
    acting: [
      "Best Actor",
      "Best Actress",
      "Best Supporting Actor",
      "Best Supporting Actress",
    ],
    directing: ["Best Picture", "Best Director"],
    writing: ["Best Adapted Screenplay", "Best Original Screenplay"],
    technical: [
      "Best Cinematography",
      "Best Film Editing",
      "Best Original Score",
      "Best Original Song",
      "Best Costume Design",
      "Best Makeup and Hairstyling",
      "Best Production Design",
      "Best Sound",
      "Best Visual Effects",
      "Best Casting",
    ],
    other: [
      "Best Animated Feature Film",
      "Best International Feature Film",
      "Best Documentary Feature Film",
      "Best Animated Short Film",
      "Best Live Action Short Film",
      "Best Documentary Short Film",
    ],
  };

  function categoriesForGroup(group) {
    if (group === "all") return null; // no filter
    return CATEGORY_GROUPS[group] || null;
  }

  // -----------------------------------------------------------------------
  // Data loading
  // -----------------------------------------------------------------------
  let DATA = null;

  async function loadData() {
    try {
      const resp = await fetch("../website_data.json");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      DATA = await resp.json();
      return true;
    } catch (err) {
      console.error("Failed to load website_data.json:", err);
      // Try alternate path
      try {
        const resp2 = await fetch("website_data.json");
        if (!resp2.ok) throw new Error(`HTTP ${resp2.status}`);
        DATA = await resp2.json();
        return true;
      } catch {
        console.error("Data file not found at either path.");
        return false;
      }
    }
  }

  // -----------------------------------------------------------------------
  // Initialization
  // -----------------------------------------------------------------------
  async function init() {
    const loaded = await loadData();
    const overlay = document.getElementById("loading-overlay");

    if (!loaded) {
      overlay.querySelector(".loader-text").textContent =
        "Could not load data. Make sure website_data.json exists in the repository root.";
      overlay.querySelector(".loader-bar").style.display = "none";
      return;
    }

    // Hide loader
    overlay.classList.add("hidden");

    // Setup nav
    setupNav();
    setupScrollObserver();
    animateHeroCounters();

    // Render all sections
    renderAccuracyRankings();
    renderHeatmap();
    setupOverTime();
    setupAgreement();
    renderUpsets();
    renderPredictability();
    renderConsensus();
    renderModels();
    renderPredictions2026();
  }

  // -----------------------------------------------------------------------
  // Navigation
  // -----------------------------------------------------------------------
  function setupNav() {
    const toggle = document.getElementById("nav-toggle");
    const sidebar = document.getElementById("sidebar");

    toggle.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });

    // Close sidebar when clicking a link on mobile
    document.querySelectorAll(".nav-links a").forEach((link) => {
      link.addEventListener("click", () => {
        sidebar.classList.remove("open");
      });
    });
  }

  function setupScrollObserver() {
    const sections = document.querySelectorAll(".section");
    const navLinks = document.querySelectorAll(".nav-links a");

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          // Fade in sections
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
          }

          // Update active nav link
          if (entry.isIntersecting && entry.intersectionRatio > 0.15) {
            const id = entry.target.id;
            navLinks.forEach((link) => {
              link.classList.toggle("active", link.getAttribute("data-section") === id);
            });
          }
        });
      },
      { threshold: [0.05, 0.15, 0.3], rootMargin: "-10% 0px -10% 0px" }
    );

    sections.forEach((s) => observer.observe(s));
  }

  // -----------------------------------------------------------------------
  // Hero animated counters
  // -----------------------------------------------------------------------
  function animateHeroCounters() {
    const counters = document.querySelectorAll(".stat-number");
    const md = DATA.metadata || {};

    // Update targets from actual data
    counters.forEach((el) => {
      const label = el.nextElementSibling?.textContent?.toLowerCase() || "";
      if (label.includes("award shows") && md.award_shows) {
        el.dataset.target = Array.isArray(md.award_shows)
          ? md.award_shows.length
          : md.award_shows;
      } else if (label.includes("years") && md.year_range) {
        const [start, end] = Array.isArray(md.year_range)
          ? md.year_range
          : [2000, 2025];
        el.dataset.target = end - start;
      } else if (label.includes("records") && md.total_records) {
        el.dataset.target = md.total_records;
      } else if (label.includes("categories") && md.oscar_categories) {
        el.dataset.target = Array.isArray(md.oscar_categories)
          ? md.oscar_categories.length
          : md.oscar_categories;
      }
    });

    counters.forEach((el) => {
      const target = parseInt(el.dataset.target) || 0;
      const suffix = el.dataset.suffix || "";
      const useComma = el.dataset.comma === "true";
      const duration = 1500;
      const start = performance.now();

      function tick(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(eased * target);
        el.textContent =
          (useComma ? current.toLocaleString() : current) + suffix;
        if (progress < 1) requestAnimationFrame(tick);
      }

      requestAnimationFrame(tick);
    });
  }

  // -----------------------------------------------------------------------
  // Section 2: Overall Accuracy Rankings
  // -----------------------------------------------------------------------
  function renderAccuracyRankings() {
    const container = document.getElementById("chart-accuracy-bar");
    const data = DATA.overall_accuracy;
    if (!data) {
      container.innerHTML = "<p style='color:var(--text-dim)'>No accuracy data available.</p>";
      return;
    }

    // Sort by accuracy descending
    const entries = Object.entries(data)
      .map(([name, d]) => ({
        name,
        accuracy: d.accuracy || 0,
        matches: d.matches || 0,
        total: d.total || 0,
      }))
      .sort((a, b) => a.accuracy - b.accuracy); // ascending for horizontal bar

    const trace = {
      type: "bar",
      orientation: "h",
      y: entries.map((e) => e.name),
      x: entries.map((e) => e.accuracy * 100),
      marker: {
        color: entries.map((e) => getAwardColor(e.name)),
        line: { width: 0 },
      },
      hovertemplate: entries.map(
        (e) =>
          `<b>${e.name}</b><br>` +
          `Accuracy: ${(e.accuracy * 100).toFixed(1)}%<br>` +
          `${e.matches} / ${e.total} correct<br>` +
          `Type: ${getAwardType(e.name)}` +
          `<extra></extra>`
      ),
      text: entries.map((e) => (e.accuracy * 100).toFixed(1) + "%"),
      textposition: "outside",
      textfont: { size: 11, color: COLORS.textDim },
    };

    const layout = mergeLayout({
      title: { text: "Overall Oscar Prediction Accuracy", font: { size: 16 } },
      xaxis: {
        title: "Accuracy (%)",
        range: [0, Math.min(105, Math.max(...entries.map((e) => e.accuracy * 100)) + 10)],
      },
      yaxis: { automargin: true, tickfont: { size: 12 } },
      margin: { l: 180, r: 60, t: 50, b: 50 },
      height: Math.max(400, entries.length * 32 + 80),
    });

    Plotly.newPlot(container, [trace], layout, PLOTLY_CONFIG);

    // Add legend for award types
    const legend = document.createElement("div");
    legend.style.cssText =
      "display:flex;gap:1.2rem;flex-wrap:wrap;margin-top:0.75rem;padding-left:0.5rem;";
    const types = ["Industry", "Major", "Critics", "Festival"];
    types.forEach((t) => {
      const item = document.createElement("span");
      item.style.cssText = `font-size:0.8rem;color:${COLORS.textDim};display:flex;align-items:center;gap:0.35rem;`;
      item.innerHTML = `<span style="width:10px;height:10px;border-radius:50%;background:${AWARD_TYPE_COLORS[t]};display:inline-block;"></span>${t}`;
      legend.appendChild(item);
    });
    container.parentElement.appendChild(legend);
  }

  // -----------------------------------------------------------------------
  // Section 3: Accuracy Heatmap
  // -----------------------------------------------------------------------
  function renderHeatmap(filterGroup) {
    const container = document.getElementById("chart-heatmap");
    const data = DATA.accuracy_by_category;
    if (!data) {
      container.innerHTML = "<p style='color:var(--text-dim)'>No heatmap data available.</p>";
      return;
    }

    const filterCats = filterGroup ? categoriesForGroup(filterGroup) : null;
    let categories = Object.keys(data);
    if (filterCats) {
      categories = categories.filter((c) => filterCats.includes(c));
    }
    categories.sort();

    // Collect all awards across categories
    const allAwards = new Set();
    categories.forEach((cat) => {
      Object.keys(data[cat]).forEach((a) => allAwards.add(a));
    });
    const awards = [...allAwards].sort();

    // Build z-matrix
    const z = categories.map((cat) =>
      awards.map((a) => {
        const entry = data[cat][a];
        if (entry == null) return null;
        // entry may be a plain number or an object with .accuracy
        return typeof entry === "object" ? (entry.accuracy ?? null) : entry;
      })
    );

    // Custom text for hover
    const hovertext = categories.map((cat) =>
      awards.map((a) => {
        const entry = data[cat][a];
        if (entry == null) return `${a}<br>${cat}<br>No data`;
        const val = typeof entry === "object" ? entry.accuracy : entry;
        if (val == null) return `${a}<br>${cat}<br>No data`;
        const extra = typeof entry === "object" && entry.matches != null
          ? `<br>${entry.matches}/${entry.total} correct`
          : "";
        return `${a}<br>${cat}<br>Accuracy: ${(val * 100).toFixed(1)}%${extra}`;
      })
    );

    const trace = {
      type: "heatmap",
      z,
      x: awards,
      y: categories,
      hovertext,
      hoverinfo: "text",
      colorscale: [
        [0, COLORS.navy],
        [0.3, "#3a2a4e"],
        [0.5, "#6a4488"],
        [0.7, COLORS.goldDim],
        [1, COLORS.goldBright],
      ],
      zmin: 0,
      zmax: 1,
      colorbar: {
        title: { text: "Accuracy", font: { size: 12, color: COLORS.textDim } },
        tickformat: ".0%",
        tickfont: { color: COLORS.textDim, size: 10 },
        len: 0.6,
      },
      xgap: 2,
      ygap: 2,
    };

    const layout = mergeLayout({
      title: {
        text: "Prediction Accuracy: Precursors × Oscar Categories",
        font: { size: 16 },
      },
      xaxis: {
        tickangle: -45,
        automargin: true,
        side: "bottom",
        tickfont: { size: 10 },
      },
      yaxis: {
        automargin: true,
        tickfont: { size: 10 },
        autorange: "reversed",
      },
      margin: { l: 200, r: 80, t: 50, b: 120 },
      height: Math.max(500, categories.length * 28 + 200),
    });

    Plotly.newPlot(container, [trace], layout, PLOTLY_CONFIG);

    // Click handler
    container.on("plotly_click", (eventData) => {
      if (!eventData.points || !eventData.points[0]) return;
      const pt = eventData.points[0];
      const award = pt.x;
      const category = pt.y;
      const accuracy = pt.z;
      showHeatmapDetail(award, category, accuracy);
    });

    // Filter dropdown
    const filter = document.getElementById("heatmap-filter");
    filter.onchange = () => renderHeatmap(filter.value);
  }

  function showHeatmapDetail(award, category, accuracy) {
    const panel = document.getElementById("heatmap-detail");
    const content = document.getElementById("heatmap-detail-content");
    panel.classList.remove("hidden");

    // Get year-by-year data if available
    const timeData = DATA.accuracy_over_time?.[category]?.[award];
    let yearHtml = "";
    if (timeData) {
      const years = Object.keys(timeData).sort();
      const correct = years.filter((y) => timeData[y] === 1);
      const wrong = years.filter((y) => timeData[y] === 0);
      yearHtml = `
        <p style="margin-top:0.75rem;"><strong style="color:${COLORS.green};">Correct years (${correct.length}):</strong>
          <span style="color:${COLORS.textDim};font-size:0.85rem;">${correct.join(", ") || "—"}</span></p>
        <p><strong style="color:${COLORS.red};">Incorrect years (${wrong.length}):</strong>
          <span style="color:${COLORS.textDim};font-size:0.85rem;">${wrong.join(", ") || "—"}</span></p>
      `;
    }

    content.innerHTML = `
      <h3 style="font-family:var(--font-display);color:var(--gold);margin-bottom:0.5rem;">${award} → ${category}</h3>
      <p style="color:var(--text);font-size:1.1rem;">
        Accuracy: <strong style="color:${COLORS.gold};">${accuracy != null ? (accuracy * 100).toFixed(1) + "%" : "N/A"}</strong>
      </p>
      ${yearHtml}
    `;
  }

  // -----------------------------------------------------------------------
  // Section 4: Accuracy Over Time
  // -----------------------------------------------------------------------
  function setupOverTime() {
    const select = document.getElementById("time-category");
    const data = DATA.accuracy_over_time;
    if (!data) return;

    const categories = Object.keys(data).sort();
    categories.forEach((cat) => {
      const opt = document.createElement("option");
      opt.value = cat;
      opt.textContent = cat;
      select.appendChild(opt);
    });

    select.addEventListener("change", () => renderOverTime(select.value));
    if (categories.length > 0) {
      // Default to Best Picture or first
      const defaultCat = categories.includes("Best Picture") ? "Best Picture" : categories[0];
      select.value = defaultCat;
      renderOverTime(defaultCat);
    }
  }

  function renderOverTime(category) {
    const container = document.getElementById("chart-over-time");
    const catData = DATA.accuracy_over_time?.[category];
    if (!catData) {
      container.innerHTML = "<p style='color:var(--text-dim)'>No time data for this category.</p>";
      return;
    }

    const WINDOW = 5;
    const traceColors = [
      COLORS.gold, COLORS.blue, COLORS.green, COLORS.red,
      COLORS.purple, COLORS.orange, COLORS.goldDim, COLORS.textDim,
      "#e0a868", "#68c0e0", "#c068a0", "#88c068",
      "#c0a068", "#6880c0", "#c08868", "#a0c068",
    ];

    const traces = [];
    const awards = Object.keys(catData).sort();

    awards.forEach((award, i) => {
      const yearData = catData[award];
      const years = Object.keys(yearData)
        .map(Number)
        .sort((a, b) => a - b);

      // Compute rolling accuracy
      const rollingYears = [];
      const rollingAcc = [];

      for (let j = WINDOW - 1; j < years.length; j++) {
        const windowYears = years.slice(j - WINDOW + 1, j + 1);
        const windowSum = windowYears.reduce(
          (sum, y) => sum + (yearData[y] || 0),
          0
        );
        rollingYears.push(years[j]);
        rollingAcc.push((windowSum / WINDOW) * 100);
      }

      traces.push({
        type: "scatter",
        mode: "lines",
        name: award,
        x: rollingYears,
        y: rollingAcc,
        line: {
          color: traceColors[i % traceColors.length],
          width: 2,
        },
        hovertemplate: `<b>${award}</b><br>Year: %{x}<br>5-yr accuracy: %{y:.1f}%<extra></extra>`,
      });
    });

    const layout = mergeLayout({
      title: {
        text: `Rolling 5-Year Accuracy: ${category}`,
        font: { size: 16 },
      },
      xaxis: { title: "Year", dtick: 2 },
      yaxis: { title: "Accuracy (%)", range: [0, 105] },
      legend: {
        font: { size: 10, color: COLORS.textDim },
        bgcolor: "rgba(0,0,0,0)",
        orientation: "h",
        y: -0.25,
        x: 0.5,
        xanchor: "center",
      },
      margin: { l: 60, r: 30, t: 50, b: 80 },
      height: 500,
    });

    Plotly.newPlot(container, traces, layout, PLOTLY_CONFIG);
  }

  // -----------------------------------------------------------------------
  // Section 5: Award Agreement Matrix
  // -----------------------------------------------------------------------
  function setupAgreement() {
    const select = document.getElementById("agreement-category");
    const data = DATA.award_agreement;
    if (!data) return;

    const categories = Object.keys(data).sort();
    categories.forEach((cat) => {
      const opt = document.createElement("option");
      opt.value = cat;
      opt.textContent = cat;
      select.appendChild(opt);
    });

    select.addEventListener("change", () => renderAgreement(select.value));
    if (categories.length > 0) {
      const defaultCat = categories.includes("Best Picture") ? "Best Picture" : categories[0];
      select.value = defaultCat;
      renderAgreement(defaultCat);
    }
  }

  function renderAgreement(category) {
    const container = document.getElementById("chart-agreement");
    const catData = DATA.award_agreement?.[category];
    if (!catData) {
      container.innerHTML = "<p style='color:var(--text-dim)'>No agreement data for this category.</p>";
      return;
    }

    // Extract unique awards from pairs
    // Pair keys may use "-", " vs ", or " × " as separators
    const PAIR_SEPARATORS = [" × ", " vs ", "-"];
    function splitPair(pair) {
      for (const sep of PAIR_SEPARATORS) {
        const idx = pair.indexOf(sep);
        if (idx !== -1) {
          return [pair.slice(0, idx), pair.slice(idx + sep.length)];
        }
      }
      return null;
    }

    const awards = new Set();
    Object.keys(catData).forEach((pair) => {
      const parts = splitPair(pair);
      if (parts) {
        awards.add(parts[0]);
        awards.add(parts[1]);
      }
    });

    const awardList = [...awards].sort();
    if (awardList.length === 0) {
      container.innerHTML = "<p style='color:var(--text-dim)'>No agreement pairs found.</p>";
      return;
    }

    // Build matrix
    const n = awardList.length;
    const z = Array.from({ length: n }, () => Array(n).fill(null));
    const indexMap = {};
    awardList.forEach((a, i) => (indexMap[a] = i));

    Object.entries(catData).forEach(([pair, val]) => {
      const parts = splitPair(pair);
      if (!parts) return;
      const [a, b] = parts;

      const i = indexMap[a];
      const j = indexMap[b];
      if (i !== undefined && j !== undefined) {
        z[i][j] = val;
        z[j][i] = val;
      }
    });

    // Diagonal = 1
    for (let i = 0; i < n; i++) z[i][i] = 1;

    const trace = {
      type: "heatmap",
      z,
      x: awardList,
      y: awardList,
      colorscale: [
        [0, COLORS.navy],
        [0.5, "#5a3a8a"],
        [0.75, COLORS.goldDim],
        [1, COLORS.goldBright],
      ],
      zmin: 0,
      zmax: 1,
      hovertemplate: "%{y} × %{x}<br>Agreement: %{z:.1%}<extra></extra>",
      colorbar: {
        title: { text: "Agreement", font: { size: 12, color: COLORS.textDim } },
        tickformat: ".0%",
        tickfont: { color: COLORS.textDim, size: 10 },
      },
      xgap: 2,
      ygap: 2,
    };

    const layout = mergeLayout({
      title: {
        text: `Award Agreement Matrix: ${category}`,
        font: { size: 16 },
      },
      xaxis: { tickangle: -45, automargin: true, tickfont: { size: 10 } },
      yaxis: { automargin: true, autorange: "reversed", tickfont: { size: 10 } },
      margin: { l: 160, r: 60, t: 50, b: 120 },
      height: Math.max(500, n * 30 + 200),
    });

    Plotly.newPlot(container, [trace], layout, PLOTLY_CONFIG);
  }

  // -----------------------------------------------------------------------
  // Section 6: Upset Gallery
  // -----------------------------------------------------------------------
  function renderUpsets() {
    const container = document.getElementById("upset-cards");
    const upsets = DATA.upsets;
    if (!upsets || upsets.length === 0) {
      container.innerHTML = "<p style='color:var(--text-dim)'>No upset data available.</p>";
      return;
    }

    const sortSelect = document.getElementById("upset-sort");
    const countSelect = document.getElementById("upset-count");

    function render() {
      let sorted = [...upsets];
      const sortBy = sortSelect.value;

      if (sortBy === "surprise") {
        sorted.sort((a, b) => (b.consensus_pct || 0) - (a.consensus_pct || 0));
      } else if (sortBy === "year-desc") {
        sorted.sort((a, b) => b.year - a.year);
      } else {
        sorted.sort((a, b) => a.year - b.year);
      }

      const count = countSelect.value;
      if (count !== "all") {
        sorted = sorted.slice(0, parseInt(count));
      }

      container.innerHTML = sorted
        .map((u) => {
          const surprise = ((u.consensus_pct || 0) * 100).toFixed(0);
          const barColor =
            surprise > 80
              ? COLORS.red
              : surprise > 50
              ? COLORS.orange
              : COLORS.goldDim;

          let picksHtml = "";
          if (u.precursor_picks && typeof u.precursor_picks === "object") {
            const picks = Object.entries(u.precursor_picks);
            if (picks.length > 0) {
              picksHtml = picks
                .slice(0, 5)
                .map(
                  ([award, pick]) =>
                    `<strong>${award}:</strong> ${pick}`
                )
                .join("<br>");
              if (picks.length > 5) {
                picksHtml += `<br><em>...and ${picks.length - 5} more</em>`;
              }
            }
          } else if (typeof u.precursor_picks === "string") {
            picksHtml = u.precursor_picks;
          }

          return `
            <div class="upset-card">
              <div class="upset-card-header">
                <span class="upset-year">${u.year}</span>
                <span class="upset-category">${u.category}</span>
              </div>
              <div class="upset-winner">${u.winner}</div>
              <div class="surprise-meter">
                <span style="font-size:0.75rem;color:${COLORS.textMuted};">Surprise</span>
                <div class="surprise-bar-bg">
                  <div class="surprise-bar-fill" style="width:${surprise}%;background:${barColor};"></div>
                </div>
                <span class="surprise-label" style="color:${barColor};">${surprise}%</span>
              </div>
              ${picksHtml ? `<div class="upset-picks">${picksHtml}</div>` : ""}
            </div>
          `;
        })
        .join("");
    }

    sortSelect.addEventListener("change", render);
    countSelect.addEventListener("change", render);
    render();
  }

  // -----------------------------------------------------------------------
  // Section 7: Category Predictability
  // -----------------------------------------------------------------------
  function renderPredictability() {
    const container = document.getElementById("chart-predictability");
    const data = DATA.category_predictability;
    if (!data) {
      container.innerHTML = "<p style='color:var(--text-dim)'>No predictability data available.</p>";
      return;
    }

    const entries = Object.entries(data)
      .map(([cat, d]) => ({
        category: cat,
        bestPrecursor: d.best_precursor || "—",
        bestAccuracy: d.best_accuracy || 0,
        avgAccuracy: d.avg_accuracy || 0,
      }))
      .sort((a, b) => b.bestAccuracy - a.bestAccuracy);

    const traceBest = {
      type: "scatter",
      mode: "markers",
      name: "Best Precursor",
      y: entries.map((e) => e.category),
      x: entries.map((e) => e.bestAccuracy * 100),
      marker: {
        color: COLORS.gold,
        size: 12,
        line: { width: 2, color: COLORS.goldBright },
      },
      hovertemplate: entries.map(
        (e) =>
          `<b>${e.category}</b><br>Best: ${e.bestPrecursor}<br>Accuracy: ${(
            e.bestAccuracy * 100
          ).toFixed(1)}%<extra></extra>`
      ),
    };

    const traceAvg = {
      type: "scatter",
      mode: "markers",
      name: "Average Across Precursors",
      y: entries.map((e) => e.category),
      x: entries.map((e) => e.avgAccuracy * 100),
      marker: {
        color: COLORS.textMuted,
        size: 8,
        symbol: "diamond",
      },
      hovertemplate: entries.map(
        (e) =>
          `<b>${e.category}</b><br>Avg accuracy: ${(e.avgAccuracy * 100).toFixed(
            1
          )}%<extra></extra>`
      ),
    };

    // Lollipop lines connecting avg to best
    const shapes = entries.map((e, i) => ({
      type: "line",
      x0: e.avgAccuracy * 100,
      x1: e.bestAccuracy * 100,
      y0: i,
      y1: i,
      yref: "y",
      line: { color: "rgba(224,192,104,0.25)", width: 2 },
    }));

    const layout = mergeLayout({
      title: {
        text: "Category Predictability (Best vs Average Precursor Accuracy)",
        font: { size: 16 },
      },
      xaxis: { title: "Accuracy (%)", range: [0, 105] },
      yaxis: {
        automargin: true,
        tickfont: { size: 11 },
        categoryorder: "array",
        categoryarray: entries.map((e) => e.category).reverse(),
      },
      shapes,
      legend: {
        font: { size: 11, color: COLORS.textDim },
        bgcolor: "rgba(0,0,0,0)",
        x: 0.95,
        xanchor: "right",
        y: 0.02,
      },
      margin: { l: 200, r: 40, t: 50, b: 50 },
      height: Math.max(500, entries.length * 28 + 100),
    });

    Plotly.newPlot(container, [traceAvg, traceBest], layout, PLOTLY_CONFIG);
  }

  // -----------------------------------------------------------------------
  // Section 8: Consensus Meter
  // -----------------------------------------------------------------------
  function renderConsensus() {
    const container = document.getElementById("chart-consensus");
    const data = DATA.consensus_vs_outcome;
    if (!data || data.length === 0) {
      container.innerHTML = "<p style='color:var(--text-dim)'>No consensus data available.</p>";
      return;
    }

    const sorted = [...data].sort((a, b) => {
      const aVal = typeof a.consensus_bin === "string"
        ? parseFloat(a.consensus_bin) || 0
        : (a.consensus_bin || 0);
      const bVal = typeof b.consensus_bin === "string"
        ? parseFloat(b.consensus_bin) || 0
        : (b.consensus_bin || 0);
      return aVal - bVal;
    });

    const trace = {
      type: "scatter",
      mode: "markers+lines",
      x: sorted.map((d) =>
        typeof d.consensus_bin === "string"
          ? d.consensus_bin
          : (d.consensus_bin * 100).toFixed(0) + "%"
      ),
      y: sorted.map((d) => (d.win_rate || 0) * 100),
      marker: {
        size: sorted.map((d) => Math.max(8, Math.min(40, Math.sqrt(d.count || 1) * 4))),
        color: COLORS.gold,
        line: { width: 2, color: COLORS.goldBright },
        opacity: 0.85,
      },
      line: { color: COLORS.goldDim, width: 2, dash: "dot" },
      hovertemplate: sorted.map(
        (d) =>
          `Consensus: ${typeof d.consensus_bin === "string" ? d.consensus_bin : (d.consensus_bin * 100).toFixed(0) + "%"}<br>` +
          `Win rate: ${((d.win_rate || 0) * 100).toFixed(1)}%<br>` +
          `N = ${d.count}<extra></extra>`
      ),
    };

    // Add reference line (y=x)
    const refTrace = {
      type: "scatter",
      mode: "lines",
      x: sorted.map((d) =>
        typeof d.consensus_bin === "string"
          ? d.consensus_bin
          : (d.consensus_bin * 100).toFixed(0) + "%"
      ),
      y: sorted.map((d) => {
        if (typeof d.consensus_bin === "number") return d.consensus_bin * 100;
        // Extract midpoint from range strings like "0-30%", "30-50%", etc.
        const match = d.consensus_bin.match(/(\d+)[^\d]+(\d+)/);
        if (match) return (parseInt(match[1]) + parseInt(match[2])) / 2;
        const single = parseFloat(d.consensus_bin);
        return isNaN(single) ? 50 : single;
      }),
      line: { color: "rgba(224,192,104,0.15)", width: 1, dash: "dash" },
      showlegend: false,
      hoverinfo: "skip",
    };

    const layout = mergeLayout({
      title: {
        text: "Precursor Consensus vs Oscar Win Probability",
        font: { size: 16 },
      },
      xaxis: { title: "Precursor Consensus" },
      yaxis: { title: "Oscar Win Rate (%)", range: [0, 105] },
      showlegend: false,
      height: 450,
      annotations: [
        {
          text: "Bubble size = number of cases",
          xref: "paper",
          yref: "paper",
          x: 0.98,
          y: 0.02,
          showarrow: false,
          font: { size: 10, color: COLORS.textMuted },
          xanchor: "right",
        },
      ],
    });

    Plotly.newPlot(container, [refTrace, trace], layout, PLOTLY_CONFIG);
  }

  // -----------------------------------------------------------------------
  // Section 9: Model Comparison Dashboard
  // -----------------------------------------------------------------------
  function renderModels() {
    const bt = DATA.backtest_results;
    if (!bt || !bt.by_model) {
      document.getElementById("chart-model-accuracy").innerHTML =
        "<p style='color:var(--text-dim)'>No backtest data available.</p>";
      return;
    }

    renderModelAccuracyChart(bt);
    renderModelTable(bt);
    renderCalibrationChart(bt);
  }

  function renderModelAccuracyChart(bt) {
    const container = document.getElementById("chart-model-accuracy");
    const models = bt.models || Object.keys(bt.by_model);

    const entries = models
      .map((m) => ({
        name: m,
        accuracy: bt.by_model[m]?.accuracy || 0,
      }))
      .sort((a, b) => b.accuracy - a.accuracy);

    const trace = {
      type: "bar",
      x: entries.map((e) => e.name),
      y: entries.map((e) => e.accuracy * 100),
      marker: {
        color: entries.map((_, i) =>
          i === 0
            ? COLORS.gold
            : i < 3
            ? COLORS.goldDim
            : COLORS.navySurface
        ),
        line: { width: 1, color: "rgba(224,192,104,0.3)" },
      },
      text: entries.map((e) => (e.accuracy * 100).toFixed(1) + "%"),
      textposition: "outside",
      textfont: { size: 11, color: COLORS.textDim },
      hovertemplate: "<b>%{x}</b><br>Accuracy: %{y:.1f}%<extra></extra>",
    };

    const layout = mergeLayout({
      title: { text: "Model Backtest Accuracy (Leave-One-Year-Out CV)", font: { size: 16 } },
      xaxis: { tickangle: -30, automargin: true, tickfont: { size: 11 } },
      yaxis: {
        title: "Accuracy (%)",
        range: [0, Math.max(...entries.map((e) => e.accuracy * 100)) + 10],
      },
      height: 400,
      margin: { b: 100 },
    });

    Plotly.newPlot(container, [trace], layout, PLOTLY_CONFIG);
  }

  function renderModelTable(bt) {
    const wrapper = document.getElementById("model-table-wrapper");
    const models = bt.models || Object.keys(bt.by_model);
    const calibration = bt.calibration || {};

    let rows = models.map((m) => {
      const d = bt.by_model[m] || {};
      const cal = calibration[m] || {};
      return {
        name: m,
        accuracy: d.accuracy || 0,
        ece: cal.ece,
        brier: cal.brier,
      };
    });

    rows.sort((a, b) => b.accuracy - a.accuracy);

    let html = `<table class="data-table" id="model-data-table">
      <thead><tr>
        <th data-col="name">Model</th>
        <th data-col="accuracy" class="sorted-desc">Accuracy</th>
        <th data-col="ece">ECE</th>
        <th data-col="brier">Brier Score</th>
      </tr></thead><tbody>`;

    rows.forEach((r, i) => {
      html += `<tr>
        <td>${i === 0 ? "&#127942; " : ""}${r.name}</td>
        <td class="num ${i === 0 ? "highlight-cell" : ""}">${(r.accuracy * 100).toFixed(1)}%</td>
        <td class="num">${r.ece != null ? r.ece.toFixed(4) : "—"}</td>
        <td class="num">${r.brier != null ? r.brier.toFixed(4) : "—"}</td>
      </tr>`;
    });

    html += `</tbody></table>`;
    wrapper.innerHTML = html;

    // Sortable columns
    const table = document.getElementById("model-data-table");
    const headers = table.querySelectorAll("th");
    headers.forEach((th) => {
      th.addEventListener("click", () => {
        const col = th.dataset.col;
        const isDesc = th.classList.contains("sorted-desc");
        headers.forEach((h) =>
          h.classList.remove("sorted-asc", "sorted-desc")
        );
        th.classList.add(isDesc ? "sorted-asc" : "sorted-desc");

        const dir = isDesc ? 1 : -1;
        rows.sort((a, b) => {
          const av = a[col] ?? -Infinity;
          const bv = b[col] ?? -Infinity;
          if (typeof av === "string") return dir * av.localeCompare(bv);
          return dir * (av - bv);
        });

        const tbody = table.querySelector("tbody");
        tbody.innerHTML = rows
          .map(
            (r, i) => `<tr>
            <td>${i === 0 && !isDesc && col === "accuracy" ? "&#127942; " : ""}${r.name}</td>
            <td class="num">${(r.accuracy * 100).toFixed(1)}%</td>
            <td class="num">${r.ece != null ? r.ece.toFixed(4) : "—"}</td>
            <td class="num">${r.brier != null ? r.brier.toFixed(4) : "—"}</td>
          </tr>`
          )
          .join("");
      });
    });
  }

  function renderCalibrationChart(bt) {
    const container = document.getElementById("chart-calibration");
    const calibration = bt.calibration;
    if (!calibration) {
      container.innerHTML = "";
      return;
    }

    const traces = [];
    const colors = [
      COLORS.gold, COLORS.blue, COLORS.green, COLORS.red,
      COLORS.purple, COLORS.orange, COLORS.goldDim, COLORS.textDim,
    ];

    let modelIndex = 0;
    for (const [model, cal] of Object.entries(calibration)) {
      if (!cal.bins || cal.bins.length === 0) continue;

      const bins = cal.bins.sort(
        (a, b) => (a.mean_confidence || a.predicted || a.bin || 0) - (b.mean_confidence || b.predicted || b.bin || 0)
      );

      traces.push({
        type: "scatter",
        mode: "markers+lines",
        name: model,
        x: bins.map((b) => (b.mean_confidence || b.predicted || b.bin || 0) * 100),
        y: bins.map((b) => (b.actual_accuracy || b.actual || b.observed || 0) * 100),
        marker: {
          color: colors[modelIndex % colors.length],
          size: bins.map((b) =>
            Math.max(5, Math.min(20, Math.sqrt(b.count || 1) * 3))
          ),
        },
        line: { color: colors[modelIndex % colors.length], width: 1.5 },
        hovertemplate: `<b>${model}</b><br>Predicted: %{x:.1f}%<br>Actual: %{y:.1f}%<extra></extra>`,
      });
      modelIndex++;
    }

    // Perfect calibration line
    traces.push({
      type: "scatter",
      mode: "lines",
      name: "Perfect calibration",
      x: [0, 100],
      y: [0, 100],
      line: { color: "rgba(224,192,104,0.2)", width: 1, dash: "dash" },
      hoverinfo: "skip",
    });

    const layout = mergeLayout({
      title: { text: "Model Calibration", font: { size: 16 } },
      xaxis: { title: "Predicted Probability (%)", range: [0, 105] },
      yaxis: { title: "Observed Win Rate (%)", range: [0, 105] },
      legend: {
        font: { size: 10, color: COLORS.textDim },
        bgcolor: "rgba(0,0,0,0)",
      },
      height: 450,
    });

    Plotly.newPlot(container, traces, layout, PLOTLY_CONFIG);
  }

  // -----------------------------------------------------------------------
  // Section 10: 2026 Predictions
  // -----------------------------------------------------------------------
  function renderPredictions2026() {
    const container = document.getElementById("prediction-cards");
    const predictions = DATA.predictions_2026;
    if (!predictions || predictions.length === 0) {
      container.innerHTML =
        "<p style='color:var(--text-dim)'>No 2026 predictions available yet.</p>";
      return;
    }

    container.innerHTML = predictions
      .map((pred, idx) => {
        const category = pred.category;
        const modelPreds = pred.predictions || {};
        const modelNames = Object.keys(modelPreds);

        // Find consensus winner (most common prediction)
        const winnerCounts = {};
        let maxConf = 0;
        let bestWinner = "TBD";

        modelNames.forEach((m) => {
          const pick = modelPreds[m]?.winner || modelPreds[m]?.prediction;
          const conf = modelPreds[m]?.confidence || 0;
          if (pick) {
            winnerCounts[pick] = (winnerCounts[pick] || 0) + 1;
            if (conf > maxConf) {
              maxConf = conf;
              bestWinner = pick;
            }
          }
        });

        // Use the most frequently predicted winner
        const sortedWinners = Object.entries(winnerCounts).sort(
          (a, b) => b[1] - a[1]
        );
        if (sortedWinners.length > 0) {
          bestWinner = sortedWinners[0][0];
        }

        // Average confidence for the consensus winner
        let confSum = 0;
        let confCount = 0;
        modelNames.forEach((m) => {
          const pick = modelPreds[m]?.winner || modelPreds[m]?.prediction;
          const conf = modelPreds[m]?.confidence || 0;
          if (pick === bestWinner) {
            confSum += conf;
            confCount++;
          }
        });
        const avgConf = confCount > 0 ? confSum / confCount : maxConf;
        const confPct = (avgConf * 100).toFixed(0);

        // Model details
        const modelDetailsHtml = modelNames
          .map((m) => {
            const pick = modelPreds[m]?.winner || modelPreds[m]?.prediction || "—";
            const conf = modelPreds[m]?.confidence;
            return `<div class="model-detail-row">
              <span class="model-detail-name">${m}</span>
              <span class="model-detail-pick">${pick}</span>
              ${conf != null ? `<span class="model-detail-conf">${(conf * 100).toFixed(0)}%</span>` : ""}
            </div>`;
          })
          .join("");

        const cardId = `pred-models-${idx}`;

        return `
          <div class="prediction-card">
            <div class="prediction-card-category">${category}</div>
            <div class="prediction-card-winner">${bestWinner}</div>
            <div class="confidence-bar">
              <span style="font-size:0.75rem;color:var(--text-muted);">Confidence</span>
              <div class="confidence-track">
                <div class="confidence-fill" style="width:${confPct}%;"></div>
              </div>
              <span class="confidence-label">${confPct}%</span>
            </div>
            ${
              modelNames.length > 0
                ? `<button class="prediction-expand-btn" data-target="${cardId}">Show model details</button>
                  <div id="${cardId}" class="prediction-models">${modelDetailsHtml}</div>`
                : ""
            }
          </div>
        `;
      })
      .join("");

    // Event delegation for expand buttons
    container.addEventListener("click", (e) => {
      const btn = e.target.closest(".prediction-expand-btn");
      if (!btn) return;
      const targetId = btn.dataset.target;
      const el = document.getElementById(targetId);
      if (!el) return;
      el.classList.toggle("expanded");
      btn.textContent = el.classList.contains("expanded")
        ? "Hide model details"
        : "Show model details";
    });
  }

  // -----------------------------------------------------------------------
  // Boot
  // -----------------------------------------------------------------------
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
