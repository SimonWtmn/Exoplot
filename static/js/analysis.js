/**
 * Exoplot — Analysis SPA
 * ----------------------
 * Orchestrates the four-step workflow (Search → Select → Pipeline →
 * Results) against ``/api/...`` routes exposed by ``routers/analysis.py``.
 * All transitions happen in-page via ``fetch`` + panel toggling; no full
 * page reload.  The backend keeps a single in-memory session, so we can
 * stay stateless on the client side and simply request fresh state when
 * we need it.
 *
 * Refinements over the first pass
 * -------------------------------
 * • Mission filter is now a row of checkbox pills instead of a native
 *   <select multiple> (which was effectively broken on desktop browsers
 *   — Cmd-click required to add, single click reset the selection).
 * • A "Stellar information" card renders above the raw LC using exactly
 *   the same numbers the DVR report uses
 *   (``ReportGenerator._auto_stellar_params``).
 * • MCMC loader is multi-stage: a dot spinner + "Pre-processing & initial
 *   optimisation…" for the first ~20 s, then the main progress bar as
 *   soon as the backend toggles ``status.stage`` to ``"sampling"``.
 * • Parameter symbols in the summary table, stellar card and metrics
 *   list are rendered via KaTeX (``$R_p/R_\star$``, ``$a/R_\star$``…).
 * • Plots are theme-aware: we send the active UI theme on every render
 *   request and automatically re-fetch the plots via
 *   ``/api/plots/pipeline`` and ``/api/plots/results`` when the user
 *   toggles dark ↔ light.
 */
(function () {
  "use strict";

  const API = {
    search: "/api/search",
    download: "/api/download",
    mcmc: "/api/mcmc",
    mcmcStatus: "/api/mcmc/status",
    results: "/api/results",
    report: "/api/report",
    reset: "/api/reset",
    plotsPipeline: "/api/plots/pipeline",
    plotsResults: "/api/plots/results",
  };

  const STEPS = ["search", "select", "pipeline", "results"];

  // --- tiny helpers --------------------------------------------------------

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function show(el, visible = true) {
    if (!el) return;
    el.hidden = !visible;
  }

  function showError(el, msg) {
    if (!el) return;
    if (!msg) {
      el.hidden = true;
      el.textContent = "";
      return;
    }
    el.hidden = false;
    el.textContent = msg;
  }

  function refreshIcons() {
    if (typeof lucide !== "undefined" && lucide.createIcons) {
      lucide.createIcons();
    }
  }

  async function jsonFetch(url, options = {}) {
    const opts = Object.assign(
      { headers: { "Content-Type": "application/json" } },
      options
    );
    if (opts.body && typeof opts.body !== "string") {
      opts.body = JSON.stringify(opts.body);
    }
    const res = await fetch(url, opts);
    const contentType = res.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await res.json()
      : await res.text();
    if (!res.ok) {
      const detail = (payload && payload.detail) || payload || res.statusText;
      throw new Error(
        typeof detail === "string" ? detail : JSON.stringify(detail)
      );
    }
    return payload;
  }

  // --- theme / lang / error-bars ------------------------------------------
  //
  // Three global bits of visual state every server-side Matplotlib
  // render depends on:
  //   • theme (dark | light)
  //   • lang  (en   | fr)
  //   • show_errors (bool)
  //
  // They live outside the SPA (theme/lang are global UI chrome; error
  // bars are a shared pipeline toggle), so we centralise the getters
  // here and re-use them in every API call + re-render helper.

  function getTheme() {
    return document.documentElement.getAttribute("data-theme") === "light"
      ? "light"
      : "dark";
  }

  function getLang() {
    // Exoplot is strictly bilingual.  ``ExoplotI18n.getLocale()`` already
    // handles the ``localStorage`` + navigator fallback and clamps the
    // output to ``en`` / ``fr``.
    if (window.ExoplotI18n && typeof ExoplotI18n.getLocale === "function") {
      return ExoplotI18n.getLocale();
    }
    return "en";
  }

  function getShowErrors() {
    const el = document.getElementById("toggle-errorbars");
    return !!(el && el.checked);
  }

  /** Build the ``?theme=…&lang=…&show_errors=…`` suffix used by every
   *  GET plot-rerender endpoint. */
  function plotQuery() {
    const params = new URLSearchParams({
      theme: getTheme(),
      lang: getLang(),
      show_errors: getShowErrors() ? "true" : "false",
    });
    return `?${params.toString()}`;
  }

  // --- LaTeX (KaTeX auto-render) ------------------------------------------
  //
  // Parameter names returned by TransitFitter are terse ("rp", "inc", "a",
  // "t0", …).  We map them to proper LaTeX so the summary/metrics table
  // reads like a textbook ("$R_p/R_\star$", "$a/R_\star$", …).
  const PARAM_LATEX = {
    rp:  "$R_{p}/R_{\\star}$",
    "rp/rs": "$R_{p}/R_{\\star}$",
    inc: "$i$",
    i:   "$i$",
    a:   "$a/R_{\\star}$",
    "a/rs": "$a/R_{\\star}$",
    t0:  "$t_{0}$",
    per: "$P$",
    p:   "$P$",
    period: "$P$",
    u1:  "$u_{1}$",
    u2:  "$u_{2}$",
    ecc: "$e$",
    w:   "$\\omega$",
  };

  /** Return a LaTeX-wrapped symbol for a parameter name, or ``null`` if
   *  we don't have a mapping. */
  function paramLatex(name) {
    if (!name) return null;
    const key = String(name).trim().toLowerCase();
    return PARAM_LATEX[key] || null;
  }

  function typesetMath(root) {
    if (typeof window.renderMathInElement !== "function") return;
    try {
      window.renderMathInElement(root || document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$",  right: "$",  display: false },
        ],
        throwOnError: false,
        strict: "ignore",
      });
    } catch (err) {
      // KaTeX doesn't block the page if math is malformed — just log.
      console.debug("KaTeX render skipped:", err);
    }
  }

  // --- step navigation -----------------------------------------------------

  function setActiveStep(stepName) {
    const order = STEPS.indexOf(stepName);
    $$(".analysis-step").forEach((li, idx) => {
      li.classList.toggle("is-active", idx === order);
      li.classList.toggle("is-complete", idx < order);
    });
    $$(".analysis-panel").forEach((panel) => {
      const match = panel.dataset.panel === stepName;
      show(panel, match);
      panel.classList.toggle("is-active", match);
    });
    try {
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e) {
      window.scrollTo(0, 0);
    }
  }

  // ═════════════════════════ STEP 1 — SEARCH ═════════════════════════════

  function getSelectedMissions() {
    return $$("#search-mission input[type=checkbox]:checked").map(
      (cb) => cb.value
    );
  }

  function bindMissionPills() {
    // The whole label is already clickable (the <input> fills the pill
    // via absolute positioning), but we also react to `change` so the
    // ``:has(input:checked)`` style update is immediate in every browser
    // and for keyboard users (space toggles the checkbox).
    $$("#search-mission .mission-pill").forEach((pill) => {
      const input = pill.querySelector("input[type=checkbox]");
      if (!input) return;
      input.addEventListener("change", () => {
        pill.classList.toggle("is-on", input.checked);
      });
      // prime the state so hydrated pills have the class as well
      pill.classList.toggle("is-on", input.checked);
    });
  }

  function bindSearch() {
    const form = $("#search-form");
    const loader = $("#search-loader");
    const errBox = $("#search-error");
    const submit = $("#search-submit");
    if (!form) return;

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      showError(errBox, null);
      const target = $("#search-target").value.trim();
      if (!target) {
        showError(errBox, "Please enter a target name or TIC ID.");
        return;
      }

      const missions = getSelectedMissions();
      const body = {
        target,
        mission: missions.length ? missions : null,
        sector: intOrNull($("#search-sector").value),
        quarter: intOrNull($("#search-quarter").value),
        campaign: intOrNull($("#search-campaign").value),
        year: intOrNull($("#search-year").value),
        author: $("#search-author").value.trim() || null,
        limit: intOrNull($("#search-limit").value),
      };

      show(loader, true);
      submit.disabled = true;
      try {
        const data = await jsonFetch(API.search, { method: "POST", body });
        renderResultsTable(data);
        setActiveStep("select");
      } catch (err) {
        showError(errBox, err.message || "Search failed.");
      } finally {
        show(loader, false);
        submit.disabled = false;
        refreshIcons();
      }
    });
  }

  function intOrNull(v) {
    if (v === "" || v === null || v === undefined) return null;
    const n = Number(v);
    return Number.isFinite(n) ? Math.trunc(n) : null;
  }

  // ═════════════════════════ STEP 2 — SELECT ═════════════════════════════

  function renderResultsTable(data) {
    const table = $("#results-table");
    const thead = table.querySelector("thead tr");
    const tbody = table.querySelector("tbody");
    const submit = $("#select-submit");
    const sub = $("#select-sub");

    $$("th:not(.col-check)", thead).forEach((th) => th.remove());
    tbody.innerHTML = "";
    submit.disabled = true;

    const count = data.count || 0;
    sub.textContent = count
      ? `${count} observation${count === 1 ? "" : "s"} found for ${data.target}. ` +
        "Select one or more rows to download and clean."
      : `No observations found for ${data.target}.`;

    const cols = data.columns || [];
    cols.forEach((col) => {
      const th = document.createElement("th");
      th.textContent = col;
      thead.appendChild(th);
    });

    data.rows.forEach((row, idx) => {
      const tr = document.createElement("tr");
      tr.dataset.index = idx;

      const checkTd = document.createElement("td");
      checkTd.className = "col-check";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.className = "row-check";
      checkbox.value = String(idx);
      checkbox.setAttribute("aria-label", `Select observation ${idx}`);
      checkbox.addEventListener("change", () => {
        tr.classList.toggle("is-selected", checkbox.checked);
        updateSubmitState();
      });
      checkTd.appendChild(checkbox);
      tr.appendChild(checkTd);

      cols.forEach((col) => {
        const td = document.createElement("td");
        const val = row[col];
        td.textContent = val === null || val === undefined ? "—" : String(val);
        tr.appendChild(td);
      });

      tr.addEventListener("click", (e) => {
        if (e.target.tagName === "INPUT") return;
        checkbox.checked = !checkbox.checked;
        checkbox.dispatchEvent(new Event("change"));
      });

      tbody.appendChild(tr);
    });

    const master = $("#results-check-all");
    if (master) {
      master.checked = false;
      master.indeterminate = false;
      master.onchange = () => {
        $$(".row-check", tbody).forEach((cb) => {
          cb.checked = master.checked;
          cb.dispatchEvent(new Event("change"));
        });
      };
    }

    function updateSubmitState() {
      const selected = $$(".row-check:checked", tbody).length;
      submit.disabled = selected === 0;
      const all = $$(".row-check", tbody).length;
      if (master) {
        master.checked = selected > 0 && selected === all;
        master.indeterminate = selected > 0 && selected < all;
      }
    }
  }

  function bindSelect() {
    const back = $("#select-back");
    const submit = $("#select-submit");
    const loader = $("#download-loader");
    const errBox = $("#select-error");

    back.addEventListener("click", () => setActiveStep("search"));

    submit.addEventListener("click", async () => {
      showError(errBox, null);
      const indices = $$("#results-table .row-check:checked").map((cb) =>
        Number(cb.value)
      );
      if (!indices.length) return;

      show(loader, true);
      submit.disabled = true;
      try {
        const data = await jsonFetch(API.download, {
          method: "POST",
          body: {
            indices,
            theme: getTheme(),
            lang: getLang(),
            show_errors: getShowErrors(),
          },
        });
        applyPipelinePlots(data);
        setActiveStep("pipeline");
      } catch (err) {
        showError(errBox, err.message || "Download failed.");
      } finally {
        show(loader, false);
        submit.disabled = false;
        refreshIcons();
      }
    });
  }

  // ═════════════════════════ STEP 3 — PIPELINE ═══════════════════════════

  function applyPipelinePlots(data) {
    $("#plot-raw").src = data.plots.raw || "";
    $("#plot-periodogram").src = data.plots.periodogram || "";
    $("#plot-folded").src = data.plots.folded || "";

    const period = Number(data.best_period).toFixed(5);
    const label = data.selection_label || "";
    const periodPill = $("#pipeline-period");
    if (periodPill) {
      periodPill.textContent = `P ≈ ${period} d` + (label ? `  ·  ${label}` : "");
    }

    renderStellarCard(data.stellar || {});
    switchTab("validate");
  }

  /** Renders the Stellar information card above the raw LC.  Uses the
   *  exact keys returned by ``ReportGenerator._auto_stellar_params``
   *  (tmag, rs, teff, logg, mh, rho) plus the FITS-meta RA/Dec/TIC.    */
  function renderStellarCard(info) {
    const card = $("#stellar-info");
    const grid = $("#stellar-grid");
    if (!card || !grid) return;

    grid.innerHTML = "";
    const rows = [];

    const name = info.name;
    const tic = info.id;
    if (name) rows.push(["Target", name]);
    if (tic && tic !== name) rows.push(["TIC / Object", tic]);
    if (info.mission) rows.push(["Source", info.mission]);
    if (info.selection) rows.push(["Selection", info.selection]);

    if (Number.isFinite(info.ra)) {
      rows.push(["RA", Number(info.ra).toFixed(4) + "°"]);
    }
    if (Number.isFinite(info.dec)) {
      rows.push(["Dec", Number(info.dec).toFixed(4) + "°"]);
    }
    if (Number.isFinite(info.tmag)) {
      rows.push(["$T_{\\mathrm{mag}}$", Number(info.tmag).toFixed(2)]);
    }
    if (Number.isFinite(info.rs)) {
      rows.push(["$R_{\\star}$  $(R_\\odot)$", Number(info.rs).toFixed(3)]);
    }
    if (Number.isFinite(info.teff)) {
      rows.push(["$T_{\\mathrm{eff}}$ (K)", Number(info.teff).toFixed(0)]);
    }
    if (Number.isFinite(info.logg)) {
      rows.push(["$\\log g$", Number(info.logg).toFixed(2)]);
    }
    if (Number.isFinite(info.mh)) {
      rows.push(["$[\\mathrm{Fe/H}]$", Number(info.mh).toFixed(2)]);
    }
    if (Number.isFinite(info.rho)) {
      rows.push(["$\\rho_\\star$ (g·cm$^{-3}$)", Number(info.rho).toFixed(3)]);
    }

    if (!rows.length) {
      card.hidden = true;
      return;
    }

    rows.forEach(([key, val]) => {
      const cell = document.createElement("div");
      const dt = document.createElement("dt");
      dt.textContent = key;              // may contain $...$ — KaTeX later
      const dd = document.createElement("dd");
      dd.textContent = String(val);
      cell.appendChild(dt);
      cell.appendChild(dd);
      grid.appendChild(cell);
    });
    card.hidden = false;
    typesetMath(card);
  }

  function switchTab(tab) {
    $$(".pipeline-tab").forEach((btn) => {
      const on = btn.dataset.tab === tab;
      btn.classList.toggle("is-active", on);
      btn.setAttribute("aria-selected", on ? "true" : "false");
    });
    $$(".pipeline-tabpanel").forEach((panel) => {
      show(panel, panel.dataset.tabpanel === tab);
      panel.classList.toggle("is-active", panel.dataset.tabpanel === tab);
    });
  }

  function bindPipeline() {
    $("#pipeline-back").addEventListener("click", () =>
      setActiveStep("select")
    );
    $$(".pipeline-tab").forEach((btn) => {
      btn.addEventListener("click", () => switchTab(btn.dataset.tab));
    });

    const autoToggle = $("#mcmc-auto");
    const manualBox = $("#mcmc-manual");
    autoToggle.addEventListener("change", () => {
      show(manualBox, !autoToggle.checked);
    });

    const mcmcForm = $("#mcmc-form");
    mcmcForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      await launchMCMC();
    });
  }

  function parseManualBounds() {
    const txt = $("#mcmc-manual-text").value.trim();
    if (!txt) return null;
    const fitted = [];
    const bounds = [];
    const x0 = [];
    txt.split(/\r?\n/).forEach((raw) => {
      const line = raw.trim();
      if (!line || line.startsWith("#")) return;
      const parts = line.split(/[\s,]+/);
      if (parts.length < 4) {
        throw new Error(`Malformed bounds line: ${raw}`);
      }
      const [name, lo, hi, guess] = parts;
      const nLo = Number(lo);
      const nHi = Number(hi);
      const nX = Number(guess);
      if (!Number.isFinite(nLo) || !Number.isFinite(nHi) || !Number.isFinite(nX)) {
        throw new Error(`Non-numeric values in bounds line: ${raw}`);
      }
      fitted.push(name);
      bounds.push([nLo, nHi]);
      x0.push(nX);
    });
    if (!fitted.length) return null;
    return { fitted, bounds, x0 };
  }

  async function launchMCMC() {
    const errBox = $("#mcmc-error");
    const progress = $("#mcmc-progress");
    const launchBtn = $("#mcmc-launch");
    showError(errBox, null);

    const fitted = $$('.mcmc-checks input[name="fitted"]:checked').map(
      (cb) => cb.value
    );
    if (!fitted.length) {
      showError(errBox, "Select at least one free parameter.");
      return;
    }

    const auto = $("#mcmc-auto").checked;
    const body = {
      auto_bounds: auto,
      nwalkers: Number($("#mcmc-walkers").value) || 32,
      nsteps: Number($("#mcmc-steps").value) || 4000,
      use_multiprocessing: $("#mcmc-mp").checked,
    };

    if (auto) {
      body.fitted_params = fitted;
    } else {
      try {
        const manual = parseManualBounds();
        if (!manual) {
          showError(errBox, "Provide manual bounds or enable auto-search.");
          return;
        }
        body.fitted_params = manual.fitted;
        body.custom_bounds = manual.bounds;
        body.custom_x0 = manual.x0;
      } catch (err) {
        showError(errBox, err.message);
        return;
      }
    }

    launchBtn.disabled = true;
    show(progress, true);
    // Start in the pre-processing stage so the user immediately sees a
    // lively indicator — otherwise the first ~20 s of optimiser warm-up
    // looks like the app is frozen.
    updateProgress({
      state: "queued",
      stage: "preprocessing",
      step: 0,
      total: body.nsteps,
      message: "Queued — warming up the sampler…",
    });

    try {
      await jsonFetch(API.mcmc, { method: "POST", body });
      await pollMCMC();
      const data = await jsonFetch(`${API.results}${plotQuery()}`);
      renderResults(data);
      setActiveStep("results");
    } catch (err) {
      showError(errBox, err.message || "MCMC failed.");
      show(progress, false);
    } finally {
      launchBtn.disabled = false;
      refreshIcons();
    }
  }

  /** Updates the two-stage MCMC progress UI.
   *
   *  ``status.stage`` can be one of:
   *    - ``"preprocessing"`` → dot spinner + message, no progress bar.
   *    - ``"sampling"``      → hide spinner, show + fill the bar.
   *    - ``"done"`` / ``"error"`` → let the outer flow take over. */
  function updateProgress(status) {
    const pre = $("#mcmc-preprocess");
    const samp = $("#mcmc-sampling");
    const msg = $("#mcmc-progress-msg");

    const stage = status.stage || status.state || "preprocessing";
    const isSampling =
      stage === "sampling" ||
      (status.step && Number(status.step) > 0 && status.state !== "queued");

    if (!isSampling) {
      show(pre, true);
      show(samp, false);
      msg.textContent =
        status.message ||
        "Pre-processing & initial optimisation — this takes a few seconds.";
      return;
    }

    show(pre, false);
    show(samp, true);
    const pct =
      status.total > 0
        ? Math.min(100, (100 * (status.step || 0)) / status.total)
        : 0;
    $("#mcmc-progress-fill").style.width = `${pct.toFixed(1)}%`;
    msg.textContent = status.message || "Sampling…";
    $("#mcmc-progress-stats").textContent = `state: ${status.state} · stage: ${
      stage
    } · step ${status.step}/${status.total} (${pct.toFixed(1)}%)`;
  }

  async function pollMCMC() {
    // Poll faster at first so the spinner→bar transition feels snappy,
    // then back off for the long sampling tail so we don't flood /status.
    let interval = 500;
    while (true) {
      const status = await jsonFetch(API.mcmcStatus);
      updateProgress(status);
      if (status.state === "done") return;
      if (status.state === "error") {
        throw new Error(status.message || "MCMC failed.");
      }
      await sleep(interval);
      // during sampling we can relax; during preprocess keep snappy
      if (status.stage === "sampling") {
        interval = Math.min(interval + 250, 2500);
      } else {
        interval = 500;
      }
    }
  }

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  // ═════════════════════════ STEP 4 — RESULTS ════════════════════════════

  function renderResults(data) {
    const summaryBody = $("#summary-table tbody");
    summaryBody.innerHTML = "";
    (data.summary || []).forEach((row) => {
      const tr = document.createElement("tr");
      // Parameter column — prefer backend-supplied LaTeX, fall back to
      // the client-side map, fall back to the raw name.
      const paramTd = document.createElement("td");
      const latex = row.latex || paramLatex(row.parameter);
      paramTd.textContent = latex || row.parameter;
      if (!latex) {
        paramTd.style.fontFamily =
          "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, monospace";
      }
      tr.appendChild(paramTd);

      [row.median, row.plus, row.minus].forEach((val) => {
        const td = document.createElement("td");
        td.textContent = Number(val).toPrecision(5);
        td.style.fontFamily =
          "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, monospace";
        tr.appendChild(td);
      });
      summaryBody.appendChild(tr);
    });

    const sub = $("#results-sub");
    sub.textContent =
      `Target: ${data.target}  ·  P = ${Number(data.best_period).toFixed(5)} d` +
      (data.snr ? `  ·  SNR ≈ ${Number(data.snr).toFixed(1)}` : "");

    const metrics = $("#results-metrics");
    metrics.innerHTML = "";
    const d = data.diagnostics || {};
    const rows = [
      ["SNR", data.snr == null ? "—" : Number(data.snr).toFixed(1), true],
      ["$P$ (d)", Number(data.best_period).toFixed(6)],
      ["$t_{0}$", Number(data.epoch_time).toFixed(6)],
      ["Walkers", d.nwalkers ?? "—"],
      ["Steps", d.nsteps ?? "—"],
      ["Burn-in", d.burn_in ?? "—"],
      ["Thin", d.thin ?? "—"],
      [
        "$\\tau$ autocorr",
        d.autocorr_time == null ? "—" : Number(d.autocorr_time).toFixed(1),
      ],
      [
        "Acceptance",
        d.mean_acceptance_fraction == null
          ? "—"
          : `${(100 * d.mean_acceptance_fraction).toFixed(1)} %`,
      ],
      ["Effective samples", d.n_effective_samples ?? "—"],
    ];
    rows.forEach(([key, val, highlight]) => {
      const dt = document.createElement("dt");
      dt.textContent = key;
      const dd = document.createElement("dd");
      dd.textContent = String(val);
      if (highlight) dd.classList.add("is-highlight");
      metrics.appendChild(dt);
      metrics.appendChild(dd);
    });

    $("#plot-folded-model").src = data.plots.folded_model || "";
    $("#plot-odd-even").src = data.plots.odd_even || "";
    $("#plot-corner").src = data.plots.corner || "";

    typesetMath($("[data-panel=results]"));
  }

  function bindResults() {
    $("#results-restart").addEventListener("click", async () => {
      try {
        await jsonFetch(API.reset, { method: "POST" });
      } catch (e) { /* ignore */ }
      location.reload();
    });

    $("#results-report").addEventListener("click", async () => {
      const btn = $("#results-report");
      const status = $("#report-status");
      showError(status, null);
      btn.disabled = true;
      const prevHTML = btn.innerHTML;
      btn.innerHTML = '<i data-lucide="loader-2"></i><span>Generating…</span>';
      refreshIcons();
      try {
        const res = await jsonFetch(API.report, {
          method: "POST",
          body: {
            movie: false,
            use_latex: false,
            lang: getLang(),
          },
        });
        if (res && res.url) {
          const a = document.createElement("a");
          a.href = res.url;
          a.download = res.filename || "exoplot_report.pdf";
          document.body.appendChild(a);
          a.click();
          a.remove();
          status.className = "analysis-error";
          status.style.color = "#86efac";
          status.style.borderColor = "rgba(34, 197, 94, 0.35)";
          status.style.background = "rgba(34, 197, 94, 0.08)";
          status.hidden = false;
          status.textContent = `Report ready: ${res.filename}`;
        }
      } catch (err) {
        showError(status, err.message || "Report generation failed.");
      } finally {
        btn.disabled = false;
        btn.innerHTML = prevHTML;
        refreshIcons();
      }
    });
  }

  // ═════════════════════════ PLOT RE-SYNC ════════════════════════════════
  //
  // Re-render every server-side plot that currently has data backing it
  // whenever any of the three visual-state bits flip: theme, language,
  // or the "Show error bars" toggle.  The API returns base64 PNGs so
  // refreshing the ``src`` attribute is enough — no layout shift, no
  // analyzer state mutation.

  let rerenderInFlight = false;
  let rerenderPending = false;

  async function rerenderPlots() {
    // If one is already in flight, remember we need another pass after
    // it finishes — otherwise rapid toggling would drop updates.
    if (rerenderInFlight) {
      rerenderPending = true;
      return;
    }
    rerenderInFlight = true;
    try {
      const query = plotQuery();

      const pipelinePanel = document.querySelector('[data-panel="pipeline"]');
      const resultsPanel  = document.querySelector('[data-panel="results"]');

      // Re-render the pipeline set if we're on it *or* if we have
      // pipeline plots already visible (user may flip the toggle after
      // the results step).
      if (pipelinePanel && !pipelinePanel.hidden) {
        try {
          const data = await jsonFetch(`${API.plotsPipeline}${query}`);
          $("#plot-raw").src = data.plots.raw || "";
          $("#plot-periodogram").src = data.plots.periodogram || "";
          $("#plot-folded").src = data.plots.folded || "";
        } catch (_) { /* no session data yet — ignore */ }
      }

      if (resultsPanel && !resultsPanel.hidden) {
        try {
          const data = await jsonFetch(`${API.plotsResults}${query}`);
          $("#plot-folded-model").src = data.plots.folded_model || "";
          $("#plot-odd-even").src = data.plots.odd_even || "";
          $("#plot-corner").src = data.plots.corner || "";
        } catch (_) { /* ignore */ }
      }
    } finally {
      rerenderInFlight = false;
      if (rerenderPending) {
        rerenderPending = false;
        rerenderPlots();
      }
    }
  }

  function bindVisualStateSync() {
    window.addEventListener("exoplot-theme", rerenderPlots);
    window.addEventListener("exoplot-lang", rerenderPlots);
    const errBox = document.getElementById("toggle-errorbars");
    if (errBox) {
      errBox.addEventListener("change", rerenderPlots);
    }
  }

  // ═════════════════════════ BOOT ════════════════════════════════════════

  function init() {
    bindMissionPills();
    bindSearch();
    bindSelect();
    bindPipeline();
    bindResults();
    bindVisualStateSync();
    refreshIcons();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
