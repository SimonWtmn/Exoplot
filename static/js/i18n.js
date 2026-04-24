/**
 * UI copy for Exoplot (client-side i18n).
 *
 * Exoplot is intentionally **strictly bilingual** — English + French.
 * The header language toggle only exposes those two locales, and this
 * module mirrors that contract so any newly-added UI string lives in
 * exactly two places.
 *
 * Keys mirror (where they overlap) the backend ``modules/i18n.py``,
 * but the frontend pack is intentionally larger: it carries all the
 * chrome/hero/about copy the Jinja templates rely on through the
 * ``data-i18n="…"`` attribute.
 */
(function (global) {
  "use strict";

  var STRINGS = {
    // ──────────────────────────────────────────────────────────────
    //  English (reference pack)
    // ──────────────────────────────────────────────────────────────
    en: {
      meta_title: "Exoplot",
      meta_description:
        "Explore exoplanet data visually: catalogs, light curves, and clear plots in one place.",
      nav_home: "Home",
      nav_analysis: "Analysis",
      hero_eyebrow: "For curious observers",
      hero_title: "Explore exoplanets visually",
      hero_subtitle:
        "Search worlds beyond the solar system, see how they compare, and follow real mission light curves.",
      hero_scroll_aria: "Scroll to the about section",
      btn_launch: "Launch Lightcurve Analysis",

      // Analysis SPA
      step_search: "Search",
      step_select: "Select",
      step_pipeline: "Pipeline",
      step_results: "Results",
      search_eyebrow: "Step 1 · Target",
      search_title: "Find a lightcurve",
      search_sub:
        "Query MAST for Kepler, K2 and TESS observations of any resolvable star or TIC ID.",
      field_target: "Target",
      field_mission: "Mission",
      field_sector: "Sector",
      field_quarter: "Quarter",
      field_campaign: "Campaign",
      field_year: "Year",
      field_author: "Author",
      field_limit: "Limit",
      btn_search: "Search",
      loading_search: "Scanning the MAST archive…",
      loading_download:
        "Downloading, normalising and stitching observations…",
      select_eyebrow: "Step 2 · Observations",
      select_title: "Select observations",
      btn_back: "Back",
      btn_download_clean: "Download & Clean Data",
      pipeline_eyebrow: "Step 3 · Pipeline",
      pipeline_title: "Data validation & MCMC configuration",
      pipeline_sub:
        "Inspect the cleaned data, then tune the transit fit.",
      tab_validate: "A · Data Validation",
      tab_configure: "B · MCMC Configuration",
      plot_raw: "Stitched raw lightcurve",
      plot_periodogram: "BLS periodogram",
      plot_folded: "Initial folded lightcurve",
      mcmc_params: "Free parameters",
      mcmc_bounds: "Bounds & initial guess",
      mcmc_auto: "Let the program auto-search bounds & x0",
      mcmc_manual_hint:
        "One parameter per line: name  low  high  x0",
      mcmc_runtime: "Runtime",
      mcmc_walkers: "Walkers",
      mcmc_steps: "Steps",
      mcmc_mp: "Multiprocessing",
      btn_launch_mcmc: "Launch MCMC",
      mcmc_running: "Running MCMC…",
      mcmc_preprocess:
        "Pre-processing & initial optimisation — this takes a few seconds.",
      stellar_header: "Stellar information",
      toggle_errorbars: "Show error bars",
      toggle_errorbars_hint:
        "Overlay σ(F) on the data points. Re-renders plots instantly.",
      results_eyebrow: "Step 4 · Results",
      results_title: "Best-fit transit parameters",
      results_summary: "Posterior summary",
      results_metrics: "Fit diagnostics",
      col_param: "Parameter",
      col_median: "Median",
      col_plus: "+1σ",
      col_minus: "−1σ",
      plot_folded_model: "Folded lightcurve + best-fit model",
      plot_odd_even: "Odd / even transits",
      plot_corner: "Corner plot",
      btn_new_analysis: "New analysis",
      btn_report: "Generate & Download Report",

      // Chrome
      footer_resources: "Useful links",
      link_nea: "NASA Exoplanet Archive",
      link_lightkurve: "Lightkurve",
      link_github: "GitHub",
      link_linkedin: "LinkedIn",
      footer_lang: "Language",
      footer_credit: "Made by Simon Wittmann",
      logo_credit: "made by A. Wittmann",
      theme_toggle_aria: "Switch between dark and light theme",
      menu_toggle_aria: "Open navigation menu",
      menu_close_aria: "Close navigation menu",
      aria_github: "Exoplot on GitHub",
      aria_linkedin: "LinkedIn profile",
      aria_lightkurve: "Lightkurve documentation",
      aria_nea: "NASA Exoplanet Archive",

      about_eyebrow: "About",
      about_title: "About this project",
      about_lead:
        "Exoplot gathers catalogs and mission light curves in one calm workspace — for anyone who wants readable plots, reproducible Python, and fewer tab switches.",
      about_card_catalog_title: "Browseable catalogs",
      about_card_catalog_body:
        "Filter published worlds and compare key measurements without losing the big picture.",
      about_card_curves_title: "Light curves that read clearly",
      about_card_curves_body:
        "Follow transits and variability with a layout tuned for long time series from Kepler, TESS, and similar missions.",
      about_card_open_title: "Transparent stack",
      about_card_open_body:
        "Open Python modules underneath — inspect, extend, and reproduce every figure you see here.",
      project_lead:
        "Here you will soon find the story of Exoplot, technical notes, and short science explainers — this page is a placeholder for now.",
    },

    // ──────────────────────────────────────────────────────────────
    //  French
    // ──────────────────────────────────────────────────────────────
    fr: {
      meta_title: "Exoplot",
      meta_description:
        "Explorez les données d’exoplanètes visuellement : catalogues, courbes de lumière et graphiques clairs au même endroit.",
      nav_home: "Accueil",
      nav_analysis: "Analyse",
      hero_eyebrow: "Pour les curieux du ciel",
      hero_title: "Explorez les exoplanètes visuellement",
      hero_subtitle:
        "Parcourez des mondes au-delà du Système solaire, comparez leurs propriétés et suivez de vraies courbes de lumière.",
      hero_scroll_aria: "Aller à la section À propos",
      btn_launch: "Lancer l'analyse de courbe",

      // Analysis SPA
      step_search: "Recherche",
      step_select: "Sélection",
      step_pipeline: "Pipeline",
      step_results: "Résultats",
      search_eyebrow: "Étape 1 · Cible",
      search_title: "Rechercher une courbe de lumière",
      search_sub:
        "Interroge MAST pour les observations Kepler, K2 et TESS d'une étoile ou d'un TIC.",
      field_target: "Cible",
      field_mission: "Mission",
      field_sector: "Secteur",
      field_quarter: "Trimestre",
      field_campaign: "Campagne",
      field_year: "Année",
      field_author: "Auteur",
      field_limit: "Limite",
      btn_search: "Rechercher",
      loading_search: "Interrogation de l'archive MAST…",
      loading_download:
        "Téléchargement, normalisation et assemblage des observations…",
      select_eyebrow: "Étape 2 · Observations",
      select_title: "Sélectionner des observations",
      btn_back: "Retour",
      btn_download_clean: "Télécharger et nettoyer",
      pipeline_eyebrow: "Étape 3 · Pipeline",
      pipeline_title: "Validation des données et configuration MCMC",
      pipeline_sub:
        "Inspectez les données nettoyées puis réglez l'ajustement de transit.",
      tab_validate: "A · Validation des données",
      tab_configure: "B · Configuration MCMC",
      plot_raw: "Courbe de lumière brute assemblée",
      plot_periodogram: "Périodogramme BLS",
      plot_folded: "Courbe repliée initiale",
      mcmc_params: "Paramètres libres",
      mcmc_bounds: "Bornes et initialisation",
      mcmc_auto: "Recherche automatique des bornes et x0",
      mcmc_manual_hint:
        "Une ligne par paramètre : nom  bas  haut  x0",
      mcmc_runtime: "Exécution",
      mcmc_walkers: "Walkers",
      mcmc_steps: "Pas",
      mcmc_mp: "Multitraitement",
      btn_launch_mcmc: "Lancer le MCMC",
      mcmc_running: "MCMC en cours…",
      mcmc_preprocess:
        "Pré-traitement & optimisation initiale — quelques secondes.",
      stellar_header: "Informations stellaires",
      toggle_errorbars: "Afficher les barres d'erreur",
      toggle_errorbars_hint:
        "Superpose σ(F) sur les points. Re-rend les graphiques instantanément.",
      results_eyebrow: "Étape 4 · Résultats",
      results_title: "Paramètres de transit ajustés",
      results_summary: "Résumé du postérieur",
      results_metrics: "Diagnostics d'ajustement",
      col_param: "Paramètre",
      col_median: "Médiane",
      col_plus: "+1σ",
      col_minus: "−1σ",
      plot_folded_model: "Courbe repliée + modèle ajusté",
      plot_odd_even: "Transits pairs / impairs",
      plot_corner: "Corner plot",
      btn_new_analysis: "Nouvelle analyse",
      btn_report: "Générer & télécharger le rapport",

      // Chrome
      footer_resources: "Liens utiles",
      link_nea: "NASA Exoplanet Archive",
      link_lightkurve: "Lightkurve",
      link_github: "GitHub",
      link_linkedin: "LinkedIn",
      footer_lang: "Langue",
      footer_credit: "Réalisé par Simon Wittmann",
      logo_credit: "made by A. Wittmann",
      theme_toggle_aria: "Basculer entre thème sombre et clair",
      menu_toggle_aria: "Ouvrir le menu de navigation",
      menu_close_aria: "Fermer le menu de navigation",
      aria_github: "Exoplot sur GitHub",
      aria_linkedin: "Profil LinkedIn",
      aria_lightkurve: "Documentation Lightkurve",
      aria_nea: "NASA Exoplanet Archive",

      about_eyebrow: "À propos",
      about_title: "À propos de ce projet",
      about_lead:
        "Exoplot réunit catalogues et courbes de mission dans un espace posé — pour des graphiques lisibles, du Python reproductible, et moins de fenêtres ouvertes.",
      about_card_catalog_title: "Catalogues explorables",
      about_card_catalog_body:
        "Filtrez les mondes publiés et comparez les mesures clés sans perdre la vue d’ensemble.",
      about_card_curves_title: "Courbes lisibles au premier regard",
      about_card_curves_body:
        "Suivez transits et variabilité avec une mise en page pensée pour les longues séries Kepler, TESS et assimilées.",
      about_card_open_title: "Stack transparente",
      about_card_open_body:
        "Des modules Python ouverts en dessous — inspectez, prolongez et reproduisez chaque figure.",
      project_lead:
        "Vous y trouverez bientôt la genèse d’Exoplot, des notes techniques et de courts rappels scientifiques — pour l’instant ce n’est qu’une page d’attente.",
    },
  };

  var SUPPORTED = ["en", "fr"];

  var HTML_LANG = {
    en: "en",
    fr: "fr",
  };

  function normaliseLocale(raw) {
    if (!raw) return null;
    var s = String(raw).toLowerCase();
    if (STRINGS[s]) return s;
    var short = s.split("-")[0].split("_")[0];
    return STRINGS[short] ? short : null;
  }

  function getLocale() {
    try {
      var stored = localStorage.getItem("exoplot-lang");
      var n = normaliseLocale(stored);
      if (n) return n;
    } catch (e) {}
    var raw =
      (global.navigator && global.navigator.language) || "en";
    var n2 = normaliseLocale(raw);
    return n2 || "en";
  }

  function setLocale(code) {
    var n = normaliseLocale(code);
    if (!n) return;
    try {
      localStorage.setItem("exoplot-lang", n);
    } catch (e) {}
    if (global.document && global.document.documentElement) {
      global.document.documentElement.lang = HTML_LANG[n] || "en";
    }
  }

  function t(key, locale) {
    var loc = normaliseLocale(locale) || getLocale();
    var pack = STRINGS[loc] || STRINGS.en;
    return pack[key] != null
      ? pack[key]
      : STRINGS.en[key] != null
      ? STRINGS.en[key]
      : key;
  }

  function apply(locale) {
    var loc = normaliseLocale(locale) || getLocale();
    var root = global.document;
    if (!root) return;
    var nodes = root.querySelectorAll("[data-i18n], [data-i18n-aria]");
    var i;
    for (i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      var ariaKey = el.getAttribute("data-i18n-aria");
      if (ariaKey) {
        el.setAttribute("aria-label", t(ariaKey, loc));
      }
      var k = el.getAttribute("data-i18n");
      if (!k) continue;
      var val = t(k, loc);
      if (el.tagName === "TITLE") {
        root.title = val;
      } else if (
        "placeholder" in el &&
        el.hasAttribute("data-i18n-placeholder")
      ) {
        el.placeholder = val;
      } else {
        el.textContent = val;
      }
    }
    var meta = root.querySelector('meta[name="description"]');
    if (meta) meta.setAttribute("content", t("meta_description", loc));
  }

  function getHtmlLang(locale) {
    return HTML_LANG[normaliseLocale(locale) || getLocale()] || "en";
  }

  global.ExoplotI18n = {
    getLocale: getLocale,
    setLocale: setLocale,
    getHtmlLang: getHtmlLang,
    t: t,
    apply: apply,
    locales: SUPPORTED.slice(),
  };
})(typeof window !== "undefined" ? window : this);
