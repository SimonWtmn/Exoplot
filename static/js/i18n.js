/**
 * UI copy for landing (client-side i18n).
 */
(function (global) {
  "use strict";

  var STRINGS = {
    en: {
      meta_title: "Exoplot",
      meta_description:
        "Explore exoplanet data visually: catalogs, light curves, and clear plots in one place.",
      nav_home: "Home",
      nav_analyse: "Analyse",
      nav_discover: "Discover",
      hero_eyebrow: "For curious observers",
      hero_title: "Explore exoplanets visually",
      hero_subtitle:
        "Search worlds beyond the solar system, see how they compare, and follow real mission light curves.",
      hero_scroll_aria: "Scroll to the about section",
      btn_analyse: "Analyse",
      btn_discover: "Discover",
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
    fr: {
      meta_title: "Exoplot",
      meta_description:
        "Explorez les données d’exoplanètes visuellement : catalogues, courbes de lumière et graphiques clairs au même endroit.",
      nav_home: "Accueil",
      nav_analyse: "Analyse",
      nav_discover: "Découvrir",
      hero_eyebrow: "Pour les curieux du ciel",
      hero_title: "Explorez les exoplanètes visuellement",
      hero_subtitle:
        "Parcourez des mondes au-delà du Système solaire, comparez leurs propriétés et suivez de vraies courbes de lumière.",
      hero_scroll_aria: "Aller à la section À propos",
      btn_analyse: "Analyse",
      btn_discover: "Découvrir",
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
    es: {
      meta_title: "Exoplot",
      meta_description:
        "Explora datos de exoplanetas de forma visual: catálogos, curvas de luz y gráficos claros en un solo lugar.",
      nav_home: "Inicio",
      nav_analyse: "Análisis",
      nav_discover: "Descubrir",
      hero_eyebrow: "Para miradas curiosas",
      hero_title: "Explora exoplanetas visualmente",
      hero_subtitle:
        "Recorre mundos más allá del sistema solar, compara sus propiedades y sigue curvas de luz reales de misiones.",
      btn_analyse: "Análisis",
      btn_discover: "Descubrir",
      footer_resources: "Enlaces útiles",
      link_nea: "NASA Exoplanet Archive",
      link_lightkurve: "Lightkurve",
      link_github: "GitHub",
      link_linkedin: "LinkedIn",
      footer_lang: "Idioma",
      footer_credit: "Hecho por Simon Wittmann",
      logo_credit: "made by A. Wittmann",
      theme_toggle_aria: "Cambiar entre tema oscuro y claro",
      aria_github: "Exoplot en GitHub",
      aria_linkedin: "Perfil de LinkedIn",
      aria_lightkurve: "Documentación de Lightkurve",
      aria_nea: "NASA Exoplanet Archive",
      about_eyebrow: "Acerca de",
      about_title: "Sobre este proyecto",
      about_lead:
        "Exoplot reúne catálogos y curvas de luz de misiones en un espacio sereno: gráficos legibles, Python reproducible y menos pestañas abiertas.",
      about_card_catalog_title: "Catálogos explorables",
      about_card_catalog_body:
        "Filtra mundos publicados y compara mediciones clave sin perder el contexto global.",
      about_card_curves_title: "Curvas de luz claras",
      about_card_curves_body:
        "Sigue tránsitos y variabilidad con un diseño pensado para series largas de Kepler, TESS y similares.",
      about_card_open_title: "Pila transparente",
      about_card_open_body:
        "Módulos Python abiertos: inspecciona, amplía y reproduce cada figura.",
      project_lead:
        "Aquí encontrarás pronto la historia de Exoplot, notas técnicas y breves explicaciones científicas — por ahora es solo una página provisional.",
    },
    de: {
      meta_title: "Exoplot",
      meta_description:
        "Exoplanetendaten visuell erkunden: Kataloge, Lichtkurven und klare Diagramme an einem Ort.",
      nav_home: "Start",
      nav_analyse: "Analyse",
      nav_discover: "Entdecken",
      hero_eyebrow: "Für neugierige Blicke",
      hero_title: "Exoplaneten visuell erkunden",
      hero_subtitle:
        "Welten jenseits des Sonnensystems durchstöbern, Eigenschaften vergleichen und echte Missions-Lichtkurven verfolgen.",
      btn_analyse: "Analyse",
      btn_discover: "Entdecken",
      footer_resources: "Nützliche Links",
      link_nea: "NASA Exoplanet Archive",
      link_lightkurve: "Lightkurve",
      link_github: "GitHub",
      link_linkedin: "LinkedIn",
      footer_lang: "Sprache",
      footer_credit: "Erstellt von Simon Wittmann",
      logo_credit: "made by A. Wittmann",
      theme_toggle_aria: "Zwischen dunklem und hellem Thema wechseln",
      aria_github: "Exoplot auf GitHub",
      aria_linkedin: "LinkedIn-Profil",
      aria_lightkurve: "Lightkurve-Dokumentation",
      aria_nea: "NASA Exoplanet Archive",
      about_eyebrow: "Über",
      about_title: "Über dieses Projekt",
      about_lead:
        "Exoplot bündelt Kataloge und Missions-Lichtkurven in einer ruhigen Oberfläche — lesbare Plots, reproduzierbares Python, weniger Tab-Chaos.",
      about_card_catalog_title: "Durchsuchbare Kataloge",
      about_card_catalog_body:
        "Filtere veröffentlichte Welten und vergleiche Kennwerte, ohne den Überblick zu verlieren.",
      about_card_curves_title: "Lesbare Lichtkurven",
      about_card_curves_body:
        "Transits und Variabilität verfolgen — Layout für lange Zeitreihen von Kepler, TESS & Co.",
      about_card_open_title: "Offener Stack",
      about_card_open_body:
        "Offene Python-Module — inspizieren, erweitern und jede Abbildung nachvollziehen.",
      project_lead:
        "Hier gibt es bald die Geschichte von Exoplot, technische Notizen und kurze wissenschaftliche Erklärungen — derzeit ist dies nur eine Platzhalterseite.",
    },
    ja: {
      meta_title: "Exoplot",
      meta_description:
        "系外惑星データを視覚的に探索：カタログ、光度曲線、すっきりしたプロットをひとつの場所で。",
      nav_home: "ホーム",
      nav_analyse: "解析",
      nav_discover: "発見",
      hero_eyebrow: "宇宙に興味のある方へ",
      hero_title: "系外惑星をビジュアルに探る",
      hero_subtitle:
        "太陽系の外の世界を検索し、性質を比べ、実際のミッションの光度曲線を追跡できます。",
      btn_analyse: "解析",
      btn_discover: "発見",
      footer_resources: "便利なリンク",
      link_nea: "NASA Exoplanet Archive",
      link_lightkurve: "Lightkurve",
      link_github: "GitHub",
      link_linkedin: "LinkedIn",
      footer_lang: "言語",
      footer_credit: "Simon Wittmann 制作",
      logo_credit: "made by A. Wittmann",
      theme_toggle_aria: "ダークテーマとライトテーマを切り替え",
      aria_github: "GitHub の Exoplot",
      aria_linkedin: "LinkedIn プロフィール",
      aria_lightkurve: "Lightkurve ドキュメント",
      aria_nea: "NASA Exoplanet Archive",
      about_eyebrow: "概要",
      about_title: "このプロジェクトについて",
      about_lead:
        "Exoplot は星表とミッションの光度曲線を落ち着いたワークスペースにまとめます。読みやすいプロット、再現可能な Python、タブ切り替えの削減を目指します。",
      about_card_catalog_title: "探索できるカタログ",
      about_card_catalog_body:
        "公開データから惑星を絞り込み、重要な測定値を比較しても全体像を失いません。",
      about_card_curves_title: "読みやすい光度曲線",
      about_card_curves_body:
        "トランジットと変動を、Kepler や TESS などの長い時系列向けに整えたレイアウトで追跡できます。",
      about_card_open_title: "透明なスタック",
      about_card_open_body:
        "下層はオープンな Python モジュール。検証・拡張・図の再現が可能です。",
      project_lead:
        "まもなく Exoplot の背景、技術メモ、短い科学解説を掲載予定です。現在はプレースホルダーページです。",
    },
    zh: {
      meta_title: "Exoplot",
      meta_description:
        "直观地探索系外行星数据：星表、光变曲线与清晰图表，集中在一处。",
      nav_home: "首页",
      nav_analyse: "分析",
      nav_discover: "发现",
      hero_eyebrow: "献给好奇的观星者",
      hero_title: "可视化探索系外行星",
      hero_subtitle:
        "浏览太阳系以外的世界，比较它们的性质，并跟踪真实任务的光变曲线.",
      btn_analyse: "分析",
      btn_discover: "发现",
      footer_resources: "实用链接",
      link_nea: "NASA Exoplanet Archive",
      link_lightkurve: "Lightkurve",
      link_github: "GitHub",
      link_linkedin: "领英",
      footer_lang: "语言",
      footer_credit: "由 Simon Wittmann 制作",
      logo_credit: "made by A. Wittmann",
      theme_toggle_aria: "在深色与浅色主题之间切换",
      aria_github: "GitHub 上的 Exoplot",
      aria_linkedin: "LinkedIn 主页",
      aria_lightkurve: "Lightkurve 文档",
      aria_nea: "NASA Exoplanet Archive",
      about_eyebrow: "关于",
      about_title: "关于本项目",
      about_lead:
        "Exoplot 将星表与任务光变曲线集中在一处清晰的工作区：易读的图表、可复现的 Python，并减少来回切换标签页。",
      about_card_catalog_title: "可浏览的星表",
      about_card_catalog_body:
        "筛选已发表的世界并比较关键测量，同时保持全局视角。",
      about_card_curves_title: "清晰的光变曲线",
      about_card_curves_body:
        "以适合 Kepler、TESS 等长时序的布局跟踪凌星与变星信号。",
      about_card_open_title: "透明的技术栈",
      about_card_open_body:
        "底层为开放的 Python 模块——检查、扩展并重现每一幅图。",
      project_lead:
        "这里很快会介绍 Exoplot 的由来、技术说明与简短科学讲解——目前仅为占位页面。",
    },
    ru: {
      meta_title: "Exoplot",
      meta_description:
        "Наглядно изучайте данные об экзопланетах: каталоги, кривые блеска и понятные графики в одном месте.",
      nav_home: "Главная",
      nav_analyse: "Анализ",
      nav_discover: "Обзор",
      hero_eyebrow: "Для любознательных",
      hero_title: "Исследуйте экзопланеты наглядно",
      hero_subtitle:
        "Просматривайте миры за пределами Солнечной системы, сравнивайте их свойства и следите за реальными кривыми блеска миссий.",
      btn_analyse: "Анализ",
      btn_discover: "Обзор",
      footer_resources: "Полезные ссылки",
      link_nea: "NASA Exoplanet Archive",
      link_lightkurve: "Lightkurve",
      link_github: "GitHub",
      link_linkedin: "LinkedIn",
      footer_lang: "Язык",
      footer_credit: "Сделано Simon Wittmann",
      logo_credit: "made by A. Wittmann",
      theme_toggle_aria: "Переключить тёмную и светлую тему",
      aria_github: "Exoplot на GitHub",
      aria_linkedin: "Профиль LinkedIn",
      aria_lightkurve: "Документация Lightkurve",
      aria_nea: "NASA Exoplanet Archive",
      about_eyebrow: "О проекте",
      about_title: "Об этом проекте",
      about_lead:
        "Exoplot собирает каталоги и кривые блеска миссий в одном спокойном интерфейсе — читаемые графики, воспроизводимый Python и меньше вкладок.",
      about_card_catalog_title: "Обозримые каталоги",
      about_card_catalog_body:
        "Фильтруйте опубликованные миры и сравнивайте ключевые величины, не теряя общей картины.",
      about_card_curves_title: "Понятные кривые блеска",
      about_card_curves_body:
        "Следите за транзитами и вариабельностью в вёрстке для длинных рядов Kepler, TESS и аналогов.",
      about_card_open_title: "Открытый стек",
      about_card_open_body:
        "Открытые Python-модули — проверяйте, расширяйте и воспроизводите каждый рисунок.",
      project_lead:
        "Здесь скоро появятся история Exoplot, технические заметки и короткие научные пояснения — сейчас это только заглушка.",
    },
    pt: {
      meta_title: "Exoplot",
      meta_description:
        "Explore dados de exoplanetas visualmente: catálogos, curvas de luz e gráficos claros num só sítio.",
      nav_home: "Início",
      nav_analyse: "Análise",
      nav_discover: "Descobrir",
      hero_eyebrow: "Para quem olha o céu",
      hero_title: "Explore exoplanetas visualmente",
      hero_subtitle:
        "Percorra mundos fora do sistema solar, compare propriedades e acompanhe curvas de luz reais de missões.",
      btn_analyse: "Análise",
      btn_discover: "Descobrir",
      footer_resources: "Ligações úteis",
      link_nea: "NASA Exoplanet Archive",
      link_lightkurve: "Lightkurve",
      link_github: "GitHub",
      link_linkedin: "LinkedIn",
      footer_lang: "Idioma",
      footer_credit: "Feito por Simon Wittmann",
      logo_credit: "made by A. Wittmann",
      theme_toggle_aria: "Alternar entre tema escuro e claro",
      aria_github: "Exoplot no GitHub",
      aria_linkedin: "Perfil no LinkedIn",
      aria_lightkurve: "Documentação Lightkurve",
      aria_nea: "NASA Exoplanet Archive",
      about_eyebrow: "Sobre",
      about_title: "Sobre este projeto",
      about_lead:
        "O Exoplot junta catálogos e curvas de luz de missões num espaço calmo — gráficos legíveis, Python reproduzível e menos separadores abertos.",
      about_card_catalog_title: "Catálogos navegáveis",
      about_card_catalog_body:
        "Filtre mundos publicados e compare grandezas-chave sem perder o panorama.",
      about_card_curves_title: "Curvas de luz claras",
      about_card_curves_body:
        "Acompanhe trânsitos e variabilidade com um layout pensado para séries longas do Kepler, TESS e similares.",
      about_card_open_title: "Stack transparente",
      about_card_open_body:
        "Módulos Python abertos por baixo — inspecione, estenda e reproduza cada figura.",
      project_lead:
        "Em breve encontrará a história do Exoplot, notas técnicas e pequenos textos de divulgação científica — por agora é apenas uma página provisória.",
    },
    ko: {
      meta_title: "Exoplot",
      meta_description:
        "외계 행성 데이터를 시각적으로 탐색: 카탈로그, 밝기 곡선, 한눈에 보는 플롯을 한곳에서.",
      nav_home: "홈",
      nav_analyse: "분석",
      nav_discover: "탐색",
      hero_eyebrow: "하늘을 향한 호기심에게",
      hero_title: "외계 행성을 시각적으로 탐험하세요",
      hero_subtitle:
        "태양계 밖의 세계를 찾아보고 성질을 비교하며 실제 임무의 밝기 곡선을 따라갈 수 있습니다. 여러 도구를 옮겨 다닐 필요가 없습니다.",
      btn_analyse: "분석",
      btn_discover: "탐색",
      footer_resources: "유용한 링크",
      link_nea: "NASA Exoplanet Archive",
      link_lightkurve: "Lightkurve",
      link_github: "GitHub",
      link_linkedin: "LinkedIn",
      footer_lang: "언어",
      footer_credit: "Simon Wittmann 제작",
      logo_credit: "made by A. Wittmann",
      theme_toggle_aria: "다크 테마와 라이트 테마 전환",
      aria_github: "GitHub의 Exoplot",
      aria_linkedin: "LinkedIn 프로필",
      aria_lightkurve: "Lightkurve 문서",
      aria_nea: "NASA Exoplanet Archive",
      about_eyebrow: "소개",
      about_title: "이 프로젝트에 대해",
      about_lead:
        "Exoplot은 성표와 임무 밝기 곡선을 한곳의 차분한 작업 공간에 모읍니다. 읽기 쉬운 플롯, 재현 가능한 Python, 탭 전환 감소를 목표로 합니다.",
      about_card_catalog_title: "탐색 가능한 성표",
      about_card_catalog_body:
        "발표된 세계를 필터링하고 핵심 측정값을 비교해도 큰 그림을 잃지 않습니다.",
      about_card_curves_title: "읽기 쉬운 밝기 곡선",
      about_card_curves_body:
        "Kepler, TESS 등 긴 시계열에 맞춘 레이아웃으로 린저와 변광을 추적합니다.",
      about_card_open_title: "투명한 스택",
      about_card_open_body:
        "아래는 열린 Python 모듈 — 검사, 확장, 모든 그림 재현이 가능합니다.",
      project_lead:
        "곧 Exoplot의 배경, 기술 노트, 짧은 과학 설명이 올라올 예정입니다. 지금은 자리 표시자 페이지입니다.",
    },
  };

  var HTML_LANG = {
    en: "en",
    fr: "fr",
    es: "es",
    de: "de",
    ja: "ja",
    zh: "zh-CN",
    ru: "ru",
    pt: "pt",
    ko: "ko",
  };

  function getLocale() {
    try {
      var stored = localStorage.getItem("exoplot-lang");
      if (stored && STRINGS[stored]) return stored;
    } catch (e) {}
    var raw = (global.navigator && global.navigator.language) || "en";
    var nav = raw.toLowerCase();
    if (nav.startsWith("fr")) return "fr";
    if (nav.startsWith("es")) return "es";
    if (nav.startsWith("de")) return "de";
    if (nav.startsWith("ja")) return "ja";
    if (nav.startsWith("zh")) return "zh";
    if (nav.startsWith("ru")) return "ru";
    if (nav.startsWith("pt")) return "pt";
    if (nav.startsWith("ko")) return "ko";
    return "en";
  }

  function setLocale(code) {
    if (!STRINGS[code]) return;
    try {
      localStorage.setItem("exoplot-lang", code);
    } catch (e) {}
    if (global.document && global.document.documentElement) {
      global.document.documentElement.lang = HTML_LANG[code] || "en";
    }
  }

  function t(key, locale) {
    var loc = locale || getLocale();
    var pack = STRINGS[loc] || STRINGS.en;
    return pack[key] != null ? pack[key] : (STRINGS.en[key] != null ? STRINGS.en[key] : key);
  }

  function apply(locale) {
    var loc = locale || getLocale();
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
      } else if ("placeholder" in el && el.hasAttribute("data-i18n-placeholder")) {
        el.placeholder = val;
      } else {
        el.textContent = val;
      }
    }
    var meta = root.querySelector('meta[name="description"]');
    if (meta) meta.setAttribute("content", t("meta_description", loc));
  }

  function getHtmlLang(locale) {
    return HTML_LANG[locale || getLocale()] || "en";
  }

  global.ExoplotI18n = {
    getLocale: getLocale,
    setLocale: setLocale,
    getHtmlLang: getHtmlLang,
    t: t,
    apply: apply,
    locales: ["en", "fr", "es", "de", "ja", "zh", "ru", "pt", "ko"],
  };
})(typeof window !== "undefined" ? window : this);
