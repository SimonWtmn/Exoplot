/**
 * Exoplot front-end: theme, language, optional LinkedIn, Lucide icons,
 * mobile navigation, scroll reveals, and hero canvas background.
 */
(function () {
  "use strict";

  var THEME_KEY = "exoplot-theme";
  var LANG_SELECT_IDS = ["lang-select", "lang-select-mobile"];

  function refreshLucideIcons() {
    if (typeof lucide !== "undefined" && lucide.createIcons) {
      lucide.createIcons();
    }
  }

  function setMobileMenuAriaLabels(isOpen) {
    if (!window.ExoplotI18n || !ExoplotI18n.t) return;
    var btn = document.getElementById("header-menu-toggle");
    if (!btn) return;
    btn.setAttribute(
      "aria-label",
      ExoplotI18n.t(isOpen ? "menu_close_aria" : "menu_toggle_aria")
    );
  }

  function refreshHeaderChrome() {
    refreshLucideIcons();
    var wrap = document.querySelector(".site-header-wrap");
    setMobileMenuAriaLabels(
      !!(wrap && wrap.classList.contains("is-mobile-menu-open"))
    );
  }

  function getTheme() {
    try {
      var v = localStorage.getItem(THEME_KEY);
      if (v === "light" || v === "dark") return v;
    } catch (e) {}
    if (
      typeof matchMedia === "function" &&
      matchMedia("(prefers-color-scheme: light)").matches
    ) {
      return "light";
    }
    return "dark";
  }

  function setTheme(mode) {
    var root = document.documentElement;
    if (mode === "light") {
      root.setAttribute("data-theme", "light");
    } else {
      root.removeAttribute("data-theme");
    }
    try {
      localStorage.setItem(THEME_KEY, mode);
    } catch (e) {}
    root.style.colorScheme = mode === "light" ? "light" : "dark";
    updateThemeToggle(mode);
    try {
      window.dispatchEvent(new CustomEvent("exoplot-theme", { detail: { theme: mode } }));
    } catch (err) {}
  }

  function updateThemeToggle(mode) {
    var btn = document.getElementById("theme-toggle");
    if (btn) btn.setAttribute("aria-pressed", mode === "dark" ? "true" : "false");
  }

  function wireLinkedIn() {
    var cfg = window.EXOPLOT_SITE_CONFIG || {};
    var url = (cfg.linkedinProfile || "").trim();
    var list = [
      document.getElementById("footer-linkedin"),
      document.getElementById("header-linkedin"),
      document.getElementById("header-mobile-linkedin"),
    ];
    var i;
    for (i = 0; i < list.length; i++) {
      var a = list[i];
      if (!a) continue;
      if (url) {
        a.href = url;
        a.removeAttribute("hidden");
      } else {
        a.setAttribute("hidden", "");
      }
    }
  }

  function fillLangSelect() {
    var selects = LANG_SELECT_IDS.map(function (id) {
      return document.getElementById(id);
    }).filter(Boolean);
    if (!selects.length || !window.ExoplotI18n) return;
    var labels = {
      en: "English",
      fr: "Français",
      es: "Español",
      de: "Deutsch",
      ja: "日本語",
      zh: "中文",
      ru: "Русский",
      pt: "Português",
      ko: "한국어",
    };
    var locales = ExoplotI18n.locales;
    var current = ExoplotI18n.getLocale();
    var s;
    for (s = 0; s < selects.length; s++) {
      var sel = selects[s];
      sel.innerHTML = "";
      var i;
      for (i = 0; i < locales.length; i++) {
        var code = locales[i];
        var opt = document.createElement("option");
        opt.value = code;
        opt.textContent = labels[code] || code;
        if (code === current) opt.selected = true;
        sel.appendChild(opt);
      }
    }
  }

  function syncLangSelects(value) {
    var i;
    for (i = 0; i < LANG_SELECT_IDS.length; i++) {
      var el = document.getElementById(LANG_SELECT_IDS[i]);
      if (el) el.value = value;
    }
  }

  function handleLangChange(ev) {
    var v = ev.target.value;
    syncLangSelects(v);
    if (window.ExoplotI18n) {
      ExoplotI18n.setLocale(v);
      ExoplotI18n.apply(v);
    }
    refreshHeaderChrome();
  }

  function initChrome() {
    setTheme(getTheme());
    wireLinkedIn();

    var loc = window.ExoplotI18n ? ExoplotI18n.getLocale() : "en";
    if (document.documentElement && window.ExoplotI18n && ExoplotI18n.getHtmlLang) {
      document.documentElement.lang = ExoplotI18n.getHtmlLang(loc);
    }
    fillLangSelect();
    if (window.ExoplotI18n) ExoplotI18n.apply(loc);

    var themeBtn = document.getElementById("theme-toggle");
    if (themeBtn) {
      themeBtn.addEventListener("click", function () {
        var next = getTheme() === "dark" ? "light" : "dark";
        setTheme(next);
      });
    }

    var langDesktop = document.getElementById("lang-select");
    var langMobile = document.getElementById("lang-select-mobile");
    if (langDesktop) langDesktop.addEventListener("change", handleLangChange);
    if (langMobile) langMobile.addEventListener("change", handleLangChange);

    refreshHeaderChrome();
  }

  function prefersReducedMotion() {
    return (
      typeof matchMedia === "function" &&
      matchMedia("(prefers-reduced-motion: reduce)").matches
    );
  }

  function initScrollReveals() {
    var reduceMotion = prefersReducedMotion();

    var nodes = document.querySelectorAll(".reveal-on-scroll");
    if (!nodes.length) return;

    if (reduceMotion) {
      var j;
      for (j = 0; j < nodes.length; j++) nodes[j].classList.add("is-visible");
      return;
    }

    var io = new IntersectionObserver(
      function (entries) {
        var i;
        for (i = 0; i < entries.length; i++) {
          if (entries[i].isIntersecting) {
            entries[i].target.classList.add("is-visible");
            io.unobserve(entries[i].target);
          }
        }
      },
      { root: null, rootMargin: "0px 0px -8% 0px", threshold: 0.08 }
    );

    var k;
    for (k = 0; k < nodes.length; k++) io.observe(nodes[k]);
  }

  function initMobileNav() {
    var wrap = document.querySelector(".site-header-wrap");
    var toggle = document.getElementById("header-menu-toggle");
    var panel = document.getElementById("header-mobile-panel");
    if (!wrap || !toggle || !panel) return;

    function setOpen(open) {
      wrap.classList.toggle("is-mobile-menu-open", open);
      panel.setAttribute("aria-hidden", open ? "false" : "true");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      setMobileMenuAriaLabels(open);
      refreshLucideIcons();
    }

    setOpen(false);

    toggle.addEventListener("click", function (e) {
      e.stopPropagation();
      setOpen(!wrap.classList.contains("is-mobile-menu-open"));
    });

    document.addEventListener("click", function (e) {
      if (!wrap.classList.contains("is-mobile-menu-open")) return;
      if (!wrap.contains(e.target)) setOpen(false);
    });

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") setOpen(false);
    });

    window.addEventListener(
      "resize",
      function () {
        if (window.innerWidth > 768 && wrap.classList.contains("is-mobile-menu-open")) {
          setOpen(false);
        }
      },
      { passive: true }
    );

    var links = panel.querySelectorAll("a");
    var i;
    for (i = 0; i < links.length; i++) {
      links[i].addEventListener("click", function () {
        setOpen(false);
      });
    }
  }

  function initDeepSky() {
    var canvas = document.getElementById("stars-canvas");
    if (!canvas || !canvas.getContext) return;

    var ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    var bodies = [];
    var rafId = 0;
    var lastW = 0;
    var lastH = 0;
    var rgb = "244, 244, 245";
    var reduceMotion = prefersReducedMotion();

    var mouseTx = 0;
    var mouseTy = 0;
    var mouseX = 0;
    var mouseY = 0;
    var hasMouse = false;

    function clamp(n, min, max) {
      return Math.min(max, Math.max(min, n));
    }

    function readStarRgb() {
      try {
        var raw = getComputedStyle(document.documentElement)
          .getPropertyValue("--star-rgb")
          .trim();
        if (raw) return raw;
      } catch (e) {}
      return "244, 244, 245";
    }

    function parseRgbTriplet(str) {
      var p = str.split(",").map(function (x) {
        return parseFloat(x.trim()) || 0;
      });
      return { r: p[0] || 200, g: p[1] || 200, b: p[2] || 210 };
    }

    function tripletToString(t) {
      return Math.round(t.r) + "," + Math.round(t.g) + "," + Math.round(t.b);
    }

    function pickDpr() {
      return clamp(typeof devicePixelRatio === "number" ? devicePixelRatio : 1, 1, 2);
    }

    function budgetForViewport(w, h) {
      var area = w * h;
      var isLight = document.documentElement.getAttribute("data-theme") === "light";
      var density = isLight ? 2100 : 1750;
      var nStars = clamp(Math.floor(area / density), 420, 1180);
      var nMoons = clamp(Math.floor(nStars * 0.045), 18, 52);
      var nPlanets = clamp(Math.floor(nStars * 0.055), 22, 60);
      return { nStars: nStars, nMoons: nMoons, nPlanets: nPlanets };
    }

    function initBodies(w, h) {
      bodies.length = 0;
      rgb = readStarRgb();
      var baseTri = parseRgbTriplet(rgb);
      var b = budgetForViewport(w, h);
      var i;

      for (i = 0; i < b.nStars; i++) {
        bodies.push({
          kind: "star",
          ox: Math.random() * w,
          oy: Math.random() * h,
          r: Math.random() * 1.05 + 0.18,
          base: 0.18 + Math.random() * 0.48,
          phase: Math.random() * Math.PI * 2,
          twSpeed: 0.05 + Math.random() * 0.14,
          jolt: 0,
          depth: 0.5 + Math.random() * 1,
        });
      }

      for (i = 0; i < b.nMoons; i++) {
        bodies.push({
          kind: "moon",
          ox: Math.random() * w,
          oy: Math.random() * h,
          r: Math.random() * 2.2 + 1.1,
          base: 0.28 + Math.random() * 0.35,
          phase: Math.random() * Math.PI * 2,
          twSpeed: 0.04 + Math.random() * 0.1,
          jolt: 0,
          depth: 0.45 + Math.random() * 0.55,
          crescentAngle: Math.random() * Math.PI * 2,
        });
      }

      for (i = 0; i < b.nPlanets; i++) {
        var pr = Math.random() * 1.85 + 0.95;
        var tintR = baseTri.r + (Math.random() - 0.5) * 42;
        var tintG = baseTri.g + (Math.random() - 0.5) * 38;
        var tintB = baseTri.b + (Math.random() - 0.15) * 50;
        if (Math.random() > 0.55) {
          tintR += 18;
          tintG += 8;
          tintB -= 12;
        } else {
          tintR -= 10;
          tintG += 22;
          tintB += 25;
        }
        bodies.push({
          kind: "planet",
          ox: Math.random() * w,
          oy: Math.random() * h,
          r: pr,
          base: 0.22 + Math.random() * 0.32,
          phase: Math.random() * Math.PI * 2,
          twSpeed: 0.035 + Math.random() * 0.09,
          jolt: 0,
          depth: 0.55 + Math.random() * 0.9,
          planetRgb: tripletToString({
            r: clamp(tintR, 0, 255),
            g: clamp(tintG, 0, 255),
            b: clamp(tintB, 0, 255),
          }),
          ringAngle: Math.random() * Math.PI * 2,
          hasRing: Math.random() > 0.62,
        });
      }

      mouseTx = w * 0.5;
      mouseTy = h * 0.5;
      mouseX = mouseTx;
      mouseY = mouseTy;
    }

    function resize() {
      var dpr = pickDpr();
      var w = window.innerWidth;
      var h = window.innerHeight;
      if (w === lastW && h === lastH) return;
      lastW = w;
      lastH = h;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      initBodies(w, h);
    }

    var resizeScheduled = false;
    function onResize() {
      if (resizeScheduled) return;
      resizeScheduled = true;
      requestAnimationFrame(function () {
        resizeScheduled = false;
        resize();
      });
    }

    function onThemeChange() {
      rgb = readStarRgb();
      if (lastW && lastH) initBodies(lastW, lastH);
    }

    function onPointerMove(e) {
      hasMouse = true;
      mouseTx = e.clientX;
      mouseTy = e.clientY;
    }

    function onTouchMove(e) {
      if (!e.touches || !e.touches.length) return;
      hasMouse = true;
      mouseTx = e.touches[0].clientX;
      mouseTy = e.touches[0].clientY;
    }

    function onPointerLeave() {
      hasMouse = false;
    }

    var t0 = performance.now();

    function drawStar(x, y, s, alpha) {
      ctx.fillStyle = "rgba(" + rgb + "," + alpha.toFixed(3) + ")";
      ctx.beginPath();
      ctx.arc(x, y, s.r, 0, Math.PI * 2);
      ctx.fill();
    }

    function drawMoon(x, y, s, alpha) {
      var a = clamp(alpha * 0.88, 0.05, 0.92);
      ctx.save();
      ctx.fillStyle = "rgba(" + rgb + "," + a.toFixed(3) + ")";
      ctx.beginPath();
      ctx.arc(x, y, s.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalCompositeOperation = "destination-out";
      var ox = Math.cos(s.crescentAngle) * s.r * 0.42;
      var oy = Math.sin(s.crescentAngle) * s.r * 0.42;
      ctx.beginPath();
      ctx.arc(x + ox, y + oy, s.r * 0.92, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    function drawPlanet(x, y, s, alpha) {
      var a = clamp(alpha, 0.06, 0.95);
      ctx.save();
      if (s.hasRing) {
        ctx.strokeStyle = "rgba(" + s.planetRgb + "," + (a * 0.45).toFixed(3) + ")";
        ctx.lineWidth = 0.55;
        ctx.beginPath();
        ctx.ellipse(x, y, s.r * 2.1, s.r * 0.38, s.ringAngle, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.fillStyle = "rgba(" + s.planetRgb + "," + (a * 0.72).toFixed(3) + ")";
      ctx.beginPath();
      ctx.arc(x, y, s.r * 0.88, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(" + s.planetRgb + "," + (a * 0.35).toFixed(3) + ")";
      ctx.lineWidth = 0.35;
      ctx.stroke();
      ctx.restore();
    }

    function drawBody(x, y, s, alpha) {
      if (s.kind === "moon") drawMoon(x, y, s, alpha);
      else if (s.kind === "planet") drawPlanet(x, y, s, alpha);
      else drawStar(x, y, s, alpha);
    }

    function drawStatic() {
      var w = lastW;
      var h = lastH;
      ctx.clearRect(0, 0, w, h);
      rgb = readStarRgb();
      var i;
      for (i = 0; i < bodies.length; i++) {
        var s = bodies[i];
        drawBody(s.ox, s.oy, s, s.base);
      }
    }

    function frame(now) {
      var w = lastW;
      var h = lastH;
      var t = (now - t0) / 1000;
      ctx.clearRect(0, 0, w, h);
      rgb = readStarRgb();

      mouseX += (mouseTx - mouseX) * 0.07;
      mouseY += (mouseTy - mouseY) * 0.07;

      var cx = hasMouse ? mouseX : w * 0.5;
      var cy = hasMouse ? mouseY : h * 0.5;
      var parallaxX = (cx - w * 0.5) * 0.014;
      var parallaxY = (cy - h * 0.5) * 0.014;

      var repelR = 140;
      var repelG = 30;

      var i;
      for (i = 0; i < bodies.length; i++) {
        var s = bodies[i];
        s.jolt += (Math.random() - 0.5) * 0.014;
        s.jolt *= 0.968;

        var px = parallaxX * s.depth;
        var py = parallaxY * s.depth;
        var x = s.ox + px;
        var y = s.oy + py;

        var dx = x - cx;
        var dy = y - cy;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < repelR && dist > 0.5) {
          var push = ((repelR - dist) / repelR) * repelG;
          var inv = push / dist;
          x += dx * inv * 0.12;
          y += dy * inv * 0.12;
        }

        var shimmer =
          s.jolt +
          0.1 * Math.sin(t * s.twSpeed + s.phase) +
          0.035 * Math.sin(t * s.twSpeed * 1.08 + s.phase * 0.65);
        var a = clamp(s.base + shimmer, 0.04, 0.97);

        drawBody(x, y, s, a);
      }
      rafId = requestAnimationFrame(frame);
    }

    function start() {
      resize();
      window.addEventListener("resize", onResize, { passive: true });
      window.addEventListener("exoplot-theme", onThemeChange);
      try {
        var obs = new MutationObserver(onThemeChange);
        obs.observe(document.documentElement, {
          attributes: true,
          attributeFilter: ["data-theme"],
        });
      } catch (e) {}

      if (!reduceMotion) {
        window.addEventListener("mousemove", onPointerMove, { passive: true });
        window.addEventListener("touchmove", onTouchMove, { passive: true });
        document.body.addEventListener("mouseleave", onPointerLeave);
      }

      if (reduceMotion) {
        drawStatic();
        return;
      }
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(frame);
    }

    start();
  }

  function init() {
    initChrome();
    initDeepSky();
    initScrollReveals();
    initMobileNav();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
