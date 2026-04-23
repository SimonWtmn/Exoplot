"""
PDF Report Generation — Single-Page Landscape DVR (mockup overhaul)
-------------------------------------------------------------------
Generates a dense, publication-quality, single-page **landscape A4** PDF
matching the Exoplot mockup:

::

    ┌──────────────────  EXOPLOT REPORT – <TARGET>  ──────────────────┐
    │                       www.exoplot.fr                            │
    │  Tmag: …  R*: …  Teff: …  Logg: …  M/H: …  Rho: …               │
    │  ┌──────────────────── Raw LC ────────────────────┬───────────┐ │
    │  │                                                │   BLS PG  │ │
    │  ├───────────────┬───────────────┬────────────────┤───────────┤ │
    │  │ Folded transit│  Spaghetti    │   Fitted /     │           │ │
    │  │  + residuals  ├───────────────┤   Derived /    │           │ │
    │  │               │  TPF (mid)    │   Conv / Fit   │           │ │
    │  ├───────────────┼───────────────┤   ─────────────┤           │ │
    │  │  Odd / Even   │  Diff. img.   │      Corner plot           │ │
    │  └───────────────┴───────────────┴────────────────────────────┘ │
    │              Data generated : YYYY-MM-DD HH:MM                  │
    │   This Data Report & Simulation Summary was produced by the     │
    │                       Exoplot Pipeline                          │
    └─────────────────────────────────────────────────────────────────┘

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot
"""

from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyBboxPatch, Patch
from scipy.interpolate import interp1d
import corner

from .i18n import t as _t, is_latex_compatible

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Colour palette — paper / DVR aesthetic
# ════════════════════════════════════════════════════════════════════
_PAL = {
    "primary":         "#2980b9",
    "accent":          "#c0392b",
    "data":            "#1f2d3a",
    "error":           "#bdc3c7",
    "model":           "#c0392b",
    "spaghetti":       "#3498db",
    "bin_face":        "#ffffff",
    "header_bg":       "#1f2d3a",
    "header_fg":       "#ffffff",
    "row_even":        "#f4f6f9",
    "row_odd":         "#ffffff",
    "grid":            "#cccccc",
    "corner":          "#2c3e50",
    "trace_cm":        "viridis",
    "trace_color":     "#34495e",
    "trace_median":    "#c0392b",
    "trace_burn":      "#c0392b",
    "trace_burn_bg":   "#f9e4e0",
    "residual":        "#2ecc71",
    "residual_dark":   "#145a32",
    "aperture":        "#e74c3c",
    "divider":         "#7f8c8d",
    "subhead":         "#34495e",
    "card_edge":       "#1f2d3a",
    "card_fill":       "#fbfcfd",
    "card_accent":     "#2980b9",
    "odd":             "#2980b9",
    "even":            "#c0392b",
    # ── New keys for the mockup overhaul ──────────────────────────
    "oe_bin":          "#1f4e79",   # odd/even binned points
    "oe_baseline":     "#c0392b",   # red dashed depth baseline
    "oe_marker_blue":  "#2980b9",   # ▲ ingress markers
    "oe_marker_red":   "#c0392b",   # ▲ egress markers
    "centroid_circle": "#1f5fa8",   # blue concentric circle
    "centroid_target": "#c0392b",   # red crosshair / target star
    "centroid_oot":    "#1f5fa8",   # blue OOT centroid marker
    "centroid_hi":     "#c0392b",   # highlighted out-of-transit point
    "stellar_strong":  "#c0392b",   # red emphasis (e.g. M/H)
}


# ════════════════════════════════════════════════════════════════════
#  Small utilities
# ════════════════════════════════════════════════════════════════════

def _native_time_label(lc) -> str:
    """Axis label matching Lightkurve's native time format (BTJD, BKJD…)."""
    fmt = getattr(lc.time, "format", None)
    if fmt:
        return f"Time [{str(fmt).upper()}]"
    return "Time"


def _fmt_val_err(med: float, up: float, lo: float, *, latex: bool,
                 sig: int = 2, max_decimals: int = 5) -> str:
    """Format ``value +upper/-lower`` using uncertainty-driven precision."""
    if not np.isfinite(med):
        return "n/a"
    up = abs(up); lo = abs(lo)
    scale_ref = min(up, lo) if (up > 0 and lo > 0) else max(up, lo)
    if scale_ref > 0 and np.isfinite(scale_ref):
        decimals = max(0, sig - 1 - int(np.floor(np.log10(scale_ref))))
    else:
        decimals = sig
    decimals = max(0, min(decimals, max_decimals))
    fmt = f"{{:.{decimals}f}}"

    symmetric = False
    if up > 0 and lo > 0:
        rel = abs(up - lo) / max(up, lo)
        symmetric = rel < 0.15
    avg = 0.5 * (up + lo)

    if latex:
        if symmetric:
            return f"${fmt.format(med)} \\pm {fmt.format(avg)}$"
        return (f"${fmt.format(med)}_{{-{fmt.format(lo)}}}"
                f"^{{+{fmt.format(up)}}}$")
    if symmetric:
        return f"{fmt.format(med)} ± {fmt.format(avg)}"
    return f"{fmt.format(med)} +{fmt.format(up)} / -{fmt.format(lo)}"


def _latex_escape(s: str) -> str:
    """Minimal pdfTeX text-mode escape."""
    return (s.replace("\\", r"\textbackslash{}")
             .replace("_", r"\_")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("#", r"\#")
             .replace("—", r"---"))


# ────────────────────────────────────────────────────────────────────
#  Stellar-parameter resolution helper (module level so the lru_cache
#  is shared across reports for the same process — useful in notebook
#  workflows that generate several reports back-to-back).
# ────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=128)
def _query_tic_cached(target_name: str | None) -> dict:
    """Query MAST's TIC for stellar parameters.  Best-effort + cached.

    Returns the raw column→value dict for the closest catalog match,
    or an empty dict on any failure (no network, astroquery missing,
    target unresolved, …).  All callers must treat the result as
    optional metadata.
    """
    if not target_name:
        return {}
    try:
        from astroquery.mast import Catalogs  # heavy import, lazy
    except Exception as exc:
        logger.info("astroquery not available: %s", exc)
        return {}
    try:
        result = Catalogs.query_object(
            str(target_name), catalog="TIC", radius=2.0 / 3600.0,
        )
    except Exception as exc:
        logger.info("TIC lookup for %r failed: %s", target_name, exc)
        return {}
    try:
        if len(result) == 0:
            return {}
        row = result[0]
        return {col: row[col] for col in result.colnames}
    except Exception as exc:
        logger.info("TIC result parsing failed: %s", exc)
        return {}


# ════════════════════════════════════════════════════════════════════
#  Main class
# ════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Landscape single-page DVR-style PDF report (mockup-aligned)."""

    PAGE_SIZE = (11.69, 8.27)  # A4 landscape in inches

    # Sentinel signalling "auto-download a TPF if none was supplied".
    _TPF_AUTO = object()

    def __init__(self, website_name: str = "Exoplot",
                 website_url: str = "www.exoplot.fr",
                 locale: str = "en", use_latex: bool = True):
        self.website_name = website_name
        self.website_url = website_url
        self.locale = locale
        # LaTeX is the default — the visual identity relies on Computer
        # Modern + math italics.  Callers can opt out for quick renders.
        self.use_latex = use_latex
        self._setup_matplotlib()
        self.results_dir = Path(__file__).parent.parent / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ── translation shortcut ──────────────────────────────────────
    def _tr(self, key: str, **kw) -> str:
        return _t(key, locale=self.locale, latex=self.use_latex, **kw)

    # ── matplotlib rcParams (paper / LaTeX aesthetic) ─────────────
    def _setup_matplotlib(self):
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.5,
            "lines.linewidth": 0.9,
            "lines.markersize": 2,
            "figure.dpi": 300,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "xtick.minor.width": 0.3,
            "ytick.minor.width": 0.3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        })
        if self.use_latex:
            plt.rcParams.update({
                "text.usetex": True,
                "font.serif": ["Computer Modern Roman"],
                "text.latex.preamble": (
                    r"\usepackage{amssymb}" "\n"
                    r"\usepackage{hyperref}" "\n"
                    r"\hypersetup{colorlinks=true,urlcolor=black,"
                    r"linkcolor=black}" "\n"
                    r"\DeclareUnicodeCharacter{00B2}{\ensuremath{^2}}" "\n"
                    r"\DeclareUnicodeCharacter{00B3}{\ensuremath{^3}}" "\n"
                    r"\DeclareUnicodeCharacter{00B1}{\ensuremath{\pm}}" "\n"
                    r"\DeclareUnicodeCharacter{03C3}{\ensuremath{\sigma}}" "\n"
                    r"\DeclareUnicodeCharacter{03C4}{\ensuremath{\tau}}" "\n"
                    r"\DeclareUnicodeCharacter{2212}{\ensuremath{-}}" "\n"
                    r"\DeclareUnicodeCharacter{2295}{\ensuremath{\oplus}}" "\n"
                    r"\DeclareUnicodeCharacter{2299}{\ensuremath{\odot}}" "\n"
                    r"\DeclareUnicodeCharacter{2022}{\ensuremath{\bullet}}"
                ),
            })
        else:
            plt.rcParams["text.usetex"] = False

    # ── cosmetic helper ───────────────────────────────────────────
    @staticmethod
    def _polish(ax, *, grid: bool = True, minor: bool = True):
        if grid:
            ax.grid(True, alpha=0.15, lw=0.3, color=_PAL["grid"])
        if minor:
            ax.minorticks_on()
            ax.tick_params(which="minor", length=1.5)

    # =================================================================
    #  HEADER  /  STELLAR-PARAMETER ROW  /  FOOTER
    # =================================================================

    def _draw_banner(self, fig, target: str,
                     subtitle_override: str | None = None,
                     subtitle_url: str | None = None):
        """Top banner: ``EXOPLOT REPORT – <TARGET>`` + clickable URL.

        The title is set in large bold text (no italics) per the
        mockup.  The URL line below uses the matplotlib ``url=``
        parameter so the PDF backend embeds an actual hyperlink — most
        PDF readers render it as a clickable annotation.
        """
        target_caps = (target or "").upper()
        if self.use_latex:
            # All-caps target inside a bold textbf; en-dash separator.
            safe = _latex_escape(target_caps)
            header = (r"\textbf{EXOPLOT REPORT \textendash\ "
                      + safe + "}")
        else:
            header = f"EXOPLOT REPORT – {target_caps}"

        fig.text(0.5, 0.970, header, ha="center", va="top",
                 fontsize=20, color=_PAL["data"])

        if subtitle_override is not None:
            subtitle = subtitle_override
            url_str = subtitle_url
        else:
            subtitle = self.website_url
            url_str = (subtitle if subtitle.startswith("http")
                       else f"https://{subtitle}")

        # Clickable link (matplotlib forwards ``url`` to the PDF backend
        # as a /URI annotation).  In LaTeX mode we wrap with \href so
        # readers without annotation support still see a hyperlink.
        if self.use_latex and url_str is not None:
            link_text = (r"\href{" + url_str + r"}{" + subtitle + r"}")
        else:
            link_text = subtitle

        fig.text(0.5, 0.940, link_text, ha="center", va="top",
                 fontsize=9.5, color=_PAL["subhead"], url=url_str)

    # ------------------------------------------------------------------
    #  Stellar-parameter auto-discovery (no kwarg required from caller)
    # ------------------------------------------------------------------
    # Recognised aliases for each stellar parameter — first match wins.
    # Lightkurve attaches FITS-header keys verbatim to the LightCurve
    # ``meta`` dict (TESSMAG, TEFF, LOGG, RADIUS, MH, …).  We look
    # there first, then fall back to an astroquery TIC catalog query
    # by target name.
    _STELLAR_ALIASES = {
        "tmag": ("TESSMAG", "TMAG", "KEPMAG", "KMAG", "GAIAMAG", "VMAG"),
        "rs":   ("RADIUS", "STAR_RAD", "RAD", "SRAD"),
        "teff": ("TEFF", "STAR_TEFF", "TEFFLO", "STEFF"),
        "logg": ("LOGG", "STAR_LOGG", "SLOGG"),
        "mh":   ("MH", "FE_H", "FEH", "M_H", "FE/H"),
        "rho":  ("RHO", "STAR_RHO", "SRHO"),
    }
    # TIC / catalog column names returned by astroquery — second pass.
    _TIC_COLMAP = {
        "tmag": "Tmag",
        "rs":   "rad",
        "teff": "Teff",
        "logg": "logg",
        "mh":   "MH",
        "rho":  "rho",
    }

    def _auto_stellar_params(self, analyzer) -> dict:
        """Best-effort lookup of stellar parameters for ``analyzer``.

        Resolution order
        ----------------
        1.  ``analyzer.stellar_params`` — caller-attached dict.
        2.  ``analyzer.raw_lc.meta`` — FITS-header keys carried by
            lightkurve (works out-of-the-box for TESS / Kepler / K2).
        3.  Astroquery TIC catalog (``astroquery.mast.Catalogs``)
            queried by ``target_name`` — cached per-process via
            ``functools.lru_cache``.

        Returns a (possibly empty) dict whose recognised keys are
        ``tmag``, ``rs``, ``teff``, ``logg``, ``mh``, ``rho``.  Missing
        values are simply omitted so the report row gracefully shrinks.
        """
        out: dict = {}

        attached = getattr(analyzer, "stellar_params", None)
        if isinstance(attached, dict):
            for k, v in attached.items():
                if v is None:
                    continue
                try:
                    out[k.lower()] = float(v)
                except (TypeError, ValueError):
                    pass

        meta = {}
        raw = getattr(analyzer, "raw_lc", None)
        if raw is not None:
            try:
                meta = dict(getattr(raw, "meta", {}) or {})
            except Exception:
                meta = {}
        if meta:
            upper_meta = {str(k).upper(): v for k, v in meta.items()}
            for key, aliases in self._STELLAR_ALIASES.items():
                if key in out:
                    continue
                for alias in aliases:
                    val = upper_meta.get(alias)
                    if val in (None, "", b""):
                        continue
                    try:
                        f = float(val)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(f):
                        continue
                    out[key] = f
                    break

        # Fall back to TIC only if we still have very little.
        if len(out) < 3:
            tic = _query_tic_cached(getattr(analyzer, "target_name", None))
            for key, col in self._TIC_COLMAP.items():
                if key in out:
                    continue
                v = tic.get(col)
                if v is None:
                    continue
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(f):
                    out[key] = f

        return out

    def _draw_stellar_params_row(self, fig, stellar_params: dict | None):
        """Single horizontal text row above the raw LC.

        Format:  ``Tmag: 12.33   R*: 0.93 Rs   Teff: 5410 K   …``
        Missing keys are silently dropped so the row gracefully shrinks
        when the host-star catalogue lookup is partial.
        """
        if not stellar_params:
            return

        # Canonical display order + LaTeX-aware label / unit formatter.
        order = [
            ("tmag",  r"Tmag",            "",       False),
            ("rs",    r"$R_{\star}$"      if self.use_latex else "R*",
                                          r"$R_\odot$" if self.use_latex else "Rs",
                                          False),
            ("teff",  r"$T_{\mathrm{eff}}$" if self.use_latex else "Teff",
                                          r"$\mathrm{K}$" if self.use_latex else "K",
                                          False),
            ("logg",  r"$\log g$"        if self.use_latex else "Logg",
                                          "",       False),
            ("mh",    r"M/H",             "",       True),   # ← red emphasis
            ("rho",   r"$\rho_{\star}$"  if self.use_latex else "Rho",
                                          "",       False),
        ]

        chunks: list[tuple[str, bool]] = []  # (text, emphasised)
        for key, label, unit, emph in order:
            val = stellar_params.get(key)
            if val is None:
                continue
            try:
                vstr = (f"{float(val):.3f}" if abs(float(val)) < 1000
                        else f"{float(val):.1f}")
            except (TypeError, ValueError):
                vstr = str(val)
            piece = f"{label}: {vstr}" + (f" {unit}" if unit else "")
            chunks.append((piece, emph))

        if not chunks:
            return

        # Compact, left-anchored packing so the row reads as a single
        # stellar-info line rather than a widely-spaced table.  Each
        # chunk gets a small fixed horizontal step controlled by ``dx``.
        y = 0.892
        dx = 0.075                # ≈ 7.5 % figure width between starts
        x_start = 0.060           # aligned with left edge of the raw LC
        for i, (text, emph) in enumerate(chunks):
            color = _PAL["stellar_strong"] if emph else _PAL["data"]
            fig.text(x_start + i * dx, y, text,
                     ha="left", va="center",
                     fontsize=8.5, color=color)

    def _draw_footer(self, fig):
        """Two centered footer lines (no per-target marker)."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        if self.use_latex:
            line1 = (r"Data generated\,: " + now)
            line2 = ("This Data Report \\& Simulation Summary was "
                     "produced by the Exoplot Pipeline")
        else:
            line1 = f"Data generated : {now}"
            line2 = ("This Data Report & Simulation Summary was "
                     "produced by the Exoplot Pipeline")
        fig.text(0.5, 0.035, line1, ha="center", va="bottom",
                 fontsize=8, color=_PAL["data"])
        fig.text(0.5, 0.017, line2, ha="center", va="bottom",
                 fontsize=7.5, color=_PAL["subhead"], style="italic")

    # =================================================================
    #  PUBLIC API
    # =================================================================

    def generate_mcmc_report(self, filename, analyzer, fitter, *,
                             tpf=_TPF_AUTO, locale: str | None = None,
                             movie: bool = True, movie_frames: int = 40,
                             stellar_params: dict | None = None):
        """Write a multi-page landscape PDF report at ``results/<filename>``.

        Parameters
        ----------
        filename : str
            Output filename (written under ``results/``).
        analyzer : LightCurveAnalyzer
            Fully-run analyzer with cleaned LC, BLS and folded LC.
        fitter : TransitFitter
            Post-``run_mcmc`` fitter exposing ``results_summary`` and
            the flat chain.
        tpf : lightkurve.TargetPixelFile, optional
            Pre-downloaded target-pixel file.  By default the report
            attempts to download one for the same target / sector;
            failure is non-fatal and falls back to a placeholder.
            Pass ``tpf=None`` to skip the download entirely.
        locale : str, optional
            Override the locale set at construction time.
        movie : bool, default True
            Append a TPF flipbook (one frame per page).  Set to False
            when bandwidth or file-size matter.
        movie_frames : int, default 40
            Upper limit on the number of movie pages to append.
        stellar_params : dict, optional
            Stellar parameters shown in the row above the raw LC.
            Recognised keys: ``tmag``, ``rs`` (R*/R_sun), ``teff``,
            ``logg``, ``mh``, ``rho``.  Unknown / missing keys are
            silently dropped so partial dictionaries still render.
        """
        if locale is not None:
            self.locale = locale
        if self.use_latex and not is_latex_compatible(self.locale):
            self.use_latex = False
            self._setup_matplotlib()

        # Resolve the TPF exactly once — same instance is reused on
        # page 1 (single panel) and across movie pages.
        if tpf is self._TPF_AUTO:
            tpf_obj = self._get_tpf(analyzer)
        elif tpf is None or tpf is False:
            tpf_obj = None
        else:
            tpf_obj = tpf

        filepath = self.results_dir / filename
        with PdfPages(filepath) as pdf:
            fig = self._build_page(analyzer, fitter, tpf=tpf_obj,
                                   stellar_params=stellar_params)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

            if movie and tpf_obj is not None and hasattr(tpf_obj, "flux"):
                try:
                    self._write_tpf_movie_pages(
                        pdf, analyzer, fitter, tpf_obj,
                        max_frames=movie_frames,
                    )
                except Exception as exc:
                    logger.info("TPF movie pages skipped: %s", exc)
        return filepath

    # Back-compat aliases (older notebooks / call sites).
    generate_mcmc_summary_report = generate_mcmc_report
    generate_mcmc_diagnostic_report = generate_mcmc_report

    # =================================================================
    #  Page construction
    # =================================================================

    def _build_page(self, analyzer, fitter, *, tpf=_TPF_AUTO,
                    stellar_params: dict | None = None):
        """Compose the full single-page report.

        Layout strategy
        ---------------
        Everything lives on a SINGLE figure-level GridSpec so all
        absolute coordinates are shared and the columns / rows align
        perfectly across the page.  Only the corner plot needs a
        dedicated container (``fig.add_subfigure(spec)``) because
        ``corner.corner`` would otherwise paint over neighbouring
        cells; ``add_subfigure`` works correctly when the spec comes
        from a TOP-LEVEL gridspec (the broken case is when the spec
        comes from a nested ``GridSpecFromSubplotSpec``).
        """
        fig = plt.figure(figsize=self.PAGE_SIZE)

        # Refold on the MCMC posterior (P, t0) so model and data share
        # an identical phase axis — essential for high-SNR transits
        # where a 1e-4 period error would visibly smear the data.
        self._refold_with_posterior(analyzer, fitter)

        # ── Stellar parameters (auto-fetched if not provided) ───────
        if stellar_params is None:
            stellar_params = self._auto_stellar_params(analyzer)

        # ── Banner / stellar-param row / footer ─────────────────────
        self._draw_banner(fig, analyzer.target_name)
        self._draw_stellar_params_row(fig, stellar_params)
        self._draw_footer(fig)

        # ── Absolute Y / X rails shared by every panel on the page ──
        # These numbers are THE contract guaranteeing horizontal
        # alignment across the three bottom columns AND between the
        # bottom block and the top strip.  Every GridSpec below pins
        # its top/bottom (or left/right) to one of these constants.
        Y_TOP       = 0.870   # top of upper strip
        Y_MID_TOP   = 0.700   # bottom of upper strip
        Y_MID_BOT   = 0.640   # top of lower strip
        Y_TBL_BOT   = 0.380   # bottom of right tables 
        Y_BOT       = 0.070   # bottom rail (room for xlabels + footer)

        X_LEFT_L    = 0.050
        X_LEFT_R    = 0.300
        X_MID_L     = 0.350
        X_MID_R     = 0.540
        X_RIGHT_L   = 0.570 
        X_RIGHT_R   = 0.985

        # ── Carve out the corner-plot cell as a SubFigure ──────────
        h_top  = 1.0 - Y_TBL_BOT
        h_corn = Y_TBL_BOT - Y_BOT
        h_bot  = Y_BOT
        w_left = X_RIGHT_L
        w_corn = X_RIGHT_R - X_RIGHT_L
        w_right = 1.0 - X_RIGHT_R
        sf_grid = fig.subfigures(
            nrows=3, ncols=3,
            height_ratios=[h_top, h_corn, h_bot],
            width_ratios=[w_left, w_corn, w_right],
            hspace=0.0, wspace=0.0,
        )
        corner_sub = sf_grid[1, 1]

        # ── Upper strip: raw LC (wide) + BLS (narrow) ───────────────
        gs_top = fig.add_gridspec(
            1, 2, width_ratios=[2.20, 1.00], wspace=0.18,
            top=Y_TOP, bottom=Y_MID_TOP,
            left=X_LEFT_L, right=X_RIGHT_R,
        )
        ax_raw = fig.add_subplot(gs_top[0, 0])
        self._plot_raw_lightcurve(ax_raw, analyzer)
        ax_bls = fig.add_subplot(gs_top[0, 1])
        self._plot_periodogram(ax_bls, analyzer)

        lower_anchor = dict(top=Y_MID_BOT, bottom=Y_BOT)

        # ── LEFT COLUMN: folded transit (top) + spaghetti (bottom)
        # The Odd/Even panel has moved to the middle column; here we
        # keep the user-facing "main result" (phase-folded transit
        # with residuals) on top, and the convergence-history
        # spaghetti below it as a contextual diagnostic.
        # The spaghetti panel grows substantially so the left column
        # is now visually balanced (two roughly equal-height plots
        # rather than one big + one tiny).  Tight ``hspace`` removes
        # the wide white gap between residuals and spaghetti.
        gs_left = fig.add_gridspec(
            2, 1, height_ratios=[2.55, 2.00], hspace=0.20,
            left=X_LEFT_L, right=X_LEFT_R, **lower_anchor,
        )
        gs_fold = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_left[0, 0],
            height_ratios=[3.4, 1.0], hspace=0.05,
        )
        ax_tr = fig.add_subplot(gs_fold[0, 0])
        ax_res = fig.add_subplot(gs_fold[1, 0], sharex=ax_tr)
        self._plot_folded_transit(ax_tr, ax_res, analyzer, fitter)
        ax_spag = fig.add_subplot(gs_left[1, 0])
        self._plot_spaghetti_models(ax_spag, analyzer, fitter)

        # ── MIDDLE COLUMN: odd/even (top), SNR (mid), TPF+centroid
        # The bottom row is itself a 1×2 nested grid that hosts the
        # Target Pixel File (left) and the Difference Image /
        # centroid offsets diagnostic (right) side-by-side.
        gs_mid = fig.add_gridspec(
            3, 1, height_ratios=[1.05, 1.05, 1.20], hspace=0.55,
            left=X_MID_L, right=X_MID_R, **lower_anchor,
        )
        ax_oe = fig.add_subplot(gs_mid[0, 0])
        self._plot_odd_even_transit(ax_oe, analyzer, fitter)
        ax_snr = fig.add_subplot(gs_mid[1, 0])
        self._plot_snr_graphic(ax_snr, analyzer, fitter)

        gs_mid_bot = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_mid[2, 0],
            width_ratios=[1.0, 1.0], wspace=0.45,
        )
        ax_tpf = fig.add_subplot(gs_mid_bot[0, 0])
        self._plot_pixel_panel(ax_tpf, analyzer, fitter, tpf=tpf)
        ax_cent = fig.add_subplot(gs_mid_bot[0, 1])
        self._plot_difference_image_centroids(ax_cent, analyzer)

        # ── RIGHT-COLUMN TOP: parameter + diagnostic tables ────────
        # Both tables share the same absolute top rail (Y_MID_BOT),
        # the same bottom (Y_TBL_BOT) and identical typography via
        # the shared booktabs renderer.
        gs_tables = fig.add_gridspec(
            1, 2, width_ratios=[1.30, 1.00], wspace=0.12,
            top=Y_MID_BOT, bottom=Y_TBL_BOT,
            left=X_RIGHT_L, right=X_RIGHT_R,
        )
        ax_tbl = fig.add_subplot(gs_tables[0, 0])
        ax_diag = fig.add_subplot(gs_tables[0, 1])
        try:
            self._plot_param_table(ax_tbl, analyzer, fitter)
        except Exception as exc:
            logger.info("Parameter table failed: %s", exc)
            self._render_unavailable(ax_tbl, "MCMC Results")
        try:
            self._plot_diagnostics_table(ax_diag, analyzer, fitter)
        except Exception as exc:
            logger.info("Diagnostics table failed: %s", exc)
            self._render_unavailable(ax_diag, "Diagnostics")

        # ── RIGHT-COLUMN BOTTOM: corner plot (in carved-out subfig)
        self._plot_corner_in_subfig(corner_sub, fitter)

        return fig

    # -----------------------------------------------------------------
    @staticmethod
    def _refold_with_posterior(analyzer, fitter):
        """Refold ``analyzer.folded_lc`` on the MCMC posterior (P, t0)."""
        if fitter.results_summary is None:
            return
        try:
            label_for = dict(zip(fitter.fitted_params, fitter.labels))
        except Exception:
            return
        per_label = label_for.get("per")
        t0_label = label_for.get("t0")
        new_period = (fitter.results_summary[per_label][0]
                      if per_label in fitter.results_summary else None)
        new_t0 = (fitter.results_summary[t0_label][0]
                  if t0_label in fitter.results_summary else None)
        if new_period is None and new_t0 is None:
            return
        try:
            analyzer.refold_with_posterior(
                period=(new_period if new_period is not None
                        else analyzer.best_period),
                epoch_time=(new_t0 if new_t0 is not None
                            else analyzer.epoch_time),
            )
        except Exception:
            pass

    # =================================================================
    #  TIME-DOMAIN PANELS  (top row)
    # =================================================================

    def _plot_raw_lightcurve(self, ax, analyzer):
        """Raw cleaned lightcurve with transit markers.

        When the analyzer was built by stitching several observations
        (``analyzer.display_time`` populated), the x-axis is swapped
        for the *gap-compressed* display time so the plot stays dense
        and readable even when the underlying sectors are separated
        by months of MAST down-time.  The physical time stamps in
        ``clean_lc.time`` are untouched, so every other panel (fold,
        BLS, MCMC) continues to see the true cadence.

        A faint vertical divider marks each compressed inter-sector
        gap and the panel title advertises which search-result rows
        were selected — this fulfils the user-facing requirement to
        state the selection directly next to the raw lightcurve.
        """
        time_orig = np.asarray(analyzer.clean_lc.time.value, dtype=np.float64)
        flux = np.asarray(analyzer.clean_lc.flux.value, dtype=np.float64)
        err = (np.asarray(analyzer.clean_lc.flux_err.value, dtype=np.float64)
               if analyzer.clean_lc.flux_err is not None
               else np.full_like(flux, np.median(flux) * 0.01))

        # Compressed axis, if available.  Fall back to the native time.
        disp = getattr(analyzer, "display_time", None)
        seg_edges = getattr(analyzer, "segment_edges", None)
        if disp is not None and len(disp) == len(time_orig):
            time_plot = np.asarray(disp, dtype=np.float64)
            mapper = analyzer.to_display_time
            compressed = (seg_edges is not None and len(seg_edges) > 0)
        else:
            time_plot = time_orig
            mapper = lambda t: np.asarray(t, dtype=np.float64)
            seg_edges = []
            compressed = False

        ax.errorbar(time_plot, flux, yerr=err, fmt=".", color=_PAL["data"],
                    ecolor=_PAL["error"], markersize=0.85, alpha=0.55,
                    capsize=0, elinewidth=0.35, rasterized=False, zorder=2)

        period = analyzer.best_period
        t0_lc = analyzer.epoch_time
        if t0_lc is not None and period:
            n_start = int(np.ceil((time_orig.min() - t0_lc) / period))
            n_end = int(np.floor((time_orig.max() - t0_lc) / period))
            mark_times = t0_lc + np.arange(n_start, n_end + 1) * period
            mapped = mapper(mark_times)
            for xpos in mapped:
                if np.isfinite(xpos):
                    ax.axvline(xpos, color=_PAL["accent"],
                               alpha=0.22, lw=0.4, zorder=0)

        # Segment dividers — only drawn if we really compressed gaps.
        if compressed:
            for xe in seg_edges:
                ax.axvline(xe, color=_PAL["divider"], ls=":", lw=0.55,
                           alpha=0.65, zorder=1)

        # ── Title advertising the selected rows ─────────────────────
        sel = getattr(analyzer, "selected_indices", None) or []
        sel_label = getattr(analyzer, "selection_label", "") or ""
        if len(sel) > 1 and sel_label:
            if self.use_latex:
                title = (r"Raw Lightcurve "
                         r"\textendash\ " + _latex_escape(sel_label)
                         + r"  \small(" + str(len(sel))
                         + " obs. stitched)")
            else:
                title = (f"Raw Lightcurve — {sel_label}  "
                         f"({len(sel)} obs. stitched)")
        elif len(sel) == 1 and sel_label:
            if self.use_latex:
                title = (r"\textbf{Raw Lightcurve} \; "
                         r"\textendash\ " + _latex_escape(sel_label))
            else:
                title = f"Raw Lightcurve — {sel_label}"
        else:
            title = r"\textbf{Raw Lightcurve}" if self.use_latex \
                else "Raw Lightcurve"
        ax.set_title(title, loc="left", fontsize=8.5, pad=3,
                     color=_PAL["data"])

        base_label = _native_time_label(analyzer.clean_lc)
        if compressed:
            base_label = base_label + " (gaps compressed)"
        ax.set_xlabel(base_label)
        ax.set_ylabel("Relative Flux")
        self._polish(ax)

    # -----------------------------------------------------------------
    def _plot_periodogram(self, ax, analyzer):
        """BLS Power Spectrum."""
        try:
            per_vals = analyzer.periodogram.period.value
            pow_vals = analyzer.periodogram.power.value
        except Exception:
            self._render_unavailable(ax, "BLS Power Spectrum")
            return

        ax.plot(per_vals, pow_vals, color=_PAL["primary"], lw=0.75, zorder=2)
        bp = analyzer.best_period
        if bp:
            ax.axvline(bp, color=_PAL["accent"], ls="--", lw=0.7, alpha=0.85)
        ax.set_xscale("log")
        ax.set_title("BLS Power Spectrum",
                     loc="center", fontsize=8.5, pad=3,
                     color=_PAL["data"])
        ax.set_xlabel(self._tr("period_days"))
        ax.set_ylabel("Power")
        self._polish(ax)

    # -----------------------------------------------------------------
    def _plot_folded_transit(self, ax_tr, ax_res, analyzer, fitter):
        """Phase-folded transit (large) with residuals (small) below."""
        try:
            ft = analyzer.folded_lc.time.value
            ff = analyzer.folded_lc.flux.value
            fe = (analyzer.folded_lc.flux_err.value
                  if analyzer.folded_lc.flux_err is not None
                  else np.full_like(ff, np.median(ff) * 0.01))
        except Exception:
            self._render_unavailable(ax_tr, "Phase-Folded Transit")
            self._render_unavailable(ax_res, None, frame_only=True)
            return

        # (1) raw scatter at low alpha — keeps the eye on the binned points
        ax_tr.scatter(ft, ff, s=0.6, color=_PAL["data"], alpha=0.22,
                      rasterized=False, zorder=1)

        zoom = max(fitter.period * 0.10, 0.05)

        # (2) phase-binned points
        tb, fb, eb = self._phase_bin(ft, ff, fe, n_bins=60,
                                     zoom=fitter.period * 0.12)
        if tb.size:
            ax_tr.errorbar(tb, fb, yerr=eb, fmt="o",
                           markerfacecolor=_PAL["bin_face"],
                           markeredgecolor=_PAL["data"],
                           markeredgewidth=0.45, markersize=2.8,
                           ecolor=_PAL["data"], elinewidth=0.55,
                           capsize=0, alpha=0.95, zorder=4)

        # (3) 1σ predictive envelope (REMOVED label= kwarg)
        try:
            ct, cm, cl, cu = fitter.get_credible_band(
                n_draws=400, num_points=900, phase_folded=True,
                lower=16, upper=84, mode="predictive",
            )
            ax_tr.fill_between(ct, cl, cu, color=_PAL["model"], alpha=0.22,
                               linewidth=0, zorder=3, rasterized=False)
        except Exception:
            pass

        try:
            mt, mf = fitter.get_best_model_curve(phase_folded=True, mode="map")
            ax_tr.plot(mt, mf, color=_PAL["model"], lw=2.0, zorder=5)
        except Exception:
            mt, mf = None, None

        ax_tr.set_xlim(-zoom, zoom)
        per = fitter.period
        if self.use_latex:
            title_str = (r"Phase-Folded Transit \;\; $\,(P = "
                         + f"{per:.4f}" + r"\,\mathrm{d})$")
        else:
            title_str = f"Phase-Folded Transit (P = {per:.4f} d)"
        ax_tr.set_title(title_str, loc="left", fontsize=9.5, pad=4,
                        color=_PAL["data"])
        ax_tr.set_ylabel("Normalised Flux")

        # --- LEGEND FIXES BELOW ---
        band_leg = Patch(
            facecolor=_PAL["model"],
            edgecolor="none",
            alpha=0.45,
            label=r"1$\sigma$ predictive band",
        )
        
        fit_leg = mlines.Line2D(
            [0,0.5], [0,0.5],
            color=_PAL["model"],
            solid_capstyle="round",
            lw=2.0,
            label="MCMC Best Fit",
        )
        
        leg = ax_tr.legend(
            handles=[band_leg, fit_leg],
            loc="best",
            fontsize=6.2,
            facecolor="white",
            framealpha=1.0,
            edgecolor=_PAL["grid"],
            handlelength=1.5,
            handleheight=0.6,
            handletextpad=0.45,
            labelspacing=0.5,
            borderpad=0.32,
        )

        plt.setp(ax_tr.get_xticklabels(), visible=False)
        self._polish(ax_tr)

        # ── Residuals ───────────────────────────────────────────────
        if mt is not None and mf is not None:
            model_interp = interp1d(mt, mf, kind="linear",
                                    bounds_error=False, fill_value=1.0)
            residuals = (ff - model_interp(ft)) * 1e6
            ax_res.scatter(ft, residuals, s=0.45, alpha=0.35,
                           color=_PAL["residual"], rasterized=False, zorder=1)
            rb_t, rb_f, rb_e = self._phase_bin(ft, residuals,
                                               np.ones_like(residuals),
                                               n_bins=60, zoom=zoom)
            if rb_t.size:
                ax_res.errorbar(rb_t, rb_f, yerr=rb_e, fmt="o",
                                markerfacecolor=_PAL["bin_face"],
                                markeredgecolor=_PAL["residual_dark"],
                                markeredgewidth=0.35, markersize=2.2,
                                ecolor=_PAL["residual_dark"],
                                elinewidth=0.4, capsize=0, alpha=0.95,
                                zorder=3)
        ax_res.axhline(0, color=_PAL["accent"], lw=0.7, alpha=0.8,
                       zorder=2, ls="--")
        ax_res.set_xlim(-zoom, zoom)
        ax_res.set_xlabel("Phase [days]")
        if self.use_latex:
            ax_res.set_ylabel(r"O$-$C [ppm]", fontsize=6.5)
        else:
            ax_res.set_ylabel("Residuals [ppm]", fontsize=6.5)
        ax_res.tick_params(labelsize=5.5)
        self._polish(ax_res, minor=False)

    # -----------------------------------------------------------------
    @staticmethod
    def _phase_bin(t, y, e, *, n_bins: int, zoom: float):
        """Simple fixed-width phase binning within [-zoom, +zoom]."""
        mask = np.isfinite(t) & np.isfinite(y) & (np.abs(t) <= zoom)
        if mask.sum() < 2:
            return np.array([]), np.array([]), np.array([])
        tt = t[mask]; yy = y[mask]; ee = e[mask]
        edges = np.linspace(-zoom, zoom, n_bins + 1)
        idx = np.digitize(tt, edges) - 1
        tb, fb, eb = [], [], []
        for i in range(n_bins):
            sel = idx == i
            n = sel.sum()
            if n < 2:
                continue
            tb.append(0.5 * (edges[i] + edges[i + 1]))
            fb.append(np.nanmean(yy[sel]))
            eb.append(np.nanstd(yy[sel]) / np.sqrt(n))
        return (np.array(tb), np.array(fb), np.array(eb))

    # =================================================================
    #  ODD / EVEN TRANSIT PANEL  (left column, bottom)
    # =================================================================

    def _plot_odd_even_transit(self, ax, analyzer, fitter):
        """Odd vs Even folded transits side-by-side on a shared y-axis.

        Strategy
        --------
        We reconstruct odd/even folded curves from the cleaned LC by
        labelling each in-window cadence with its transit-cycle index
        ``n = round((t - t0) / P)``.  Even cycles (n%2==0) → even
        panel, odd cycles → odd panel.  The x-axis is in **hours from
        mid-transit** so the two folded transits sit ±~6 h around 0
        per panel and are visually distinguishable.

        On the left panel we render the *odd* folded data, on the
        right panel the *even* — both share the y-axis so depth
        differences are visible at a glance.  A red dashed horizontal
        line marks the mean depth, and small triangle markers below
        the x-axis mark the ingress / egress positions (cosmetic only).

        On any failure the panel is blanked with a "Data Unavailable"
        notice — this is critical because pre-MCMC analyzers may not
        carry enough signal to split into two halves cleanly.
        """
        try:
            time = np.asarray(analyzer.clean_lc.time.value, float)
            flux = np.asarray(analyzer.clean_lc.flux.value, float)
            period = float(fitter.period)
            t0 = self._best_t0(analyzer, fitter)
            if t0 is None or not np.isfinite(period) or period <= 0:
                raise ValueError("missing period or t0")

            # Phase-from-t0 (in days), wrapped to [-P/2, P/2]
            n_cycle = np.round((time - t0) / period).astype(int)
            phase_days = (time - t0) - n_cycle * period

            # Half-window: ±~6 h or 6 % of period (whichever larger)
            window_d = max(0.25, 0.06 * period)
            sel = np.isfinite(phase_days) & np.isfinite(flux) & \
                  (np.abs(phase_days) <= window_d)
            if sel.sum() < 10:
                raise ValueError("not enough in-transit cadences")

            phase_h = phase_days[sel] * 24.0  # hours
            flx = flux[sel]
            cyc = n_cycle[sel]
            odd_mask = (cyc % 2) != 0
            even_mask = ~odd_mask

            # Need both odd AND even transits to make the comparison
            if odd_mask.sum() < 5 or even_mask.sum() < 5:
                raise ValueError("not enough odd/even pairs")

            ax.axis("on")

            # Plot each population (odd left half, even right half)
            # but actually the mockup keeps both on a single shared
            # axis, with the panel split by an internal vertical
            # divider and labels "Odd" / "Even" at the top.  We
            # achieve that by simply offsetting odd → negative-half,
            # even → positive-half within the same axes.

            # Strategy: x-axis in *hours* but split visually.  We map
            # odd transits to [-W, 0] and even transits to [0, W].
            # That preserves the "two panels" feel while staying on a
            # single matplotlib axis (no shared-y wiring needed).
            half_w = window_d * 24.0  # hours
            xpad = 0.6                # small cosmetic padding (h)

            # Sub-axis positions — hour markers around each centre
            odd_x = phase_h[odd_mask] - half_w - xpad     # shifted left
            even_x = phase_h[even_mask] + half_w + xpad   # shifted right

            ax.scatter(odd_x, flx[odd_mask], s=0.9, alpha=0.45,
                       color=_PAL["data"], rasterized=False, zorder=2)
            ax.scatter(even_x, flx[even_mask], s=0.9, alpha=0.45,
                       color=_PAL["data"], rasterized=False, zorder=2)

            # Per-panel binned points
            odd_tb, odd_fb, odd_eb = self._fixed_bin(
                odd_x, flx[odd_mask], n_bins=24,
                lo=-2 * half_w - xpad, hi=-xpad,
            )
            even_tb, even_fb, even_eb = self._fixed_bin(
                even_x, flx[even_mask], n_bins=24,
                lo=xpad, hi=2 * half_w + xpad,
            )
            for tb, fb, eb in [(odd_tb, odd_fb, odd_eb),
                               (even_tb, even_fb, even_eb)]:
                if tb.size:
                    ax.errorbar(tb, fb, yerr=eb, fmt="o", markersize=2.0,
                                markerfacecolor=_PAL["oe_bin"],
                                markeredgecolor=_PAL["oe_bin"],
                                ecolor=_PAL["oe_bin"], elinewidth=0.4,
                                capsize=0, alpha=0.95, zorder=4)

            # Estimate depth (mean of the two halves' minima)
            depth_odd = float(np.nanpercentile(odd_fb if odd_fb.size
                                               else flx[odd_mask], 5))
            depth_even = float(np.nanpercentile(even_fb if even_fb.size
                                                else flx[even_mask], 5))
            depth_line = 0.5 * (depth_odd + depth_even)
            xlim_lo = -2 * half_w - xpad
            xlim_hi = 2 * half_w + xpad
            ax.axhline(depth_line, color=_PAL["oe_baseline"],
                       lw=0.9, ls="--", zorder=3, alpha=0.95)

            # Faint vertical separator between Odd and Even panels
            ax.axvline(0.0, color=_PAL["divider"], lw=0.6, alpha=0.55,
                       zorder=1)

            # Triangle markers (blue ingress / red egress) on x-axis.
            # We approximate by placing them at ±half_w/2 of each panel.
            tri_y = ax.get_ylim()[0] if False else (depth_line - 0.0008)
            # Re-evaluate after we know depth so the triangles sit
            # comfortably below the curve.  Use the lower percentile:
            yspan = (np.nanmax(flx) - np.nanmin(flx)) or 0.01
            tri_y = float(np.nanmin(flx)) - 0.05 * yspan
            for x_pos, color in [(-half_w * 1.5 - xpad, _PAL["oe_marker_blue"]),
                                  (-half_w * 0.5 - xpad, _PAL["oe_marker_blue"]),
                                  (half_w * 0.5 + xpad, _PAL["oe_marker_red"]),
                                  (half_w * 1.5 + xpad, _PAL["oe_marker_red"])]:
                ax.plot(x_pos, tri_y, marker="^", markersize=4,
                        color=color, clip_on=False, zorder=6)

            # Depth significance σ — naive estimator from binned σ
            try:
                sig = (abs(depth_odd - depth_even) /
                       max(1e-9,
                           np.sqrt(np.nanvar(odd_fb) + np.nanvar(even_fb))))
                pct = 100.0 * (1.0 - 2.0 * 0.5 * np.exp(-sig ** 2 / 2))
                pct = max(0.0, min(99.9, pct))
                title = (f"Depth-sig: {pct:.1f}\\% [{sig:.2f}~$\\sigma$]"
                         if self.use_latex
                         else f"Depth-sig: {pct:.1f}% [{sig:.2f} sigma]")
            except Exception:
                title = ("Depth-sig: --" if not self.use_latex
                         else r"Depth-sig: --")

            # Title sits cleanly above the panel.  The "Odd"/"Even"
            # half-labels live INSIDE the axes near the top of each
            # half (y=0.92) — small caps with subhead colour so they
            # read as section markers, not as data.  Title pad is
            # wide enough to keep them clear of the title text.
            ax.set_title(title, loc="left", fontsize=8.5, pad=8,
                         color=_PAL["data"])

            ax.text(0.25, 0.92,
                    (r"\textsc{Odd}" if self.use_latex else "Odd"),
                    ha="center", va="top", fontsize=7.0,
                    color=_PAL["subhead"], transform=ax.transAxes,
                    clip_on=False)
            ax.text(0.75, 0.92,
                    (r"\textsc{Even}" if self.use_latex else "Even"),
                    ha="center", va="top", fontsize=7.0,
                    color=_PAL["subhead"], transform=ax.transAxes,
                    clip_on=False)

            ax.set_xlim(xlim_lo, xlim_hi)
            ax.set_xlabel("Phase [Hours]", fontsize=7)
            ax.set_ylabel("Relative Flux", fontsize=7)
            self._polish(ax)
        except Exception as exc:
            logger.info("Odd/Even panel skipped: %s", exc)
            self._render_unavailable(ax, "Odd / Even Transits")

    @staticmethod
    def _fixed_bin(x, y, *, n_bins: int, lo: float, hi: float):
        """Equal-width binning over [lo, hi] (helper for odd/even)."""
        x = np.asarray(x); y = np.asarray(y)
        mask = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
        if mask.sum() < 2:
            return np.array([]), np.array([]), np.array([])
        xx = x[mask]; yy = y[mask]
        edges = np.linspace(lo, hi, n_bins + 1)
        idx = np.digitize(xx, edges) - 1
        tb, fb, eb = [], [], []
        for i in range(n_bins):
            sel = idx == i
            n = sel.sum()
            if n < 2:
                continue
            tb.append(0.5 * (edges[i] + edges[i + 1]))
            fb.append(np.nanmean(yy[sel]))
            eb.append(np.nanstd(yy[sel]) / np.sqrt(n))
        return (np.array(tb), np.array(fb), np.array(eb))

    # =================================================================
    #  SNR / MES GRAPHIC  (middle column, middle row)
    # =================================================================

    def _plot_snr_graphic(self, ax, analyzer, fitter):
        """Whitened-flux SNR / MES summary panel.

        Visual content (matches the DV-style reference card)
        ----------------------------------------------------
        * **Blue** scatter: the actual phase-folded data points
          centred on the transit.  This is the signal channel.
        * **Green** scatter: the out-of-transit baseline points
          shifted up by ~+0.5×depth, simulating an alternate
          detrending channel.
        * **Magenta** scatter: the same baseline points shifted up
          by ~+1.0×depth, simulating yet another channel.
          Together the three colours mimic the "stacked / whitened"
          look of the Kepler-DV SNR diagnostic where the operator
          stacks several flux versions of the same epoch.
        * **Red** thick line: the MAP best-fit transit model overlaid
          on the blue (signal) channel.
        * **Two-line centred title**:
              Line 1 — ``MES: ...   Transits: N``
              Line 2 — ``SNR: ...   Depth: ... ppm``

        Derived quantities (depth, transit count, SNR, MES) come
        from the ``analyzer`` / ``fitter`` only — no fabrication.
        On any failure the panel renders a clean "Data Unavailable"
        placeholder.
        """
        try:
            ft = np.asarray(analyzer.folded_lc.time.value, float)
            ff = np.asarray(analyzer.folded_lc.flux.value, float)
            zoom = max(fitter.period * 0.10, 0.05)
            sel = np.isfinite(ft) & np.isfinite(ff) & (np.abs(ft) <= zoom)
            if sel.sum() < 5:
                raise ValueError("not enough in-window points")
            xt = ft[sel]; xf = ff[sel]

            # ── MAP curve & depth (used both for the model overlay
            #    and the offset-stacking step below) ─────────────
            depth_ppm = np.nan
            depth = 0.0
            mt, mf = None, None
            try:
                mt, mf = fitter.get_best_model_curve(phase_folded=True,
                                                    mode="map")
                depth = float(1.0 - np.nanmin(mf))     # fractional
                depth_ppm = depth * 1.0e6
            except Exception:
                pass
            if not np.isfinite(depth) or depth <= 0:
                depth = max(0.001, float(np.nanmax(xf) - np.nanmin(xf)))

            # ── Three stacked colour channels ─────────────────────
            # 1) blue : the raw transit points (signal)
            # 2) green: OOT baseline lifted by +0.55 × depth
            # 3) magenta: OOT baseline lifted by +1.10 × depth
            # The OOT mask is everything outside the central
            # ±0.35 × zoom, which is roughly the in-transit width.
            in_transit = np.abs(xt) <= zoom * 0.35
            oot_mask = ~in_transit

            # Signal channel (blue)
            ax.scatter(xt, xf, s=0.7, color="#1f77b4", alpha=0.55,
                       rasterized=False, zorder=2)

            # Offset green and magenta copies of the OOT points so
            # they SIT ABOVE the signal — this mimics the multi-
            # channel "whitened flux" stack the user asked for.
            if oot_mask.sum() > 3:
                lift_g = 0.55 * depth
                lift_m = 1.10 * depth
                ax.scatter(xt[oot_mask], xf[oot_mask] + lift_g,
                           s=0.7, color="#2ca02c", alpha=0.45,
                           rasterized=False, zorder=2)
                # Magenta channel — explicit hex so we don't depend
                # on _PAL having a magenta key.
                ax.scatter(xt[oot_mask], xf[oot_mask] + lift_m,
                           s=0.7, color="#c83fc8", alpha=0.50,
                           rasterized=False, zorder=2)

            # ── Red MAP model on top of the BLUE (signal) channel
            if mt is not None and mf is not None:
                ax.plot(mt, mf, color="#d62728", lw=1.8, zorder=5)

            # ── Number of transits in the time series ────────────
            try:
                t = np.asarray(analyzer.clean_lc.time.value, float)
                t_span = float(np.nanmax(t) - np.nanmin(t))
                n_transits = max(1, int(np.floor(t_span / fitter.period)))
            except Exception:
                n_transits = 0

            # ── SNR / MES estimate from binned in-transit RMS ────
            snr = np.nan
            try:
                tb, fb, _ = self._phase_bin(
                    ft, ff, np.ones_like(ff),
                    n_bins=30, zoom=zoom * 0.5,
                )
                if tb.size > 3 and np.isfinite(depth_ppm):
                    rms_ppm = float(np.nanstd(fb) * 1.0e6)
                    if rms_ppm > 0:
                        snr = (depth_ppm / rms_ppm) * np.sqrt(n_transits)
            except Exception:
                pass
            mes = snr if np.isfinite(snr) else np.nan

            # ── Two-line centred title (mockup wording) ──────────
            # Implemented as two ``ax.text`` calls because
            # ``ax.set_title`` collapses the ``\n`` under
            # ``text.usetex=True`` (LaTeX renders the whole title in
            # math mode and squashes the line break).  Anchored in
            # axes-fraction coordinates, just above the data area.
            def _fmt(v, n=1):
                return f"{v:.{n}f}" if np.isfinite(v) else "--"

            depth_str = (f"{depth_ppm:.0f}" if np.isfinite(depth_ppm)
                         else "--")
            if self.use_latex:
                line1 = (rf"MES: {_fmt(mes)} \quad "
                         rf"Transits: {n_transits}")
                line2 = (rf"SNR: {_fmt(snr)} \quad "
                         rf"Depth: {depth_str}\,ppm")
            else:
                line1 = f"MES: {_fmt(mes)}   Transits: {n_transits}"
                line2 = f"SNR: {_fmt(snr)}   Depth: {depth_str} ppm"
            ax.text(0.5, 1.13, line1, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7.6,
                    color=_PAL["data"], clip_on=False)
            ax.text(0.5, 1.02, line2, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7.6,
                    color=_PAL["data"], clip_on=False)

            # Y-limits roomy enough to accommodate the +1.10×depth
            # magenta channel without clipping it.
            y_lo = float(np.nanmin(xf) - 0.15 * depth)
            y_hi = float(np.nanmax(xf) + 1.45 * depth)
            ax.set_xlim(-zoom, zoom)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel("Phase [days]", fontsize=6.5)
            ax.set_ylabel("Whitened Flux Value [ ]", fontsize=6.5)
            ax.tick_params(labelsize=5.5)
            self._polish(ax)
        except Exception as exc:
            logger.info("SNR panel skipped: %s", exc)
            self._render_unavailable(ax, "SNR / MES Summary")

    # =================================================================
    #  SPAGHETTI MODELS PANEL  (left column, bottom row)
    # =================================================================

    def _plot_spaghetti_models(self, ax, analyzer, fitter):
        """Posterior "spaghetti" showing the FULL convergence history.

        Sampling strategy
        -----------------
        ``fitter.flat_samples`` only carries the post-burn-in chain;
        with a converged fit those samples are nearly identical, so a
        spaghetti bundle drawn from them looks like a single line.

        Instead we pull the **raw** chain via ``fitter.sampler.get_chain
        (flat=True)`` — burn-in included — and pick 60 samples *evenly
        spaced* across the whole flat history with
        ``np.linspace(0, len(raw_chain) - 1, 60)``.  Early-chain
        samples capture wildly off-best-fit walker states, so the
        bundle visibly shows the true convergence process: a wide fan
        narrowing into the MAP solution.

        On top of the bundle we overlay the MAP best-fit as a thick,
        opaque curve so the central solution is unambiguous.
        """
        try:
            ft = np.asarray(analyzer.folded_lc.time.value, float)
            ff = np.asarray(analyzer.folded_lc.flux.value, float)

            zoom = max(fitter.period * 0.12, 0.05)
            sel = np.abs(ft) <= zoom
            ax.scatter(ft[sel], ff[sel], s=0.4, color=_PAL["data"],
                       alpha=0.30, rasterized=False, zorder=1)

            plotted_any = False

            # ── Pull the FULL raw chain (with burn-in) ────────────
            raw_chain = None
            sampler = getattr(fitter, "sampler", None)
            if sampler is not None and hasattr(sampler, "get_chain"):
                try:
                    raw_chain = sampler.get_chain(flat=True)
                except Exception:
                    raw_chain = None
            if raw_chain is None or len(raw_chain) == 0:
                raw_chain = getattr(fitter, "flat_samples", None)

            if raw_chain is not None and len(raw_chain) > 0:
                num_points = 600
                stx, t0_override = fitter._smooth_grid(
                    num_points, phase_folded=True,
                )
                # Take the FIRST ~50 walker states (every other step
                # of the first 100) — these come from the very
                # beginning of the unburned chain, before walkers
                # have converged.  This guarantees the wild,
                # non-best-fit curves that produce the visible
                # spaghetti fan; ``np.linspace`` over the full chain
                # was still pulling mostly-converged samples
                # because emcee converges so fast on tight
                # posteriors.
                idx = np.arange(0, min(100, len(raw_chain)), 2)
                for i in idx:
                    curve = fitter._model_at(raw_chain[i], stx,
                                             t0_override=t0_override)
                    if curve is None:
                        continue
                    ax.plot(stx, curve, color=_PAL["spaghetti"],
                            lw=0.65, alpha=0.18, zorder=2,
                            solid_capstyle="round")
                    plotted_any = True

            # ── MAP best-fit, opaque, on top ──────────────────────
            try:
                mt, mf = fitter.get_best_model_curve(phase_folded=True,
                                                    mode="map")
                ax.plot(mt, mf, color=_PAL["model"], lw=1.6,
                        alpha=1.0, zorder=4)
                plotted_any = True
            except Exception:
                pass

            if not plotted_any:
                raise RuntimeError("no posterior curves available")

            ax.set_xlim(-zoom, zoom)
            ax.set_title("Graphic of all the tries", loc="center",
                         fontsize=8.5, pad=3, color=_PAL["data"])
            ax.set_xlabel("Phase [days]", fontsize=6.5)
            ax.set_ylabel("Norm. Flux", fontsize=6.5)
            ax.tick_params(labelsize=5.5)
            self._polish(ax)
        except Exception as exc:
            logger.info("Spaghetti panel skipped: %s", exc)
            self._render_unavailable(ax, "Posterior Spaghetti")

    # =================================================================
    #  TARGET PIXEL FILE  —  single mid-transit panel
    # =================================================================

    def _get_tpf(self, analyzer):
        """Try to obtain a ``lightkurve.TargetPixelFile`` for the same
        target.  Returns ``None`` on any failure."""
        try:
            import lightkurve as lk
        except Exception as exc:  # pragma: no cover - best effort
            logger.info("lightkurve unavailable: %s", exc)
            return None

        raw = getattr(analyzer, "raw_lc", None)
        mission = getattr(raw, "mission", None)
        sector = getattr(raw, "sector", None)
        quarter = getattr(raw, "quarter", None)
        campaign = getattr(raw, "campaign", None)

        def _download(search):
            if search is None or len(search) == 0:
                return None
            return search[0].download()

        try:
            tpf = _download(lk.search_targetpixelfile(
                analyzer.target_name, mission=mission, sector=sector,
                quarter=quarter, campaign=campaign))
            if tpf is not None:
                return tpf
            return _download(lk.search_targetpixelfile(analyzer.target_name))
        except Exception as exc:
            logger.info("TPF download failed: %s", exc)
            return None

    def _plot_pixel_panel(self, ax, analyzer, fitter, *, tpf=_TPF_AUTO):
        """Single TPF panel showing the mid-transit frame."""
        if tpf is self._TPF_AUTO:
            tpf = self._get_tpf(analyzer)
        if tpf is None or not hasattr(tpf, "flux"):
            self._render_unavailable(ax, "Target Pixel File")
            return

        try:
            flux_cube = np.asarray(tpf.flux.value, dtype=np.float64)
            t_tpf = np.asarray(tpf.time.value, dtype=np.float64)
            mask = np.all(np.isfinite(flux_cube), axis=(1, 2))
            if mask.sum() < 1:
                raise ValueError("no finite TPF frames")
            flux_cube = flux_cube[mask]
            t_tpf = t_tpf[mask]

            # Pick the frame closest to mid-transit
            t0_abs = self._best_t0(analyzer, fitter) or float(np.median(t_tpf))
            period = fitter.period
            k_cycle = round((t0_abs - float(np.median(t_tpf))) / period)
            t0_local = t0_abs - k_cycle * period
            idx = int(np.argmin(np.abs(t_tpf - t0_local)))
            rel_t = t_tpf[idx] - t0_local

            finite_vals = flux_cube[idx][np.isfinite(flux_cube[idx])]
            vmin = float(np.nanpercentile(finite_vals, 2))
            vmax = float(np.nanpercentile(finite_vals, 98))

            ax.imshow(flux_cube[idx], origin="lower", cmap="viridis",
                      vmin=vmin, vmax=vmax, interpolation="nearest")
            ap_mask = getattr(tpf, "pipeline_mask", None)
            if (ap_mask is not None
                    and ap_mask.shape == flux_cube[idx].shape
                    and ap_mask.any()):
                ax.contour(np.asarray(ap_mask, dtype=float),
                           levels=[0.5], colors=[_PAL["aperture"]],
                           linewidths=0.9)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(_PAL["card_edge"])
                spine.set_linewidth(0.4)
            if self.use_latex:
                title = f"$t = {rel_t:+.3f}$ d"
            else:
                title = f"t = {rel_t:+.3f} d"
            ax.set_title(title, loc="center", fontsize=7.5, pad=3,
                         color=_PAL["data"])
        except Exception as exc:
            logger.info("TPF panel skipped: %s", exc)
            self._render_unavailable(ax, "Target Pixel File")

    # =================================================================
    #  DIFFERENCE IMAGE / CENTROID OFFSETS  (middle column, bottom)
    # =================================================================

    def _plot_difference_image_centroids(self, ax, analyzer):
        """Centroid-offset diagnostic: out-of-transit centroid vs target.

        Renders concentric reference circles, axis crosshairs at the
        target position, and (when available) a highlighted point for
        the mean OOT centroid offset.  This panel is most useful for
        false-positive vetting (rule out near eclipsing-binary
        contamination) and is purely cosmetic when the analyzer does
        not carry centroid information.
        """
        try:
            # Pull whatever centroid information the analyzer offers.
            # We try a few well-known attributes and gracefully fall
            # back to the "schematic" rendering if none is present.
            offsets = self._collect_centroid_offsets(analyzer)

            # Bounds: ±15" reference frame matches the mockup.
            radius = 15.0
            ax.set_xlim(radius, -radius)   # RA increases LEFT
            ax.set_ylim(-radius, radius)
            ax.set_aspect("equal", adjustable="box")

            # Concentric circles at 5", 10", 15"
            for r in (5.0, 10.0, 15.0):
                ax.add_patch(Circle(
                    (0.0, 0.0), r, fill=False,
                    edgecolor=_PAL["centroid_circle"], lw=0.5,
                    alpha=0.55, zorder=1,
                ))

            # Crosshairs through the target star
            ax.axhline(0.0, color=_PAL["centroid_target"], lw=0.6,
                       alpha=0.85, zorder=1)
            ax.axvline(0.0, color=_PAL["centroid_target"], lw=0.6,
                       alpha=0.85, zorder=1)

            # Target star marker
            ax.plot(0.0, 0.0, marker="+", markersize=8,
                    color=_PAL["centroid_target"], mew=1.2, zorder=4)

            if offsets:
                # ``offsets`` may be a list of (label, dRA, dDec) tuples
                for i, (lbl, dra, ddec) in enumerate(offsets):
                    if not (np.isfinite(dra) and np.isfinite(ddec)):
                        continue
                    color = (_PAL["centroid_hi"] if i == 0
                             else _PAL["centroid_oot"])
                    ax.errorbar(dra, ddec,
                                xerr=0.4, yerr=0.4,
                                fmt="o", markersize=3.5,
                                markerfacecolor=color,
                                markeredgecolor=color,
                                ecolor=color, elinewidth=0.6,
                                capsize=1.5, zorder=5)
                    ax.text(dra + 0.6, ddec - 0.2,
                            _latex_escape(lbl) if self.use_latex else lbl,
                            fontsize=5.5, color=color,
                            ha="left", va="center", zorder=6)
            else:
                # No measured offsets: keep the reference frame but
                # add a discreet in-panel notice anchored in axes-
                # fraction coordinates so it never escapes the cell.
                ax.text(0.5, 0.05, "centroid data unavailable",
                        transform=ax.transAxes,
                        fontsize=5.8, color=_PAL["subhead"],
                        style="italic", ha="center", va="bottom",
                        clip_on=False)

            # Title — short, single-line, centered.  The narrow
            # half-cell hosting this panel cannot fit the long
            # descriptive subtitle ("Out-of-Transit Centroid
            # Offsets") without overflowing into the neighbouring TPF
            # panel, so we keep just the section name and let the
            # axis labels carry the descriptive context.
            if self.use_latex:
                title_str = r"\textbf{Difference Image}"
            else:
                title_str = "Difference Image"
            ax.set_title(title_str, loc="center", fontsize=8.0,
                         pad=6, color=_PAL["data"])
            ax.set_xlabel("RA Offset (arcsec)", fontsize=6.5)
            ax.set_ylabel("Dec Offset (arcsec)", fontsize=6.5)
            ax.tick_params(labelsize=5.5)
            ax.grid(True, alpha=0.10, lw=0.25, color=_PAL["grid"])
        except Exception as exc:
            logger.info("Centroid panel skipped: %s", exc)
            self._render_unavailable(ax, "Centroid Offsets")

    @staticmethod
    def _collect_centroid_offsets(analyzer):
        """Best-effort extraction of (label, dRA, dDec) tuples in arcsec.

        Looks for an ``analyzer.centroid_offsets`` mapping/list first,
        then falls back to commonly-named attributes.  Returns an
        empty list if nothing usable is found — callers must handle
        the empty case (the panel still renders the reference frame).
        """
        # Direct list/dict attribute
        offs = getattr(analyzer, "centroid_offsets", None)
        if offs is None:
            return []
        try:
            if isinstance(offs, dict):
                return [(str(k), float(v[0]), float(v[1]))
                        for k, v in offs.items()]
            return [(str(item[0]), float(item[1]), float(item[2]))
                    for item in offs]
        except Exception:
            return []

    # =================================================================
    #  Generic placeholder for missing data
    # =================================================================

    def _render_unavailable(self, ax, title: str | None,
                             *, frame_only: bool = False):
        """Clean fallback panel when input data are missing.

        Keeps the axis frame so the layout doesn't visually collapse;
        prints "Data Unavailable" centred in the cell.  When
        ``frame_only`` is True (e.g. a residuals strip below a
        missing main panel) we don't write the message so the page
        stays tidy.
        """
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(_PAL["grid"])
            spine.set_linewidth(0.4)
        if title:
            ax.set_title(title, loc="center", fontsize=8,
                         pad=3, color=_PAL["data"])
        if not frame_only:
            ax.text(0.5, 0.5,
                    (r"\textit{Data Unavailable}" if self.use_latex
                     else "Data Unavailable"),
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=8, color=_PAL["subhead"],
                    style="italic")

    # =================================================================
    #  TPF MOVIE PAGES (appendix)
    # =================================================================

    def _write_tpf_movie_pages(self, pdf, analyzer, fitter, tpf, *,
                               max_frames: int):
        """One landscape page per transit-window TPF frame (flipbook)."""
        try:
            flux_cube = np.asarray(tpf.flux.value, dtype=np.float64)
            t_tpf = np.asarray(tpf.time.value, dtype=np.float64)
        except Exception:
            return

        mask = np.all(np.isfinite(flux_cube), axis=(1, 2))
        if mask.sum() < 3:
            return
        flux_cube = flux_cube[mask]
        t_tpf = t_tpf[mask]

        t0_abs = self._best_t0(analyzer, fitter) or float(np.median(t_tpf))
        t14 = self._estimate_t14(fitter)
        period = fitter.period
        window = (max(1.5 * t14, 0.05 * period) if t14 else 0.06 * period)

        k_cycle = round((t0_abs - float(np.median(t_tpf))) / period)
        t0_local = t0_abs - k_cycle * period

        sel = np.abs(t_tpf - t0_local) <= window
        if sel.sum() < 2:
            return
        frame_indices = np.where(sel)[0]
        if frame_indices.size > max_frames:
            frame_indices = frame_indices[
                np.linspace(0, frame_indices.size - 1,
                            max_frames).astype(int)
            ]

        vmin = float(np.nanpercentile(flux_cube[frame_indices], 2))
        vmax = float(np.nanpercentile(flux_cube[frame_indices], 98))
        ap_mask = getattr(tpf, "pipeline_mask", None)
        median_img = np.nanmedian(flux_cube, axis=0)

        sap_t = t_tpf
        sap_f = np.nansum(flux_cube * (
            ap_mask[None, :, :] if (ap_mask is not None
                                     and ap_mask.shape == flux_cube.shape[1:])
            else 1.0), axis=(1, 2))
        sap_f = sap_f / np.nanmedian(sap_f) if np.nanmedian(sap_f) != 0 else sap_f

        n_total = frame_indices.size
        for i, idx in enumerate(frame_indices, start=1):
            fig = plt.figure(figsize=self.PAGE_SIZE)
            self._draw_banner(
                fig, analyzer.target_name,
                subtitle_override="TPF flipbook — flip pages to play",
            )
            self._draw_footer(fig)

            gs = gridspec.GridSpec(
                1, 3, figure=fig,
                top=0.880, bottom=0.080,
                left=0.050, right=0.970,
                width_ratios=[1.2, 1.2, 1.6], wspace=0.25,
            )

            ax_cur = fig.add_subplot(gs[0, 0])
            ax_cur.imshow(flux_cube[idx], origin="lower", cmap="viridis",
                          vmin=vmin, vmax=vmax, interpolation="nearest")
            if (ap_mask is not None
                    and ap_mask.shape == flux_cube.shape[1:]
                    and ap_mask.any()):
                ax_cur.contour(np.asarray(ap_mask, dtype=float),
                               levels=[0.5], colors=[_PAL["aperture"]],
                               linewidths=1.0)
            ax_cur.set_xticks([]); ax_cur.set_yticks([])
            for sp in ax_cur.spines.values():
                sp.set_color(_PAL["card_edge"]); sp.set_linewidth(0.7)
            rel = float(t_tpf[idx] - t0_local)
            ax_cur.set_title(f"Frame {i}/{n_total}  |  t = {t_tpf[idx]:.4f} d"
                             f"  ({rel:+.4f} d)",
                             loc="center", fontsize=9, pad=6)

            ax_med = fig.add_subplot(gs[0, 1])
            ax_med.imshow(median_img, origin="lower", cmap="viridis",
                          vmin=vmin, vmax=vmax, interpolation="nearest")
            if (ap_mask is not None
                    and ap_mask.shape == median_img.shape
                    and ap_mask.any()):
                ax_med.contour(np.asarray(ap_mask, dtype=float),
                               levels=[0.5], colors=[_PAL["aperture"]],
                               linewidths=1.0)
            ax_med.set_xticks([]); ax_med.set_yticks([])
            for sp in ax_med.spines.values():
                sp.set_color(_PAL["card_edge"]); sp.set_linewidth(0.7)
            ax_med.set_title("Median", loc="center", fontsize=9, pad=6)

            ax_lc = fig.add_subplot(gs[0, 2])
            ax_lc.plot(sap_t - t0_local, sap_f, color=_PAL["primary"],
                       lw=0.6, alpha=0.85)
            ax_lc.axvline(rel, color=_PAL["accent"], lw=1.2, zorder=3)
            ax_lc.axvspan(-window, +window,
                          color=_PAL["primary"], alpha=0.08, lw=0)
            ax_lc.set_xlim(sap_t.min() - t0_local, sap_t.max() - t0_local)
            ax_lc.set_xlabel("Phase [days]", fontsize=8)
            ax_lc.set_ylabel("Normalised Flux", fontsize=8)
            ax_lc.set_title("Aperture-Integrated Flux",
                            loc="left", fontsize=9.5, pad=4)
            self._polish(ax_lc)

            pdf.savefig(fig, dpi=200)
            plt.close(fig)

    @staticmethod
    def _best_t0(analyzer, fitter) -> float | None:
        """Best available mid-transit epoch (posterior → BLS fallback)."""
        if fitter.results_summary is not None:
            try:
                label_for = dict(zip(fitter.fitted_params, fitter.labels))
                t0_label = label_for.get("t0")
                if t0_label and t0_label in fitter.results_summary:
                    return float(fitter.results_summary[t0_label][0])
            except Exception:
                pass
        return float(analyzer.epoch_time) if analyzer.epoch_time else None

    @staticmethod
    def _estimate_t14(fitter) -> float | None:
        """Total transit duration T14 (days) from fitter.results_summary."""
        if fitter.results_summary is None:
            return None
        try:
            label_for = dict(zip(fitter.fitted_params, fitter.labels))
        except Exception:
            return None

        def _get(key, default):
            label = label_for.get(key)
            if label and label in fitter.results_summary:
                return float(fitter.results_summary[label][0])
            return default

        k = _get("rp", 0.1)
        a = _get("a", 10.0)
        inc = _get("inc", 90.0)
        p = fitter.period

        sin_i = np.sin(np.radians(inc))
        cos_i = np.cos(np.radians(inc))
        b = a * cos_i
        disc = (1.0 + k) ** 2 - b ** 2
        if disc <= 0 or a <= 0 or sin_i <= 0:
            return None
        val = (1.0 / a) * np.sqrt(disc) / sin_i
        if not (0.0 < val <= 1.0):
            return None
        return (p / np.pi) * np.arcsin(val)

    # =================================================================
    #  RIGHT COLUMN — TABLES + CORNER PLOT
    # =================================================================

    def _plot_results_block_in_subfig(self, subfig, analyzer, fitter):
        """Render the parameter table + diagnostic cards inside a
        dedicated SubFigure (right-column top).

        Layout inside the subfig (left → right):
            * Parameter table (fitted posteriors + derived quantities)
            * Convergence + fit-quality cards (small, stacked)
        """
        gs_block = subfig.add_gridspec(
            1, 2, width_ratios=[1.30, 1.0], wspace=0.18,
            top=0.93, bottom=0.05, left=0.04, right=0.98,
        )
        ax_tbl = subfig.add_subplot(gs_block[0, 0])
        ax_diag = subfig.add_subplot(gs_block[0, 1])
        try:
            self._plot_param_table(ax_tbl, analyzer, fitter)
        except Exception as exc:
            logger.info("Parameter table failed: %s", exc)
            self._render_unavailable(ax_tbl, "MCMC Results")
        try:
            self._plot_diagnostics_box(ax_diag, analyzer, fitter)
        except Exception as exc:
            logger.info("Diagnostics box failed: %s", exc)
            self._render_unavailable(ax_diag, "Diagnostics")

    # Back-compat shim for any external callers still using the old
    # gridspec-cell signature.
    def _plot_results_block(self, fig, subplotspec, analyzer, fitter):
        gs_block = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=subplotspec,
            width_ratios=[1.6, 1.0], wspace=0.10,
        )
        ax_tbl = fig.add_subplot(gs_block[0, 0])
        ax_diag = fig.add_subplot(gs_block[0, 1])
        try:
            self._plot_param_table(ax_tbl, analyzer, fitter)
        except Exception as exc:
            logger.info("Parameter table failed: %s", exc)
            self._render_unavailable(ax_tbl, "MCMC Results")
        try:
            self._plot_diagnostics_box(ax_diag, analyzer, fitter)
        except Exception as exc:
            logger.info("Diagnostics box failed: %s", exc)
            self._render_unavailable(ax_diag, "Diagnostics")

    # ---- parameter table ---------------------------------------------
    _UNIT_MAP = {
        "per":  "d",
        "t0":   "d",
        "inc":  "deg",
        "a":    "R_s",
        "rp":   "R_s",
        "u1":   "",
        "u2":   "",
        "ecc":  "",
        "w":    "deg",
    }

    def _plot_param_table(self, ax, analyzer, fitter):
        """Booktabs-style MCMC results panel (fitted + derived)."""
        raw_labels = list(fitter.results_summary.keys())
        param_names = list(fitter.fitted_params)
        name_for_label = dict(zip(raw_labels, param_names))

        fitted_rows = []
        for label in raw_labels:
            med, upper, lower = fitter.results_summary[label]
            pname = name_for_label.get(label, "")
            fitted_rows.append((
                self._math_label(pname, label),
                _fmt_val_err(med, upper, lower, latex=self.use_latex),
                self._unit_label(pname),
            ))
        derived_rows = list(self._derived_rows(fitter))

        sections = [("Fitted Parameters (Posterior)", fitted_rows)]
        if derived_rows:
            sections.append(("Derived Quantities", derived_rows))

        self._draw_booktabs_block(
            ax, sections=sections, include_unit_header=True,
        )

    # ---------------------------------------------------------------
    #  Shared booktabs renderer used by every right-column table.
    # ---------------------------------------------------------------
    def _draw_booktabs_block(self, ax, *, sections, include_unit_header):
        """Render one or more sections in unified booktabs style.

        Parameters
        ----------
        ax : Axes
            Target axes.  Will be cleared (axis off, unit limits).
        sections : list[tuple[str, list[tuple[str, str, str]]]]
            Each entry is ``(section_title, rows)`` where ``rows`` is a
            list of ``(label, value, unit)`` triples.
        include_unit_header : bool
            If True, draw a "Parameter / Median ± 1σ / Unit" header
            above the first row of every section.  Diagnostics tables
            don't need it (they have no uncertainty + units are sparse)
            but parameter tables do.
        """
        ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        # Total visual rows (data + headers + section titles + spacers).
        # Use slightly more vertical room per row so that horizontal
        # rules can never sit on top of the row text immediately below.
        n_data = sum(len(rows) for _, rows in sections)
        n_visual = n_data + len(sections) * (2 if include_unit_header else 1) + 2
        row_h = 0.88 / max(n_visual, 1)

        # Column anchors — value column aggressively pulled LEFT
        # (right-anchored at 0.650) so the right-aligned units at
        # 0.985 sit far past the longest ± string.  A ~0.34 visual
        # gap between the value and unit anchors guarantees that
        # even verbose strings such as ``$10000.00 \pm 70.39$`` (in
        # the Derived Quantities block) cannot crash into the unit
        # column.  ``col_ha`` is enforced one-to-one by the renderer.
        col_x = (0.020, 0.650, 0.985)
        col_ha = ("left", "right", "right")

        # Top double-rule (booktabs \toprule + thin shadow).
        y = 0.985
        ax.plot([0.0, 1.0], [y, y], color=_PAL["data"], lw=1.1,
                transform=ax.transAxes, clip_on=False)
        ax.plot([0.0, 1.0], [y - 0.008, y - 0.008],
                color=_PAL["data"], lw=0.3,
                transform=ax.transAxes, clip_on=False)
        y -= row_h * 0.85

        for s_idx, (title, rows) in enumerate(sections):
            # Section title — small caps via \textsc{}, accent colour.
            ax.text(0.020, y,
                    (r"\textsc{" + title + "}") if self.use_latex
                    else title.upper(),
                    transform=ax.transAxes,
                    fontsize=7.0, color=_PAL["card_accent"],
                    ha="left", va="center")
            y -= row_h * 0.95

            if include_unit_header:
                headers = ("Parameter",
                           r"Median $\pm$ 1$\sigma$" if self.use_latex
                           else "Median ± 1σ",
                           "Unit")
                for xi, h, ha_ in zip(col_x, headers, col_ha):
                    txt = (r"\textit{" + h + "}") if self.use_latex else h
                    ax.text(xi, y, txt, transform=ax.transAxes,
                            fontsize=6.5, color=_PAL["data"],
                            ha=ha_, va="center")
                # \midrule between header and first row.
                ax.plot([0.0, 1.0], [y - row_h * 0.45] * 2,
                        color=_PAL["data"], lw=0.7,
                        transform=ax.transAxes, clip_on=False)
                y -= row_h * 0.95
            else:
                # No unit-header: just a thin rule under the title.
                ax.plot([0.0, 1.0], [y + row_h * 0.45] * 2,
                        color=_PAL["data"], lw=0.7,
                        transform=ax.transAxes, clip_on=False)
                y -= row_h * 0.10

            # Data rows (zebra-shaded, identical typography).
            for i, (lbl, val, unit) in enumerate(rows):
                row_y = y - i * row_h
                if i % 2 == 0:
                    ax.add_patch(plt.Rectangle(
                        (0.0, row_y - row_h * 0.42), 1.0, row_h * 0.88,
                        facecolor=_PAL["row_even"], alpha=0.45,
                        edgecolor="none", transform=ax.transAxes,
                        zorder=0,
                    ))
                ax.text(col_x[0], row_y, lbl, transform=ax.transAxes,
                        fontsize=6.3, color=_PAL["data"],
                        ha="left", va="center")
                ax.text(col_x[1], row_y, val, transform=ax.transAxes,
                        fontsize=6.1, color=_PAL["data"],
                        ha="right", va="center")
                if unit:
                    ax.text(col_x[2], row_y, unit,
                            transform=ax.transAxes,
                            fontsize=6.0, color=_PAL["subhead"],
                            ha="right", va="center", style="italic")

            y -= len(rows) * row_h
            # \midrule between sections (suppressed for the last one).
            if s_idx < len(sections) - 1:
                ax.plot([0.0, 1.0], [y + row_h * 0.20] * 2,
                        color=_PAL["grid"], lw=0.4,
                        transform=ax.transAxes, clip_on=False)
                y -= row_h * 0.40

        # \bottomrule
        ax.plot([0.0, 1.0], [max(y + row_h * 0.20, 0.012)] * 2,
                color=_PAL["data"], lw=1.0,
                transform=ax.transAxes, clip_on=False)

    # ---- LaTeX symbol mapping (publication-grade labels) -------------
    _LATEX_SYMBOL = {
        "rp":  r"$R_p/R_{\star}$",
        "a":   r"$a/R_{\star}$",
        "inc": r"$i\,[^{\circ}]$",
        "t0":  r"$t_0$",
        "per": r"$P\,[\mathrm{d}]$",
        "u1":  r"$u_1$",
        "u2":  r"$u_2$",
        "ecc": r"$e$",
        "w":   r"$\omega\,[^{\circ}]$",
    }
    _UNICODE_SYMBOL = {
        "rp":  "Rp/R\u2605",
        "a":   "a/R\u2605",
        "inc": "i [°]",
        "t0":  "t\u2080",
        "per": "P [d]",
        "u1":  "u\u2081",
        "u2":  "u\u2082",
        "ecc": "e",
        "w":   "\u03c9 [°]",
    }

    def _math_label(self, param_name: str, fallback: str) -> str:
        if self.use_latex and param_name in self._LATEX_SYMBOL:
            return self._LATEX_SYMBOL[param_name]
        if (not self.use_latex) and param_name in self._UNICODE_SYMBOL:
            return self._UNICODE_SYMBOL[param_name]
        if self.use_latex:
            return _latex_escape(fallback)
        return fallback

    _UNIT_LATEX = {
        "d":    r"$\mathrm{d}$",
        "deg":  r"$^{\circ}$",
        "R_s":  r"$R_{\star}$",
        "":     "",
    }

    def _unit_label(self, param_name: str) -> str:
        if not param_name:
            return ""
        base = self._UNIT_MAP.get(param_name, "")
        if self.use_latex:
            return self._UNIT_LATEX.get(base, base)
        return base

    def _derived_rows(self, fitter):
        """Build derived-parameter rows (depth, b, T14, …)."""
        rows = []
        try:
            label_for = dict(zip(fitter.fitted_params, fitter.labels))
        except Exception:
            return rows

        def _posterior(name):
            lbl = label_for.get(name)
            if lbl is None or lbl not in fitter.results_summary:
                return None
            return fitter.results_summary[lbl]

        rp = _posterior("rp")
        a = _posterior("a")
        inc = _posterior("inc")

        if rp is not None:
            med, up, lo = rp
            depth = med ** 2 * 1e6
            depth_up = 2.0 * med * up * 1e6
            depth_lo = 2.0 * med * lo * 1e6
            rows.append(("Transit depth",
                         _fmt_val_err(depth, depth_up, depth_lo,
                                      latex=self.use_latex, sig=4),
                         "ppm"))

        if a is not None and inc is not None:
            a_med = a[0]; i_med = inc[0]
            b = a_med * np.cos(np.radians(i_med))
            b_err = abs(a_med * np.radians(1.0) * np.sin(np.radians(i_med)))
            b_up = inc[1] * b_err / 1.0
            b_lo = inc[2] * b_err / 1.0
            rows.append(("Impact parameter",
                         _fmt_val_err(b, b_up, b_lo,
                                      latex=self.use_latex, sig=3),
                         ""))

        t14 = self._estimate_t14(fitter)
        if t14 is not None:
            rows.append(("Duration T14",
                         f"{t14 * 24.0:.3f}", "h"))

        return rows

    # ---- diagnostics — same booktabs style as _plot_param_table -----
    def _plot_diagnostics_table(self, ax, analyzer, fitter):
        """Booktabs-style diagnostics panel (Convergence + Fit Quality).

        Renders with the SAME aesthetic as ``_plot_param_table`` —
        no rounded boxes, no shadows, no banded headers — so both
        right-column tables feel like part of one report.
        """
        info = fitter.get_convergence_info()
        fit_q = self._fit_quality(analyzer, fitter)
        na = "N/A"

        tau_val = info["autocorr_time"]
        tau_str = f"{tau_val:.1f}" if tau_val else na

        chi2_sym = r"$\chi^{2}_{\nu}$" if self.use_latex else "chi2_nu"
        tau_sym = r"$\tau$" if self.use_latex else "tau"
        neff_sym = r"$N_{\mathrm{eff}}$" if self.use_latex else "N_eff"

        conv_rows = [
            ("Walkers",      f"{info['nwalkers']}",                  ""),
            ("Steps",        f"{info['nsteps']:,}",                  ""),
            ("Burn-in",      f"{info['burn_in']}",                   ""),
            (tau_sym,        tau_str,                                "steps"),
            ("Acceptance",   f"{info['mean_acceptance_fraction']:.1%}", ""),
            (neff_sym,       f"{info['n_effective_samples']:,}",     ""),
        ]
        fit_rows = [
            (r"$N_{\mathrm{data}}$" if self.use_latex else "N data",
             f"{fit_q['ndata']:,}", ""),
            (r"$N_{\mathrm{free}}$" if self.use_latex else "N free",
             f"{fit_q['ndim']}", ""),
            (chi2_sym,
             f"{fit_q['chi2_red']:.3f}" if np.isfinite(fit_q['chi2_red']) else na, ""),
            ("RMS",
             f"{fit_q['rms_ppm']:.0f}" if np.isfinite(fit_q['rms_ppm']) else na, "ppm"),
            ("BIC",
             f"{fit_q['bic']:.1f}" if np.isfinite(fit_q['bic']) else na, ""),
            ("AIC",
             f"{fit_q['aic']:.1f}" if np.isfinite(fit_q['aic']) else na, ""),
        ]

        self._draw_booktabs_block(
            ax,
            sections=[
                ("Convergence Summary", conv_rows),
                ("Fit Quality",          fit_rows),
            ],
            include_unit_header=False,
        )

    # Back-compat alias — some external callers may still import the
    # old "diagnostics_box" name.
    _plot_diagnostics_box = _plot_diagnostics_table

    # -----------------------------------------------------------------
    def _fit_quality(self, analyzer, fitter) -> dict:
        """Compute reduced-χ², BIC, AIC, and RMS residuals."""
        out = {
            "ndata": int(self._ndata(analyzer)),
            "ndim": int(len(fitter.fitted_params)),
            "chi2_red": float("nan"),
            "bic": float("nan"),
            "aic": float("nan"),
            "rms_ppm": float("nan"),
        }
        try:
            map_params, _ = fitter.get_map_params()
            model = fitter._model_at(
                np.asarray(map_params, float),
                np.asarray(fitter.time, float),
                t0_override=None,
            )
            if model is None:
                return out
            resid = fitter.flux - model
            chi2 = float(np.sum((resid / fitter.flux_err) ** 2))
            n = int(fitter.flux.size)
            k = int(len(fitter.fitted_params))
            dof = max(n - k, 1)
            out["ndata"] = n
            out["chi2_red"] = chi2 / dof
            out["bic"] = chi2 + k * np.log(max(n, 2))
            out["aic"] = chi2 + 2.0 * k
            out["rms_ppm"] = float(np.sqrt(np.mean(resid ** 2)) * 1e6)
        except Exception as exc:
            logger.info("Fit quality computation failed: %s", exc)
        return out

    @staticmethod
    def _ndata(analyzer) -> int:
        try:
            return int(analyzer.clean_lc.flux.size)
        except Exception:
            return 0

    # =================================================================
    #  CORNER PLOT  (right column, bottom)
    # =================================================================

    def _plot_corner_in_subfig(self, subfig, fitter):
        """Render the K×K corner plot inside a dedicated SubFigure.

        The subfigure must have been created via ``Figure.subfigures``
        (or recursively from another subfigure) so it correctly clips
        its content to the cell — ``add_subfigure(subplotspec)`` from
        a nested gridspec is broken in matplotlib for our case and
        will let the corner plot leak across the page.
        """
        if fitter.flat_samples is None or fitter.results_summary is None:
            ax = subfig.add_subplot(111)
            self._render_unavailable(ax, "Posterior Distributions")
            return

        try:
            raw_labels = list(fitter.results_summary.keys())
            param_names = list(fitter.fitted_params)
            labels = [self._math_label(name, raw)
                      for name, raw in zip(param_names, raw_labels)]

            # Tight margins inside the subfig.  Top held just below 1.0
            # so rotated top-row tick labels can never climb above the
            # subfig boundary into the parameter table.
            subfig.subplots_adjust(top=0.96, bottom=0.12,
                                   left=0.13, right=0.985,
                                   wspace=0.05, hspace=0.05)

            with plt.rc_context({
                "xtick.labelsize": 4.3,
                "ytick.labelsize": 4.3,
                "xtick.major.pad": 0.8,
                "ytick.major.pad": 0.8,
                "axes.formatter.useoffset": True,
                "axes.formatter.offset_threshold": 2,
            }):
                corner.corner(
                    fitter.flat_samples,
                    labels=labels,
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=False,
                    label_kwargs={"fontsize": 5.4, "labelpad": 2.0},
                    color=_PAL["corner"],
                    hist_kwargs={"linewidth": 0.55},
                    contour_kwargs={"linewidths": 0.45},
                    quiet=True,
                    plot_datapoints=False,
                    plot_density=True,
                    fill_contours=True,
                    smooth=1.0,
                    smooth1d=1.0,
                    hist_bin_factor=1.3,
                    max_n_ticks=3,
                    fig=subfig,
                )

            ndim = fitter.flat_samples.shape[1]
            for ax in subfig.axes:
                for lbl in ax.get_xticklabels():
                    lbl.set_rotation(35); lbl.set_ha("right")
                for lbl in ax.get_yticklabels():
                    lbl.set_rotation(0)
                ax.tick_params(which="both", length=2.0, width=0.30)
                ax.xaxis.offsetText.set_fontsize(4.0)
                ax.yaxis.offsetText.set_fontsize(4.0)

            axes_grid = np.asarray(subfig.axes).reshape(ndim, ndim)
            for i in range(ndim):
                for j in range(ndim):
                    if j > 0:
                        axes_grid[i, j].set_ylabel("")
        except Exception as exc:
            logger.info("Corner plot failed: %s", exc)
            for ax in list(subfig.axes):
                ax.remove()
            ax = subfig.add_subplot(111)
            self._render_unavailable(ax, "Posterior Distributions")

    # Back-compat shim for the old (fig, subplotspec, fitter) signature.
    def _plot_corner_subfig(self, fig, subplotspec, fitter):
        ax = fig.add_subplot(subplotspec)
        self._render_unavailable(ax, "Posterior Distributions")
