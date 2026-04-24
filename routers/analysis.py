"""
Exoplot Analysis API
--------------------
FastAPI router exposing the lightcurve → MCMC → report pipeline as a
tiny JSON API consumed by the single-page analysis front-end
(``templates/analysis.html`` + ``static/js/analysis.js``).

Design choices
--------------
* **Single in-memory session** – the app is a local scientific tool,
  not a multi-tenant service, so the analyzer / fitter live on a
  module-level ``_SESSION`` dict.  Restart the server for a clean slate.
* **Plots are rendered server-side** – each endpoint returns base64
  PNGs built with a lightweight Matplotlib (Agg) style that matches the
  DVR report aesthetic (``modules/reports.py``): serif + Computer-Modern
  mathtext for LaTeX-like labels, transparent background so the UI card
  shows through, and a theme-aware palette so plots look native in both
  dark and light UI modes.  The palette intentionally picks vibrant
  scientific colours (deep blue data, red model overlay) that "pop"
  without looking garish.
* **MCMC runs in a background thread** – progress is written to the
  shared session dict and polled by the UI.  A ``stage`` field
  (``preprocessing`` | ``sampling`` | ``done``) disambiguates the
  20-second optimiser warm-up from the emcee main loop so the UI can
  show a meaningful multi-stage loader.

Author: S. Wittmann
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from modules.lightcurve import LightCurveAnalyzer
from modules.mcmc import TransitFitter
from modules.i18n import t as _t, normalise_locale

#: Single source of truth for the set of locales every server endpoint
#: accepts.  Anything else is coerced to ``en`` by :func:`normalise_locale`.
SUPPORTED_LANGS: tuple[str, ...] = ("en", "fr")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analysis"])

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
#  Session state (single-user, in-process)
# ---------------------------------------------------------------------

@dataclass
class AnalysisSession:
    target: str | None = None
    analyzer: LightCurveAnalyzer | None = None
    fitter: TransitFitter | None = None
    stellar_info: dict[str, Any] = field(default_factory=dict)
    mcmc_status: dict[str, Any] = field(default_factory=lambda: {
        "state": "idle",      # idle | queued | running | done | error
        "stage": "idle",      # idle | preprocessing | sampling | done | error
        "step": 0,
        "total": 0,
        "message": "",
    })
    mcmc_thread: threading.Thread | None = None
    report_filename: str | None = None


_SESSION = AnalysisSession()
_LOCK = threading.Lock()


def _reset_session() -> None:
    with _LOCK:
        _SESSION.target = None
        _SESSION.analyzer = None
        _SESSION.fitter = None
        _SESSION.stellar_info = {}
        _SESSION.mcmc_status = {
            "state": "idle",
            "stage": "idle",
            "step": 0,
            "total": 0,
            "message": "",
        }
        _SESSION.mcmc_thread = None
        _SESSION.report_filename = None


# ---------------------------------------------------------------------
#  Plotting — theme-aware, academic style with mathtext
# ---------------------------------------------------------------------
#
#  The look mirrors ``modules/reports.py`` (serif, Computer-Modern math,
#  tick-in axes, subtle grid) but uses matplotlib's *mathtext* engine
#  rather than a real LaTeX install — so ``$R_p/R_\star$`` still
#  renders with italic math even when pdflatex is unavailable.  All
#  figures are saved with ``transparent=True`` and ``facecolor='none'``
#  so the UI card background shows through; text, ticks, and grid
#  adapt to the active theme via the palette below.
# ---------------------------------------------------------------------

_STYLE_BASE: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": [
        "DejaVu Serif", "Bitstream Vera Serif",
        "Computer Modern Roman", "serif",
    ],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 2.6,
    "figure.autolayout": True,
    "figure.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
    "axes.facecolor": "none",
    "axes.grid": True,
    "grid.linewidth": 0.45,
    "axes.linewidth": 0.9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
}


def _theme_style(theme: str) -> dict[str, Any]:
    dark = (theme or "dark").lower() == "dark"
    fg = "#f4f4f5" if dark else "#1d1d1f"
    muted = "#a1a1aa" if dark else "#5c5c61"
    grid = "#52525b" if dark else "#c9c9d0"
    return {
        **_STYLE_BASE,
        "text.color": fg,
        "axes.labelcolor": fg,
        "axes.titlecolor": fg,
        "axes.edgecolor": muted,
        "xtick.color": muted,
        "ytick.color": muted,
        "grid.color": grid,
        "grid.alpha": 0.32 if dark else 0.55,
        "legend.labelcolor": fg,
    }


def _theme_palette(theme: str) -> dict[str, str]:
    """Vibrant-but-academic palette that picks colours with enough
    luminance contrast against the card background in each theme."""
    dark = (theme or "dark").lower() == "dark"
    if dark:
        return {
            "data":        "#5dade2",  # soft sky blue — data scatter
            "data_strong": "#19d3f3",  # cyan — binned / highlighted data
            "model":       "#ff7675",  # warm red — best-fit model overlay
            "accent":      "#ffa15a",  # orange — secondary emphasis
            "segment":     "#ffa15a",
            "peak":        "#ffa15a",
            "odd":         "#ff7675",
            "even":        "#19d3f3",
        }
    return {
        "data":        "#1f4e79",      # deep academic blue
        "data_strong": "#154a86",
        "model":       "#c0392b",      # classic deep red
        "accent":      "#e67e22",
        "segment":     "#d35400",
        "peak":        "#e67e22",
        "odd":         "#c0392b",
        "even":        "#2980b9",
    }


def _polish(ax) -> None:
    """Paper-style grid tuning (mirrors ``ReportGenerator._polish``)."""
    ax.grid(True, which="major", alpha=0.35, lw=0.4)
    ax.grid(True, which="minor", alpha=0.15, lw=0.3)
    ax.tick_params(which="minor", length=1.8)


def _fig_to_b64(fig, *, dpi: int = 500) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _time_axis_label(analyzer: LightCurveAnalyzer, lang: str) -> str:
    """Match the native time-system shorthand used in the DVR report."""
    try:
        fmt = str(analyzer.clean_lc.time.format).upper()
    except Exception:
        fmt = ""
    if fmt == "BTJD":
        return _t("ui_axis_time_btjd", lang)
    if fmt == "BKJD":
        return _t("ui_axis_time_bkjd", lang)
    return _t("ui_axis_time_generic", lang)


def _flux_err_array(lc) -> "np.ndarray | None":
    """Return a 1-D float array of flux errors, or ``None`` if the
    lightcurve doesn't carry meaningful uncertainties."""
    if lc is None:
        return None
    raw = getattr(lc, "flux_err", None)
    if raw is None:
        return None
    try:
        arr = np.asarray(
            raw.value if hasattr(raw, "value") else raw, dtype=float)
    except Exception:
        return None
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return None
    return arr


def _draw_points(ax, x, y, *, yerr=None, color: str, show_errors: bool,
                 size: float = 2.5, alpha_scatter: float = 0.75,
                 alpha_err: float = 0.28,
                 label: str | None = None) -> None:
    """Draw "scatter ± σ" depending on the error-bar toggle.

    ``plt.errorbar`` is more expensive than ``scatter``, so we only
    route through it when the caller actually asked for error bars and
    the lightcurve carries a usable ``flux_err`` column."""
    if show_errors and yerr is not None and yerr.shape == y.shape:
        ax.errorbar(
            x, y, yerr=yerr,
            fmt="o", ms=max(np.sqrt(size), 1.5),
            mfc=color, mec=color, ecolor=color,
            elinewidth=0.5, capsize=0,
            alpha=alpha_scatter, lw=0,
            zorder=2, rasterized=True,
            label=label,
        )
    else:
        ax.scatter(x, y, s=size, c=color, alpha=alpha_scatter,
                   edgecolors="none", rasterized=True, label=label)


def _plot_raw_lightcurve(analyzer: LightCurveAnalyzer, *,
                         theme: str = "dark", lang: str = "en",
                         show_errors: bool = False) -> str:
    pal = _theme_palette(theme)
    with plt.rc_context(_theme_style(theme)):
        fig, ax = plt.subplots(figsize=(9.0, 3.1))
        t = np.asarray(analyzer.clean_lc.time.value, dtype=float)
        f = np.asarray(analyzer.clean_lc.flux.value, dtype=float)
        ferr = _flux_err_array(analyzer.clean_lc)
        if analyzer.display_time is not None:
            x = analyzer.display_time
            ax.set_xlabel(_t("ui_axis_time_obs", lang))
        else:
            x = t
            ax.set_xlabel(_time_axis_label(analyzer, lang))
        _draw_points(ax, x, f, yerr=ferr, color=pal["data"],
                     show_errors=show_errors,
                     size=2.5, alpha_scatter=0.75)
        if analyzer.segment_edges is not None and len(analyzer.segment_edges):
            for e in analyzer.segment_edges:
                ax.axvline(e, color=pal["segment"], alpha=0.35, lw=0.6)
        ax.set_ylabel(_t("ui_axis_flux", lang))
        title = _t("ui_title_raw_lc", lang, target=analyzer.target_name)
        if analyzer.selection_label:
            title += rf"  [{analyzer.selection_label}]"
        ax.set_title(title, pad=8)
        _polish(ax)
        return _fig_to_b64(fig)


def _plot_periodogram(analyzer: LightCurveAnalyzer, *,
                      theme: str = "dark", lang: str = "en",
                      show_errors: bool = False) -> str:
    # Periodograms carry no per-point errors; accept the flag for
    # API symmetry but ignore it.
    del show_errors
    pal = _theme_palette(theme)
    with plt.rc_context(_theme_style(theme)):
        fig, ax = plt.subplots(figsize=(9.0, 3.0))
        pg = analyzer.periodogram
        period = pg.period.value
        power = pg.power.value
        ax.plot(period, power, color=pal["data"], lw=1.0)
        ax.axvline(
            analyzer.best_period, color=pal["peak"], lw=1.0, ls="--",
            label=_t("ui_legend_peak_period", lang,
                     period=float(analyzer.best_period)))
        ax.set_xscale("log")
        ax.set_xlabel(_t("ui_axis_period", lang))
        ax.set_ylabel(_t("ui_axis_power", lang))
        ax.set_title(_t("ui_title_periodogram", lang), pad=8)
        ax.legend(loc="best")
        _polish(ax)
        return _fig_to_b64(fig)


def _plot_folded(analyzer: LightCurveAnalyzer, *,
                 theme: str = "dark", lang: str = "en",
                 show_errors: bool = False) -> str:
    pal = _theme_palette(theme)
    with plt.rc_context(_theme_style(theme)):
        fig, ax = plt.subplots(figsize=(9.0, 3.1))
        f = analyzer.folded_lc
        phase = np.asarray(f.time.value, dtype=float)
        flux = np.asarray(f.flux.value, dtype=float)
        ferr = _flux_err_array(f)
        _draw_points(ax, phase, flux, yerr=ferr, color=pal["data"],
                     show_errors=show_errors,
                     size=2.5, alpha_scatter=0.55)
        ax.set_xlabel(_t("ui_axis_phase", lang))
        ax.set_ylabel(_t("ui_axis_flux", lang))
        ax.set_title(
            _t("ui_title_folded_initial", lang,
               period=float(analyzer.best_period)),
            pad=8)
        p = float(analyzer.best_period)
        ax.set_xlim(-0.15 * p, 0.15 * p)
        _polish(ax)
        return _fig_to_b64(fig)


def _plot_folded_with_model(analyzer: LightCurveAnalyzer,
                            fitter: TransitFitter, *,
                            theme: str = "dark", lang: str = "en",
                            show_errors: bool = False) -> str:
    pal = _theme_palette(theme)
    with plt.rc_context(_theme_style(theme)):
        fig, ax = plt.subplots(figsize=(9.0, 3.4))
        f = analyzer.folded_lc
        phase = np.asarray(f.time.value, dtype=float)
        flux = np.asarray(f.flux.value, dtype=float)
        ferr = _flux_err_array(f)
        _draw_points(ax, phase, flux, yerr=ferr, color=pal["data"],
                     show_errors=show_errors,
                     size=2.8, alpha_scatter=0.55,
                     label=_t("ui_legend_folded_data", lang))
        try:
            t_model, model = fitter.get_best_model_curve(
                num_points=2000, phase_folded=True, mode="map")
            ax.plot(t_model, model, color=pal["model"], lw=1.8,
                    label=_t("ui_legend_best_fit_map", lang))
        except Exception as exc:  # pragma: no cover
            logger.warning("Model overlay failed: %s", exc)
        ax.set_xlabel(_t("ui_axis_phase", lang))
        ax.set_ylabel(_t("ui_axis_flux", lang))
        ax.set_title(_t("ui_title_folded_with_model", lang), pad=8)
        p = float(analyzer.best_period)
        ax.set_xlim(-0.1 * p, 0.1 * p)
        ax.legend(loc="lower right")
        _polish(ax)
        return _fig_to_b64(fig)


def _plot_odd_even(analyzer: LightCurveAnalyzer, *,
                   theme: str = "dark", lang: str = "en",
                   show_errors: bool = False) -> str:
    """Odd/even transit comparison — flags eclipsing-binary impostors
    whose alternating depths only show up when you split the transits."""
    pal = _theme_palette(theme)
    with plt.rc_context(_theme_style(theme)):
        fig, (ax_odd, ax_even) = plt.subplots(
            1, 2, figsize=(9.0, 3.3), sharey=True)
        lc = analyzer.clean_lc
        period = float(analyzer.best_period)
        t0 = float(analyzer.epoch_time)
        t = np.asarray(lc.time.value, dtype=float)
        f = np.asarray(lc.flux.value, dtype=float)
        ferr = _flux_err_array(lc)
        cycle = np.floor((t - t0 + 0.5 * period) / period).astype(int)
        phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
        win = np.abs(phase) < 0.15 * period

        odd_mask = win & (cycle % 2 == 1)
        even_mask = win & (cycle % 2 == 0)
        for ax, mask, title_key, color in (
            (ax_odd, odd_mask, "ui_odd_transits", pal["odd"]),
            (ax_even, even_mask, "ui_even_transits", pal["even"]),
        ):
            sub_err = ferr[mask] if ferr is not None else None
            _draw_points(ax, phase[mask], f[mask], yerr=sub_err,
                         color=color, show_errors=show_errors,
                         size=3.0, alpha_scatter=0.6)
            ax.set_xlabel(_t("ui_axis_phase_short", lang))
            ax.set_title(_t(title_key, lang), pad=6)
            ax.set_xlim(-0.08 * period, 0.08 * period)
            _polish(ax)
        ax_odd.set_ylabel(_t("ui_axis_flux", lang))
        return _fig_to_b64(fig)


def _plot_corner(fitter: TransitFitter, *,
                 theme: str = "dark", lang: str = "en",
                 show_errors: bool = False) -> str:
    # A corner plot has no per-point errors; accept for API symmetry.
    del show_errors, lang
    try:
        import corner  # type: ignore
    except Exception as exc:
        logger.warning("Corner not available: %s", exc)
        return ""
    pal = _theme_palette(theme)
    # Pretty-print parameter names in LaTeX via mathtext.
    latex_labels = [_LATEX_NAME.get(lbl, lbl) for lbl in fitter.labels]
    style = _theme_style(theme)
    style_no_titles = {**style, "axes.titlepad": 2.0}
    with plt.rc_context(style_no_titles):
        fig = corner.corner(
            fitter.flat_samples,
            labels=latex_labels,
            show_titles=True,
            title_fmt=".4f",
            title_kwargs={"fontsize": 8.5},
            label_kwargs={"fontsize": 9.5},
            color=pal["data_strong"],
            hist_kwargs={"color": pal["data_strong"], "linewidth": 1.2},
            plot_datapoints=False,
            fill_contours=True,
        )
        fig.patch.set_alpha(0.0)
        for ax in fig.get_axes():
            ax.patch.set_alpha(0.0)
            ax.tick_params(labelsize=7)
        return _fig_to_b64(fig, dpi=500)


# ---------------------------------------------------------------------
#  LaTeX label map for the UI + corner plot
# ---------------------------------------------------------------------

_LATEX_NAME: dict[str, str] = {
    "rp":  r"$R_p/R_{\star}$",
    "Rp/Rs": r"$R_p/R_{\star}$",
    "inc": r"$i$",
    "i":   r"$i$",
    "a":   r"$a/R_{\star}$",
    "a/Rs": r"$a/R_{\star}$",
    "t0":  r"$t_0$",
    "per": r"$P$",
    "P":   r"$P$",
    "u1":  r"$u_1$",
    "u2":  r"$u_2$",
    "ecc": r"$e$",
    "w":   r"$\omega$",
}


# ---------------------------------------------------------------------
#  Stellar information lookup
# ---------------------------------------------------------------------
#  Reuses ``ReportGenerator._auto_stellar_params`` (meta-header first,
#  TIC catalogue fallback) so the UI and the DVR report show identical
#  numbers.  The ReportGenerator is instantiated once, lazily, to avoid
#  re-running its matplotlib setup on every request.
# ---------------------------------------------------------------------

_REPORT_HELPER = None


def _get_report_helper():
    global _REPORT_HELPER
    if _REPORT_HELPER is None:
        from modules.reports import ReportGenerator  # local: heavy import
        _REPORT_HELPER = ReportGenerator(use_latex=False)
    return _REPORT_HELPER


def _extract_stellar_info(analyzer: LightCurveAnalyzer) -> dict[str, Any]:
    """Pull together star identifiers + stellar parameters for the
    small summary card shown above the raw LC plot."""
    info: dict[str, Any] = {"name": analyzer.target_name}

    raw = getattr(analyzer, "raw_lc", None)
    meta = {}
    if raw is not None:
        try:
            meta = dict(getattr(raw, "meta", {}) or {})
        except Exception:
            meta = {}
    upper = {str(k).upper(): v for k, v in meta.items()}

    def _first(keys, *, numeric: bool = False):
        for key in keys:
            v = upper.get(key)
            if v in (None, "", b""):
                continue
            if numeric:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
            return str(v).strip()
        return None

    tic = _first(("TICID", "TICID_MAST", "OBJECT", "TARGETID", "TARGET_NAME"))
    if tic:
        info["id"] = tic
    ra = _first(("RA_OBJ", "RA", "RA_TARG"), numeric=True)
    if ra is not None:
        info["ra"] = ra
    dec = _first(("DEC_OBJ", "DEC", "DEC_TARG"), numeric=True)
    if dec is not None:
        info["dec"] = dec
    for key in ("TELESCOP", "MISSION", "INSTRUME"):
        val = upper.get(key)
        if val:
            info["mission"] = str(val).strip()
            break

    # Stellar parameters (tmag, rs, teff, logg, mh, rho) via the same
    # helper the DVR report uses — so the UI card and the report row
    # agree to the last digit.
    try:
        params = _get_report_helper()._auto_stellar_params(analyzer)
        for k, v in params.items():
            if v is None:
                continue
            try:
                info[k] = float(v)
            except (TypeError, ValueError):
                info[k] = v
    except Exception as exc:
        logger.info("Stellar auto-fetch skipped: %s", exc)

    if analyzer.selection_label:
        info["selection"] = analyzer.selection_label
    return info


def _compute_snr(analyzer: LightCurveAnalyzer,
                 fitter: TransitFitter) -> float | None:
    """Transit depth divided by the per-point noise, scaled by √(in-transit
    samples).  Mirrors the SNR formula used inside
    ``TransitFitter._estimate_bounds``."""
    try:
        t = np.asarray(analyzer.clean_lc.time.value, float)
        f = np.asarray(analyzer.clean_lc.flux.value, float)
        period = float(analyzer.best_period)
        t0 = float(analyzer.epoch_time)
        phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period

        wing = np.abs(phase) > 0.15 * period
        if wing.sum() < 20:
            wing = np.abs(phase) > 0.05 * period
        baseline = float(np.median(f[wing])) if wing.any() else float(np.median(f))
        mad = float(np.median(np.abs(f[wing] - baseline))) if wing.any() else 0.0
        sigma = 1.4826 * mad if mad > 0 else float(np.std(f))
        sigma = max(sigma, 1e-6)

        summary = fitter.results_summary or {}
        rp_entry = summary.get("rp") or summary.get("Rp/Rs") or summary.get("r_p")
        if rp_entry is None:
            rp = 0.1
        else:
            rp = float(rp_entry[0])
        depth = rp ** 2
        in_transit = np.abs(phase) < 0.02 * period
        n_in = max(int(in_transit.sum()), 1)
        return float((depth / sigma) * np.sqrt(n_in))
    except Exception as exc:
        logger.warning("SNR calc failed: %s", exc)
        return None


# ---------------------------------------------------------------------
#  Request / Response models
# ---------------------------------------------------------------------

class SearchRequest(BaseModel):
    target: str = Field(..., min_length=1, description="Star name or TIC ID")
    mission: list[str] | None = None
    author: str | None = None
    year: int | None = None
    quarter: int | None = None
    campaign: int | None = None
    sector: int | None = None
    exptime: float | str | None = None
    limit: int | None = Field(None, ge=1, le=500)


class DownloadRequest(BaseModel):
    indices: list[int] = Field(..., min_length=1)
    theme: str = Field("dark", pattern="^(dark|light)$")
    lang: str = Field("en", pattern="^(en|fr)$")
    show_errors: bool = False


class MCMCRequest(BaseModel):
    fitted_params: list[str] | None = None
    auto_bounds: bool = True
    custom_bounds: list[list[float]] | None = None
    custom_x0: list[float] | None = None
    nwalkers: int = Field(32, ge=8, le=512)
    nsteps: int = Field(4000, ge=100, le=200_000)
    use_multiprocessing: bool = False
    n_cores: int | None = None


class ReportRequest(BaseModel):
    filename: str | None = None
    movie: bool = False
    use_latex: bool = False
    # ``en`` | ``fr`` only — anything else is coerced server-side.
    lang: str = Field("en", pattern="^(en|fr)$")


# ---------------------------------------------------------------------
#  Routes
# ---------------------------------------------------------------------

@router.post("/reset")
async def reset_session():
    _reset_session()
    return {"ok": True}


@router.get("/state")
async def state():
    return {
        "target": _SESSION.target,
        "has_analyzer": _SESSION.analyzer is not None,
        "has_fitter": _SESSION.fitter is not None,
        "mcmc_status": _SESSION.mcmc_status,
        "report_filename": _SESSION.report_filename,
    }


# ---- STEP 2 -----------------------------------------------------------------

@router.post("/search")
async def search(req: SearchRequest):
    _reset_session()
    _SESSION.target = req.target.strip()
    analyzer = LightCurveAnalyzer(_SESSION.target)

    kwargs: dict[str, Any] = {}
    if req.mission:
        kwargs["mission"] = tuple(req.mission)
    if req.author:
        kwargs["author"] = req.author
    if req.sector is not None:
        kwargs["sector"] = req.sector
    if req.quarter is not None:
        kwargs["quarter"] = req.quarter
    if req.campaign is not None:
        kwargs["campaign"] = req.campaign
    if req.exptime is not None:
        kwargs["exptime"] = req.exptime
    if req.limit is not None:
        kwargs["limit"] = req.limit

    try:
        df = analyzer.search(**kwargs)
    except Exception as exc:
        logger.exception("MAST search failed")
        raise HTTPException(status_code=502,
                            detail=f"MAST search failed: {exc}")

    if req.year is not None and not df.empty and "year" in df.columns:
        df = df[pd.to_numeric(df["year"], errors="coerce") == req.year].copy()

    _SESSION.analyzer = analyzer

    rows: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                v = None
            clean[str(k)] = v
        rows.append(clean)

    return {
        "target": analyzer.target_name,
        "count": len(rows),
        "columns": [str(c) for c in df.columns],
        "rows": rows,
    }


# ---- STEP 3 -----------------------------------------------------------------

@router.post("/download")
async def download(req: DownloadRequest):
    if _SESSION.analyzer is None:
        raise HTTPException(status_code=400,
                            detail="No active search — call /api/search first.")
    analyzer = _SESSION.analyzer

    try:
        analyzer.download_and_clean(indices=list(req.indices))
        analyzer.compute_periodogram()
        analyzer.fold_lightcurve()
    except Exception as exc:
        logger.exception("Download / processing failed")
        raise HTTPException(status_code=500, detail=str(exc))

    _SESSION.stellar_info = _extract_stellar_info(analyzer)

    lang = normalise_locale(req.lang)
    return {
        "target": analyzer.target_name,
        "selection_label": analyzer.selection_label,
        "best_period": float(analyzer.best_period),
        "best_power": float(analyzer.best_power),
        "epoch_time": float(analyzer.epoch_time),
        "n_points": int(len(analyzer.clean_lc.flux)),
        "stellar": _SESSION.stellar_info,
        "plots": {
            "raw": _plot_raw_lightcurve(
                analyzer, theme=req.theme, lang=lang,
                show_errors=req.show_errors),
            "periodogram": _plot_periodogram(
                analyzer, theme=req.theme, lang=lang,
                show_errors=req.show_errors),
            "folded": _plot_folded(
                analyzer, theme=req.theme, lang=lang,
                show_errors=req.show_errors),
        },
    }


@router.get("/plots/pipeline")
async def plots_pipeline(theme: str = "dark", lang: str = "en",
                         show_errors: bool = False):
    """Re-render the 3 validation plots in a new theme/language/error-bar
    state without touching the analyzer — invoked whenever the user
    toggles dark/light, EN/FR or the "Show error bars" checkbox."""
    analyzer = _SESSION.analyzer
    if analyzer is None or analyzer.clean_lc is None:
        raise HTTPException(status_code=400,
                            detail="No pipeline data in session.")
    theme = "light" if str(theme).lower() == "light" else "dark"
    lang = normalise_locale(lang)
    return {
        "plots": {
            "raw": _plot_raw_lightcurve(
                analyzer, theme=theme, lang=lang, show_errors=show_errors),
            "periodogram": _plot_periodogram(
                analyzer, theme=theme, lang=lang, show_errors=show_errors),
            "folded": _plot_folded(
                analyzer, theme=theme, lang=lang, show_errors=show_errors),
        }
    }


# ---- STEP 4 (MCMC) ----------------------------------------------------------

def _run_mcmc_job(req: MCMCRequest) -> None:
    """Thread body: builds the fitter and runs emcee.  All exceptions
    are captured and surfaced via ``mcmc_status``.

    We explicitly stamp the ``stage`` field so the UI can distinguish
    the ~20 s L-BFGS-B + DE optimiser warm-up from the main sampling
    loop (which only then starts emitting step/total updates)."""
    try:
        analyzer = _SESSION.analyzer
        if analyzer is None:
            raise RuntimeError("No analyzer available.")

        _SESSION.mcmc_status.update(
            state="running", stage="preprocessing",
            step=0, total=int(req.nsteps),
            message="Pre-processing & initial optimisation…")

        t, f, ferr, per, t0 = analyzer.get_mcmc_data(folded=False)

        kw: dict[str, Any] = dict(auto_bounds=req.auto_bounds)
        if req.fitted_params:
            kw["fitted_params"] = list(req.fitted_params)
        if req.custom_bounds:
            kw["custom_bounds"] = [tuple(pair) for pair in req.custom_bounds]
        if req.custom_x0:
            kw["custom_x0"] = list(req.custom_x0)

        fitter = TransitFitter(t, f, ferr, period=per, t0=t0, **kw)
        _SESSION.fitter = fitter

        _SESSION.mcmc_status.update(
            state="running", stage="preprocessing",
            step=0, total=int(req.nsteps),
            message="Polishing initial guess (L-BFGS-B + DE)…")

        def _cb(step: int, total: int) -> None:
            _SESSION.mcmc_status.update(
                state="running", stage="sampling",
                step=int(step), total=int(total),
                message=f"Sampling ({step}/{total})")

        fitter.run_mcmc(
            nwalkers=req.nwalkers,
            nsteps=req.nsteps,
            progress_callback=_cb,
            use_multiprocessing=req.use_multiprocessing,
            n_cores=req.n_cores,
        )

        # Keep data and model on the same phase axis (posterior P/t0).
        try:
            summary = fitter.results_summary
            if summary:
                per_entry = summary.get("per") or summary.get("Period") \
                    or summary.get("period")
                t0_entry = summary.get("t0")
                if per_entry and t0_entry:
                    analyzer.refold_with_posterior(
                        period=float(per_entry[0]),
                        epoch_time=float(t0_entry[0]))
                elif t0_entry:
                    analyzer.refold_with_posterior(
                        period=analyzer.best_period,
                        epoch_time=float(t0_entry[0]))
        except Exception as exc:  # pragma: no cover
            logger.info("Refold with posterior failed: %s", exc)

        _SESSION.mcmc_status.update(
            state="done", stage="done",
            step=int(req.nsteps), total=int(req.nsteps),
            message="MCMC complete.")
    except Exception as exc:
        logger.exception("MCMC run failed")
        _SESSION.mcmc_status.update(
            state="error", stage="error",
            message=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )


@router.post("/mcmc")
async def launch_mcmc(req: MCMCRequest):
    if _SESSION.analyzer is None or _SESSION.analyzer.clean_lc is None:
        raise HTTPException(status_code=400,
                            detail="No clean lightcurve — run /api/download first.")
    if (_SESSION.mcmc_thread is not None
            and _SESSION.mcmc_thread.is_alive()):
        raise HTTPException(status_code=409,
                            detail="An MCMC run is already in progress.")

    _SESSION.mcmc_status = {
        "state": "queued",
        "stage": "queued",
        "step": 0,
        "total": int(req.nsteps),
        "message": "Queued…",
    }
    _SESSION.fitter = None
    thread = threading.Thread(
        target=_run_mcmc_job, args=(req,),
        name="exoplot-mcmc", daemon=True)
    _SESSION.mcmc_thread = thread
    thread.start()
    return {"ok": True, "status": _SESSION.mcmc_status}


@router.get("/mcmc/status")
async def mcmc_status():
    return _SESSION.mcmc_status


# ---- STEP 5 (Results) -------------------------------------------------------

@router.get("/results")
async def results(theme: str = "dark", lang: str = "en",
                  show_errors: bool = False):
    fitter = _SESSION.fitter
    analyzer = _SESSION.analyzer
    if fitter is None or fitter.results_summary is None:
        raise HTTPException(status_code=400, detail="MCMC results not ready.")
    if analyzer is None:
        raise HTTPException(status_code=400, detail="No analyzer in session.")

    theme = "light" if str(theme).lower() == "light" else "dark"
    lang = normalise_locale(lang)

    summary_rows = []
    for label, (med, up, lo) in fitter.results_summary.items():
        summary_rows.append({
            "parameter": label,
            "latex": _LATEX_NAME.get(label, label),
            "median": float(med),
            "plus": float(up),
            "minus": float(lo),
        })

    diagnostics: dict[str, Any] = {}
    try:
        conv = fitter.get_convergence_info()
        diagnostics = {
            "nwalkers": conv.get("nwalkers"),
            "nsteps": conv.get("nsteps"),
            "burn_in": conv.get("burn_in"),
            "thin": conv.get("thin"),
            "autocorr_time": conv.get("autocorr_time"),
            "mean_acceptance_fraction": conv.get("mean_acceptance_fraction"),
            "n_effective_samples": conv.get("n_effective_samples"),
        }
    except Exception as exc:
        logger.info("Convergence diagnostics unavailable: %s", exc)

    snr = _compute_snr(analyzer, fitter)

    return {
        "target": analyzer.target_name,
        "summary": summary_rows,
        "diagnostics": diagnostics,
        "snr": snr,
        "best_period": float(analyzer.best_period),
        "epoch_time": float(analyzer.epoch_time),
        "plots": {
            "folded_model": _plot_folded_with_model(
                analyzer, fitter, theme=theme, lang=lang,
                show_errors=show_errors),
            "odd_even": _plot_odd_even(
                analyzer, theme=theme, lang=lang,
                show_errors=show_errors),
            "corner": _plot_corner(
                fitter, theme=theme, lang=lang,
                show_errors=show_errors),
        },
    }


@router.get("/plots/results")
async def plots_results(theme: str = "dark", lang: str = "en",
                        show_errors: bool = False):
    """Re-render the results plots in a new theme / language /
    error-bar state."""
    fitter = _SESSION.fitter
    analyzer = _SESSION.analyzer
    if fitter is None or fitter.results_summary is None or analyzer is None:
        raise HTTPException(status_code=400,
                            detail="No MCMC results in session.")
    theme = "light" if str(theme).lower() == "light" else "dark"
    lang = normalise_locale(lang)
    return {
        "plots": {
            "folded_model": _plot_folded_with_model(
                analyzer, fitter, theme=theme, lang=lang,
                show_errors=show_errors),
            "odd_even": _plot_odd_even(
                analyzer, theme=theme, lang=lang,
                show_errors=show_errors),
            "corner": _plot_corner(
                fitter, theme=theme, lang=lang,
                show_errors=show_errors),
        }
    }


# ---- Report -----------------------------------------------------------------

@router.post("/report")
async def generate_report(req: ReportRequest):
    fitter = _SESSION.fitter
    analyzer = _SESSION.analyzer
    if fitter is None or fitter.results_summary is None or analyzer is None:
        raise HTTPException(status_code=400,
                            detail="Run MCMC before generating a report.")

    from modules.reports import ReportGenerator

    target = analyzer.target_name or "exoplot"
    safe = "".join(c if c.isalnum() or c in "-_" else "_"
                   for c in target).strip("_") or "exoplot"
    filename = (req.filename or f"{safe}_DVR.pdf").strip()
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    try:
        gen = ReportGenerator(website_name="Exoplot",
                              website_url="www.exoplot.fr",
                              locale=normalise_locale(req.lang),
                              use_latex=req.use_latex)
        gen.generate_mcmc_report(
            filename, analyzer, fitter,
            tpf=False,
            movie=req.movie,
        )
    except Exception as exc:
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500,
                            detail=f"Report generation failed: {exc}")

    _SESSION.report_filename = filename
    return {"ok": True, "filename": filename,
            "url": f"/results/{filename}"}


@router.get("/report/download")
async def download_report():
    if not _SESSION.report_filename:
        raise HTTPException(status_code=404, detail="No report generated yet.")
    path = RESULTS_DIR / _SESSION.report_filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report file missing on disk.")
    return FileResponse(path, media_type="application/pdf",
                        filename=_SESSION.report_filename)
