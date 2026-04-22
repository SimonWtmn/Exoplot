"""
Publication-Quality Plotting Utilities
--------------------------------------
Provides `PlotStyle`, `CatalogPlotter`, and `TransitPlotter` classes built
on Matplotlib for generating print-ready figures of exoplanet populations,
lightcurves, periodograms, and MCMC diagnostics.

All methods return ``(fig, axes)`` tuples so the caller can display inline,
tweak further, or save to any format.

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.ndimage import gaussian_filter

from .constants import LABEL_MAP
from .models import MassRadiusModels


# ===========================================================
# Base Styling Configuration
# ===========================================================

_PALETTE = {
    "primary":    "#2980b9",
    "secondary":  "#e67e22",
    "accent":     "#e74c3c",
    "highlight":  "#c0392b",
    # Light-curve panels (match ``modules.reports`` DVR style)
    "data":       "#2c3e50",
    "error":      "#bdc3c7",
    "model_cycle": [
        "#e74c3c", "#2980b9", "#27ae60", "#8e44ad",
        "#f39c12", "#1abc9c", "#d35400", "#2c3e50",
    ],
}


class PlotStyle:
    """
    Static utility for consistent, publication-quality Matplotlib styling
    with LaTeX rendering by default (Computer Modern Roman).
    Call ``PlotStyle.configure()`` once at the start of a session.
    """

    _configured = False

    @staticmethod
    def configure(*, usetex: bool = True, font_family: str = "serif",
                  base_size: float = 10, dpi: int = 500):
        """Apply global rcParams for publication figures.

        LaTeX is enabled by default for crisp typesetting.  Pass
        ``usetex=False`` to fall back to Matplotlib's mathtext engine.
        """
        plt.rcParams.update({
            "font.family": font_family,
            "font.size": base_size,
            "axes.labelsize": base_size + 1,
            "axes.titlesize": base_size + 2,
            "xtick.labelsize": base_size - 1,
            "ytick.labelsize": base_size - 1,
            "legend.fontsize": base_size - 1,
            "legend.framealpha": 0.85,
            "legend.edgecolor": "0.7",
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": False,
            "text.usetex": usetex,
        })
        if usetex:
            plt.rcParams.update({
                "font.serif": ["Computer Modern Roman"],
                "text.latex.preamble": (
                    r"\usepackage{amssymb}" "\n"
                    r"\DeclareUnicodeCharacter{00B2}{\ensuremath{^2}}" "\n"
                    r"\DeclareUnicodeCharacter{00B3}{\ensuremath{^3}}" "\n"
                    r"\DeclareUnicodeCharacter{00B1}{\ensuremath{\pm}}" "\n"
                    r"\DeclareUnicodeCharacter{03C3}{\ensuremath{\sigma}}" "\n"
                    r"\DeclareUnicodeCharacter{03C4}{\ensuremath{\tau}}" "\n"
                    r"\DeclareUnicodeCharacter{2212}{\ensuremath{-}}" "\n"
                    r"\DeclareUnicodeCharacter{2295}{\ensuremath{\oplus}}" "\n"
                    r"\DeclareUnicodeCharacter{2299}{\ensuremath{\odot}}"
                ),
            })
        PlotStyle._configured = True

    @staticmethod
    def _ensure_configured():
        if not PlotStyle._configured:
            PlotStyle.configure()

    @staticmethod
    def get_label(col_name: str) -> str:
        """Translate a DataFrame column name into a human-readable label.

        HTML sub/superscript tags from the NEA label map are converted into
        LaTeX math so they render correctly with both ``usetex=True`` and
        Matplotlib's built-in mathtext engine.
        """
        import re
        raw = LABEL_MAP.get(col_name, col_name)
        raw = re.sub(r"<sub>(.*?)</sub>", r"$_{\1}$", raw)
        raw = re.sub(r"<sup>(.*?)</sup>", r"$^{\1}$", raw)
        return raw

    @staticmethod
    def polish(ax: plt.Axes, *, grid: bool = True, minor: bool = True):
        """Apply final cosmetic touches to an Axes."""
        if grid:
            ax.grid(True, alpha=0.20, linewidth=0.5, color="#cccccc")
        if minor:
            ax.minorticks_on()
            ax.tick_params(which="minor", length=2)


# ===========================================================
# Population & Catalog Visualization
# ===========================================================

class CatalogPlotter:
    """
    Generates publication-quality statistical plots for exoplanet populations.

    Every public method returns ``(fig, ax)`` or ``(fig, axes)`` so the
    caller can ``plt.show()`` or ``fig.savefig(...)`` as needed.
    """

    def __init__(self):
        PlotStyle._ensure_configured()
        self.model_loader = MassRadiusModels()

    def __repr__(self):
        return "CatalogPlotter(backend='matplotlib')"

    # ── private helpers ───────────────────────────────────────────

    def _add_model_overlays(self, ax: plt.Axes, x_col: str, y_col: str,
                            overlay_models: list | None):
        if not overlay_models:
            return
        valid_axes = {("pl_bmasse", "pl_rade"), ("pl_rade", "pl_bmasse")}
        if (x_col, y_col) not in valid_axes:
            return

        colors = _PALETTE["model_cycle"]
        for i, model_key in enumerate(overlay_models):
            try:
                mdf = self.model_loader.get_model_curve(model_key)
                label = self.model_loader.get_model_label(model_key)
                xm, ym = mdf["mass"].values, mdf["radius"].values
                if x_col == "pl_rade":
                    xm, ym = ym, xm
                ax.plot(xm, ym, ls="--", lw=1.8,
                        color=colors[i % len(colors)], label=label, zorder=5)
            except Exception as e:
                print(f"Warning: could not load model '{model_key}': {e}")

    @staticmethod
    def _clean(df: pd.DataFrame, cols: list, log_cols: list | None = None):
        """Drop NaN rows and non-positive values for log-scaled columns."""
        out = df.dropna(subset=cols).copy()
        for c in (log_cols or []):
            if c in out.columns:
                out = out[out[c] > 0]
        return out

    # ── public methods ────────────────────────────────────────────

    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, *,
                     color_by: str | None = None,
                     highlight_planets: list | None = None,
                     log_x: bool = False, log_y: bool = False,
                     overlay_models: list | None = None,
                     cmap: str = "plasma",
                     figsize: tuple = (7, 5.5),
                     title: str | None = None) -> tuple[plt.Figure, plt.Axes]:
        """Scatter plot with optional color mapping, model overlays, and highlights."""
        PlotStyle._ensure_configured()

        log_cols = []
        if log_x: log_cols.append(x_col)
        if log_y: log_cols.append(y_col)
        cdf = self._clean(df, [x_col, y_col], log_cols)

        fig, ax = plt.subplots(figsize=figsize)

        scatter_kw = dict(s=18, alpha=0.75, edgecolors="0.3",
                          linewidths=0.3, zorder=2, rasterized=True)

        if color_by and color_by in cdf.columns:
            cdf = cdf.dropna(subset=[color_by])
            sc = ax.scatter(cdf[x_col], cdf[y_col], c=cdf[color_by],
                            cmap=cmap, **scatter_kw)
            cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
            cbar.set_label(PlotStyle.get_label(color_by))
        else:
            ax.scatter(cdf[x_col], cdf[y_col],
                       color=_PALETTE["primary"], **scatter_kw)

        if highlight_planets:
            for name in highlight_planets:
                hp = cdf[cdf["pl_name"] == name]
                if hp.empty:
                    continue
                ax.scatter(hp[x_col], hp[y_col], marker="*", s=220, zorder=10,
                           color=_PALETTE["accent"], edgecolors="k", linewidths=0.6)
                ax.annotate(name, xy=(hp[x_col].iloc[0], hp[y_col].iloc[0]),
                            xytext=(6, 6), textcoords="offset points",
                            fontsize=8, fontweight="bold",
                            color=_PALETTE["accent"])

        self._add_model_overlays(ax, x_col, y_col, overlay_models)

        if log_x: ax.set_xscale("log")
        if log_y: ax.set_yscale("log")
        ax.set_xlabel(PlotStyle.get_label(x_col))
        ax.set_ylabel(PlotStyle.get_label(y_col))
        ax.set_title(title or f"{PlotStyle.get_label(y_col)} vs {PlotStyle.get_label(x_col)}")
        if overlay_models or highlight_planets:
            ax.legend(loc="best", fontsize=8)
        PlotStyle.polish(ax)
        fig.tight_layout()
        return fig, ax

    def plot_density(self, df: pd.DataFrame, x_col: str, y_col: str, *,
                     log_x: bool = False, log_y: bool = False,
                     cmap: str = "YlOrRd", sigma: float = 6.0,
                     bins: int = 100,
                     overlay_models: list | None = None,
                     show_points: bool = True,
                     figsize: tuple = (7, 5.5),
                     title: str | None = None) -> tuple[plt.Figure, plt.Axes]:
        """2-D Gaussian-smoothed density heatmap with optional point overlay."""
        PlotStyle._ensure_configured()

        log_cols = []
        if log_x: log_cols.append(x_col)
        if log_y: log_cols.append(y_col)
        cdf = self._clean(df, [x_col, y_col], log_cols)

        xd = cdf[x_col].values
        yd = cdf[y_col].values
        xh = np.log10(xd) if log_x else xd
        yh = np.log10(yd) if log_y else yd

        H, xe, ye = np.histogram2d(xh, yh, bins=bins)
        H = gaussian_filter(H, sigma=sigma)

        xc = (xe[:-1] + xe[1:]) / 2
        yc = (ye[:-1] + ye[1:]) / 2
        if log_x: xc = 10**xc
        if log_y: yc = 10**yc

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(xc, yc, H.T, cmap=cmap, shading="auto",
                           rasterized=True, zorder=1)
        fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046, label="Density")

        if show_points:
            ax.scatter(xd, yd, s=1, alpha=0.15, color="k", zorder=2, rasterized=True)

        self._add_model_overlays(ax, x_col, y_col, overlay_models)

        if log_x: ax.set_xscale("log")
        if log_y: ax.set_yscale("log")
        ax.set_xlabel(PlotStyle.get_label(x_col))
        ax.set_ylabel(PlotStyle.get_label(y_col))
        ax.set_title(title or f"Population Density: {PlotStyle.get_label(y_col)} vs {PlotStyle.get_label(x_col)}")
        PlotStyle.polish(ax, grid=False)
        fig.tight_layout()
        return fig, ax

    def plot_histogram(self, df: pd.DataFrame, column: str, *,
                       bins: int = 50,
                       log_x: bool = False, log_y: bool = False,
                       color: str | None = None,
                       figsize: tuple = (7, 4.5),
                       title: str | None = None) -> tuple[plt.Figure, plt.Axes]:
        """1-D distribution histogram."""
        PlotStyle._ensure_configured()

        cdf = self._clean(df, [column], [column] if log_x else None)
        vals = cdf[column].values
        label = PlotStyle.get_label(column)

        if log_x:
            bin_edges = np.logspace(np.log10(vals.min()), np.log10(vals.max()), bins + 1)
        else:
            bin_edges = np.linspace(vals.min(), vals.max(), bins + 1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(vals, bins=bin_edges,
                color=color or _PALETTE["primary"],
                edgecolor="white", linewidth=0.5, zorder=2)

        if log_x: ax.set_xscale("log")
        if log_y: ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(title or f"Distribution of {label}")
        PlotStyle.polish(ax)
        fig.tight_layout()
        return fig, ax


# ===========================================================
# Transit & MCMC Visualization
# ===========================================================

class TransitPlotter:
    """
    Generates publication-quality figures for individual transit analyses:
    lightcurves, periodograms, MCMC traces, and corner plots.

    All methods are static and return ``(fig, axes)``.
    """

    @staticmethod
    def plot_lightcurve(x: np.ndarray, y: np.ndarray, *,
                        err: np.ndarray | None = None,
                        model_x: np.ndarray | None = None,
                        model_y: np.ndarray | None = None,
                        style: str = "scatter",
                        n_bins: int | None = None,
                        xlabel: str = "Time",
                        ylabel: str = "Normalised flux",
                        title: str = "Light curve",
                        figsize: tuple = (8, 4)) -> tuple[plt.Figure, plt.Axes]:
        """Plot a lightcurve with optional errors, optional binning, model overlay.

        Styling matches the DVR report: dark grey points, light grey error bars,
        red model.  Binning is **off** by default (set ``n_bins`` only if needed).
        """
        PlotStyle._ensure_configured()
        fig, ax = plt.subplots(figsize=figsize)

        if style == "line":
            ax.plot(x, y, lw=0.75, alpha=0.75, color=_PALETTE["data"],
                    zorder=2, rasterized=True, label="Data")
        elif err is not None:
            ax.errorbar(x, y, yerr=err, fmt=".", ms=1.8, alpha=0.45,
                        color=_PALETTE["data"], ecolor=_PALETTE["error"],
                        elinewidth=0.45, capsize=0, zorder=2, rasterized=True,
                        label="Data")
        else:
            ax.scatter(x, y, s=4, alpha=0.45, color=_PALETTE["data"],
                       edgecolors="none", zorder=2, rasterized=True, label="Data")

        if n_bins:
            bm, be, _ = stats.binned_statistic(x, y, statistic="mean", bins=n_bins)
            bc = 0.5 * (be[1:] + be[:-1])
            ax.plot(bc, bm, "o-", ms=3, lw=1.0, color=_PALETTE["primary"],
                    zorder=4, label=f"Binned ({n_bins})")

        if model_x is not None and model_y is not None:
            ax.plot(model_x, model_y, lw=1.8, color=_PALETTE["accent"],
                    zorder=5, label="Model")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        PlotStyle.polish(ax)
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def plot_periodogram(x: np.ndarray, y: np.ndarray, *,
                         xaxis_type: str = "period",
                         title: str = "BLS Periodogram",
                         figsize: tuple = (8, 3.5)) -> tuple[plt.Figure, plt.Axes]:
        """Plot a Box Least Squares periodogram."""
        PlotStyle._ensure_configured()
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(x, y, lw=1.0, color=_PALETTE["primary"], zorder=2)

        if xaxis_type == "frequency":
            ax.set_xlabel("Frequency [1/day]")
        else:
            ax.set_xlabel("Period [days]")
            ax.set_xscale("log")

        ax.set_ylabel("Power")
        ax.set_title(title)
        PlotStyle.polish(ax)
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def plot_mcmc_traces(flat_samples: np.ndarray, labels: list, *,
                         title: str = "MCMC Walker Traces",
                         figsize: tuple | None = None) -> tuple[plt.Figure, np.ndarray]:
        """Plot the trace of each MCMC parameter."""
        PlotStyle._ensure_configured()
        ndim = len(labels)
        if figsize is None:
            figsize = (8, 1.8 * ndim)

        fig, axes = plt.subplots(ndim, 1, figsize=figsize, sharex=True)
        if ndim == 1:
            axes = np.array([axes])

        for i, ax in enumerate(axes):
            ax.plot(flat_samples[:, i], lw=0.3, alpha=0.5,
                    color=_PALETTE["primary"], rasterized=True)
            ax.set_ylabel(labels[i], fontsize=9)
            PlotStyle.polish(ax, minor=False)

        axes[-1].set_xlabel("Sample step")
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig, axes

    @staticmethod
    def plot_mcmc_corner(flat_samples: np.ndarray, labels: list, *,
                         title: str = "Posterior Distributions",
                         color: str | None = None) -> tuple[plt.Figure, np.ndarray]:
        """Corner plot using the ``corner`` library.

        Returns ``(fig, axes)`` where *axes* is the 2-D array of subplot axes.
        """
        PlotStyle._ensure_configured()
        import corner

        fig = corner.corner(
            flat_samples, labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 10},
            label_kwargs={"fontsize": 10},
            color=color or _PALETTE["primary"],
            hist_kwargs={"linewidth": 1.0},
            plot_datapoints=False,
            plot_density=True,
            smooth=1.0,
            smooth1d=1.0,
            hist_bin_factor=1.5,
            quiet=True,
        )
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
        return fig, np.array(fig.axes)
