"""
PDF Report Generation Utilities
-------------------------------
Produces a unified, publication-quality, multi-page PDF report combining
the transit-fit summary and MCMC diagnostics.

Pages
    1  Summary — cleaned lightcurve, BLS periodogram, phase-folded transit
       with best-fit overlay, and a formatted parameter table.
    2  Diagnostics — walker trace plots and a corner plot with proper label
       spacing.

All visible text respects the locale set in ``modules.i18n``.

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot_ENS
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import corner
from datetime import datetime
from pathlib import Path

from .i18n import t as _t, is_latex_compatible

# ── colour palette ──────────────────────────────────────────────────
_PALETTE = {
    "primary": "#2980b9",
    "accent": "#e74c3c",
    "data": "#2c3e50",
    "error": "#bdc3c7",
    "header_bg": "#2c3e50",
    "header_fg": "#ffffff",
    "row_even": "#f7f9fb",
    "row_odd": "#ffffff",
    "grid": "#cccccc",
    "corner": "#1f77b4",
}


class ReportGenerator:
    """
    Generates unified, multi-page PDF reports with i18n support.
    """

    def __init__(self, website_name: str = "Exoplot",
                 website_url: str = "www.exoplot.ens.fr",
                 locale: str = "en",
                 use_latex: bool = False):
        self.website_name = website_name
        self.website_url = website_url
        self.locale = locale
        self.use_latex = use_latex
        self._setup_matplotlib()
        self.results_dir = Path(__file__).parent.parent / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _tr(self, key: str, **kwargs) -> str:
        return _t(key, locale=self.locale, latex=self.use_latex, **kwargs)

    # ── matplotlib defaults ─────────────────────────────────────────
    def _setup_matplotlib(self):
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 9,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'lines.linewidth': 1.2,
            'lines.markersize': 3,
            'figure.dpi': 300,
        })
        if self.use_latex:
            plt.rcParams.update({
                'text.usetex': True,
                'font.serif': ['Computer Modern Roman'],
            })
        else:
            plt.rcParams['text.usetex'] = False

    # ── header / footer ─────────────────────────────────────────────
    def _draw_header(self, fig, target_name: str, subtitle: str):
        fig.text(0.50, 0.975, f"{target_name}",
                 ha='center', va='top', fontsize=18, fontweight='bold',
                 color=_PALETTE["data"])
        fig.text(0.50, 0.955, subtitle,
                 ha='center', va='top', fontsize=11, color='gray')

    def _draw_footer(self, fig, target_name: str):
        now = datetime.now().strftime('%Y-%m-%d')
        left = self._tr("target_label", target=target_name)
        center = self._tr("generated_by", name=self.website_name) + f"  |  {now}"
        fig.text(0.05, 0.012, left, fontsize=7, color='gray', ha='left')
        fig.text(0.50, 0.012, center, fontsize=7, color='gray', ha='center')
        fig.text(0.95, 0.012, self.website_url, fontsize=7, color='gray', ha='right')

    # ═════════════════════════════════════════════════════════════════
    #  PUBLIC: unified MCMC report
    # ═════════════════════════════════════════════════════════════════

    def generate_mcmc_report(self, filename: str, analyzer, fitter,
                             locale: str | None = None):
        """
        Generate a 2-page A4 portrait PDF:
            Page 1 – Summary (lightcurve · periodogram · transit · table)
            Page 2 – Diagnostics (traces · corner plot)

        Parameters
        ----------
        filename : str   Output file name (saved under ``results/``).
        analyzer : LightCurveAnalyzer   Must have clean_lc, folded_lc, periodogram populated.
        fitter   : TransitFitter        Must have been run (results_summary, sampler, flat_samples).
        locale   : str or None          Override the instance locale for this report.
        """
        if locale is not None:
            self.locale = locale

        if self.use_latex and not is_latex_compatible(self.locale):
            self.use_latex = False
            self._setup_matplotlib()

        filepath = self.results_dir / filename
        with PdfPages(filepath) as pdf:
            self._page_summary(pdf, analyzer, fitter)
            self._page_diagnostics(pdf, analyzer, fitter)

        return filepath

    # kept for backward-compat; just delegates
    def generate_mcmc_summary_report(self, filename, analyzer, fitter):
        return self.generate_mcmc_report(filename, analyzer, fitter)

    def generate_mcmc_diagnostic_report(self, filename, analyzer, fitter):
        return self.generate_mcmc_report(filename, analyzer, fitter)

    # ── page 1: summary ─────────────────────────────────────────────
    def _page_summary(self, pdf, analyzer, fitter):
        fig = plt.figure(figsize=(8.27, 11.69))
        self._draw_header(fig, analyzer.target_name,
                          self._tr("page_summary"))

        gs = gridspec.GridSpec(4, 1, figure=fig,
                               top=0.93, bottom=0.06, left=0.12, right=0.92,
                               height_ratios=[1, 1, 1.5, 1.2], hspace=0.40)

        self._plot_lightcurve(fig.add_subplot(gs[0]), analyzer)
        self._plot_periodogram(fig.add_subplot(gs[1]), analyzer)
        self._plot_transit_fit(fig.add_subplot(gs[2]), analyzer, fitter)
        self._plot_param_table(fig.add_subplot(gs[3]), fitter)

        self._draw_footer(fig, analyzer.target_name)
        pdf.savefig(fig, dpi=200)
        plt.close(fig)

    def _plot_lightcurve(self, ax, analyzer):
        time = analyzer.clean_lc.time.value
        flux = analyzer.clean_lc.flux.value
        err = (analyzer.clean_lc.flux_err.value
               if analyzer.clean_lc.flux_err is not None
               else np.full_like(flux, np.median(flux) * 0.01))
        ax.errorbar(time, flux, yerr=err, fmt='.', color=_PALETTE["data"],
                     ecolor=_PALETTE["error"], markersize=1.5, alpha=0.5,
                     capsize=0, elinewidth=0.6)
        ax.set_title(self._tr("section_lightcurve"), fontweight='bold', loc='left')
        ax.set_xlabel(self._tr("time_bjd"))
        ax.set_ylabel(self._tr("normalized_flux"))
        ax.grid(True, alpha=0.25, color=_PALETTE["grid"])

    def _plot_periodogram(self, ax, analyzer):
        ax.plot(analyzer.periodogram.period.value,
                analyzer.periodogram.power.value,
                color=_PALETTE["primary"], linewidth=1.2)
        ax.axvline(analyzer.best_period, color=_PALETTE["accent"],
                    linestyle='--', alpha=0.8,
                    label=self._tr("best_period_label",
                                   period=analyzer.best_period))
        ax.set_title(self._tr("section_periodogram"), fontweight='bold', loc='left')
        ax.set_xlabel(self._tr("period_days"))
        ax.set_ylabel(self._tr("power"))
        ax.set_xscale('log')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.25, color=_PALETTE["grid"])

    def _plot_transit_fit(self, ax, analyzer, fitter):
        ft = analyzer.folded_lc.time.value
        ff = analyzer.folded_lc.flux.value
        fe = (analyzer.folded_lc.flux_err.value
              if analyzer.folded_lc.flux_err is not None
              else np.full_like(ff, np.median(ff) * 0.01))
        mt, mf = fitter.get_best_model_curve(phase_folded=True)

        ax.errorbar(ft, ff, yerr=fe, fmt='.', color=_PALETTE["data"],
                     ecolor=_PALETTE["error"], markersize=2, alpha=0.5,
                     capsize=0, elinewidth=0.7, zorder=1)
        ax.plot(mt, mf, color=_PALETTE["accent"], linewidth=2.0, zorder=2,
                label=self._tr("mcmc_best_fit"))

        zoom = fitter.period * 0.1
        ax.set_xlim(-zoom, zoom)
        ax.set_title(
            self._tr("section_transit_fit", period=fitter.period),
            fontweight='bold', loc='left')
        ax.set_xlabel(self._tr("phase_days"))
        ax.set_ylabel(self._tr("normalized_flux"))
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.25, color=_PALETTE["grid"])

    def _plot_param_table(self, ax, fitter):
        ax.axis('off')
        ax.set_title(self._tr("section_parameters"), fontweight='bold',
                      loc='left', pad=12)

        labels = list(fitter.results_summary.keys())
        cell_text = []
        for label in labels:
            med, upper, lower = fitter.results_summary[label]
            if self.use_latex:
                val = f"${med:.6f}_{{-{lower:.6f}}}^{{+{upper:.6f}}}$"
            else:
                val = f"{med:.6f}  (+{upper:.6f} / -{lower:.6f})"
            cell_text.append([label, val])

        col_labels = [self._tr("col_parameter"), self._tr("col_value")]
        table = ax.table(
            cellText=cell_text, colLabels=col_labels,
            loc='center', cellLoc='center',
            bbox=[0.05, 0.0, 0.90, 0.85]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.4)
            cell.set_edgecolor(_PALETTE["grid"])
            if row == 0:
                cell.set_facecolor(_PALETTE["header_bg"])
                cell.set_text_props(weight='bold', color=_PALETTE["header_fg"])
                cell.set_height(0.08)
            else:
                cell.set_facecolor(
                    _PALETTE["row_even"] if row % 2 == 0 else _PALETTE["row_odd"])
                cell.set_height(0.065)

        # Convergence summary below the table
        info = fitter.get_convergence_info()
        tau_str = (f"{info['autocorr_time']:.1f}"
                   if info['autocorr_time'] else self._tr("not_estimated"))
        conv = (f"{self._tr('n_walkers')}: {info['nwalkers']}  |  "
                f"{self._tr('n_steps')}: {info['nsteps']}  |  "
                f"{self._tr('burn_in')}: {info['burn_in']}  |  "
                f"{self._tr('autocorr')}: {tau_str}  |  "
                f"{self._tr('acceptance')}: {info['mean_acceptance_fraction']:.2%}  |  "
                f"{self._tr('n_eff')}: {info['n_effective_samples']}")
        ax.text(0.50, -0.05, conv, transform=ax.transAxes,
                fontsize=6.5, ha='center', color='gray')

    # ── page 2: diagnostics ─────────────────────────────────────────
    def _page_diagnostics(self, pdf, analyzer, fitter):
        labels = list(fitter.results_summary.keys())
        ndim = len(labels)

        fig = plt.figure(figsize=(8.27, 11.69))
        self._draw_header(fig, analyzer.target_name,
                          self._tr("page_diagnostics"))
        self._draw_footer(fig, analyzer.target_name)

        # ---- Trace plots (top ~28 % of the page) -------------------
        gs_traces = gridspec.GridSpec(
            ndim, 1, figure=fig,
            top=0.93, bottom=0.65, left=0.14, right=0.94, hspace=0.08)

        try:
            chain = fitter.sampler.get_chain(
                thin=max(1, fitter.sampler.iteration // 200))
        except Exception:
            chain = None

        trace_axes = []
        for i in range(ndim):
            ax = fig.add_subplot(gs_traces[i])
            trace_axes.append(ax)
            if chain is not None:
                nw = chain.shape[1]
                colors = plt.cm.viridis(np.linspace(0.15, 0.85, nw))
                for w in range(nw):
                    ax.plot(chain[:, w, i], color=colors[w], alpha=0.35,
                            linewidth=0.4, rasterized=True)
            else:
                ax.plot(fitter.flat_samples[::5, i],
                        color=_PALETTE["corner"], alpha=0.3, linewidth=0.4)
            ax.set_ylabel(labels[i], fontsize=8)
            ax.tick_params(labelsize=7)
            if i < ndim - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(self._tr("step_number"), fontsize=8)
            ax.grid(True, alpha=0.2, color=_PALETTE["grid"])

        trace_axes_set = set(id(a) for a in trace_axes)

        # ---- Corner plot (bottom ~60 % of the page) ----------------
        corner_left, corner_right = 0.12, 0.95
        corner_bottom, corner_top = 0.04, 0.60

        with plt.rc_context({'xtick.labelsize': 7, 'ytick.labelsize': 7}):
            corner.corner(
                fitter.flat_samples, labels=labels,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 8, "pad": 6},
                label_kwargs={"fontsize": 8},
                color=_PALETTE["corner"],
                hist_kwargs={'linewidth': 1.0},
                quiet=True,
                plot_datapoints=False,
                plot_density=True,
                fig=fig,
            )

        corner_axes = [a for a in fig.axes if id(a) not in trace_axes_set]

        if corner_axes:
            all_pos = np.array([[a.get_position().x0, a.get_position().y0,
                                  a.get_position().x1, a.get_position().y1]
                                for a in corner_axes])
            cx0, cy0 = all_pos[:, 0].min(), all_pos[:, 1].min()
            cx1, cy1 = all_pos[:, 2].max(), all_pos[:, 3].max()
            sx = (corner_right - corner_left) / max(cx1 - cx0, 1e-9)
            sy = (corner_top - corner_bottom) / max(cy1 - cy0, 1e-9)
            for a in corner_axes:
                pos = a.get_position()
                new_x0 = corner_left + (pos.x0 - cx0) * sx
                new_y0 = corner_bottom + (pos.y0 - cy0) * sy
                new_w = pos.width * sx
                new_h = pos.height * sy
                a.set_position([new_x0, new_y0, new_w, new_h])

            for a in corner_axes:
                for lbl in a.get_xticklabels():
                    lbl.set_rotation(45)
                    lbl.set_ha('right')
                    lbl.set_fontsize(6)
                for lbl in a.get_yticklabels():
                    lbl.set_fontsize(6)
                if a.get_xlabel():
                    a.set_xlabel(a.get_xlabel(), fontsize=7)
                if a.get_ylabel():
                    a.set_ylabel(a.get_ylabel(), fontsize=7)

        pdf.savefig(fig, dpi=200)
        plt.close(fig)

    # ═════════════════════════════════════════════════════════════════
    #  Catalog report (unchanged API)
    # ═════════════════════════════════════════════════════════════════

    def generate_catalog_report(self, filename: str, df, plot_configs: list,
                                title="Exoplanet Population Summary"):
        filepath = self.results_dir / filename
        with PdfPages(filepath) as pdf:
            fig = plt.figure(figsize=(11.69, 8.27))
            self._draw_header(fig, "Catalog Sample", title)
            n_plots = len(plot_configs)
            if n_plots == 1:
                gs = gridspec.GridSpec(1, 1, figure=fig)
            elif n_plots == 2:
                gs = gridspec.GridSpec(1, 2, figure=fig)
            elif n_plots <= 4:
                gs = gridspec.GridSpec(2, 2, figure=fig)
            else:
                raise ValueError("Maximum 4 plots per summary sheet.")

            for i, cfg in enumerate(plot_configs):
                ax = fig.add_subplot(gs[i])
                x_col, y_col = cfg.get('x'), cfg.get('y')
                cdf = df.dropna(subset=[x_col, y_col]) if y_col else df.dropna(subset=[x_col])
                if cfg.get('type') == 'scatter':
                    ax.scatter(cdf[x_col], cdf[y_col], alpha=0.6, s=15,
                               color=_PALETTE["primary"], edgecolor='black', linewidth=0.5)
                    if cfg.get('log_x'): ax.set_xscale('log')
                    if cfg.get('log_y'): ax.set_yscale('log')
                    ax.set_ylabel(y_col.replace('_', ' ').title())
                elif cfg.get('type') == 'histogram':
                    ax.hist(cdf[x_col], bins=30, color='#ff7f0e', edgecolor='black')
                    ax.set_ylabel("Count")
                ax.set_xlabel(x_col.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)

            fig.tight_layout(rect=[0, 0.05, 1, 0.94], w_pad=2.0, h_pad=2.0)
            self._draw_footer(fig, "Catalog Sample")
            pdf.savefig(fig)
            plt.close(fig)
        return filepath
