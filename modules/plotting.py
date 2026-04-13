"""
Interactive Plotting Utilities
------------------------------
Provides `TransitPlotter` and `CatalogPlotter` classes to generate responsive 
Plotly HTML figures for both the Exoplanet Catalog and individual transit fits.

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot_ENS
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS

from .constants import LABEL_MAP
from .models import MassRadiusModels

# ===========================================================
# Base Styling Configuration
# ===========================================================

class PlotStyle:
    """
    A purely static utility class to ensure consistent styling, fonts, 
    and HTML conversion across all plots in the application.
    """

    @staticmethod
    def to_html(fig: go.Figure) -> str:
        """
        Converts a Plotly Figure object into a lightweight, embeddable HTML string.
        Includes the Plotly.js library from the CDN to keep the output string small.
        """
        return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})

    @staticmethod
    def apply_theme(fig: go.Figure, x_label: str, y_label: str, 
                    log_x: bool = False, log_y: bool = False, theme: str = 'dark') -> go.Figure:
        """
        Applies a consistent theme (dark or light), gridlines, and axis scaling to a figure.
        """
        is_dark = (theme == 'dark')
        
        # Define dynamic colors based on the chosen theme
        font_color = "white" if is_dark else "#2C3E50"
        grid_color = 'rgba(255,255,255,0.1)' if is_dark else 'rgba(0,0,0,0.1)'
        line_color = 'white' if is_dark else '#2C3E50'
        bg_color = 'rgba(68,68,68,0.5)' if is_dark else 'rgba(255,255,255,0.8)'
        template = 'plotly_dark' if is_dark else 'plotly_white'

        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=14, color=font_color),
            xaxis=dict(
                title=x_label, 
                type='log' if log_x else 'linear',
                showgrid=True, gridcolor=grid_color, 
                showline=True, linecolor=line_color, mirror=True
            ),
            yaxis=dict(
                title=y_label, 
                type='log' if log_y else 'linear',
                showgrid=True, gridcolor=grid_color, 
                showline=True, linecolor=line_color, mirror=True
            ),
            margin=dict(l=80, r=80, t=80, b=80), 
            template=template,
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(bgcolor=bg_color, bordercolor=line_color, borderwidth=1)
        )
        return fig

    @staticmethod
    def get_label(col_name: str) -> str:
        """
        Translates a raw dataframe column name into a human-readable HTML label.
        """
        return LABEL_MAP.get(col_name, col_name)

# ===========================================================
# Transit & MCMC Visualization
# ===========================================================

class TransitPlotter:
    """
    Generates plots related to individual star systems, lightcurves, and MCMC fitting.
    """

    @staticmethod
    def plot_lightcurve(x: np.ndarray, y: np.ndarray, err: np.ndarray = None, 
                        model_x: np.ndarray = None, model_y: np.ndarray = None,
                        title: str = "Light Curve", style: str = 'scatter', 
                        bins: int = None, xlabel: str = "Time", ylabel: str = "Flux", 
                        theme: str = 'dark') -> str:
        """
        Plots a standard or folded lightcurve, optionally overlaying a theoretical fit.
        """
        fig = go.Figure()

        if style == 'line':
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='orange'), name='Data'))
        elif style == 'errorbar' and err is not None:
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='markers',
                error_y=dict(type='data', array=err, visible=True, color='rgba(255, 165, 0, 0.4)'),
                marker=dict(size=3, color='orange'), name='Data'
            ))
        else:
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=3, color='orange'), name='Data'))

        if bins:
            bin_means, bin_edges, _ = stats.binned_statistic(x, y, statistic='mean', bins=bins)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            cyan_color = '#00E5FF' if theme == 'dark' else '#00BFFF'
            fig.add_trace(go.Scatter(
                x=bin_centers, y=bin_means, mode='markers+lines', 
                marker=dict(size=6, color=cyan_color), line=dict(color=cyan_color), 
                name=f'Binned (N={bins})'
            ))

        if model_x is not None and model_y is not None:
            model_color = '#00E5FF' if theme == 'dark' else '#00BFFF'
            fig.add_trace(go.Scatter(
                x=model_x, y=model_y, mode='lines', 
                line=dict(color=model_color, width=3), name='Transit Model'
            ))

        fig = PlotStyle.apply_theme(fig, xlabel, ylabel, theme=theme)
        fig.update_layout(title=title)
        return PlotStyle.to_html(fig)

    @staticmethod
    def plot_periodogram(x: np.ndarray, y: np.ndarray, title: str = "Periodogram",
                         xaxis_type: str = 'period', theme: str = 'dark') -> str:
        """
        Plots the Box Least Squares (BLS) periodogram power spectrum.
        """
        if xaxis_type == 'frequency':
            xlabel, logx = "Frequency [1/day]", False
        else:
            xlabel, logx = "Period [day]", True

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='orange'), name='Power'))

        fig = PlotStyle.apply_theme(fig, xlabel, "Power", log_x=logx, log_y=False, theme=theme)
        fig.update_layout(title=title)
        return PlotStyle.to_html(fig)

    @staticmethod
    def plot_mcmc_traces(flat_samples: np.ndarray, labels: list, theme: str = 'dark') -> str:
        """
        Plots the raw trace of the MCMC walkers.
        """
        ndim = len(labels)
        fig = make_subplots(rows=ndim, cols=1, shared_xaxes=True, subplot_titles=labels, vertical_spacing=0.05)

        for i in range(ndim):
            fig.add_trace(
                go.Scatter(y=flat_samples[:, i], mode='lines', line=dict(width=0.5, color="orange"), opacity=0.5, showlegend=False),
                row=i+1, col=1
            )
            fig.update_yaxes(title_text=labels[i], row=i+1, col=1)

        template = "plotly_dark" if theme == 'dark' else "plotly_white"
        font_color = "white" if theme == 'dark' else "#2C3E50"
        
        fig.update_xaxes(title_text="Sample Step", row=ndim, col=1)
        fig.update_layout(
            autosize=True, template=template, title_text="MCMC Posterior Traces", height=800,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=font_color)
        )
        return PlotStyle.to_html(fig)

    @staticmethod
    def plot_mcmc_corner(flat_samples: np.ndarray, labels: list, theme: str = 'dark') -> str:
        """
        Generates a professional astronomical corner plot with 1D histograms
        on the diagonal and 2D density contours on the off-diagonals.
        Tick labels are rotated and font sizes are reduced to prevent overlap.
        """
        ndim = len(labels)
        is_dark = (theme == 'dark')

        main_color = '#00E5FF' if is_dark else '#1E90FF'
        line_color = 'white' if is_dark else 'black'
        contour_colorscale = 'Blues' if not is_dark else 'Blues_r'
        template = "plotly_dark" if is_dark else "plotly_white"
        font_color = "white" if is_dark else "#2C3E50"

        spacing = max(0.04, 0.12 / ndim)
        fig = make_subplots(
            rows=ndim, cols=ndim,
            shared_xaxes=False, shared_yaxes=False,
            horizontal_spacing=spacing, vertical_spacing=spacing,
        )

        tick_font_size = max(7, 11 - ndim)
        label_font_size = max(9, 13 - ndim)

        for i in range(ndim):
            for j in range(i + 1):
                x_data = flat_samples[:, j]

                if i == j:
                    fig.add_trace(
                        go.Histogram(x=x_data, nbinsx=30, marker_color=main_color, showlegend=False),
                        row=i+1, col=j+1
                    )
                    q16, q50, q84 = np.percentile(x_data, [16, 50, 84])
                    for q in [q16, q50, q84]:
                        fig.add_vline(x=q, line_dash="dash", line_color=line_color,
                                      line_width=1, row=i+1, col=j+1)
                else:
                    y_data = flat_samples[:, i]
                    fig.add_trace(
                        go.Histogram2dContour(
                            x=x_data, y=y_data, colorscale=contour_colorscale,
                            showscale=False, ncontours=6, line=dict(width=1.5)
                        ),
                        row=i+1, col=j+1
                    )

                if i == ndim - 1:
                    fig.update_xaxes(
                        title_text=labels[j], title_font_size=label_font_size,
                        tickfont_size=tick_font_size, tickangle=45,
                        nticks=5, row=i+1, col=j+1)
                else:
                    fig.update_xaxes(showticklabels=False, row=i+1, col=j+1)

                if j == 0 and i != 0:
                    fig.update_yaxes(
                        title_text=labels[i], title_font_size=label_font_size,
                        tickfont_size=tick_font_size, nticks=5,
                        row=i+1, col=j+1)
                else:
                    fig.update_yaxes(showticklabels=False, row=i+1, col=j+1)

        side = max(600, 200 * ndim)
        fig.update_layout(
            autosize=False, height=side, width=side,
            template=template,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=font_color, size=tick_font_size),
            title_text="MCMC Posterior Distributions",
            margin=dict(l=60, r=30, t=60, b=60),
        )

        return PlotStyle.to_html(fig)

# ===========================================================
# Population & Catalog Visualization
# ===========================================================

class CatalogPlotter:
    """
    Generates macroscopic statistical plots comparing hundreds or thousands of exoplanets.
    """

    def __init__(self):
        self.model_loader = MassRadiusModels()

    def _add_model_overlays(self, fig: go.Figure, x_col: str, y_col: str, overlay_models: list):
        if not overlay_models:
            return
        valid_axes = {("pl_bmasse", "pl_rade"), ("pl_rade", "pl_bmasse")}
        if (x_col, y_col) not in valid_axes:
            return

        for i, model_key in enumerate(overlay_models):
            try:
                model_df = self.model_loader.get_model_curve(model_key)
                label = self.model_loader.get_model_label(model_key)
                x_model, y_model = model_df['mass'], model_df['radius']
                if x_col == "pl_rade":
                    x_model, y_model = y_model, x_model

                fig.add_trace(go.Scatter(
                    x=x_model, y=y_model, mode='lines', name=label,
                    line=dict(dash='dash', width=2, color=DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]),
                    hoverinfo='name', showlegend=True
                ))
            except Exception as e:
                print(f"Warning: Failed to load model {model_key}: {e}")

    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, 
                     color_by: str = None, highlight_planets: list = None, 
                     log_x: bool = False, log_y: bool = False, 
                     overlay_models: list = None, theme: str = 'dark') -> str:
        x_label = PlotStyle.get_label(x_col)
        y_label = PlotStyle.get_label(y_col)
        clean_df = df.dropna(subset=[x_col, y_col]).copy()
        
        if log_x: clean_df = clean_df[clean_df[x_col] > 0]
        if log_y: clean_df = clean_df[clean_df[y_col] > 0]

        fig = go.Figure()
        marker_line_color = 'rgba(255,255,255,0.4)' if theme == 'dark' else 'rgba(0,0,0,0.4)'
        marker_style = dict(opacity=0.9, size=7, line=dict(width=0.5, color=marker_line_color))
        
        if color_by and color_by in clean_df.columns:
            clean_df = clean_df.dropna(subset=[color_by])
            color_label = PlotStyle.get_label(color_by)
            marker_style.update(dict(
                color=clean_df[color_by], colorscale='Plasma',
                colorbar=dict(title=color_label, x=1.02, y=0.5, len=0.7), showscale=True
            ))
            hovertemplate = f"<b>%{{text}}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>{color_label}: %{{marker.color}}<extra></extra>"
        else:
            default_color = '#00E5FF' if theme == 'dark' else '#1E90FF'
            marker_style.update(dict(color=default_color))
            hovertemplate = f"<b>%{{text}}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>"

        fig.add_trace(go.Scatter(
            x=clean_df[x_col], y=clean_df[y_col], mode='markers', text=clean_df['pl_name'],
            name='Exoplanets', marker=marker_style, hovertemplate=hovertemplate
        ))

        if highlight_planets:
            for planet in highlight_planets:
                hp = clean_df[clean_df['pl_name'] == planet]
                if not hp.empty:
                    fig.add_trace(go.Scatter(
                        x=hp[x_col], y=hp[y_col], mode='markers+text',
                        text=[planet]*len(hp), textposition='top center', name=planet,
                        marker=dict(symbol='star', size=16, color='#FF3366', line=dict(width=1, color='white')),
                        hovertemplate=f"<b>%{{text}}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>"
                    ))

        self._add_model_overlays(fig, x_col, y_col, overlay_models)
        fig = PlotStyle.apply_theme(fig, x_label, y_label, log_x, log_y, theme)
        fig.update_layout(title=f"Exoplanet Distribution: {y_label} vs {x_label}")
        return PlotStyle.to_html(fig)

    def plot_density(self, df: pd.DataFrame, x_col: str, y_col: str, 
                     log_x: bool = False, log_y: bool = False, 
                     cmap: str = 'YlOrRd', overlay_models: list = None, theme: str = 'dark') -> str:
        x_label = PlotStyle.get_label(x_col)
        y_label = PlotStyle.get_label(y_col)
        clean_df = df.dropna(subset=[x_col, y_col]).copy()
        
        if log_x: clean_df = clean_df[clean_df[x_col] > 0]
        if log_y: clean_df = clean_df[clean_df[y_col] > 0]

        x_data, y_data = clean_df[x_col].to_numpy(), clean_df[y_col].to_numpy()
        x_hist = np.log10(x_data) if log_x else x_data
        y_hist = np.log10(y_data) if log_y else y_data

        bins = 100
        H, xedges, yedges = np.histogram2d(x_hist, y_hist, bins=bins)
        H = gaussian_filter(H, sigma=6)

        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2
        
        if log_x: x_centers = 10**x_centers
        if log_y: y_centers = 10**y_centers

        fig = go.Figure()
        fig.add_trace(go.Heatmap(x=x_centers, y=y_centers, z=H.T, colorscale=cmap, opacity=0.8, name='Density', showscale=False))

        # Change dot color based on theme
        dot_color = 'white' if theme == 'dark' else 'black'
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data, mode='markers', text=clean_df['pl_name'],
            name='Exoplanets', marker=dict(color=dot_color, size=2, opacity=0.3),
            hovertemplate=f"<b>%{{text}}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>"
        ))

        self._add_model_overlays(fig, x_col, y_col, overlay_models)
        fig = PlotStyle.apply_theme(fig, x_label, y_label, log_x, log_y, theme)
        fig.update_layout(title=f"Population Density: {y_label} vs {x_label}")
        return PlotStyle.to_html(fig)

    def plot_histogram(self, df: pd.DataFrame, column: str, bins: int = 50, 
                       log_x: bool = False, log_y: bool = False, color: str = None, theme: str = 'dark') -> str:
        label = PlotStyle.get_label(column)
        clean_df = df.dropna(subset=[column]).copy()
        
        if log_x: clean_df = clean_df[clean_df[column] > 0]
            
        fig = go.Figure()
        min_val, max_val = clean_df[column].min(), clean_df[column].max()
        if log_x:
            bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), bins + 1)
        else:
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
        counts, edges = np.histogram(clean_df[column], bins=bin_edges)
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        
        bar_color = color if color else ('#00E5FF' if theme == 'dark' else '#1E90FF')
        border_color = 'black' if theme == 'dark' else 'white'
        
        fig.add_trace(go.Bar(
            x=centers, y=counts, width=widths, name='Count',
            marker=dict(color=bar_color, line=dict(color=border_color, width=1)),
            hovertemplate=f"<b>{label}</b>: %{{x}}<br><b>Count</b>: %{{y}}<extra></extra>"
        ))
        
        fig = PlotStyle.apply_theme(fig, label, "Count", log_x=log_x, log_y=log_y, theme=theme)
        fig.update_layout(title=f"Distribution of {label}", barmode='relative')
        return PlotStyle.to_html(fig)