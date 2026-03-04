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
    def apply_dark_theme(fig: go.Figure, x_label: str, y_label: str, 
                         log_x: bool = False, log_y: bool = False) -> go.Figure:
        """
        Applies a consistent dark theme, gridlines, and axis scaling to a figure.
        """
        fig.update_layout(
            font=dict(family="Inter, sans-serif", size=14, color="white"),
            xaxis=dict(
                title=x_label, 
                type='log' if log_x else 'linear',
                showgrid=True, gridcolor='rgba(255,255,255,0.1)', 
                showline=True, linecolor='white', mirror=True
            ),
            yaxis=dict(
                title=y_label, 
                type='log' if log_y else 'linear',
                showgrid=True, gridcolor='rgba(255,255,255,0.1)', 
                showline=True, linecolor='white', mirror=True
            ),
            margin=dict(l=80, r=80, t=80, b=80), 
            template='plotly_dark',
            legend=dict(bgcolor='rgba(68,68,68,0.5)', bordercolor='white', borderwidth=1)
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
                        bins: int = None, xlabel: str = "Time", ylabel: str = "Flux") -> str:
        """
        Plots a standard or folded lightcurve, optionally overlaying a theoretical fit 
        and binning the data for clarity.
        """
        fig = go.Figure()

        # 1. Plot the raw observational data
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

        # 2. Add binned data if requested (reduces visual noise)
        if bins:
            bin_means, bin_edges, _ = stats.binned_statistic(x, y, statistic='mean', bins=bins)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            fig.add_trace(go.Scatter(
                x=bin_centers, y=bin_means, mode='markers+lines', 
                marker=dict(size=6, color='cyan'), line=dict(color='cyan'), 
                name=f'Binned (N={bins})'
            ))

        # 3. Add the theoretical model fit (from Batman)
        if model_x is not None and model_y is not None:
            fig.add_trace(go.Scatter(
                x=model_x, y=model_y, mode='lines', 
                line=dict(color='cyan', width=3), name='Transit Model'
            ))

        fig = PlotStyle.apply_dark_theme(fig, xlabel, ylabel)
        fig.update_layout(title=title)
        
        return PlotStyle.to_html(fig)

    @staticmethod
    def plot_periodogram(x: np.ndarray, y: np.ndarray, title: str = "Periodogram",
                         xaxis_type: str = 'period') -> str:
        """
        Plots the Box Least Squares (BLS) periodogram power spectrum.
        """
        if xaxis_type == 'frequency':
            xlabel, logx = "Frequency [1/day]", False
        else:
            xlabel, logx = "Period [day]", True

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines', 
            line=dict(color='orange'), name='Power'
        ))

        fig = PlotStyle.apply_dark_theme(fig, xlabel, "Power", log_x=logx, log_y=False)
        fig.update_layout(title=title)
        
        return PlotStyle.to_html(fig)

    @staticmethod
    def plot_mcmc_traces(flat_samples: np.ndarray, labels: list) -> str:
        """
        Plots the raw trace of the MCMC walkers to verify convergence.
        """
        ndim = len(labels)
        # We reshape the flat samples back to (steps, walkers, dimensions) for plotting
        # Note: In a real app, you might want to pass the unflattened chain directly here instead
        # For simplicity, we assume we want to plot the distributions of the flat samples as histograms or simple lines
        
        fig = make_subplots(rows=ndim, cols=1, shared_xaxes=True, subplot_titles=labels, vertical_spacing=0.05)

        for i in range(ndim):
            fig.add_trace(
                go.Scatter(
                    y=flat_samples[:, i], mode='lines',
                    line=dict(width=0.5, color="orange"), opacity=0.5, showlegend=False
                ),
                row=i+1, col=1
            )
            fig.update_yaxes(title_text=labels[i], row=i+1, col=1)

        fig.update_xaxes(title_text="Sample Step", row=ndim, col=1)
        fig.update_layout(autosize=True, template="plotly_dark", title_text="MCMC Posterior Traces", height=800)

        return PlotStyle.to_html(fig)

    @staticmethod
    def plot_mcmc_corner(flat_samples: np.ndarray, labels: list) -> str:
        """
        Generates a scatter matrix (corner plot) to show covariances between MCMC parameters.
        """
        df = pd.DataFrame(flat_samples, columns=labels)

        fig = px.scatter_matrix(
            df, dimensions=labels, title="MCMC Posterior Distributions", template="plotly_dark"
        )
        fig.update_traces(diagonal_visible=True, marker=dict(opacity=0.3, size=2, color="orange"))
        fig.update_layout(autosize=True, height=800)
        
        return PlotStyle.to_html(fig)




# ===========================================================
# Population & Catalog Visualization
# ===========================================================

class CatalogPlotter:
    """
    Generates macroscopic statistical plots comparing hundreds or thousands of exoplanets.
    """

    def __init__(self):
        """
        Initializes the plotter and the theoretical mass-radius model loader.
        """
        self.model_loader = MassRadiusModels()

    def _add_model_overlays(self, fig: go.Figure, x_col: str, y_col: str, overlay_models: list):
        """
        Internal helper: Draws theoretical mass-radius lines if the current axes match.
        """
        if not overlay_models:
            return

        # Models only make sense if the axes are Mass and Radius
        valid_axes = {("pl_bmasse", "pl_rade"), ("pl_rade", "pl_bmasse")}
        if (x_col, y_col) not in valid_axes:
            return

        for i, model_key in enumerate(overlay_models):
            try:
                model_df = self.model_loader.get_model_curve(model_key)
                label = self.model_loader.get_model_label(model_key)
                
                # Determine which axis is mass and which is radius
                x_model, y_model = model_df['mass'], model_df['radius']
                if x_col == "pl_rade":
                    x_model, y_model = y_model, x_model

                fig.add_trace(go.Scatter(
                    x=x_model, y=y_model, mode='lines', name=label,
                    line=dict(dash='dash', width=2, color=DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]),
                    hoverinfo='name', showlegend=True
                ))
            except Exception as e:
                # If a model fails to load, we skip it rather than crashing the whole graph
                print(f"Warning: Failed to load model {model_key}: {e}")

    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, 
                     color_by: str = None, highlight_planets: list = None, 
                     log_x: bool = False, log_y: bool = False, 
                     overlay_models: list = None) -> str:
        """
        Generates a standard or color-mapped scatter plot of the planetary population.
        """
        # Resolve human readable labels
        x_label = PlotStyle.get_label(x_col)
        y_label = PlotStyle.get_label(y_col)
        
        # Ensure we don't plot NaNs for our primary axes
        clean_df = df.dropna(subset=[x_col, y_col]).copy()
        
        if log_x: clean_df = clean_df[clean_df[x_col] > 0]
        if log_y: clean_df = clean_df[clean_df[y_col] > 0]

        fig = go.Figure()

        # 1. Base Scatter trace (with or without a color gradient)
        marker_style = dict(opacity=0.8, size=6)
        
        if color_by and color_by in clean_df.columns:
            clean_df = clean_df.dropna(subset=[color_by])
            color_label = PlotStyle.get_label(color_by)
            marker_style.update(dict(
                color=clean_df[color_by],
                colorscale='Viridis',
                colorbar=dict(title=color_label, x=1.02, y=0.5, len=0.7),
                showscale=True
            ))
            hovertemplate = f"<b>%{{text}}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>{color_label}: %{{marker.color}}<extra></extra>"
        else:
            marker_style.update(dict(color='#00BFFF'))
            hovertemplate = f"<b>%{{text}}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>"

        # Main planetary population
        fig.add_trace(go.Scatter(
            x=clean_df[x_col], y=clean_df[y_col], 
            mode='markers', text=clean_df['pl_name'],
            name='Exoplanets', marker=marker_style,
            hovertemplate=hovertemplate
        ))

        # 2. Highlight specific planets (e.g., Earth, Jupiter, or user-selected targets)
        if highlight_planets:
            for planet in highlight_planets:
                hp = clean_df[clean_df['pl_name'] == planet]
                if not hp.empty:
                    fig.add_trace(go.Scatter(
                        x=hp[x_col], y=hp[y_col], mode='markers+text',
                        text=[planet]*len(hp), textposition='top center', name=planet,
                        marker=dict(symbol='star', size=14, color='red', line=dict(width=1, color='white')),
                        hovertemplate=f"<b>%{{text}}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>"
                    ))

        # 3. Add mass-radius theoretical curves if applicable
        self._add_model_overlays(fig, x_col, y_col, overlay_models)

        # Apply formatting
        fig = PlotStyle.apply_dark_theme(fig, x_label, y_label, log_x, log_y)
        fig.update_layout(title=f"Exoplanet Distribution: {y_label} vs {x_label}")

        return PlotStyle.to_html(fig)

    def plot_density(self, df: pd.DataFrame, x_col: str, y_col: str, 
                     log_x: bool = False, log_y: bool = False, 
                     cmap: str = 'YlOrRd', overlay_models: list = None) -> str:
        """
        Generates a 2D Gaussian density heatmap, overlaying the raw scatter points on top.
        Excellent for visualizing highly congested datasets like the Kepler sample.
        """
        x_label = PlotStyle.get_label(x_col)
        y_label = PlotStyle.get_label(y_col)
        
        clean_df = df.dropna(subset=[x_col, y_col]).copy()
        if log_x: clean_df = clean_df[clean_df[x_col] > 0]
        if log_y: clean_df = clean_df[clean_df[y_col] > 0]

        # Extract raw numpy arrays for histogram computation
        x_data, y_data = clean_df[x_col].to_numpy(), clean_df[y_col].to_numpy()
        
        # If the axes are logarithmic, we must compute the density map in log space
        x_hist = np.log10(x_data) if log_x else x_data
        y_hist = np.log10(y_data) if log_y else y_data

        # Compute 2D Histogram and apply Gaussian smoothing
        bins = 100
        H, xedges, yedges = np.histogram2d(x_hist, y_hist, bins=bins)
        H = gaussian_filter(H, sigma=6)

        # Transform bin edges back to normal space for plotting
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2
        
        if log_x: x_centers = 10**x_centers
        if log_y: y_centers = 10**y_centers

        fig = go.Figure()

        # 1. Base Density Heatmap
        fig.add_trace(go.Heatmap(
            x=x_centers, y=y_centers, z=H.T, 
            colorscale=cmap, opacity=0.8, name='Density', showscale=False
        ))

        # 2. Overlay faint scatter points for outliers
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data, mode='markers', text=clean_df['pl_name'],
            name='Exoplanets', marker=dict(color='white', size=2, opacity=0.3),
            hovertemplate=f"<b>%{{text}}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>"
        ))

        self._add_model_overlays(fig, x_col, y_col, overlay_models)
        fig = PlotStyle.apply_dark_theme(fig, x_label, y_label, log_x, log_y)
        fig.update_layout(title=f"Population Density: {y_label} vs {x_label}")

        return PlotStyle.to_html(fig)