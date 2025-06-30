'''Plots Module for NEA Exoplanet Dataset
---------------------------------------
This module provides functions for creating static visualizations of exoplanet datasets,
including scatter plots, mass–radius plots with per-preset colorbars, and density plots.

Author: S.WITTMANN & V.RAGNER (enhanced to select data sources & presets by key)
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from .label import label_map
from .presets import (
    ALL_DATA,
    STELLAR_PRESETS,
    MISSION_PRESETS,
    LIT_PRESETS,
    HZ_PRESETS,
    PLANET_PRESETS,
)

# Flatten preset dictionaries (filter functions only)
type_dicts = [STELLAR_PRESETS, MISSION_PRESETS, LIT_PRESETS, HZ_PRESETS, PLANET_PRESETS]
ALL_PRESETS = {k: fn for d in type_dicts for k, fn in d.items()}


def combine_samples(df_list):
    """
    Combine a list of (label, DataFrame) or dict of {label: DataFrame}
    into one DataFrame with a 'source' column.
    """
    if isinstance(df_list, dict):
        sources = list(df_list.items())
    elif df_list and isinstance(df_list[0], tuple):
        sources = df_list
    else:
        sources = [(f"Sample {i+1}", df) for i, df in enumerate(df_list)]
    pieces = []
    for label, df in sources:
        tmp = df.copy()
        tmp['source'] = label
        pieces.append(tmp)
    return pd.concat(pieces, ignore_index=True)


# ---------------------------------- Scatter ----------------------------------
def plot_scatter(
    df,
    x,
    y,
    highlight=None,
    log_x=False,
    log_y=False,
    show_error=True
):
    """
    Simplified Plotly scatter plot with a Matplotlib-like white template.

    Args:
        df (pd.DataFrame): Input dataframe with columns x, y, and optional error columns.
        x (str): Column name for x-axis.
        y (str): Column name for y-axis.
        highlight (list[str], optional): List of planet names to highlight.
        log_x (bool, optional): Use logarithmic scale on x-axis.
        log_y (bool, optional): Use logarithmic scale on y-axis.
        show_error (bool, optional): Display error bars if error columns are present.

    Returns:
        plotly.graph_objs.Figure
    """
    # Friendly labels
    labels = {
        x: label_map.get(x, x),
        y: label_map.get(y, y),
        'source': 'Dataset',
        'pl_name': 'Planet'
    }

    # Determine error columns
    err_x = f"{x}err1" if show_error and f"{x}err1" in df.columns else None
    err_y = f"{y}err1" if show_error and f"{y}err1" in df.columns else None

    # Base dataframe (exclude highlights)
    df_base = df[~df['pl_name'].isin(highlight)] if highlight else df

    # Create base scatter with Plotly Express
    fig = px.scatter(
        df_base,
        x=x,
        y=y,
        color='source',
        labels=labels,
        hover_name='pl_name',
        error_x=err_x,
        error_y=err_y,
        log_x=log_x,
        log_y=log_y,
        template='plotly_white',
        opacity=0.8,
        height=600
    )

    # Highlight specified planets on top
    if highlight:
        for planet in highlight:
            df_h = df[df['pl_name'] == planet]
            if df_h.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=df_h[x],
                    y=df_h[y],
                    mode='markers+text',
                    text=[planet] * len(df_h),
                    textposition='top center',
                    name=planet,
                    marker=dict(
                        symbol='star',
                        size=14,
                        color='red',
                        line=dict(width=1, color='black')
                    ),
                )
            )

    # Final layout tweaks
    fig.update_layout(
        legend=dict(title='Source', traceorder='normal'),
        xaxis=dict(title=labels[x], type='log' if log_x else 'linear'),
        yaxis=dict(title=labels[y], type='log' if log_y else 'linear'),
        margin=dict(l=60, r=20, t=40, b=60)
    )

    return fig





# ----------------------------- Mass–Radius Plot ------------------------------
def plot_mass_radius(df, x, y, color_by, highlight=None, log_x=False, log_y=False, show_error=True):
    highlight = highlight or []
    labels = {x: label_map.get(x, x), y: label_map.get(y, y), color_by: label_map.get(color_by, color_by), 'pl_name': 'Planet'}
    palettes = ['YlOrRd', 'Blues', 'Greens', 'Purples', 'Oranges', 'Viridis']
    sources = df['source'].unique()
    vmin, vmax = df[color_by].min(), df[color_by].max()

    fig = go.Figure()
    for i, src in enumerate(sources):
        sub = df[df['source'] == src]
        palette = palettes[i % len(palettes)]
        fig.add_trace(go.Scatter(
            x=sub[x], y=sub[y], mode='markers',
            marker=dict(
                color=sub[color_by], colorscale=palette,
                cmin=vmin, cmax=vmax,
                showscale=True,
                colorbar=dict(x=1.02 + i*0.08, len=0.6, thickness=12)
            ),
            name=src, showlegend=True,
            text=sub['pl_name'],
            hovertemplate=(
                "%{text}<br>"
                f"{labels[x]} = " + "%{x}<br>"
                f"{labels[y]} = " + "%{y}<extra></extra>"
            )
        ))
    for i, src in enumerate(sources):
        fig.add_annotation(
            x=1.02 + i*0.08, y=-0.1, text=src,
            xref='paper', yref='paper', showarrow=False,
            xanchor='center', yanchor='top', font=dict(size=12)
        )
    if len(sources):
        x0 = 1.02 + ((len(sources)-1)*0.08)/2
        fig.add_annotation(
            x=x0, y=-0.15, text=labels[color_by],
            xref='paper', yref='paper', showarrow=False,
            xanchor='center', yanchor='top', font=dict(size=12)
        )

    fig.update_layout(
        template='plotly_white',
        margin=dict(b=100),
        title=f"{labels[x]} vs {labels[y]} (colored by {labels[color_by]})",
        xaxis=dict(title=labels[x], type='log' if log_x else 'linear'),
        yaxis=dict(title=labels[y], type='log' if log_y else 'linear')
    )

    for planet in highlight:
        sub = df[df['pl_name'] == planet]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y], mode='markers+text', text=[planet],
                marker=dict(size=16, color='red', symbol='star'),
                name=f"★ {planet}", showlegend=True,
                hovertemplate=f"{planet}<br>{labels[x]} = %{{x}}<br>{labels[y]} = %{{y}}<extra></extra>"
            ))
    return fig


# ----------------------------- Density Plot ----------------------------------
def plot_density(df, x, y, highlight=None, log_x=False, log_y=False, show_error=True, cmap='YlOrBr'):
    highlight = highlight or []
    clean = df[[x, y, 'pl_name']].replace([np.inf, -np.inf], np.nan).dropna()
    if log_x:
        clean = clean[clean[x] > 0]
    if log_y:
        clean = clean[clean[y] > 0]
    if len(clean) < 10:
        print("Too few points for density.")
        return go.Figure()

    labels = {x: label_map.get(x, x), y: label_map.get(y, y), 'pl_name': 'Planet'}
    fig = px.density_contour(clean, x=x, y=y, labels=labels, height=600, template="plotly_white")
    fig.update_traces(contours_coloring='fill', contours_showlines=False, opacity=0.7, colorscale=cmap)
    fig.add_trace(go.Scatter(x=clean[x], y=clean[y], mode='markers', marker=dict(size=4, opacity=0.5), showlegend=False))
    for planet in highlight:
        sub = clean[clean['pl_name'] == planet]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y], mode='markers+text', text=[planet],
                marker=dict(size=16, color='red', symbol='star'),
                name=planet, showlegend=True,
                hovertemplate=f"{planet}<br>{labels[x]} = %{{x}}<br>{labels[y]} = %{{y}}<extra></extra>"
            ))

    fig.update_layout(
        template="plotly_white",
        title=f"Density: {labels[x]} vs {labels[y]}",
        xaxis=dict(title=labels[x], type='log' if log_x else 'linear'),
        yaxis=dict(title=labels[y], type='log' if log_y else 'linear')
    )
    return fig


# ----------------------------- Main Plot Function ----------------------------
def main_plot(
    plot_type,
    df_list=None,
    preset_keys=None,
    df_full=None,
    x_axis=None,
    y_axis=None,
    highlight_planets=None,
    color_by=None,
    log_x=False,
    log_y=False,
    show_error=True,
    cmap='YlOrBr'
):
    """
    Create a combined plot from either a list of (name, df) pairs or a list of preset and/or data keys.

    - If `preset_keys` is provided, `df_full` must also be provided.
    - `df_full` can be a DataFrame or a key in ALL_DATA dict.
    - `preset_keys` may include keys from ALL_PRESETS (filter presets)
      or keys from ALL_DATA (entire datasets).
    """
    if preset_keys is not None:
        if df_full is None:
            raise ValueError("`df_full` is required when using `preset_keys`")
        # resolve df_full reference
        if isinstance(df_full, str):
            if df_full in ALL_DATA:
                df_full = ALL_DATA[df_full]
            else:
                raise KeyError(f"No data named '{df_full}' in ALL_DATA")
        samples = []
        missing = []
        for key in preset_keys:
            if key in ALL_PRESETS:
                samples.append((key, ALL_PRESETS[key](df_full)))
            elif key in ALL_DATA:
                samples.append((key, ALL_DATA[key]))
            else:
                missing.append(key)
        if missing:
            raise KeyError(f"No such presets or data: {', '.join(missing)}")
        df_list = samples

    if df_list is None:
        raise ValueError("Either `df_list` or `preset_keys` must be provided.")

    df = combine_samples(df_list)
    missing_planets = [p for p in (highlight_planets or []) if p not in df['pl_name'].values]
    if missing_planets:
        print(f"⚠️ Planets not found: {', '.join(missing_planets)}")

    if plot_type == 'scatter':
        fig = plot_scatter(df, x_axis, y_axis, highlight_planets, log_x, log_y, show_error)
    elif plot_type == 'mr':
        if color_by is None:
            raise ValueError("`color_by` is required for MR plot.")
        fig = plot_mass_radius(df, x_axis, y_axis, color_by, highlight_planets, log_x, log_y, show_error)
    elif plot_type == 'density':
        return plot_density(df, x_axis, y_axis, highlight_planets, log_x, log_y, show_error, cmap)
    else:
        fig = ValueError(f"Unknown plot_type: {plot_type}")
    
    fig.show()
    return None
