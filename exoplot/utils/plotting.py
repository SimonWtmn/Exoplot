"""
Presets for Plotting Exoplanet Dataset
--------------------------------------
Provides predefined plotting functions.

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage import gaussian_filter


from .label import label_map
from .presets import *
from .models import get_model_curve

PRESET_GROUPS = [STELLAR_PRESETS, MISSION_PRESETS, LIT_PRESETS, HZ_PRESETS, PLANET_PRESETS]
ALL_PRESETS = {k: v for group in PRESET_GROUPS for k, v in group.items()}


def combine_samples(samples):
    if isinstance(samples, dict):
        samples = samples.items()
    elif samples and not isinstance(samples[0], tuple):
        samples = [(f"Sample {i+1}", df) for i, df in enumerate(samples)]
    return pd.concat([df.assign(source=label) for label, df in samples], ignore_index=True)


def prepare_labels(*keys):
    return {k: label_map.get(k, k) for k in keys}


def get_error_columns(df, x, y, show_error):
    return (
        f"{x}err1" if show_error and f"{x}err1" in df else None,
        f"{y}err1" if show_error and f"{y}err1" in df else None
    )


def clean_data(df, x, y=None, color_by=None, log_x=False, log_y=False, show_error=False):
    cols = [x]
    if y: cols.append(y)
    if color_by: cols.append(color_by)
    err_x, err_y = get_error_columns(df, x, y, show_error)
    if err_x: cols.append(err_x)
    if err_y: cols.append(err_y)
    cols += ['pl_name', 'source']

    df = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

    if log_x: df = df[df[x] > 0].copy()
    if log_y and y: df = df[df[y] > 0].copy()

    return df


def add_scatter_trace(fig, group, x, y, label_x, label_y,
                      name=None, color=None, err_x=None, err_y=None, marker_extra=None):
    marker = dict(opacity=0.8)
    if color and (not marker_extra or 'color' not in marker_extra):
        marker['color'] = color
    if marker_extra:
        marker.update(marker_extra)

    fig.add_trace(go.Scatter(
        x=group[x].tolist(),
        y=group[y].tolist(),
        mode='markers',
        name=name or (group['source'].iloc[0] if 'source' in group and not group.empty else 'Sample'),
        text=group['pl_name'].tolist(),
        marker=marker,
        error_x=dict(array=group[err_x].tolist()) if err_x else None,
        error_y=dict(array=group[err_y].tolist()) if err_y else None,
        hovertemplate=f"%{{text}}<br>{label_x} = %{{x}}<br>{label_y} = %{{y}}<extra></extra>",
        showlegend=True
    ))


def add_highlight_traces(fig, df, x, y, highlight):
    if not highlight:
        return
    for planet in highlight:
        if planet not in df['pl_name'].values:
            print(f"⚠️ Planet {planet} not found in current data, skipping")
            continue
        hp = df[df['pl_name'] == planet]
        if not hp.empty:
            fig.add_trace(go.Scatter(
                x=hp[x], y=hp[y],
                mode='markers+text',
                text=[planet]*len(hp),
                textposition='top center',
                name=planet,
                marker=dict(symbol='star', size=14, color='red', line=dict(width=1, color='black')),
                customdata=hp[['pl_name', 'source']],
                hovertemplate="%{text}<extra></extra>"
            ))


def add_model_overlay_traces(fig, x, y, overlay_models):
    if not overlay_models:
        return
    valid_axes = {("pl_bmasse", "pl_rade"), ("pl_rade", "pl_bmasse")}
    if (x, y) not in valid_axes and (y, x) not in valid_axes:
        return

    from plotly.colors import DEFAULT_PLOTLY_COLORS
    for i, model_key in enumerate(overlay_models):
        model_df = get_model_curve(model_key)
        print(f"Model '{model_key}' -> {len(model_df)} rows")
        x_model = model_df['mass']
        y_model = model_df['radius']
        if x == "pl_rade":
            x_model, y_model = y_model, x_model

        fig.add_trace(go.Scatter(
            x=x_model, y=y_model,
            mode='lines',
            name=model_key.replace('_', ' ').title(),
            line=dict(dash='dash', width=2, color=DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]),
            showlegend=True
        ))
        print(f"✔️ Added model trace: {model_key}, x[0]={x_model.iloc[0]}, y[0]={y_model.iloc[0]}")


def plot_scatter(df, x, y, highlight, log_x, log_y, show_error, overlay_models):
    labels = prepare_labels(x, y)
    err_x, err_y = get_error_columns(df, x, y, show_error)
    base_df = df if not highlight else df[~df['pl_name'].isin(highlight)]
    fig = go.Figure()

    for src, group in base_df.groupby('source'):
        add_scatter_trace(fig, group, x, y, labels[x], labels[y], name=src, err_x=err_x, err_y=err_y)

    add_highlight_traces(fig, df, x, y, highlight)
    add_model_overlay_traces(fig, x, y, overlay_models)

    fig.update_layout(
        title=f"{labels[y]} vs {labels[x]}",
        xaxis=dict(title=labels[x], type='log' if log_x else 'linear'),
        yaxis=dict(title=labels[y], type='log' if log_y else 'linear'),
        margin=dict(l=60, r=20, t=40, b=60),
        template='plotly_white', height=500, width=1100
    )
    return fig


def plot_colored(df, x, y, color_by, highlight=None, log_x=False, log_y=False, show_error=False, colorscale_list=None, overlay_models=None):
    labels = prepare_labels(x, y, color_by)
    palettes = colorscale_list or ['YlOrRd', 'Blues', 'Greens', 'Purples', 'Oranges', 'Viridis']
    vmin, vmax = df[color_by].min(), df[color_by].max()
    err_x, err_y = get_error_columns(df, x, y, show_error)
    base_df = df if not highlight else df[~df['pl_name'].isin(highlight)]
    fig = go.Figure()

    for i, (src, group) in enumerate(base_df.groupby('source')):
        fig.add_trace(go.Scatter(
            x=group[x].tolist(), y=group[y].tolist(), mode='markers', name=src,
            text=group['pl_name'].tolist(),
            marker=dict(
                color=group[color_by].tolist(), colorscale=palettes[i % len(palettes)],
                cmin=vmin, cmax=vmax, showscale=True,
                colorbar=dict(x=1.02 + i * 0.08, y=0.4, len=0.6, thickness=12)
            ),
            error_x=dict(array=group[err_x].tolist()) if err_x else None,
            error_y=dict(array=group[err_y].tolist()) if err_y else None,
            hovertemplate=f"%{{text}}<br>{labels[x]} = %{{x}}<br>{labels[y]} = %{{y}}<br>{labels[color_by]} = %{{marker.color}}<extra></extra>"
        ))
        fig.add_annotation(x=1.02 + i * 0.08 + 0.02, y=0.1, text=src, xref='paper', yref='paper',
                           showarrow=False, xanchor='center', yanchor='top', font=dict(size=12))

    if base_df['source'].nunique():
        fig.add_annotation(x=1.02 + (len(base_df['source'].unique()) - 1) * 0.08 / 1.6, y=0.05,
                           text=labels[color_by], xref='paper', yref='paper',
                           showarrow=False, xanchor='center', yanchor='top', font=dict(size=12))

    add_highlight_traces(fig, df, x, y, highlight)
    add_model_overlay_traces(fig, x, y, overlay_models)

    fig.update_layout(
        title=f"{labels[x]} vs {labels[y]} (colored by {labels[color_by]})",
        xaxis=dict(title=labels[x], type='log' if log_x else 'linear'),
        yaxis=dict(title=labels[y], type='log' if log_y else 'linear'),
        margin=dict(l=60, r=100, t=60, b=100),
        template='plotly_white', height=500, width=1100
    )
    return fig


def plot_density(
    df, x, y, highlight=None, log_x=False, log_y=False,
    show_error=False, cmap='Oranges', overlay_models=None
):
    labels = prepare_labels(x, y)
    err_x, err_y = get_error_columns(df, x, y, show_error)
    base_df = df if not highlight else df[~df['pl_name'].isin(highlight)]
    fig = go.Figure()

    x_data = df[x].to_numpy()
    y_data = df[y].to_numpy()

    if log_x:
        x_data = np.log10(x_data)
    if log_y:
        y_data = np.log10(y_data)

    bins = 50
    x_bins = np.linspace(x_data.min(), x_data.max(), bins)
    y_bins = np.linspace(y_data.min(), y_data.max(), bins)

    H, xedges, yedges = np.histogram2d(x_data, y_data, bins=[x_bins, y_bins])
    H = gaussian_filter(H, sigma=8) 

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    fig.add_trace(go.Heatmap(
        x=x_centers if not log_x else 10**x_centers,
        y=y_centers if not log_y else 10**y_centers,
        z=H.T,
        colorscale=cmap,
        colorbar=dict(title='Density', x=1.02, y=0.5, len=0.6, thickness=12),
        opacity=0.7,
        name='Density'
    ))

    palette = px.colors.qualitative.Plotly
    for i, (src, group) in enumerate(base_df.groupby('source')):
        add_scatter_trace(fig, group, x, y, labels[x], labels[y], name=src,
                          color=palette[i % len(palette)], err_x=err_x, err_y=err_y,
                          marker_extra={'opacity': 0.6})

    add_highlight_traces(fig, df, x, y, highlight)
    add_model_overlay_traces(fig, x, y, overlay_models)

    fig.update_layout(
        xaxis=dict(title=labels[x], type='log' if log_x else 'linear'),
        yaxis=dict(title=labels[y], type='log' if log_y else 'linear'),
        title=f"{labels[y]} vs {labels[x]} (with Density)",
        margin=dict(l=60, r=100, t=60, b=60),
        template='plotly_white', height=500, width=1100
    )
    return fig



def plot_histogram(df, column, by=None, bins=None, log_x=False, log_y=False):
    labels = prepare_labels(column, by) if by else prepare_labels(column)
    palette = px.colors.qualitative.Plotly
    fig = go.Figure()

    if by and by in df.columns:
        for i, (name, group) in enumerate(df.groupby(by)):
            fig.add_trace(go.Histogram(
                x=group[column], name=str(name), nbinsx=bins,
                opacity=0.75,
                marker=dict(color=palette[i % len(palette)], line=dict(color='black', width=1)),
                hovertemplate=f"{labels[by]}: {name}<br>{labels[column]}: %{{x}}<br>Count: %{{y}}<extra></extra>"
            ))
        barmode = 'stack'
    else:
        fig.add_trace(go.Histogram(
            x=df[column], nbinsx=bins,
            marker=dict(color=palette[0], line=dict(color='black', width=1)),
            hovertemplate=f"{labels[column]}: %{{x}}<br>Count: %{{y}}<extra></extra>"
        ))
        barmode = 'relative'

    fig.update_layout(
        barmode=barmode,
        title=f"Histogram of {labels[column]}" + (f" by {labels[by]}" if by else ''),
        xaxis=dict(title=labels[column], type='log' if log_x else 'linear'),
        yaxis=dict(title='Count'),
        margin=dict(l=60, r=60, t=60, b=60),
        template='plotly_white', height=500, width=1100
    )
    return fig


def main_plot(plot_type, preset_keys=None, df_full='NEA',
              x_axis='pl_bmasse', y_axis='pl_rade', highlight_planets=None,
              color_by=None, log_x=False, log_y=False,
              show_error=False, cmap='YlOrBr', bins=None,
              overlay_models=None):
    if isinstance(df_full, str):
        df_full = ALL_DATA.get(df_full)
        if df_full is None:
            raise KeyError(f"No data named '{df_full}' in ALL_DATA")

    if not preset_keys or preset_keys == ['NEA']:
        # No presets or explicitly requested NEA: use full dataset
        df = df_full.copy()
        df['source'] = 'NEA'
    else:
        df_list = [(key, ALL_PRESETS[key](df_full)) for key in preset_keys if key in ALL_PRESETS]
        df = combine_samples(df_list)

    plot_funcs = {
        'scatter': plot_scatter,
        'colored': plot_colored,
        'density': plot_density,
        'histogram': plot_histogram
    }

    if plot_type == 'histogram':
        df = clean_data(df, x_axis, color_by, log_x=log_x, log_y=log_y)
        return plot_histogram(df, column=x_axis, by=color_by, bins=bins, log_x=log_x, log_y=log_y)
    else:
        df = clean_data(df, x_axis, y_axis, color_by=color_by, log_x=log_x, log_y=log_y, show_error=show_error)

        if plot_type == 'colored':
            return plot_colored(
                df, x_axis, y_axis, color_by, highlight_planets,
                log_x, log_y, show_error,
                overlay_models=overlay_models
            )
        elif plot_type == 'density':
            return plot_density(
                df, x_axis, y_axis, highlight_planets,
                log_x, log_y, show_error, cmap=cmap if cmap else 'YlOrBr',
                overlay_models=overlay_models
            )
        else:
            return plot_funcs[plot_type](
                df, x_axis, y_axis, highlight_planets, log_x, log_y, show_error, overlay_models
            )

