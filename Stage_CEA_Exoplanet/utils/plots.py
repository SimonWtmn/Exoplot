"""
Exoplanet Plotting Utilities
----------------------------
Provides interactive plotting functions for visualizing exoplanet datasets, 
including scatter, density, and colored plots.

Author: S.WITTMANN
Repository: https://github.com/SimonWtmn/Stage_CEA_Exoplanet
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from .label import label_map
from .presets import *

# ---------------------------------- Presets and Globals ----------------------------------

PRESET_GROUPS = [STELLAR_PRESETS, MISSION_PRESETS, LIT_PRESETS, HZ_PRESETS, PLANET_PRESETS]
ALL_PRESETS = {k: v for group in PRESET_GROUPS for k, v in group.items()}

# ---------------------------------- Utility Functions ----------------------------------

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

# ---------------------------------- Trace Adders ----------------------------------

def add_highlight_traces(fig, df, x, y, highlight):
    if not highlight:
        return
    for planet in highlight:
        hp = df[df['pl_name'] == planet]
        if not hp.empty:
            fig.add_trace(go.Scatter(
                x=hp[x], y=hp[y], mode='markers+text', text=[planet]*len(hp),
                textposition='top center', name=planet,
                marker=dict(symbol='star', size=14, color='red', line=dict(width=1, color='black')),
                customdata=hp[['pl_name', 'source']],
                hovertemplate="%{text}<extra></extra>"
            ))

def add_scatter_trace(fig, group, x, y, label_x, label_y,
                      name=None, color=None, err_x=None, err_y=None, marker_extra=None):
    marker = dict(opacity=0.8)
    if color is not None and (not marker_extra or 'color' not in marker_extra):
        marker['color'] = color
    if marker_extra:
        marker.update(marker_extra)
    fig.add_trace(go.Scatter(
        x=group[x], y=group[y], mode='markers', name=name or group.get('source', 'Sample'),
        text=group['pl_name'], marker=marker,
        error_x=dict(array=group[err_x]) if err_x else None,
        error_y=dict(array=group[err_y]) if err_y else None,
        customdata=group[['pl_name', 'source']],
        hovertemplate=f"%{{text}}<br>{label_x} = %{{x}}<br>{label_y} = %{{y}}<extra></extra>"
    ))

# ---------------------------------- Plot Functions ----------------------------------

def plot_scatter(df, x, y, highlight=None, log_x=False, log_y=False, show_error=False):
    labels = prepare_labels(x, y)
    labels.update({'source': 'Dataset', 'pl_name': 'Planet'})
    err_x, err_y = get_error_columns(df, x, y, show_error)
    fig, base_df = go.Figure(), df if not highlight else df[~df['pl_name'].isin(highlight)]
    for src, group in base_df.groupby('source'):
        add_scatter_trace(fig, group, x, y, labels[x], labels[y], name=src, err_x=err_x, err_y=err_y)
    add_highlight_traces(fig, df, x, y, highlight)
    fig.update_layout(
        legend_traceorder='normal',
        xaxis=dict(title=labels[x], type='log' if log_x else 'linear'),
        yaxis=dict(title=labels[y], type='log' if log_y else 'linear'),
        title=f"{labels[y]} vs {labels[x]}",
        margin=dict(l=60, r=20, t=40, b=60),
        template='plotly_white',
        height=800
    )
    return fig

def plot_colored(df, x, y, color_by, highlight=None, log_x=False, log_y=False, show_error=False, colorscale_list=None):
    labels = prepare_labels(x, y, color_by)
    labels.update({'source': 'Dataset', 'pl_name': 'Planet'})
    palettes = colorscale_list or ['YlOrRd', 'Blues', 'Greens', 'Purples', 'Oranges', 'Viridis']
    vmin, vmax = df[color_by].min(), df[color_by].max()
    err_x, err_y = get_error_columns(df, x, y, show_error)
    fig, base_df = go.Figure(), df if not highlight else df[~df['pl_name'].isin(highlight)]

    for i, (src, group) in enumerate(base_df.groupby('source')):
        fig.add_trace(go.Scatter(
            x=group[x], y=group[y], mode='markers', name=src, text=group['pl_name'],
            marker=dict(color=group[color_by], colorscale=palettes[i % len(palettes)],
                        cmin=vmin, cmax=vmax, showscale=True,
                        colorbar=dict(x=1.02 + i * 0.08, y=0.4, len=0.6, thickness=12)),
            error_x=dict(array=group[err_x]) if err_x else None,
            error_y=dict(array=group[err_y]) if err_y else None,
            hovertemplate=f"%{{text}}<br>{labels[x]} = %{{x}}<br>{labels[y]} = %{{y}}<br>{labels[color_by]} = %{{marker.color}}<extra></extra>"
        ))
        fig.add_annotation(x=1.02 + i * 0.08 + 0.02, y=0.1, text=src, xref='paper', yref='paper',
                           showarrow=False, xanchor='center', yanchor='top', font=dict(size=12))

    if base_df['source'].nunique():
        fig.add_annotation(
            x=1.02 + (len(base_df['source'].unique()) - 1) * 0.08 / 1.6, y=0.05,
            text=labels[color_by], xref='paper', yref='paper',
            showarrow=False, xanchor='center', yanchor='top', font=dict(size=12)
        )

    add_highlight_traces(fig, df, x, y, highlight)
    fig.update_layout(
        legend_traceorder='normal',
        xaxis=dict(title=labels[x], type='log' if log_x else 'linear'),
        yaxis=dict(title=labels[y], type='log' if log_y else 'linear'),
        title=f"{labels[x]} vs {labels[y]} (colored by {labels[color_by]})",
        margin=dict(l=60, r=100, t=60, b=100),
        template='plotly_white',
        height=800
    )
    return fig

def plot_density(df, x, y, highlight=None, log_x=False, log_y=False, show_error=False, cmap='YlOrBr'):
    labels = prepare_labels(x, y)
    labels.update({'source': 'Dataset', 'pl_name': 'Planet'})
    err_x, err_y = get_error_columns(df, x, y, show_error)
    cols = [x, y, 'pl_name', 'source'] + [c for c in (err_x, err_y) if c]
    df = df[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y])

    if log_x:
        df = df[df[x] > 0].copy(); df[x] = np.log10(df[x])
    if log_y:
        df = df[df[y] > 0].copy(); df[y] = np.log10(df[y])

    fig = go.Figure()
    if len(df) >= 10:
        fig.add_trace(go.Histogram2dContour(
            x=df[x], y=df[y], colorscale=cmap,
            contours=dict(coloring='fill', showlines=False),
            showscale=True, colorbar=dict(title='Density', x=1.02, y=0.5, len=0.6, thickness=12),
            hoverinfo='skip', name='Density', opacity=0.5
        ))

    palette = px.colors.qualitative.Plotly
    base_df = df if not highlight else df[~df['pl_name'].isin(highlight)]
    for i, (src, group) in enumerate(base_df.groupby('source')):
        add_scatter_trace(fig, group, x, y, labels[x], labels[y], name=src,
                          color=palette[i % len(palette)], err_x=err_x, err_y=err_y)
    add_highlight_traces(fig, df, x, y, highlight)

    def log_ticks(series, enabled):
        if not enabled: return None, None
        exponents = range(int(np.floor(series.min())), int(np.ceil(series.max())) + 1)
        return list(exponents), [f"10<sup>{e}</sup>" for e in exponents]

    xticks, xticklabels = log_ticks(df[x], log_x)
    yticks, yticklabels = log_ticks(df[y], log_y)

    fig.update_layout(
        xaxis=dict(title=labels[x], tickvals=xticks, ticktext=xticklabels),
        yaxis=dict(title=labels[y], tickvals=yticks, ticktext=yticklabels),
        title=f"{labels[y]} vs {labels[x]} (with Occurrence Pattern)",
        margin=dict(l=60, r=100, t=60, b=60),
        template='plotly_white',
        height=800
    )
    return fig


# ---------------------------------- Histogram Plot ----------------------------------

def plot_histogram(df, column, by=None, bins=None, log_x=False, log_y=False):
    labels = prepare_labels(column, by) if by else prepare_labels(column)
    labels.update({'source': 'Dataset', 'pl_name': 'Planet'})
    palette = px.colors.qualitative.Plotly

    data = df[[column] + ([by] if by else [])].replace([np.inf, -np.inf], np.nan).dropna(subset=[column])

    fig = go.Figure()
    if by and by in data.columns:
        for i, (name, group) in enumerate(data.groupby(by)):
            fig.add_trace(go.Histogram(
                x=group[column], name=str(name),
                opacity=0.75, 
                marker=dict(
                    color=palette[i % len(palette)],
                    line=dict(color='black', width=1)
                ),
                nbinsx=bins,
                hovertemplate=f"{labels[by]}: {name}<br>{labels[column]}: %{{x}}<br>Count: %{{y}}<extra></extra>"
            ))
        barmode = 'stack'
    else:
        fig.add_trace(go.Histogram(
            x=data[column],
            nbinsx=bins,
            marker=dict(
                color=palette[0],
                line=dict(color='black', width=1)
            ),
            hovertemplate=f"{labels[column]}: %{{x}}<br>Count: %{{y}}<extra></extra>"
        ))
        barmode = 'relative'

    fig.update_layout(
        barmode=barmode,
        title=f"Histogram of {labels[column]}" + (f" by {labels[by]}" if by else ''),
        xaxis=dict(title=labels[column], type='log' if log_x else 'linear'),
        yaxis=dict(title='Count'),
        margin=dict(l=60, r=60, t=60, b=60),
        template='plotly_white',
        height=600
    )
    
    return fig



# ---------------------------------- Main Dispatcher ----------------------------------

def main_plot(plot_type, preset_keys=None, df_full=None,
              x_axis=None, y_axis=None, highlight_planets=None,
              color_by=None, log_x=False, log_y=False,
              show_error=True, cmap='YlOrBr', show_points=True, bins=None):
    # Load data if using presets
    if df_full is None:
        raise ValueError("`df_full` is required when using `preset_keys`")
    if isinstance(df_full, str):
        df_full = ALL_DATA.get(df_full)
        if df_full is None:
            raise KeyError(f"No data named '{df_full}' in ALL_DATA")

    df_list = []
    for key in preset_keys:
        if key in ALL_PRESETS:
            df_list.append((key, ALL_PRESETS[key](df_full)))
        elif key in ALL_DATA:
            df_list.append((key, ALL_DATA[key]))
        else:
            raise KeyError(f"No such presets or data: {key}")

    df = combine_samples(df_list)
    if 'pl_name' not in df.columns:
        raise KeyError("Combined DataFrame must contain a 'pl_name' column.")

    if highlight_planets:
        missing = [p for p in highlight_planets if p not in df['pl_name'].values]
        if missing:
            print(f"⚠️ Planets not found: {', '.join(missing)}")

    # Supported plot functions
    plot_funcs = {
        'scatter': plot_scatter,
        'colored': plot_colored,
        'density': plot_density,
        'histogram': plot_histogram
    }

    if plot_type not in plot_funcs:
        raise ValueError(f"Unknown plot_type: {plot_type}")

    # Prepare arguments per plot type
    if plot_type == 'colored':
        kwargs = dict(df=df, x=x_axis, y=y_axis, highlight=highlight_planets,
                      log_x=log_x, log_y=log_y, show_error=show_error, color_by=color_by)
    elif plot_type == 'density':
        kwargs = dict(df=df, x=x_axis, y=y_axis, highlight=highlight_planets,
                      log_x=log_x, log_y=log_y, show_error=show_error, cmap=cmap)
    elif plot_type == 'histogram':
        kwargs = dict(df=df, column=x_axis, by=color_by,
                      log_x=log_x, log_y=log_y, bins=bins)
    else:  # 'scatter'
        kwargs = dict(df=df, x=x_axis, y=y_axis, highlight=highlight_planets,
                      log_x=log_x, log_y=log_y, show_error=show_error)

    fig = plot_funcs[plot_type](**kwargs)
    fig.show()


