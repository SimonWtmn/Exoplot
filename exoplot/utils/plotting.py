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





# -------------------------------------- DATA HELPER --------------------------------------

#  // COMBINED SAMPLES FROM THE LIST OF SELECTED PRESETS 
def combine_samples(samples):
    if isinstance(samples, dict):
        samples = samples.items()
    elif samples and not isinstance(samples[0], tuple):
        samples = [(f"Sample {i+1}", df) for i, df in enumerate(samples)]
    return pd.concat([df.assign(source=str(label)) for label, df in samples], ignore_index=True)


# // ADD READABALE LABEL 
def prepare_labels(*keys):
    return {k: label_map.get(k, k) for k in keys}


# // FIND ERRROS (SKETCH : ONLY GET ERR1) 
def get_error_columns(df, x, y, show_error):
    return (
        f"{x}err1" if show_error and f"{x}err1" in df else None,
        f"{y}err1" if show_error and f"{y}err1" in df else None
    )


# // ENSURE CLEAN DATA FOR JSON 
def clean_data(df, x, y=None, color_by=None, log_x=False, log_y=False, show_error=False):
    cols = [x]
    if y: cols.append(y)
    if color_by: cols.append(color_by)
    err_x, err_y = get_error_columns(df, x, y, show_error)
    if err_x: cols.append(err_x)
    if err_y: cols.append(err_y)
    cols += ['pl_name', 'source']
    df = df[cols].replace([np.inf, -np.inf], np.nan)
    for col in [x, y, color_by, err_x, err_y]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    if log_x:
        df = df[df[x] > 0].copy()
    if log_y and y:
        df = df[df[y] > 0].copy()
    return df






# -------------------------------------- TRACE HELPER -------------------------------------

# // ADD SCATTER DATA
def add_scatter_trace(
    fig, group, x, y, label_x, label_y, name=None, color=None, err_x=None, err_y=None, colorscale=None, cmin=None, cmax=None, colorbar=None, color_by=None):
    marker = dict(opacity=0.8)
    if color is not None:
        marker['color'] = color
    if colorscale:
        marker['colorscale'] = colorscale
    if cmin is not None:
        marker['cmin'] = cmin
    if cmax is not None:
        marker['cmax'] = cmax
    if colorbar:
        marker['colorbar'] = colorbar
    fig.add_trace(go.Scatter(x=group[x].tolist(), y=group[y].tolist(), mode='markers',
        name=name or (group['source'].iloc[0] if 'source' in group and not group.empty else 'Sample'),
        text=group['pl_name'].tolist(),
        marker=marker,
        error_x=dict(array=group[err_x].tolist()) if err_x else None,
        error_y=dict(array=group[err_y].tolist()) if err_y else None,
        hovertemplate=(f"%{{text}}<br>{label_x} = %{{x}}<br>{label_y} = %{{y}}" + (f"<br>{color_by} = %{{marker.color}}" if colorscale else "") + "<extra></extra>"
        ),
        showlegend=True
    ))



# // ADD SCATTER HIGHLIGHTED PLANETS
def add_highlight_traces(fig, df, x, y, label_x, label_y, highlight):
    if not highlight:
        return
    for planet in highlight:
        if planet not in df['pl_name'].values:
            print(f"Planet {planet} not found in current data, skipping")
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
                hovertemplate=f"%{{text}}<br>{label_x} = %{{x}}<br>{label_y} = %{{y}}<extra></extra>",
                showlegend=True
            ))


# // ADD LINE MODELS
def add_model_overlay_traces(fig, x, y, overlay_models):
    if not overlay_models:
        return
    valid_axes = {("pl_bmasse", "pl_rade"), ("pl_rade", "pl_bmasse")}
    if (x, y) not in valid_axes and (y, x) not in valid_axes:
        return
    
    from plotly.colors import DEFAULT_PLOTLY_COLORS

    for i, model_key in enumerate(overlay_models):
        model_df = get_model_curve(model_key)
        x_model = model_df['mass']
        y_model = model_df['radius']

        if x == "pl_rade":
            x_model, y_model = y_model, x_model

        fig.add_trace(go.Scatter(
            x=x_model,
            y=y_model,
            mode='lines',
            name=model_key.replace('_', ' ').title(),
            line=dict(dash='dash', width=2, color=DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]),
            hoverinfo='name',
            showlegend=True
        ))
        fig.update_layout(hovermode='closest')


# // ADD PLOT STYLE
def apply_style(fig, labels, x, y, log_x=False, log_y=False, width=1100, height=500, font_family="Inter, sans-serif", font_size=14, font_color="white"):
    fig.update_layout( title=f"{labels[y]} vs {labels[x]}",
                      font=dict(family=font_family, size=font_size, color=font_color),
                      xaxis=dict(title=labels[x], type='log' if log_x else 'linear'),
                      yaxis=dict(title=labels[y], type='log' if log_y else 'linear'),
                      margin=dict(l=80, r=80, t=80, b=80),
                      template='plotly_dark',
                      height=height,
                      width=width,
                      legend=dict(bgcolor='rgba(68,68,68,0.5)', bordercolor='white', borderwidth=1)
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', showline=True, linecolor='white', mirror=True, title_standoff=20)

    fig.update_yaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', showline=True, linecolor='white', mirror=True, title_standoff=20)

    return fig





# ------------------------------------- PLOT FUNCTIONS ------------------------------------

# // SIMPLE SCATTER
def plot_scatter(df, x, y, highlight, log_x, log_y, show_error, overlay_models):
    labels = prepare_labels(x, y)
    err_x, err_y = get_error_columns(df, x, y, show_error)
    base_df = df if not highlight else df[~df['pl_name'].isin(highlight)]

    fig = go.Figure()

    for src, group in base_df.groupby('source'):
        add_scatter_trace(fig, group, x, y, labels[x], labels[y], name=src, err_x=err_x, err_y=err_y)

    add_highlight_traces(fig, df, x, y, labels[x], labels[y], highlight)
    add_model_overlay_traces(fig, x, y, overlay_models)

    apply_style(fig, labels, x, y, log_x, log_y)

    return fig


# // SCATTER WITH CMAPP
def plot_colored(df, x, y, color_by, highlight=None, log_x=False, log_y=False, show_error=False, colorscale_list=None, overlay_models=None):
    labels = prepare_labels(x, y, color_by)
    palettes = colorscale_list or ['YlOrRd', 'Blues', 'Greens', 'Purples', 'Oranges', 'Viridis']
    vmin, vmax = df[color_by].min(), df[color_by].max()
    err_x, err_y = get_error_columns(df, x, y, show_error)
    base_df = df if not highlight else df[~df['pl_name'].isin(highlight)]

    fig = go.Figure()

    sources = list(base_df['source'].unique())
    n_sources = len(sources)
    min_x = 1.08
    max_x = 1.4
    if n_sources > 1:
        step_x = (max_x - min_x) / (n_sources - 1)
    else:
        step_x = 0

    for i, src in enumerate(sources):
        group = base_df[base_df['source'] == src]
        bar_x = min_x + i * step_x

        add_scatter_trace(
            fig, group, x, y, labels[x], labels[y], name=src,
            color=group[color_by].tolist(),
            err_x=err_x, err_y=err_y,
            colorscale=palettes[i % len(palettes)],
            cmin=vmin, cmax=vmax,
            colorbar=dict(x=bar_x, y=0.315, len=0.7, thickness=12, xanchor='center', yanchor='middle'), 
            color_by=labels[color_by]
        )

        fig.add_annotation(x=bar_x-0.02, y=-0.03, text=src, xref='paper', yref='paper', showarrow=False, xanchor='center', yanchor='top', font=dict(size=12, family="Inter, sans-serif", color="white"))

    center_x = (min_x + max_x) / 2
    fig.add_annotation(x=center_x-0.01, y=-0.1, text=labels[color_by],xref='paper', yref='paper',showarrow=False,xanchor='center', yanchor='top', font=dict(size=12, family="Inter, sans-serif", color="white"))

    add_highlight_traces(fig, df, x, y, labels[x], labels[y], highlight)
    add_model_overlay_traces(fig, x, y, overlay_models)

    apply_style(fig, labels, x, y, log_x, log_y)
    fig.update_layout(title=f"{labels[x]} vs {labels[y]} (colored by {labels[color_by]})")

    return fig


# // SCATTER WITH DENSITY MAP
def plot_density(df, x, y, highlight=None, log_x=False, log_y=False, show_error=False, cmap='YlOrRd', overlay_models=None):
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
    bins = 100
    x_bins = np.linspace(x_data.min() - ((x_data.max() - x_data.min()) * 0.1), x_data.max() + ((x_data.max() - x_data.min()) * 0.1), bins)
    y_bins = np.linspace(y_data.min() - ((y_data.max() - y_data.min()) * 0.1), y_data.max() + ((y_data.max() - y_data.min()) * 0.1), bins)
    H, xedges, yedges = np.histogram2d(x_data, y_data, bins=[x_bins, y_bins])
    H = gaussian_filter(H, sigma=8)
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    fig.add_trace(go.Heatmap(x=x_centers if not log_x else 10**x_centers, y=y_centers if not log_y else 10**y_centers, z=H.T,
                             colorscale=cmap,
                             colorbar=dict(x=1.02, y=0.315, len=0.7, thickness=40),
                             opacity=0.7,
                             name='Density'
    ))

    fig.add_annotation(x=1.06, y=-0.03, text='Density', xref='paper', yref='paper', showarrow=False, xanchor='center', yanchor='top', font=dict(size=15, family='Inter, sans-serif', color='white'))

    palette = px.colors.qualitative.Plotly
    for i, (src, group) in enumerate(base_df.groupby('source')):
        add_scatter_trace(fig, group, x, y, labels[x], labels[y], name=src, color=palette[i % len(palette)], err_x=err_x, err_y=err_y)

    add_highlight_traces(fig, df, x, y, labels[x], labels[y], highlight)
    add_model_overlay_traces(fig, x, y, overlay_models)

    apply_style(fig, labels, x, y, log_x, log_y)

    fig.update_layout(title=f"{labels[y]} vs {labels[x]} (with Density)")

    return fig



# // HISTOGRAM WITH OPTIONAL GROUPS
def plot_histogram(df, column, bins=50, log_x=False, log_y=False):
    labels = {column: label_map.get(column, column), 'count': 'Count'}
    palette = px.colors.qualitative.Plotly

    if log_x:
        df = df[df[column] > 0]

    min_val, max_val = df[column].min(), df[column].max()
    if log_x:
        bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), bins + 1)
    else:
        bin_edges = np.linspace(min_val, max_val, bins + 1)

    counts, edges = np.histogram(df[column], bins=bin_edges)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)

    fig = go.Figure()

    fig.add_trace(go.Bar(x=centers, y=counts, width=widths, name=labels[column],
                         marker=dict(
                             color=palette[0],
                             line=dict(color='black', width=1)),
                         hovertemplate=(f"{labels[column]}: %{{x}}<br>Count: %{{y}}<extra></extra>")
    ))

    fig.update_layout(barmode='relative', title=f"Histogram of {labels[column]}", xaxis_type='log' if log_x else 'linear', yaxis_type='log' if log_y else 'linear')

    apply_style(fig, labels, x=column, y='count', log_x=log_x, log_y=log_y)

    fig.update_yaxes(title='Count')

    return fig





# ------------------------------------- MAIN PLOT TOOL ------------------------------------

def main_plot(plot_type, preset_keys=None, df_full=None, pairs=None,
              x_axis=None, y_axis=None, 
              highlight_planets=None, overlay_models=None,
              color_by=None, log_x=False, log_y=False, show_error=False, cmap='YlOrBr', bins=None):

    if pairs is not None:
        df_list = []
        for preset_key, dataset_name in pairs:
            df_data = ALL_DATA.get(dataset_name)
            if df_data is None:
                raise KeyError(f"No data named '{dataset_name}' in ALL_DATA")
            if preset_key not in ALL_PRESETS:
                raise KeyError(f"No preset named '{preset_key}' in ALL_PRESETS")
            df_filtered = ALL_PRESETS[preset_key](df_data)
            df_list.append((f"{preset_key} ({dataset_name})", df_filtered))
        for preset_key, dataset_name in pairs:
            df_data = ALL_DATA.get(dataset_name)
            df_filtered = ALL_PRESETS[preset_key](df_data)
            print(f"{preset_key} ({dataset_name}): {len(df_filtered)} rows after filtering")
            df_list.append((f"{preset_key} ({dataset_name})", df_filtered))
        df = combine_samples(df_list)

    else:
        if isinstance(df_full, str):
            df_full = ALL_DATA.get(df_full)
            if df_full is None:
                raise KeyError(f"No data named '{df_full}' in ALL_DATA")

        if not preset_keys:
            df = df_full.copy()
            if 'source' not in df.columns:
                df['source'] = 'NEA'
        elif all(key in ALL_PRESETS for key in preset_keys):
            df_list = [(key, ALL_PRESETS[key](df_full)) for key in preset_keys]
            df = combine_samples(df_list)
        else:
            raise ValueError(
                f"preset_keys must be preset names in ALL_PRESETS when using single-dataset mode: {preset_keys}"
            )

    if plot_type == 'histogram':
        df = clean_data(df, x_axis, None, log_x=log_x, log_y=log_y)
    else:
        df = clean_data(df, x_axis, y_axis, color_by=color_by, log_x=log_x, log_y=log_y, show_error=show_error)

    plot_funcs = {
        'scatter': plot_scatter,
        'colored': plot_colored,
        'density': plot_density,
        'histogram': plot_histogram
    }

    if plot_type == 'histogram':
        return plot_histogram(df, column=x_axis, bins=bins, log_x=log_x, log_y=log_y)
    elif plot_type == 'colored':
        return plot_colored(df, x_axis, y_axis, color_by, highlight_planets, log_x, log_y, show_error, overlay_models=overlay_models)
    elif plot_type == 'density':
        return plot_density( df, x_axis, y_axis, highlight_planets, log_x, log_y, show_error, cmap=cmap, overlay_models=overlay_models)
    else: 
        return plot_scatter(df, x_axis, y_axis, highlight_planets, log_x, log_y, show_error, overlay_models)


