"""
Exoplanet Visualization Utilities
---------------------------------
This module provides plotting functions to visualize planetary and
stellar parameters from the NEA dataset.

Functions:
    - plot_sample: Scatter plot of all vs. filtered samples with optional highlight and error bars.
    - mass_radius_plot: Mass vs. radius plot colored by a third parameter.
    - histogram_by_feature: 1D histogram with optional hue variable (continuous or categorical).
    - plot_2d_density: Density map of planet occurrence with 2D binning and optional uncertainties.

Author: S.WITTMANN & V.RAGNER
Repository: https://github.com/SimonWtmn/Stage_CEA_Exoplanet
"""

import inspect
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import numpy as np
from utils.label import label_map
import itertools


# ------------------ Main Scatter Plot ------------------

def plot_sample(sample_list, x_axis, y_axis, highlight_planet=None, log_x=False, log_y=False):
    """Scatter plot comparing full vs. filtered sample with optional highlighted planet and error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get variable names from the caller's scope for legend use
    caller_vars = inspect.currentframe().f_back.f_locals
    name_lookup = {id(val): name for name, val in caller_vars.items() if hasattr(val, 'shape')}

    markers = itertools.cycle(['o', 's', 'D', '^', 'v', 'P', 'X'])
    colors = itertools.cycle(sns.color_palette("tab10"))

    for df in sample_list:
        if df.empty:
            continue

        label = name_lookup.get(id(df), "Unknown Sample").replace("_sample", "")

        ax.errorbar(
            df[x_axis], df[y_axis],
            xerr=df.get(f"{x_axis}err1", None),
            yerr=df.get(f"{y_axis}err1", None),
            fmt=next(markers),
            color=next(colors),
            label=label,
            linestyle='None',
            markersize=5,
            alpha=0.8,
            zorder=2
        )

        if highlight_planet and highlight_planet in df['pl_name'].values:
            df_highlight = df[df['pl_name'] == highlight_planet]
            ax.scatter(
                df_highlight[x_axis], df_highlight[y_axis],
                c='red', edgecolors='black', s=90, marker='*', zorder=4,
                label=f"Highlighted: {highlight_planet}"
            )

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())

    if "teff" in x_axis or "temp" in x_axis:
        ax.invert_xaxis()

    ax.set_xlabel(label_map.get(x_axis, x_axis))
    ax.set_ylabel(label_map.get(y_axis, y_axis))
    ax.legend()
    plt.tight_layout()
    plt.show()


# ------------------ Mass-Radius Plot ------------------

def mass_radius_plot(df, yaxis="pl_bmasse", color_by="st_teff", log_x=True, log_y=True):
    """Mass-radius scatter plot with optional color map based on a third variable."""
    if color_by not in df.columns:
        raise ValueError(f"Column '{color_by}' not found in dataframe.")

    if df.empty:
        print("No valid data for plotting.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color mapping setup
    cmap = cm.get_cmap("viridis")
    norm = colors.Normalize(vmin=df[color_by].min(), vmax=df[color_by].max())

    # Base scatter with color mapping
    sc = ax.scatter(
        df["pl_rade"], df[yaxis],
        c=df[color_by],
        cmap=cmap,
        norm=norm,
        s=40,
        edgecolor='k',
        alpha=0.8,
        zorder=2
    )

    # Overlay error bars (uniform color for clarity)
    ax.errorbar(
        df["pl_rade"], df[yaxis],
        xerr=df.get("pl_radeerr1", None),
        yerr=df.get(yaxis + "err1", None),
        fmt='none',
        ecolor='gray',
        alpha=0.5,
        zorder=1
    )

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())

    ax.set_xlabel(label_map.get("pl_rade", "pl_rade"))
    ax.set_ylabel(label_map.get(yaxis, yaxis))

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(label_map.get(color_by, color_by))

    plt.tight_layout()
    plt.show()


# ------------------ Histogram Plot ------------------

def histogram_by_feature(df, x_axis, hue=None, bins=30, log_x=False):
    """Histogram with optional grouping hue variable (categorical or continuous)."""

    # Track original hue (for labeling) and plotting hue (actual data column used)
    original_hue = hue
    plot_hue = hue

    # Special case: simplify spectral type
    if hue == "st_spectype":
        df = df.copy()
        df["st_spectype_simple"] = df["st_spectype"].astype(str).str.strip().str.upper().str[0]
        plot_hue = "st_spectype_simple"

    # Drop rows with missing data
    required_cols = [x_axis] + ([plot_hue] if plot_hue else [])
    df_valid = df.dropna(subset=required_cols)
    if df_valid.empty:
        print("No data available for the specified columns.")
        return

    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        data=df_valid,
        x=x_axis,
        hue=plot_hue,
        bins=bins,
        palette="muted",
        multiple="stack" if plot_hue else "layer",
        edgecolor="black"
    )

    # Apply log scale if needed
    if log_x:
        plt.xscale("log")

    # Labels
    plt.xlabel(label_map.get(x_axis, x_axis))
    plt.ylabel("Count")

    # Legend fix — always use original hue to fetch human-readable label
    if hue:
        title = label_map.get(original_hue, original_hue)
        legend = ax.get_legend()
        if legend:
            legend.set_title(title)

    plt.tight_layout()
    plt.show()





# ------------------ 2D Density Plot ------------------

def plot_2d_density(df, x_axis, y_axis="pl_rade", bins=50, cmap="YlOrBr", log_x=False, log_y=False, add_uncertainty=False):
    """2D density map showing relative occurrence of planets with optional uncertainties."""
    df_valid = df[[x_axis, y_axis]].dropna()
    if df_valid.empty:
        print("No valid data to plot.")
        return

    x = df_valid[x_axis]
    y = df_valid[y_axis]
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(xedges, yedges, hist.T, cmap=cmap)
    plt.colorbar(pcm, ax=ax, label="Relative Density of Planets")
    ax.scatter(x, y, s=8, c='black', edgecolors='white', linewidths=0.3, alpha=0.7)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    if add_uncertainty:
        ax.errorbar(0.7, 2.5, xerr=0.05, yerr=0.2,
                    fmt='o', color='black', capsize=3, markersize=0)
        ax.text(0.82, 2.9, 'typical\nuncert.', fontsize=9)

    ax.set_xlabel(label_map.get(x_axis, x_axis))
    ax.set_ylabel(label_map.get(y_axis, y_axis))
    plt.tight_layout()
    plt.show()

