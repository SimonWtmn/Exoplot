"""
Utils Package
-------------
This package contains utility modules for exoplanet data visualization,
filtering, labeling, and preset selection used in the stage_L1_CEA project.

Modules:
    - plots: Visualization utilities (scatter, histograms, density maps)
    - filters: Filtering logic for exoplanet data based on various parameters
    - presets: Predefined filters for stellar types, missions, and papers
    - label: Label mapping for plot axes and human-readable names
"""

from .plots import plot_sample, mass_radius_plot, histogram_by_feature, plot_2d_density
from .filters import apply_filters
from .presets import (
    STELLAR_TYPE_PRESETS,
    MISSION_PRESETS,
    PAPER_PRESETS,
    USER_PRESET
)
from .label import label_map
