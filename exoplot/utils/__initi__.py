"""
Utils Package
-------------
This package contains utility modules for exoplanet data visualization.

Modules:
    - filters: Filtering logic for exoplanet data based on various parameters
    - presets: Predefined filters for stellar types, missions, and papers
    - label: Label mapping for plot axes and human-readable names
    - models: Compiled model from mr-plotter data
    - plotting: Visualization utilities (scatter, histograms, density maps)

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot
"""

from .filters import apply_filters

from .presets import ALL_DATA, STELLAR_PRESETS, MISSION_PRESETS, LIT_PRESETS, HZ_PRESETS, PLANET_PRESETS, CUSTOM_PRESETS

from .label import label_map

from .models import get_model_curve, get_model_label

from .plotting import main_plot
