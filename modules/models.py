"""
Mass-Radius Theoretical Models Loader
-------------------------------------
Provides the `MassRadiusModels` class to safely load and retrieve theoretical 
mass-radius relationship curves (e.g., Zeng, Marcus) for plot overlays.

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot_ENS
"""

import pandas as pd

# Import the pre-configured paths and catalog dictionary from our constants
from .constants import MODELS_DIR, MODEL_CATALOG


class MassRadiusModels:
    """
    A utility class to handle the loading and formatting of theoretical 
    mass-radius curve data from local text files.
    """

    def __init__(self):
        """
        Initializes the model loader.
        """
        self.catalog = MODEL_CATALOG
        self.models_directory = MODELS_DIR

    def list_available_models(self) -> dict:
        """
        Returns the full dictionary of available models.

        Returns:
            dict: The catalog where keys are internal IDs and values are (filename, label).
        """
        # We return a copy to prevent accidental modifications to the original dictionary
        return self.catalog.copy()

    def get_model_label(self, key: str) -> str:
        """
        Retrieves the human-readable label for a specific model key.
        
        Args:
            key (str): The internal identifier for the model (e.g., 'zeng_rocky').
            
        Returns:
            str: The formatting label intended for plot legends.
        """
        return self.catalog.get(key, (None, key))[1]

    def get_model_curve(self, key: str) -> pd.DataFrame:
        """
        Loads the numerical data for the requested mass-radius curve.
        
        Args:
            key (str): The internal identifier for the model.
            
        Returns:
            pd.DataFrame: A dataframe containing exactly two columns: ['mass', 'radius'].
            
        Raises:
            KeyError: If the requested key is not in the catalog.
            FileNotFoundError: If the physical text file is missing from the directory.
            ValueError: If the file does not have exactly two columns.
        """
        # 1. Verify the key is valid
        if key not in self.catalog:
            raise KeyError(f"Invalid model key '{key}'. Use list_available_models() to view options.")