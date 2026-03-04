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

    def list_available_models(self):
        """
        Returns the full dictionary of available models.

        Returns:
            dict: The catalog where keys are internal IDs and values are (filename, label).
        """
        return self.catalog.copy()

    def get_model_label(self, key: str):
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
        Loads the numerical data for the requested mass-radius curve using 
        specific reading instructions if provided in the catalog.
        """
        if key not in self.catalog:
            raise KeyError(f"Invalid model key '{key}'.")

        # Extract file info
        model_info = self.catalog[key]
        filename = model_info[0]
        filepath = self.models_directory / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Model data file not found at: {filepath}")

        # Default reading parameters (works for Zeng and Marcus)
        read_params = {
            'sep': r'\s+|\t+',
            'header': None,
            'engine': 'python',
            'comment': '#'
        }
        if len(model_info) > 2:
            read_params.update(model_info[2])

        try:
            # Load the dataframe with the dynamic instructions
            df = pd.read_csv(filepath, **read_params)
            
            # If 'usecols' wasn't specified, we aggressively force the first two columns
            if 'usecols' not in read_params:
                df = df.iloc[:, :2]
                
        except Exception as e:
            raise ValueError(f"Failed to read file {filename}: {e}")
            
        # Ensure we don't have stray string headers in our data rows
        if isinstance(df.iloc[0, 0], str):
            df = df.iloc[1:].copy()
            df = df.apply(pd.to_numeric, errors='coerce')

        # Standardize the output
        df.columns = ['mass', 'radius']
        
        return df.dropna().reset_index(drop=True)