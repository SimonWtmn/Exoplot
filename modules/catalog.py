"""
Exoplanet Catalog Manager
-------------------------
Provides the `ExoplanetCatalog` class to load, manage, and filter exoplanetary data
using an object-oriented approach and method chaining.

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot_ENS
"""

import pandas as pd
from pathlib import Path

# We import the required constants from our newly created constants file
from .constants import SPECTRAL_TYPE_TEMPERATURES, DATA_PATHS


class ExoplanetCatalog:
    """
    Represent and filter an Exoplanet dataset.
    
    Attributes:
        original_df (pd.DataFrame): The unaltered, freshly loaded dataset.
        df (pd.DataFrame): The current dataset, modified by applied filters.
        name (str): The name identifier of the catalog (e.g., 'NEA', 'TOI').
    """

    def __init__(self, dataset_name="NEA", custom_path=None):
        """
        Initializes the catalog, loading the data into memory.
        
        Args:
            dataset_name (str): Identifier from DATA_PATHS (default: 'NEA').
            custom_path (str or Path): Optional override if loading a non-standard file.
        """
        self.name = dataset_name
        
        # Determine the correct path: either the custom one, or one from the constants
        if custom_path:
            file_path = Path(custom_path)
        else:
            file_path = DATA_PATHS.get(dataset_name)
            
        if file_path is None or not file_path.exists():
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found at {file_path}")

        # Load the data once and store the untouched original
        self.original_df = self._load_data(file_path)
        
        # self.df is the working copy that gets modified by filters
        self.df = self.original_df.copy()

    def _load_data(self, path: Path):
        """
        Internal method to read CSV data and clean up any trailing whitespace in headers.
        """
        df = pd.read_csv(path, comment='#')
        df.columns = df.columns.str.strip()
        return df

    def reset(self):
        """
        Removes all applied filters and restores the dataset to its original loaded state.
        
        Returns:
            self: Allows method chaining.
        """
        self.df = self.original_df.copy()
        return self

    def get_data(self):
        """
        Returns the currently filtered dataframe with a clean, reset index.
        This is typically the final method called in a chain.
        
        Returns:
            pd.DataFrame: The filtered dataframe.
        """
        return self.df.reset_index(drop=True)


    # ===========================================================
    # Internal Filter Helpers
    # ===========================================================

    def _apply_range(self, col: str, min_val, max_val):
        """
        Internal helper to apply a minimum and/or maximum threshold to a specific column.
        Updates self.df in place.
        """
        # If the column doesn't exist in this dataset, do nothing
        if col not in self.df.columns:
            return

        # Ensure min is not greater than max to prevent illogical queries
        if min_val is not None and max_val is not None and min_val > max_val:
            raise ValueError(f"Min value ({min_val}) cannot be greater than max value ({max_val})")

        # Start with a mask of all rows where the column data is not NaN (missing)
        mask = self.df[col].notna()
        
        if min_val is not None:
            mask &= self.df[col] >= min_val
        if max_val is not None:
            mask &= self.df[col] <= max_val
            
        # Apply the mask to keep only matching rows
        self.df = self.df[mask]

    def _snr_mask(self, col: str, min_snr: float):
        """
        Internal helper to filter based on Signal-to-Noise Ratio (SNR).
        Requires the dataset to have standard NASA Exoplanet Archive error columns (err1, err2).
        Updates self.df in place.
        """
        err_col1, err_col2 = f"{col}err1", f"{col}err2"
        
        # Check if the error columns actually exist before trying to use them
        if err_col1 not in self.df.columns or err_col2 not in self.df.columns:
            return

        # Keep rows where the main value is present AND at least one error value is present
        mask = self.df[col].notna() & (self.df[err_col1].notna() | self.df[err_col2].notna())
        
        # Calculate the absolute maximum error between upper and lower bounds
        errs = self.df.loc[mask, [err_col1, err_col2]].abs().max(axis=1)
        
        # Calculate the SNR
        snr = self.df.loc[mask, col] / errs
        
        # Get the index of rows that meet the minimum SNR requirement
        valid_indices = mask[mask].index[snr >= min_snr]
        self.df = self.df.loc[valid_indices]


    # ===========================================================
    # Public Filter Methods (Method Chaining)
    # ===========================================================
    
    def filter_discovery(self, mission=None, method=None, year_min=None, year_max=None, kp_max=None):
        """
        Filters the dataset based on how and when the planets were discovered.
        """
        if mission:
            self.df = self.df[self.df['disc_facility'].notna() & (self.df['disc_facility'] == mission)]
            
        if method:
            self.df = self.df[self.df['discoverymethod'].notna() & (self.df['discoverymethod'] == method)]
            
        self._apply_range('disc_year', year_min, year_max)
        self._apply_range('sy_kepmag', None, kp_max)
        
        return self

    def filter_stellar(self, st_type=None, teff_min=None, teff_max=None, lum_min=None, lum_max=None,
                       met_min=None, met_max=None, age_min=None, age_max=None,
                       rad_min=None, rad_max=None, rad_err=None):
        """
        Filters the dataset based on the properties of the host star.
        """
        if st_type:
            mask = self.df['st_spectype'].notna() & self.df['st_spectype'].str.upper().str.startswith(st_type.upper())
            self.df = self.df[mask]
            
        self._apply_range('st_teff', teff_min, teff_max)
        self._apply_range('st_lum', lum_min, lum_max)
        self._apply_range('st_met', met_min, met_max)
        self._apply_range('st_age', age_min, age_max)
        self._apply_range('st_rad', rad_min, rad_max)
        
        if rad_err is not None:
            self._snr_mask('st_rad', rad_err)
            
        return self
    
    def filter_spectral_type(self, st_class: str):
        """
        Filters the dataset by stellar spectral class (O, B, A, F, G, K, M, L, T).
        If the 'st_spectype' string is missing for a star, it falls back to checking 
        if the effective temperature (st_teff) falls within the expected range for that class.
        """
        st_class = st_class.upper()
        
        # 1. Mask for explicit spectral type string match
        mask_str = self.df['st_spectype'].notna() & self.df['st_spectype'].str.upper().str.startswith(st_class)
        
        # 2. Mask for temperature fallback
        if st_class in SPECTRAL_TYPE_TEMPERATURES:
            teff_min, teff_max = SPECTRAL_TYPE_TEMPERATURES[st_class]
            
            mask_teff = self.df['st_teff'].notna()
            if teff_min is not None:
                mask_teff &= self.df['st_teff'] >= teff_min
            if teff_max is not None:
                mask_teff &= self.df['st_teff'] <= teff_max
                
            # Combine: keep if it explicitly matches the string, OR if the string is missing 
            # but the temperature falls perfectly into the category
            final_mask = mask_str | (self.df['st_spectype'].isna() & mask_teff)
        else:
            final_mask = mask_str
            
        self.df = self.df[final_mask]
        
        return self

    def filter_planet(self, rade_min=None, rade_max=None, rade_err=None, 
                      mass_min=None, mass_max=None, mass_err=None, 
                      density_min=None, density_max=None, 
                      eqt_min=None, eqt_max=None, 
                      tdyson_min=None, tdyson_max=None):
        """
        Filters the dataset based on the physical properties of the planet itself.
        """
        self._apply_range('pl_rade', rade_min, rade_max)
        
        if rade_err is not None:
            self._snr_mask('pl_rade', rade_err)
            
        self._apply_range('pl_bmasse', mass_min, mass_max)
        
        if mass_err is not None:
            self._snr_mask('pl_bmasse', mass_err)
            
        self._apply_range('pl_dens', density_min, density_max)
        self._apply_range('pl_eqt', eqt_min, eqt_max)

        # Calculate and filter by Dyson sphere temperature approximation if required
        if tdyson_min is not None or tdyson_max is not None:
            # We calculate this column on the fly based on existing stellar/orbital parameters
            self.df['pl_tdyson'] = self.df['st_teff'] * ((self.df['st_rad'] * 0.00465) / self.df['pl_orbsmax'])**0.5
            self._apply_range('pl_tdyson', tdyson_min, tdyson_max)
            
        return self

    def filter_orbit(self, distance_min=None, distance_max=None, eccentricity_max=None, 
                     transit_depth_min=None, transit_depth_max=None, 
                     period_max=None, impact_param_max=None):
        """
        Filters the dataset based on the orbital parameters of the planet.
        """
        self._apply_range('pl_orbsmax', distance_min, distance_max)
        self._apply_range('pl_orbeccen', None, eccentricity_max)
        self._apply_range('pl_trandep', transit_depth_min, transit_depth_max)
        self._apply_range('pl_orbper', None, period_max)
        self._apply_range('pl_imppar', None, impact_param_max)
        
        return self

    def filter_system(self, multiplicity_min=None, multiplicity_max=None):
        """
        Filters the dataset based on the macroscopic properties of the planetary system.
        """
        self._apply_range('sy_pnum', multiplicity_min, multiplicity_max)
        return self

    def filter_fulton_gap(self):
        """
        Applies the specific empirical Fulton Gap cutoff threshold for planetary radius 
        relative to stellar effective temperature.
        """
        mask_valid = self.df['st_teff'].notna() & self.df['st_rad'].notna()
        
        # Calculate the threshold line equation based on Fulton et al. 2017
        threshold = 10 ** (0.00025 * (self.df.loc[mask_valid, 'st_teff'] - 5500) + 0.20)
        
        # Identify which planets fall below the radius threshold
        pass_fulton = self.df.loc[mask_valid, 'st_rad'] < threshold
        
        # We retain rows that pass the fulton test, AND rows that lacked data to be tested at all
        valid_indices = mask_valid[~mask_valid | pass_fulton].index
        self.df = self.df.loc[valid_indices]
        
        return self