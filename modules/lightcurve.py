"""
Lightcurve Processing Utilities
-------------------------------
Provides the `LightCurveAnalyzer` class to search, download, clean, 
and fold exoplanetary lightcurve data using the Lightkurve package.

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot_ENS
"""

import numpy as np
import pandas as pd
import lightkurve as lk
import astropy.units as u


class LightCurveAnalyzer:
    """
    A class used to manage the lifecycle of a lightcurve analysis,
    from searching the MAST archive to folding the data on a specific period.
    """

    def __init__(self, target_name: str):
        """
        Initializes the analyzer for a specific astronomical target.
        
        Args:
            target_name (str): The name of the star or planet (e.g., 'Kepler-10', 'TOI-700').
        """
        self.target_name = target_name
        
        # We initialize all our state variables to None. They will be populated as the user progresses through the analysis steps.
        self.search_result = None
        self.raw_lc = None
        self.clean_lc = None
        self.periodogram = None
        self.folded_lc = None
        
        # Extracted BLS (Box Least Squares) parameters
        self.best_period = None
        self.best_freq = None
        self.best_power = None
        self.epoch_time = None
        self.transit_time = None
        self.transit_depth = None

    def search(self, radius=None, exptime=None, cadence=None, 
               mission=('Kepler', 'K2', 'TESS'), author=None, 
               quarter=None, month=None, campaign=None, sector=None, 
               limit=None) -> pd.DataFrame:
        """
        Searches the MAST archive for available lightcurves matching the target.
        
        Returns:
            pd.DataFrame: A table containing the metadata of all found observations.
                          This can be easily rendered as an HTML table by Flask.
        """
        self.search_result = lk.search_lightcurve(
            self.target_name, radius=radius, exptime=exptime, cadence=cadence,
            mission=mission, author=author, quarter=quarter, month=month, 
            campaign=campaign, sector=sector, limit=limit
        )

        if len(self.search_result) == 0:
            return pd.DataFrame() # Return empty dataframe if nothing is found
        df = self.search_result.table.to_pandas()
        cols_to_show = ['mission', 'year', 'author', 'exptime', 'target_name', 'distance']
        available_cols = [col for col in cols_to_show if col in df.columns]
        
        return df[available_cols]

    def download_and_clean(self, index: int):
        """
        Downloads a specific lightcurve from the search results, normalizes it,
        and removes invalid data points (NaNs).
        
        Args:
            index (int): The row number from the search result dataframe.
        """
        if self.search_result is None or len(self.search_result) == 0:
            raise ValueError("No search results available. Call search() first.")  
        if index < 0 or index >= len(self.search_result):
            raise IndexError(f"Invalid index {index}. Must be between 0 and {len(self.search_result)-1}.")
        self.raw_lc = self.search_result[index].download()
        self.clean_lc = self.raw_lc.normalize().remove_nans()
        
        return self

    def compute_periodogram(self):
        """
        Computes the Box Least Squares (BLS) periodogram to find the most likely
        orbital period of the transiting exoplanet.
        """
        if self.clean_lc is None:
            raise ValueError("No clean lightcurve available. Call download_and_clean() first.")
        self.periodogram = self.clean_lc.to_periodogram(method='bls')
        max_power_idx = np.argmax(self.periodogram.power)
        
        # Extract physical parameters from that peak
        self.best_period = self.periodogram.period[max_power_idx].to_value(u.day)
        self.best_freq = self.periodogram.frequency[max_power_idx].to_value(1/u.day)
        self.best_power = self.periodogram.power[max_power_idx].value
        
        # Find the timestamp of the deepest transit to use as our folding epoch (t0)
        self.epoch_time = self.clean_lc.time[np.argmin(self.clean_lc.flux)].value
        
        return self

    def fold_lightcurve(self, harmonic: int = 1):
        """
        Folds the time series data over the calculated orbital period so that all 
        transits stack on top of each other at phase 0.
        
        Args:
            harmonic (int): Multiplier for the period (e.g., 2 to check for secondary eclipses).
        """
        if self.clean_lc is None or self.best_period is None:
            raise ValueError("Cannot fold. Ensure download_and_clean() and compute_periodogram() are called.")

        if harmonic <= 0:
            harmonic = 1

        fold_period = harmonic * self.best_period
        
        # Perform the actual folding operation
        self.folded_lc = self.clean_lc.fold(period=fold_period, epoch_time=self.epoch_time)
        
        return self

    def get_mcmc_data(self, folded: bool = True) -> tuple:
        """
        Extracts the raw numpy arrays needed by the Emcee and Batman packages.
        
        Args:
            folded (bool): If True, returns the phase-folded data. If False, returns the raw un-folded data.
            
        Returns:
            tuple: (time_array, flux_array, flux_error_array, best_period, epoch_time_t0)
        """
        if folded:
            if self.folded_lc is None:
                raise ValueError("No folded lightcurve available. Call fold_lightcurve() first.")
            lc = self.folded_lc
        else:
            if self.clean_lc is None:
                raise ValueError("No clean lightcurve available. Call download_and_clean() first.")
            lc = self.clean_lc

        # We use Julian Days (.jd) for time as it is standard for transit modeling
        time_jd = lc.time.jd
        flux_val = lc.flux.value
        
        # If the telescope didn't provide error margins, we estimate it as 1% of the median flux
        if lc.flux_err is not None:
            flux_err = lc.flux_err.value
        else:
            flux_err = np.full_like(flux_val, np.median(flux_val) * 0.01)

        return time_jd, flux_val, flux_err, self.best_period, self.epoch_time