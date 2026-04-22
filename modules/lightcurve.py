"""
Lightcurve Processing Utilities
-------------------------------
Provides the `LightCurveAnalyzer` class to search, download, clean, 
and fold exoplanetary lightcurve data using the Lightkurve package.

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot
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

    def download_and_clean(self, index: int, sigma: float = 5.0):
        """
        Downloads a specific lightcurve from the search results, normalizes it,
        removes invalid data points (NaNs), and clips ``sigma``-σ outliers.

        The outlier-clipping step is critical for transit fitting: a single
        cosmic-ray hit or scattered-light spike at the wrong phase will
        otherwise be picked up as the deepest point and corrupt the BLS
        epoch estimate (see ``compute_periodogram``).

        Args:
            index (int): The row number from the search result dataframe.
            sigma (float): σ-clipping threshold for outlier removal.
                ``5.0`` is conservative — high enough to keep the in-transit
                points, low enough to remove most thruster firings / cosmic
                rays.  Pass ``sigma=None`` to disable clipping entirely.
        """
        if self.search_result is None or len(self.search_result) == 0:
            raise ValueError("No search results available. Call search() first.")  
        if index < 0 or index >= len(self.search_result):
            raise IndexError(f"Invalid index {index}. Must be between 0 and {len(self.search_result)-1}.")
        self.raw_lc = self.search_result[index].download()
        clean = self.raw_lc.normalize().remove_nans()
        if sigma is not None:
            clean = clean.remove_outliers(sigma=sigma)
        self.clean_lc = clean

        return self

    def compute_periodogram(self):
        """
        Computes the Box Least Squares (BLS) periodogram to find the most likely
        orbital period of the transiting exoplanet, and uses the BLS fit
        itself to derive a robust mid-transit epoch.
        """
        if self.clean_lc is None:
            raise ValueError("No clean lightcurve available. Call download_and_clean() first.")
        self.periodogram = self.clean_lc.to_periodogram(method='bls')
        max_power_idx = np.argmax(self.periodogram.power)

        # Extract physical parameters from that peak
        self.best_period = self.periodogram.period[max_power_idx].to_value(u.day)
        self.best_freq = self.periodogram.frequency[max_power_idx].to_value(1/u.day)
        self.best_power = self.periodogram.power[max_power_idx].value

        # Mid-transit epoch.  ``np.argmin(flux)`` is a 1-sample noise
        # estimator: photon noise picks the most negatively-displaced
        # individual cadence, which on a sharp deep transit can sit
        # 5–15 min away from the true centre and then leaks into every
        # downstream parameter via the LD ↔ t0 degeneracy (WASP-121 b
        # symptom).  ``periodogram.transit_time_at_max_power`` is the
        # bin centre of the deepest BLS box, evaluated against *all*
        # stacked transits — much more reliable.
        try:
            self.epoch_time = self.periodogram.transit_time_at_max_power.value
        except AttributeError:
            # Older lightkurve versions did not expose this attribute;
            # fall back to the previous (worse) estimate so existing
            # workflows still run.
            self.epoch_time = self.clean_lc.time[
                np.argmin(self.clean_lc.flux)].value

        return self

    def fold_lightcurve(self, harmonic: int = 1,
                        period: float | None = None,
                        epoch_time: float | None = None):
        """
        Folds the time series data over the calculated orbital period so that
        all transits stack on top of each other at phase 0.

        Args:
            harmonic (int): Multiplier for the period (e.g., 2 to check for
                secondary eclipses).
            period (float, optional): Override the period used for folding.
                Useful for re-folding with the MCMC posterior period after
                a fit so the data-vs-model overlay isn't smeared by an
                incorrect BLS period (typical drift across a 30-day TESS
                sector: ~10 min for a 0.04 % period error).
            epoch_time (float, optional): Override the mid-transit epoch
                used for folding.  Defaults to ``self.epoch_time``.
        """
        if self.clean_lc is None or self.best_period is None:
            raise ValueError("Cannot fold. Ensure download_and_clean() and compute_periodogram() are called.")

        if harmonic <= 0:
            harmonic = 1

        base_period = period if period is not None else self.best_period
        fold_period = harmonic * base_period
        epoch = epoch_time if epoch_time is not None else self.epoch_time

        self.folded_lc = self.clean_lc.fold(period=fold_period,
                                            epoch_time=epoch)
        return self

    def refold_with_posterior(self, period: float, epoch_time: float,
                              harmonic: int = 1):
        """
        Re-fold the cleaned lightcurve using post-MCMC (period, t0) values
        and update ``self.best_period`` / ``self.epoch_time`` so any
        downstream plotting (the DVR report in particular) sees the
        same numbers as the model curve.

        Without this step, the plot folds the data with the BLS period
        and overlays the model with the MCMC period; the visible
        result is a "shifted" or smeared transit even when the fit
        itself is excellent.
        """
        self.best_period = float(period)
        self.epoch_time = float(epoch_time)
        return self.fold_lightcurve(harmonic=harmonic)

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

        # Use the same time system as Lightkurve's native scale (.value), e.g. BTJD
        # for TESS or BKJD for Kepler — must match ``epoch_time`` and any plots using
        # ``clean_lc.time.value``.  Mixing ``.jd`` here with ``.value`` elsewhere
        # breaks Batman overlays (model transits shifted w.r.t. data).
        time_arr = np.asarray(lc.time.value, dtype=np.float64)
        flux_val = lc.flux.value
        
        # If the telescope didn't provide error margins, we estimate it as 1% of the median flux
        if lc.flux_err is not None:
            flux_err = lc.flux_err.value
        else:
            flux_err = np.full_like(flux_val, np.median(flux_val) * 0.01)

        return time_arr, flux_val, flux_err, self.best_period, self.epoch_time