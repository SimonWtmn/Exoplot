"""
MCMC Transit Fitting Utilities
------------------------------
Provides the `TransitFitter` class to run Markov Chain Monte Carlo simulations 
on folded lightcurves to extract physical planetary parameters.
Supports dynamic priors and multiprocessing.

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot_ENS
"""

import os
import numpy as np
import emcee
import batman
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count

# Import the default physical boundaries and assumptions
from .constants import (
    MCMC_BOUNDS, MCMC_X0, MCMC_LABELS, 
    LIMB_DARKENING_COEFFS, LIMB_DARKENING_MODEL, 
    ECCENTRICITY, ARG_PERI
)

class TransitFitter:
    """
    A class to handle the mathematical modeling of a planetary transit and 
    the statistical MCMC exploration of the parameter space.
    """

    def __init__(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, period: float,
                 custom_bounds: list = None, custom_x0: list = None):
        """
        Initializes the fitter with the folded observational data and optional custom priors.
        
        Args:
            time (np.ndarray): The folded time array (usually in Julian Days).
            flux (np.ndarray): The normalized flux array.
            flux_err (np.ndarray): The estimated error for each flux measurement.
            period (float): The orbital period of the planet in days.
            custom_bounds (list): Optional custom min/max boundaries [(min, max), ...].
            custom_x0 (list): Optional custom initial guesses for the optimizer.
        """
        # FIX: Force conversion to pure float64 NumPy arrays. 
        self.time = np.asarray(time, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        self.flux_err = np.asarray(flux_err, dtype=np.float64)
        self.period = period
        
        # Override constants with user inputs if provided, otherwise fallback to defaults
        self.bounds = custom_bounds if custom_bounds is not None else MCMC_BOUNDS
        self.x0 = custom_x0 if custom_x0 is not None else MCMC_X0
        
        # State variables to hold the outputs
        self.best_fit_params = None
        self.sampler = None
        self.flat_samples = None
        self.results_summary = None

    # ===========================================================
    # Internal Math & Modeling Methods
    # ===========================================================

    def _check_bounds(self, params: list) -> bool:
        """Checks if the proposed MCMC step falls within the boundaries."""
        for p, (low, high) in zip(params, self.bounds):
            if not (low <= p <= high):
                return False

        # Physical geometry check
        rp_rs, inc_deg, a_rs, _ = params
        b = a_rs * np.cos(np.radians(inc_deg))
        if b > 1 + rp_rs:
            return False

        return True

    def _make_transit_model(self, params: list, t_array: np.ndarray) -> np.ndarray:
        """Generates a synthetic lightcurve using the Batman package."""
        rp_rs, inc_deg, a_rs, t0 = params

        transit_params = batman.TransitParams()
        transit_params.t0 = t0
        transit_params.per = self.period
        transit_params.rp = rp_rs
        transit_params.a = a_rs
        transit_params.inc = inc_deg
        
        transit_params.ecc = ECCENTRICITY
        transit_params.w = ARG_PERI
        transit_params.u = LIMB_DARKENING_COEFFS
        transit_params.limb_dark = LIMB_DARKENING_MODEL

        model = batman.TransitModel(transit_params, t_array)
        return model.light_curve(transit_params)

    def _log_likelihood(self, params: list) -> float:
        """Calculates the log-likelihood."""
        if not self._check_bounds(params):
            return -np.inf

        try:
            model_flux = self._make_transit_model(params, self.time)
        except Exception:
            return -np.inf

        if not np.all(np.isfinite(model_flux)):
            return -np.inf

        chi2 = np.sum(((self.flux - model_flux) / self.flux_err) ** 2)

        if not np.isfinite(chi2):
            return -np.inf

        return -0.5 * chi2

    def _neg_loglike(self, params: list) -> float:
        """Negative log-likelihood purely for Scipy minimization."""
        val = -self._log_likelihood(params)
        if not np.isfinite(val):
            return 1e10  
        return val

    # ===========================================================
    # Execution Methods
    # ===========================================================

    def optimize_initial_guess(self):
        """Finds a decent starting point for the MCMC walkers."""
        result = minimize(
            self._neg_loglike, 
            self.x0,
            bounds=self.bounds,
            method="L-BFGS-B" # FIX: Using L-BFGS-B rather than Poweel to respect bounds.
        )
        if not result.success:
            print("Warning: Initial optimization failed. Falling back to initial guess.")
            self.best_fit_params = self.x0
        else:
            self.best_fit_params = result.x
            
        return self.best_fit_params

    def run_mcmc(self, nwalkers: int = 32, nsteps: int = 5000, 
                 progress_callback=None, use_multiprocessing: bool = True, n_cores: int = None):
        """
        Executes the Markov Chain Monte Carlo simulation.
        Now supports multiprocessing to drastically speed up calculation.
        """
        ndim = len(MCMC_LABELS)
        self.optimize_initial_guess()
        # Initialize walkers in a tiny Gaussian ball around the best fit
        p0 = self.best_fit_params + 1e-4 * np.random.randn(nwalkers, ndim)
        # FIX: Clip the walkers to ensure the random noise didn't push them outside the bounds
        for i, (low, high) in enumerate(self.bounds):
            p0[:, i] = np.clip(p0[:, i], low, high)
        # Determine number of CPU cores to use
        cores = n_cores if n_cores else cpu_count()
        # Disable OpenMP multithreading internally to prevent CPU thrashing
        if use_multiprocessing:
            os.environ["OMP_NUM_THREADS"] = "1"
            
            with Pool(cores) as pool:
                self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_likelihood, pool=pool)
                self._run_sampling(p0, nsteps, progress_callback)
        else:
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_likelihood)
            self._run_sampling(p0, nsteps, progress_callback)
        # Discard the first 20% of steps (burn-in) and thin the chain by 10
        burn_in = int(nsteps * 0.2)
        self.flat_samples = self.sampler.get_chain(discard=burn_in, thin=10, flat=True)
        self.results_summary = self._compute_summary()

        return self.results_summary

    def _run_sampling(self, p0, nsteps, progress_callback):
        """Helper to run the sampler and emit progress."""
        for step, _ in enumerate(self.sampler.sample(p0, iterations=nsteps, progress=False)):
            if step % 10 == 0 and progress_callback is not None:
                progress_callback(step, nsteps)
        if progress_callback is not None:
            progress_callback(nsteps, nsteps)

    def _compute_summary(self) -> dict:
        """Calculates the median and confidence intervals."""
        results = {}
        for i in range(self.flat_samples.shape[1]):
            q16, q50, q84 = np.percentile(self.flat_samples[:, i], [16, 50, 84])
            results[MCMC_LABELS[i]] = (q50, q84 - q50, q50 - q16)
        return results

    def get_best_model_curve(self, num_points: int = 5000) -> tuple:
        """Generates a smooth, high-resolution transit curve."""
        if self.results_summary is None:
            raise ValueError("MCMC has not been run yet. Call run_mcmc() first.")

        final_params = [self.results_summary[label][0] for label in MCMC_LABELS]
        smooth_time = np.linspace(min(self.time), max(self.time), num_points)
        synthetic_flux = self._make_transit_model(final_params, smooth_time)
        
        return smooth_time, synthetic_flux