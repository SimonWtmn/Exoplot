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
    Fully dynamic: supports variable numbers of fitted parameters.
    """

    def __init__(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, 
                 period: float, t0: float = 0.0, fitted_params: list = None, 
                 custom_bounds: list = None, custom_x0: list = None, custom_labels: list = None):
        """
        Initializes the fitter with data and dynamic parameters.
        
        Args:
            fitted_params (list): List of batman attribute strings to fit, e.g., ['rp', 'inc', 'a', 't0', 'ecc', 'w']
            custom_bounds (list): Custom min/max boundaries [(min, max), ...] matching fitted_params.
            custom_x0 (list): Custom initial guesses matching fitted_params.
            custom_labels (list): LaTeX formatted labels for the plots.
        """
        self.time = np.asarray(time, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        self.flux_err = np.asarray(flux_err, dtype=np.float64)
        self.period = period
        self.t0 = t0 
        
        self.fitted_params = fitted_params if fitted_params is not None else ['rp', 'inc', 'a', 't0']
        n_params = len(self.fitted_params)
        self.bounds = custom_bounds[:n_params] if custom_bounds is not None else MCMC_BOUNDS[:n_params]
        self.x0 = custom_x0[:n_params] if custom_x0 is not None else MCMC_X0[:n_params]
        self.labels = custom_labels[:n_params] if custom_labels is not None else MCMC_LABELS[:n_params]
        
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

        # Extract current geometry to check impact parameter (b)
        # We assume safe defaults, but overwrite them if the MCMC is currently fitting them
        current_vals = {'rp': 0.1, 'inc': 90.0, 'a': 10.0}
        for name, val in zip(self.fitted_params, params):
            if name in current_vals:
                current_vals[name] = val
                
        b = current_vals['a'] * np.cos(np.radians(current_vals['inc']))
        if b > 1 + current_vals['rp']:
            return False

        return True

    def _make_transit_model(self, params: list, t_array: np.ndarray) -> np.ndarray:
        transit_params = batman.TransitParams()
        transit_params.limb_dark = LIMB_DARKENING_MODEL
        
        current_vals = {
            'rp': 0.1, 'inc': 90.0, 'a': 10.0, 't0': self.t0, 
            'ecc': ECCENTRICITY, 'w': ARG_PERI, 
            'u1': LIMB_DARKENING_COEFFS[0], 'u2': LIMB_DARKENING_COEFFS[1],
            'per': self.period
        }
        
        for name, val in zip(self.fitted_params, params):
            current_vals[name] = val
            
        transit_params.t0 = current_vals['t0']
        transit_params.per = current_vals['per']
        transit_params.rp = current_vals['rp']
        transit_params.a = current_vals['a']
        transit_params.inc = current_vals['inc']
        transit_params.ecc = current_vals['ecc']
        transit_params.w = current_vals['w']
        transit_params.u = [current_vals['u1'], current_vals['u2']]

        model = batman.TransitModel(transit_params, t_array)
        return model.light_curve(transit_params)

    def _log_likelihood(self, params: list) -> float:
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
        val = -self._log_likelihood(params)
        if not np.isfinite(val):
            return 1e10  
        return val

    # ===========================================================
    # Execution Methods
    # ===========================================================

    def optimize_initial_guess(self):
        result = minimize(
            self._neg_loglike, 
            self.x0,
            bounds=self.bounds,
            method="L-BFGS-B"
        )
        if not result.success:
            print("Warning: Initial optimization failed. Falling back to initial guess.")
            self.best_fit_params = self.x0
        else:
            self.best_fit_params = result.x
            
        return self.best_fit_params

    def run_mcmc(self, nwalkers: int = 32, nsteps: int = 5000, 
                 progress_callback=None, use_multiprocessing: bool = True, n_cores: int = None):
        ndim = len(self.fitted_params)
        self.optimize_initial_guess()

        p0 = self.best_fit_params + 1e-4 * np.random.randn(nwalkers, ndim)
        for i, (low, high) in enumerate(self.bounds):
            p0[:, i] = np.clip(p0[:, i], low, high)

        cores = n_cores if n_cores else cpu_count()

        if use_multiprocessing:
            os.environ["OMP_NUM_THREADS"] = "1"
            with Pool(cores) as pool:
                self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_likelihood, pool=pool)
                self._run_sampling(p0, nsteps, progress_callback)
        else:
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_likelihood)
            self._run_sampling(p0, nsteps, progress_callback)

        burn_in = int(nsteps * 0.2)
        self.flat_samples = self.sampler.get_chain(discard=burn_in, thin=10, flat=True)
        self.results_summary = self._compute_summary()

        return self.results_summary

    def _run_sampling(self, p0, nsteps, progress_callback):
        for step, _ in enumerate(self.sampler.sample(p0, iterations=nsteps, progress=False)):
            if step % 10 == 0 and progress_callback is not None:
                progress_callback(step, nsteps)
        if progress_callback is not None:
            progress_callback(nsteps, nsteps)

    def _compute_summary(self) -> dict:
        results = {}
        for i in range(self.flat_samples.shape[1]):
            q16, q50, q84 = np.percentile(self.flat_samples[:, i], [16, 50, 84])
            # Maps the specific LaTeX label defined by the user to the result!
            results[self.labels[i]] = (q50, q84 - q50, q50 - q16)
        return results

    def get_best_model_curve(self, num_points: int = 5000, phase_folded: bool = False) -> tuple:
        if self.results_summary is None:
            raise ValueError("MCMC has not been run yet. Call run_mcmc() first.")

        n_params = len(self.fitted_params)
        final_params = [self.results_summary[self.labels[i]][0] for i in range(n_params)]
        
        current_vals = {
            'rp': 0.1, 'inc': 90.0, 'a': 10.0, 't0': self.t0, 
            'ecc': ECCENTRICITY, 'w': ARG_PERI, 
            'u1': LIMB_DARKENING_COEFFS[0], 'u2': LIMB_DARKENING_COEFFS[1],
            'per': self.period
        }
        for name, val in zip(self.fitted_params, final_params):
            current_vals[name] = val
            
        transit_params = batman.TransitParams()
        transit_params.limb_dark = LIMB_DARKENING_MODEL
        transit_params.rp = current_vals['rp']
        transit_params.a = current_vals['a']
        transit_params.inc = current_vals['inc']
        transit_params.ecc = current_vals['ecc']
        transit_params.w = current_vals['w']
        transit_params.u = [current_vals['u1'], current_vals['u2']]
        transit_params.per = current_vals['per']
        
        if phase_folded:
            smooth_time = np.linspace(-current_vals['per']/10, current_vals['per']/10, num_points)
            transit_params.t0 = 0.0 
        else:
            smooth_time = np.linspace(min(self.time), max(self.time), num_points)
            transit_params.t0 = current_vals['t0']

        model = batman.TransitModel(transit_params, smooth_time)
        synthetic_flux = model.light_curve(transit_params)
        
        return smooth_time, synthetic_flux