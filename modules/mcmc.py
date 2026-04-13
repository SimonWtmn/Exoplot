"""
MCMC Transit Fitting Utilities
------------------------------
Provides the `TransitFitter` class to run Markov Chain Monte Carlo simulations 
on folded lightcurves to extract physical planetary parameters.

Uses DEMove + DESnookerMove for robust exploration and autocorrelation-based
burn-in estimation to avoid stuck walkers and patchy corner plots.

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot_ENS
"""

import os
import logging
import numpy as np
import emcee
import batman
from scipy.optimize import minimize, differential_evolution
from concurrent.futures import ProcessPoolExecutor

from .constants import (
    MCMC_BOUNDS, MCMC_X0, MCMC_LABELS, 
    LIMB_DARKENING_COEFFS, LIMB_DARKENING_MODEL, 
    ECCENTRICITY, ARG_PERI
)

logger = logging.getLogger(__name__)


def _module_level_log_likelihood(params, time, flux, flux_err, period, t0,
                                  fitted_params, bounds):
    """
    Module-level (picklable) log-likelihood for multiprocessing.
    Recreates the transit model from scratch so it can be serialized across
    processes without needing to pickle bound methods.
    """
    for p, (low, high) in zip(params, bounds):
        if not (low <= p <= high):
            return -np.inf

    current_vals = {'rp': 0.1, 'inc': 90.0, 'a': 10.0}
    for name, val in zip(fitted_params, params):
        if name in current_vals:
            current_vals[name] = val

    b = current_vals['a'] * np.cos(np.radians(current_vals['inc']))
    if b > 1 + current_vals['rp']:
        return -np.inf

    full_vals = {
        'rp': 0.1, 'inc': 90.0, 'a': 10.0, 't0': t0,
        'ecc': ECCENTRICITY, 'w': ARG_PERI,
        'u1': LIMB_DARKENING_COEFFS[0], 'u2': LIMB_DARKENING_COEFFS[1],
        'per': period
    }
    for name, val in zip(fitted_params, params):
        full_vals[name] = val

    try:
        tp = batman.TransitParams()
        tp.limb_dark = LIMB_DARKENING_MODEL
        tp.t0 = full_vals['t0']
        tp.per = full_vals['per']
        tp.rp = full_vals['rp']
        tp.a = full_vals['a']
        tp.inc = full_vals['inc']
        tp.ecc = full_vals['ecc']
        tp.w = full_vals['w']
        tp.u = [full_vals['u1'], full_vals['u2']]
        model = batman.TransitModel(tp, time)
        model_flux = model.light_curve(tp)
    except Exception:
        return -np.inf

    if not np.all(np.isfinite(model_flux)):
        return -np.inf

    chi2 = np.sum(((flux - model_flux) / flux_err) ** 2)
    if not np.isfinite(chi2):
        return -np.inf

    return -0.5 * chi2


class TransitFitter:
    """
    MCMC-based transit fitter with robust sampler moves and autocorrelation
    burn-in.  Designed to be usable both from a Jupyter notebook and from
    an async web backend (FastAPI).
    """

    def __init__(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, 
                 period: float, t0: float = 0.0, fitted_params: list = None, 
                 custom_bounds: list = None, custom_x0: list = None,
                 custom_labels: list = None, auto_bounds: bool = True):
        """
        Parameters
        ----------
        auto_bounds : bool
            If True *and* custom_bounds / custom_x0 are not provided,
            derive physically motivated bounds and initial guesses from
            the data using transit geometry.
        """
        self.time = np.asarray(time, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        self.flux_err = np.asarray(flux_err, dtype=np.float64)
        self.period = period
        self.t0 = t0 

        self.fitted_params = fitted_params or ['rp', 'inc', 'a', 't0']
        n = len(self.fitted_params)

        if auto_bounds and custom_bounds is None and custom_x0 is None:
            est_bounds, est_x0 = self.estimate_bounds(
                self.time, self.flux, self.period, self.fitted_params)
            self.bounds = est_bounds
            self.x0 = est_x0
        else:
            self.bounds = (custom_bounds or MCMC_BOUNDS)[:n]
            self.x0 = (custom_x0 or MCMC_X0)[:n]

        self.labels = (custom_labels or MCMC_LABELS)[:n]

        self.best_fit_params = None
        self.sampler = None
        self.flat_samples = None
        self.results_summary = None
        self.autocorr_time = None
        self.burn_in = None

    # =================================================================
    # Physics-based bounds estimation
    # =================================================================

    @staticmethod
    def estimate_bounds(time, flux, period, fitted_params):
        """
        Derive MCMC bounds and initial guesses from the lightcurve data
        using transit geometry.

        Physics used
        ------------
        * **Rp/Rs**: transit depth  delta ~= (Rp/Rs)^2
        * **a/Rs** : from period P and total transit duration T_14 via
                     a/Rs ~= (P / (pi * T_14)) * sqrt((1+k)^2 - b^2)
                     approximated with b~0.
        * **inc**  : near 90 deg for transiting planets; bounded by
                     the requirement b = a * cos(i) < 1 + Rp/Rs.
        * **t0**   : centred on zero for phase-folded data, otherwise
                     on the deepest-dip timestamp.

        Returns (bounds_list, x0_list), each of length len(fitted_params).
        """
        flux = np.asarray(flux, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64)

        baseline = np.median(flux)
        depth = baseline - np.min(flux)
        depth = max(depth, 1e-6)

        rp_est = np.sqrt(depth / baseline)
        rp_lo = max(rp_est * 0.3, 0.001)
        rp_hi = min(rp_est * 3.0, 0.5)

        in_transit = flux < (baseline - 0.25 * depth)
        if np.sum(in_transit) >= 2:
            transit_times = time[in_transit]
            t14 = transit_times.max() - transit_times.min()
        else:
            t14 = period * 0.05
        t14 = max(t14, 1e-6)

        a_est = (period / (np.pi * t14)) * (1 + rp_est)
        a_lo = max(a_est * 0.4, 1.5)
        a_hi = a_est * 2.5

        inc_lo = max(np.degrees(np.arccos(min(1.0, (1 + rp_hi) / a_lo))), 70.0)
        inc_hi = 90.0
        inc_est = max(min(90.0 - 0.5, inc_hi), inc_lo)

        t0_est = 0.0
        span = time.max() - time.min()
        t0_margin = max(t14 * 0.5, span * 0.02)
        t0_lo = t0_est - t0_margin
        t0_hi = t0_est + t0_margin

        param_map = {
            'rp':  ((rp_lo, rp_hi),  rp_est),
            'inc': ((inc_lo, inc_hi), inc_est),
            'a':   ((a_lo, a_hi),     a_est),
            't0':  ((t0_lo, t0_hi),   t0_est),
            'ecc': ((0.0, 0.5),       ECCENTRICITY),
            'w':   ((0.0, 180.0),     ARG_PERI),
            'u1':  ((0.0, 1.0),       LIMB_DARKENING_COEFFS[0]),
            'u2':  ((-0.5, 1.0),      LIMB_DARKENING_COEFFS[1]),
        }

        bounds, x0 = [], []
        for name in fitted_params:
            if name in param_map:
                b, g = param_map[name]
                bounds.append(b)
                x0.append(np.clip(g, b[0] + 1e-10, b[1] - 1e-10))
            else:
                bounds.append(MCMC_BOUNDS[0])
                x0.append(MCMC_X0[0])

        logger.info(
            "Auto-bounds  rp=%.4f [%.4f, %.4f]  a/Rs=%.2f [%.2f, %.2f]  "
            "inc=%.2f [%.2f, %.2f]  T14=%.4f d",
            rp_est, rp_lo, rp_hi, a_est, a_lo, a_hi,
            inc_est, inc_lo, inc_hi, t14
        )
        return bounds, x0

    # =================================================================
    # Internal helpers
    # =================================================================

    def _check_bounds(self, params):
        for p, (low, high) in zip(params, self.bounds):
            if not (low <= p <= high):
                return False
        current_vals = {'rp': 0.1, 'inc': 90.0, 'a': 10.0}
        for name, val in zip(self.fitted_params, params):
            if name in current_vals:
                current_vals[name] = val
        b = current_vals['a'] * np.cos(np.radians(current_vals['inc']))
        if b > 1 + current_vals['rp']:
            return False
        return True

    def _make_transit_model(self, params, t_array):
        tp = batman.TransitParams()
        tp.limb_dark = LIMB_DARKENING_MODEL
        full = {
            'rp': 0.1, 'inc': 90.0, 'a': 10.0, 't0': self.t0,
            'ecc': ECCENTRICITY, 'w': ARG_PERI,
            'u1': LIMB_DARKENING_COEFFS[0], 'u2': LIMB_DARKENING_COEFFS[1],
            'per': self.period
        }
        for name, val in zip(self.fitted_params, params):
            full[name] = val
        tp.t0 = full['t0'];  tp.per = full['per'];  tp.rp = full['rp']
        tp.a  = full['a'];   tp.inc = full['inc'];   tp.ecc = full['ecc']
        tp.w  = full['w'];   tp.u   = [full['u1'], full['u2']]
        return batman.TransitModel(tp, t_array).light_curve(tp)

    def _log_likelihood(self, params):
        if not self._check_bounds(params):
            return -np.inf
        try:
            model_flux = self._make_transit_model(params, self.time)
        except Exception:
            return -np.inf
        if not np.all(np.isfinite(model_flux)):
            return -np.inf
        chi2 = np.sum(((self.flux - model_flux) / self.flux_err) ** 2)
        return -0.5 * chi2 if np.isfinite(chi2) else -np.inf

    def _neg_loglike(self, params):
        val = -self._log_likelihood(params)
        return val if np.isfinite(val) else 1e10

    # =================================================================
    # Pre-optimization
    # =================================================================

    def optimize_initial_guess(self, method="de"):
        """
        Find a robust starting point.
        
        method: 'de' for differential evolution (global, slower, more robust),
                'lbfgsb' for L-BFGS-B (local, fast).
        """
        if method == "de":
            result = differential_evolution(
                self._neg_loglike, bounds=self.bounds,
                seed=42, maxiter=500, tol=1e-8, polish=True
            )
        else:
            result = minimize(
                self._neg_loglike, self.x0,
                bounds=self.bounds, method="L-BFGS-B"
            )

        if not result.success:
            logger.warning("Initial optimisation did not converge; falling back to x0.")
            self.best_fit_params = np.array(self.x0)
        else:
            self.best_fit_params = result.x
        return self.best_fit_params

    # =================================================================
    # Walker initialisation
    # =================================================================

    def _init_walkers(self, nwalkers, ndim):
        """
        Scatter walkers in a small Gaussian ball whose width scales with
        each parameter's allowed range (5 % of the range), clipped to bounds.
        """
        scales = np.array([(hi - lo) * 0.05 for lo, hi in self.bounds])
        p0 = self.best_fit_params + scales * np.random.randn(nwalkers, ndim)
        for i, (lo, hi) in enumerate(self.bounds):
            p0[:, i] = np.clip(p0[:, i], lo + 1e-10, hi - 1e-10)
        return p0

    # =================================================================
    # Main MCMC routine
    # =================================================================

    def run_mcmc(self, nwalkers: int = 64, nsteps: int = 5000,
                 progress_callback=None, use_multiprocessing: bool = False,
                 n_cores: int = None, thin_by: int = 1,
                 optimize_method: str = "de"):
        """
        Run the MCMC sampler.

        Key improvements over the previous version
        -------------------------------------------
        * DEMove + DESnookerMove: avoids stuck walkers far better than the
          default StretchMove in > 2-D parameter spaces.
        * Autocorrelation-based burn-in: discards the right number of steps
          instead of a blind 20 %.
        * Scaled walker initialisation: perturbation proportional to the
          allowed parameter range, not a global 1e-2.
        * Web-safe multiprocessing with ProcessPoolExecutor and a picklable
          module-level likelihood (off by default for web use).
        """
        ndim = len(self.fitted_params)
        self.optimize_initial_guess(method=optimize_method)
        p0 = self._init_walkers(nwalkers, ndim)

        moves = [
            (emcee.moves.DEMove(), 0.8),
            (emcee.moves.DESnookerMove(), 0.2),
        ]

        pool = None
        if use_multiprocessing:
            os.environ["OMP_NUM_THREADS"] = "1"
            cores = n_cores or (os.cpu_count() or 2)
            pool = ProcessPoolExecutor(max_workers=cores)
            self.sampler = emcee.EnsembleSampler(
                nwalkers, ndim,
                _module_level_log_likelihood,
                args=(self.time, self.flux, self.flux_err, self.period,
                      self.t0, self.fitted_params, self.bounds),
                moves=moves, pool=pool
            )
        else:
            self.sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self._log_likelihood, moves=moves
            )

        try:
            self._run_sampling(p0, nsteps, progress_callback)
        finally:
            if pool is not None:
                pool.shutdown(wait=False)

        self._estimate_burnin(nsteps)
        thin = max(1, int(self.autocorr_time / 2)) if self.autocorr_time else thin_by
        self.flat_samples = self.sampler.get_chain(
            discard=self.burn_in, thin=thin, flat=True
        )
        self.results_summary = self._compute_summary()
        return self.results_summary

    def _run_sampling(self, p0, nsteps, progress_callback):
        for step, _ in enumerate(self.sampler.sample(p0, iterations=nsteps,
                                                     progress=False)):
            if progress_callback and step % 10 == 0:
                progress_callback(step, nsteps)
        if progress_callback:
            progress_callback(nsteps, nsteps)

    # =================================================================
    # Convergence diagnostics
    # =================================================================

    def _estimate_burnin(self, nsteps):
        """
        Use emcee's integrated autocorrelation time to pick the burn-in.
        Falls back to 30 % of the chain if the estimate is unreliable.
        """
        try:
            tau = self.sampler.get_autocorr_time(quiet=True)
            self.autocorr_time = np.mean(tau)
            self.burn_in = int(2.5 * self.autocorr_time)
            if self.burn_in >= nsteps:
                raise ValueError("burn-in >= nsteps")
            logger.info("Autocorrelation τ ≈ %.1f → burn-in = %d", self.autocorr_time, self.burn_in)
        except Exception:
            self.autocorr_time = None
            self.burn_in = int(nsteps * 0.30)
            logger.warning("Could not estimate τ; using 30%% burn-in (%d steps).", self.burn_in)

    def get_convergence_info(self) -> dict:
        """Return a dict with diagnostics for the report / UI."""
        chain = self.sampler.get_chain()
        acc = np.mean(self.sampler.acceptance_fraction)
        info = {
            "nwalkers": chain.shape[1],
            "nsteps": chain.shape[0],
            "burn_in": self.burn_in,
            "thin": max(1, int(self.autocorr_time / 2)) if self.autocorr_time else 10,
            "autocorr_time": float(self.autocorr_time) if self.autocorr_time else None,
            "mean_acceptance_fraction": float(acc),
            "n_effective_samples": self.flat_samples.shape[0] if self.flat_samples is not None else 0,
        }
        return info

    # =================================================================
    # Results
    # =================================================================

    def _compute_summary(self) -> dict:
        results = {}
        for i in range(self.flat_samples.shape[1]):
            q16, q50, q84 = np.percentile(self.flat_samples[:, i], [16, 50, 84])
            results[self.labels[i]] = (q50, q84 - q50, q50 - q16)
        return results

    def get_best_model_curve(self, num_points: int = 5000,
                             phase_folded: bool = False) -> tuple:
        if self.results_summary is None:
            raise ValueError("MCMC has not been run yet. Call run_mcmc() first.")

        n = len(self.fitted_params)
        final_params = [self.results_summary[self.labels[i]][0] for i in range(n)]

        full = {
            'rp': 0.1, 'inc': 90.0, 'a': 10.0, 't0': self.t0,
            'ecc': ECCENTRICITY, 'w': ARG_PERI,
            'u1': LIMB_DARKENING_COEFFS[0], 'u2': LIMB_DARKENING_COEFFS[1],
            'per': self.period
        }
        for name, val in zip(self.fitted_params, final_params):
            full[name] = val

        tp = batman.TransitParams()
        tp.limb_dark = LIMB_DARKENING_MODEL
        tp.rp = full['rp'];    tp.a = full['a'];     tp.inc = full['inc']
        tp.ecc = full['ecc'];  tp.w = full['w']
        tp.u = [full['u1'], full['u2']];  tp.per = full['per']

        if phase_folded:
            smooth_time = np.linspace(-full['per'] / 10, full['per'] / 10,
                                      num_points)
            tp.t0 = 0.0
        else:
            smooth_time = np.linspace(self.time.min(), self.time.max(),
                                      num_points)
            tp.t0 = full['t0']

        model = batman.TransitModel(tp, smooth_time)
        return smooth_time, model.light_curve(tp)
