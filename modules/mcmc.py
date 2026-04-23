"""
MCMC Transit Fitting Utilities
------------------------------
Provides the `TransitFitter` class to run Markov Chain Monte Carlo simulations
on folded lightcurves to extract physical planetary parameters.

Uses DEMove + DESnookerMove for robust exploration and autocorrelation-based
burn-in estimation.

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot
"""

import os
import logging
import numpy as np
import emcee
import batman
from scipy.optimize import minimize, differential_evolution
from concurrent.futures import ProcessPoolExecutor

from .constants import (
    MCMC_LABELS,
    LIMB_DARKENING_COEFFS, LIMB_DARKENING_MODEL,
    ECCENTRICITY, ARG_PERI
)

logger = logging.getLogger(__name__)


# =================================================================
# Soft limb-darkening prior
# =================================================================
# For F/G/K dwarfs observed in the TESS / Kepler bandpasses, Claret
# tables give (u1, u2) clustered around (0.35, 0.22) with a typical
# spread of ~0.15-0.20.  Without this prior, MCMC can slide into the
# classical LD ↔ impact-parameter degeneracy — using an unphysically
# low u1 and high u2 to mimic the soft ingress of a grazing transit.
# The prior widths are generous enough to allow genuine M-dwarf /
# A-star cases while penalising the degenerate corner.
#
# The dictionary below is the *default*; pass ``ld_prior={…}`` to
# ``TransitFitter`` to recentre on Claret coefficients appropriate for
# the actual host star (e.g. (0.30, 0.26) for an F6V like WASP-121).
DEFAULT_LD_PRIOR = {
    'u1': (0.35, 0.20),
    'u2': (0.22, 0.15),
}


# =================================================================
# Module-level (picklable) likelihood for multiprocessing
# =================================================================

def _module_level_log_prob(params, time, flux, flux_err, period, t0,
                           fitted_params, bounds, ld_prior):
    """Picklable log-probability (prior + likelihood)."""
    lp = _log_prior(params, fitted_params, bounds, ld_prior)
    if not np.isfinite(lp):
        return -np.inf
    ll = _log_likelihood_core(params, time, flux, flux_err, period, t0,
                              fitted_params)
    return lp + ll if np.isfinite(ll) else -np.inf


def _log_prior(params, fitted_params, bounds, ld_prior=None):
    """Smooth prior: uniform inside bounds with soft Gaussian walls,
    plus a weakly-informative Gaussian prior on u1/u2 *and* the strict
    physical (Kipping 2013) inequalities for quadratic limb darkening:

        u1 + u2 <= 1     (positive intensity at the stellar limb)
        u1 >= 0           (intensity decreases away from disk centre)
        u1 + 2 u2 >= 0    (no central brightening; monotonic profile)

    Without these constraints, the LD ↔ impact-parameter degeneracy
    lets the chain settle in the unphysical (u1+u2 > 1) corner where
    `batman` clamps the intensity internally — producing apparently
    converged but physically nonsense parameters (small Rp/Rs, low inc,
    small a/Rs all driven by the LD compensation).
    """
    if ld_prior is None:
        ld_prior = DEFAULT_LD_PRIOR
    lp = 0.0
    current = {'rp': 0.1, 'inc': 90.0, 'a': 10.0,
               'u1': LIMB_DARKENING_COEFFS[0], 'u2': LIMB_DARKENING_COEFFS[1]}
    for name, val, (lo, hi) in zip(fitted_params, params, bounds):
        if val < lo or val > hi:
            return -np.inf
        width = (hi - lo) * 0.02
        if val - lo < width:
            lp -= 0.5 * ((val - lo - width) / width) ** 2
        if hi - val < width:
            lp -= 0.5 * ((hi - val - width) / width) ** 2
        if name in ld_prior:
            mu, sigma = ld_prior[name]
            lp -= 0.5 * ((val - mu) / sigma) ** 2
        if name in current:
            current[name] = val

    # Geometric: planet must actually transit at this (a, inc, rp).
    b_impact = current['a'] * np.cos(np.radians(current['inc']))
    if b_impact > 1 + current['rp']:
        return -np.inf

    # Kipping (2013) physical envelope for quadratic LD.  These are
    # hard cuts (return -inf), not soft penalties, because the
    # unphysical region is genuinely meaningless rather than just
    # "less likely".
    u1, u2 = current['u1'], current['u2']
    if (u1 + u2) > 1.0:
        return -np.inf
    if u1 < 0.0:
        return -np.inf
    if (u1 + 2.0 * u2) < 0.0:
        return -np.inf

    return lp


def _log_likelihood_core(params, time, flux, flux_err, period, t0,
                         fitted_params):
    """Core chi-squared likelihood shared by all entry points."""
    full = {
        'rp': 0.1, 'inc': 90.0, 'a': 10.0, 't0': t0,
        'ecc': ECCENTRICITY, 'w': ARG_PERI,
        'u1': LIMB_DARKENING_COEFFS[0], 'u2': LIMB_DARKENING_COEFFS[1],
        'per': period
    }
    for name, val in zip(fitted_params, params):
        full[name] = val

    try:
        tp = batman.TransitParams()
        tp.limb_dark = LIMB_DARKENING_MODEL
        tp.t0 = full['t0'];  tp.per = full['per'];  tp.rp = full['rp']
        tp.a = full['a'];    tp.inc = full['inc'];   tp.ecc = full['ecc']
        tp.w = full['w'];    tp.u = [full['u1'], full['u2']]
        model_flux = batman.TransitModel(tp, time).light_curve(tp)
    except Exception:
        return -np.inf

    if not np.all(np.isfinite(model_flux)):
        return -np.inf

    chi2 = np.sum(((flux - model_flux) / flux_err) ** 2)
    return -0.5 * chi2 if np.isfinite(chi2) else -np.inf


# =================================================================

class TransitFitter:
    """
    MCMC-based transit fitter with robust optimisation, tight walker
    initialisation, and smooth priors for healthy acceptance rates
    (target 25-50 %).
    """

    def __init__(self, time, flux, flux_err, period, t0=0.0,
                 fitted_params=None, custom_bounds=None, custom_x0=None,
                 custom_labels=None, auto_bounds=True, ld_prior=None):
        # *time* and *t0* must use the same unit system (e.g. BTJD + BTJD).
        self.time = np.asarray(time, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        self.flux_err = np.asarray(flux_err, dtype=np.float64)
        self.period = float(period)
        self.t0 = float(t0)

        # Soft Gaussian prior on (u1, u2).  Defaults are TESS Claret
        # values for a G dwarf — pass ``ld_prior={'u1': (mu, sigma),
        # 'u2': (mu, sigma)}`` to recentre on the appropriate Claret
        # coefficients for the actual host star.
        self.ld_prior = dict(ld_prior) if ld_prior is not None \
            else dict(DEFAULT_LD_PRIOR)

        self.fitted_params = fitted_params or ['rp', 'inc', 'a', 't0']
        n = len(self.fitted_params)

        # Strong recommendation: t0 should always be a free parameter.
        # BLS picks the deepest *bin* of its grid, not the true mid-
        # transit time, so its epoch is routinely off by several
        # minutes (up to half a TESS long-cadence frame, ~15 min).
        # Holding t0 fixed at the BLS epoch turns that small offset
        # into a large degeneracy chain — the fit smears the model
        # via stronger LD (u1+u2 → 1), drops a/Rs and inc, and
        # shrinks Rp/Rs to compensate.  Only suppress this warning if
        # the caller has truly externally-validated t0 (e.g. from a
        # high-cadence ground-based reference transit).
        if 't0' not in self.fitted_params:
            logger.warning(
                "`t0` is not in fitted_params. The model will be "
                "locked to the input epoch (BLS / user-supplied) and "
                "any small mid-transit-time error will leak into "
                "(Rp/Rs, a/Rs, inc, u1, u2) via the LD ↔ time-shift "
                "degeneracy.  Strongly recommended: add 't0' to "
                "fitted_params unless you have a high-cadence "
                "reference epoch."
            )

        # Filled by `_estimate_bounds`; consumed by `optimize_initial_guess`
        # to build self-consistent alternative starts.
        self._transit_geometry = {}
        self._map_cache = None

        # Auto-estimate whenever *either* bounds or x0 is missing, so the
        # user can override just one of them.  There is no longer a generic
        # ``MCMC_BOUNDS`` / ``MCMC_X0`` fallback: silently fitting an
        # unobserved planet with the wrong parameter ordering is far worse
        # than a clear error message.
        auto_b, auto_x0 = None, None
        if auto_bounds and (custom_bounds is None or custom_x0 is None):
            auto_b, auto_x0 = self._estimate_bounds()

        if custom_bounds is not None:
            self.bounds = custom_bounds
        elif auto_b is not None:
            self.bounds = auto_b
        else:
            raise ValueError(
                "TransitFitter: no bounds available.  Either pass "
                "`custom_bounds` explicitly or set `auto_bounds=True` "
                "with enough finite light-curve points (>=20)."
            )
        if custom_x0 is not None:
            self.x0 = custom_x0
        elif auto_x0 is not None:
            self.x0 = auto_x0
        else:
            raise ValueError(
                "TransitFitter: no initial guess (x0) available.  Either "
                "pass `custom_x0` explicitly or rely on `auto_bounds`."
            )

        # Final sanity clamp: guarantee lo < hi and x0 strictly inside bounds.
        # Prevents "An upper bound is less than the corresponding lower bound"
        # crashes from SciPy's differential_evolution / L-BFGS-B.
        self.bounds, self.x0 = self._sanitize_bounds(self.bounds, self.x0)

        # Labels default to the user-provided list, then to MCMC_LABELS,
        # and finally fall back to the fitted-parameter names so a
        # 6-D / 8-D run never crashes `_compute_summary` with an
        # IndexError on `self.labels[i]`.
        if custom_labels is not None:
            self.labels = list(custom_labels)[:n]
        else:
            self.labels = list(MCMC_LABELS[:n])
        while len(self.labels) < n:
            self.labels.append(self.fitted_params[len(self.labels)])

        self.best_fit_params = None
        self.sampler = None
        self.flat_samples = None
        self.results_summary = None
        self.autocorr_time = None
        self.burn_in = None

    # =================================================================
    # Physics-based bounds estimation
    # =================================================================

    def _estimate_bounds(self):
        """
        Derive bounds and initial guesses from transit geometry alone.

        Pipeline
        --------
        1.  Phase-fold the lightcurve around ``self.t0`` so the same code
            handles both raw and already-folded inputs.
        2.  Robust baseline and per-point noise σ from the out-of-transit
            wings (median + MAD).
        3.  Phase-binned median smoothing to beat photon noise and find
            the transit center, depth, and duration markers.
        4.  Depth δ from the median of the deepest bins, floored at 3σ.
        5.  T14 (total duration) from half-depth crossings, T23 (flat
            bottom) from 0.9·δ crossings.
        6.  Seager & Mallén-Ornelas (2003), eqs. 7-8, to invert (δ, T14,
            T23, P) into (k = Rp/Rs, impact parameter b, a/Rs, inc).
        7.  SNR-adaptive half-widths: tight bounds for clean transits,
            generous ones when the signal is marginal.

        Returns
        -------
        (bounds, x0) : lists in the order of ``self.fitted_params``.
        """
        period = self.period
        t0_ref = self.t0

        # ---- 1. Clean the input and fold into [-P/2, +P/2] -----------
        finite = (np.isfinite(self.time) & np.isfinite(self.flux)
                  & np.isfinite(self.flux_err) & (self.flux_err > 0))
        time = self.time[finite]
        flux = self.flux[finite]
        flux_err = self.flux_err[finite]
        if time.size < 20:
            raise ValueError("Too few finite points to estimate bounds.")

        phase = ((time - t0_ref + 0.5 * period) % period) - 0.5 * period
        order = np.argsort(phase)
        ph = phase[order]
        fl = flux[order]

        # ---- 2. Robust out-of-transit baseline + noise (MAD) ---------
        wing = np.abs(ph) > 0.15 * period
        if wing.sum() < 20:
            wing = np.abs(ph) > 0.05 * period
        if wing.sum() < 5:
            wing = np.ones_like(ph, dtype=bool)

        baseline = float(np.median(fl[wing]))
        mad = float(np.median(np.abs(fl[wing] - baseline)))
        sigma = 1.4826 * mad if mad > 0 else float(np.median(flux_err))
        sigma = max(sigma, 1e-6)

        # ---- 3. Phase-binned median smoothing ------------------------
        n_bins = int(np.clip(ph.size / 25, 60, 400))
        edges = np.linspace(ph.min(), ph.max(), n_bins + 1)
        idx = np.clip(np.digitize(ph, edges) - 1, 0, n_bins - 1)
        bin_ph = 0.5 * (edges[:-1] + edges[1:])
        bin_fl = np.full(n_bins, np.nan)
        for i in range(n_bins):
            mask_i = (idx == i)
            if mask_i.sum() >= 2:
                bin_fl[i] = np.median(fl[mask_i])
        ok = np.isfinite(bin_fl)
        if ok.sum() < 10:
            raise ValueError("Phase-binning failed (too few populated bins).")
        bph = bin_ph[ok]
        bfl = bin_fl[ok]

        # ---- 4. Transit center & depth (robust) ----------------------
        # Center = phase of the deepest smoothed bin.
        t_center = float(bph[np.argmin(bfl)])

        # Depth from the 5 deepest bins (or top 5 % if plenty of bins)
        k_deep = max(3, min(5, int(round(0.05 * bfl.size))))
        f_min = float(np.median(np.sort(bfl)[:k_deep]))
        depth = max(baseline - f_min, 3.0 * sigma)

        # ---- 5. T14 and T23 from level crossings ---------------------
        # Restrict the search to a window centered on the transit
        # (±30 % of the period) so a secondary eclipse or systematics
        # at the wings can't inflate the duration estimate.
        win = np.abs(bph - t_center) < 0.30 * period
        bph_w, bfl_w = bph[win], bfl[win]

        # Use lenient thresholds: quadratic limb darkening rounds the
        # ingress/egress so that the classical "half-depth" width sits
        # well inside the true geometric T14.  0.25·δ is closer to first
        # contact; 0.75·δ is closer to the true flat bottom.
        t14_level = baseline - 0.25 * depth
        t23_level = baseline - 0.75 * depth

        below_t14 = bfl_w <= t14_level
        t14 = (np.ptp(bph_w[below_t14]) if below_t14.sum() >= 2
               else period * 0.05)
        # Small residual LD correction (measured width still ~10-15 %
        # short of the true T14 at this threshold for typical coeffs).
        t14 *= 1.10
        t14 = float(np.clip(t14, period * 5e-4, period * 0.3))

        below_t23 = bfl_w <= t23_level
        t23 = (np.ptp(bph_w[below_t23]) if below_t23.sum() >= 2 else 0.0)
        t23 = float(np.clip(t23, 0.0, 0.95 * t14))

        # ---- 6. Seager & Mallén-Ornelas (2003) inversion -------------
        k = float(np.clip(np.sqrt(depth / max(baseline, 1e-6)), 0.005, 0.35))

        sT14 = np.sin(np.pi * t14 / period)
        sT23 = np.sin(np.pi * t23 / period)
        if t23 > 0 and sT14 > sT23 > 0:
            r = (sT23 / sT14) ** 2
            num = (1.0 - k) ** 2 - r * (1.0 + k) ** 2
            den = 1.0 - r
            b2 = num / den if den > 1e-6 else 0.2 ** 2
            b_est = float(np.sqrt(np.clip(b2, 0.0, 0.95 ** 2)))
        else:
            # Flat-bottom not resolved ⇒ assume a grazing-free central
            # transit with a conservative mid-range impact parameter.
            b_est = 0.3

        a_sq = (((1.0 + k) ** 2 - b_est ** 2) / max(sT14 ** 2, 1e-12)
                + b_est ** 2)
        a_est = float(np.clip(np.sqrt(max(a_sq, (1.0 + k) ** 2)),
                              2.0, 500.0))

        inc_est = float(np.degrees(
            np.arccos(np.clip(b_est / a_est, 0.0, 0.999))))

        # ---- 7. SNR-adaptive widths ----------------------------------
        in_transit = np.abs(ph - t_center) < 0.5 * t14
        n_in = max(int(in_transit.sum()), 1)
        snr = (depth / sigma) * np.sqrt(n_in)
        # Width multiplier: ~0.2 for a 30σ transit, ~1.5 for a 3σ one.
        w = float(np.clip(3.0 / np.sqrt(max(snr, 1.0)), 0.15, 1.5))

        # ---- 7b. Geometry-confidence regime --------------------------
        # The S&MO inversion produces (k, b, a/Rs, inc) from (depth, T14,
        # T23).  When T23 is well-resolved and b_est is small, the
        # inversion is locked-down: we should *tighten* the bounds on a
        # and inc so the optimiser cannot wander into the spurious
        # grazing mode that LD ↔ b degeneracy makes nearly equally
        # likely on noisy data.  Conversely, when T23 is unresolved or
        # b_est is large, the inversion is loose and we open the
        # bounds.  Three regimes:
        #
        #   high-confidence central : T23 resolved, b_est < 0.30
        #   low-confidence / grazing : T23 unresolved or b_est > 0.65
        #                              or k < 0.02 (very shallow)
        #   standard                 : everything else
        if t23 > 0 and b_est < 0.30:
            confidence = 'high'
            a_down = max(0.18, 0.35 * w)
            a_up   = max(0.18, 0.40 * w)
            inc_span = max(3.5, 6.0 * w)
            rp_down = max(0.20, 0.4 * w)
            rp_up   = max(0.25, 0.5 * w)
        elif (t23 <= 0.0) or (b_est > 0.65) or (k < 0.02):
            confidence = 'low'
            a_down = max(0.70, 1.0 * w)
            a_up   = max(1.00, 1.5 * w)
            inc_span = max(25.0, 35.0 * w)
            rp_down = max(0.55, 0.8 * w)
            rp_up   = max(1.00, 1.2 * w)
        else:
            confidence = 'standard'
            a_down = max(0.40, 0.7 * w)
            a_up   = max(0.50, 0.9 * w)
            inc_span = max(10.0, 15.0 * w)
            rp_down = max(0.35, 0.6 * w)
            rp_up   = max(0.45, 0.8 * w)
        grazing = (confidence == 'low')

        # ---- 8. Build bounds -----------------------------------------
        # Minimum widths absorb residual limb-darkening bias on T14/T23
        # that a pure SNR scaling cannot capture.
        rp_lo = max(k * (1.0 - rp_down), 0.002)
        rp_hi = min(k * (1.0 + rp_up), 0.5)

        # a/Rs: tightness scaled by confidence regime.
        a_lo = max(a_est * (1.0 - a_down), 1.01 * (1.0 + rp_hi))
        a_hi = min(a_est * (1.0 + a_up), 500.0)

        # inc: geometric lower bound (so the transit still occurs at the
        # widest plausible radius ratio), with a confidence-scaled
        # margin on top.
        inc_geom = np.degrees(np.arccos(min(0.999, (1.0 + rp_hi) / a_lo)))
        inc_lo = float(max(inc_geom, inc_est - inc_span))
        inc_hi = 90.0
        if inc_lo >= inc_hi - 1e-3:
            inc_lo = max(inc_geom, 60.0)

        # ``t_center`` is a *phase offset* from ``t0_ref`` (i.e. lives
        # in [-P/2, +P/2]); the batman model, however, expects ``t0``
        # in the *same absolute time system* as ``self.time``.  Adding
        # ``t0_ref`` back puts the bound and x0 in absolute time so a
        # fitted ``t0`` actually corresponds to the right BTJD epoch.
        # (Original bug: bounds were generated in phase units while the
        # likelihood treated ``t0`` as absolute, so the chain happily
        # converged to a value 1500 days off the truth.)
        t0_abs = t0_ref + t_center
        t0_margin = max(0.6 * t14, 0.005 * period)
        t0_lo = t0_abs - t0_margin
        t0_hi = t0_abs + t0_margin

        # Period: tight — BLS gives it to a few 1e-4 at worst
        per_margin = max(0.002 * period, 5 * np.median(np.diff(np.sort(time))))
        per_lo = period - per_margin
        per_hi = period + per_margin

        param_map = {
            'rp':  ((rp_lo, rp_hi), k),
            'inc': ((inc_lo, inc_hi), inc_est),
            'a':   ((a_lo, a_hi), a_est),
            't0':  ((t0_lo, t0_hi), t0_abs),
            'per': ((per_lo, per_hi), period),
            'ecc': ((0.0, 0.7), ECCENTRICITY),
            'w':   ((0.0, 360.0), ARG_PERI),
            # Quadratic-LD physical envelope: u1 ∈ [0, 1] and
            # u1 + u2 ∈ [0, 1], u1 + 2u2 ≥ 0 keep the surface intensity
            # positive and monotonic.  The bounds here are wide enough
            # to encompass everything from late-M (high u1, low u2) to
            # early-A stars (low u1, modest u2).  The walls are *not*
            # what should constrain LD; the soft Gaussian prior in
            # `_log_prior` (centred on TESS Claret coefficients for a
            # G dwarf) is.  These previously-tight bounds were the
            # source of "u2 stuck at 0.4" reports — widening them lets
            # the data speak when it really wants stronger LD, while
            # the prior keeps unphysical regions out of reach.
            'u1':  ((0.0, 0.9), 0.35),
            'u2':  ((-0.2, 0.7), 0.22),
        }

        bounds, x0 = [], []
        for name in self.fitted_params:
            b, g = param_map.get(name, ((0.0, 1.0), 0.5))
            bounds.append(b)
            x0.append(float(np.clip(g, b[0] + 1e-10, b[1] - 1e-10)))

        # Stash the transit geometry so optimize_initial_guess can build
        # physically self-consistent alternative starts (different impact
        # parameters at the *same* T14, k, P) and break the LD/b
        # degeneracy when the data are LD-dominated.
        self._transit_geometry = {
            't14': t14, 't23': t23, 'k': k, 't_center': t_center,
            'period': period,
            'b_est': b_est, 'a_est': a_est, 'inc_est': inc_est,
            'depth': depth, 'baseline': baseline, 'sigma': sigma, 'snr': snr,
            'confidence': confidence,
        }

        logger.info(
            "Auto-bounds (SNR=%.1f, w=%.2f, confidence=%s):\n"
            "  depth=%.4g  σ=%.4g  T14=%.4f d  T23=%.4f d\n"
            "  k=Rp/Rs=%.4f [%.4f, %.4f]   b=%.3f\n"
            "  a/Rs=%.2f [%.2f, %.2f]     inc=%.2f° [%.2f°, %.2f°]\n"
            "  t0 offset=%.5f d  (±%.5f d from reference)",
            snr, w, confidence,
            depth, sigma, t14, t23,
            k, rp_lo, rp_hi, b_est,
            a_est, a_lo, a_hi, inc_est, inc_lo, inc_hi,
            t_center, t0_margin,
        )
        self._print_bounds_and_x0(bounds, x0)
        return bounds, x0

    # -----------------------------------------------------------------

    def _print_bounds_and_x0(self, bounds, x0):
        """Print the auto-generated bounds and x0 in a copy-pasteable
        Python-literal format so the user can lift them straight into
        a notebook, tighten ranges, or tweak starting values.
        """
        def fmt(v):
            """Readable 6-sig-fig number, never in scientific notation."""
            return f"{v:.6g}"

        name_col = max(3, max(len(n) for n in self.fitted_params))
        lo_col = max(len(fmt(b[0])) for b in bounds)
        hi_col = max(len(fmt(b[1])) for b in bounds)
        x0_col = max(len(fmt(x)) for x in x0)

        lines = [
            f"\n──── Auto-generated MCMC bounds & x0 "
            f"({len(self.fitted_params)}-D) ────────────────────",
            "custom_bounds = [",
        ]
        for name, (lo, hi) in zip(self.fitted_params, bounds):
            lines.append(
                f"    ({fmt(lo):>{lo_col}}, {fmt(hi):>{hi_col}}),"
                f"   # {name:<{name_col}}"
            )
        lines.append("]")
        lines.append("custom_x0 = [")
        for name, val in zip(self.fitted_params, x0):
            lines.append(
                f"    {fmt(val):>{x0_col}},"
                f"{' ' * (lo_col + hi_col - x0_col + 2)}"
                f"   # {name:<{name_col}}"
            )
        lines.append("]")
        lines.append("─" * 68)
        # Use a bare print so this block always surfaces, independent
        # of the user's logging level — the user explicitly asked for
        # it on every auto-bounds call.
        print("\n".join(lines))

    @staticmethod
    def _sanitize_bounds(bounds, x0):
        """Guarantee each (lo, hi) pair is non-degenerate and that every
        element of ``x0`` lies strictly inside its interval."""
        clean_b, clean_x = [], []
        for (lo, hi), val in zip(bounds, x0):
            lo, hi = float(lo), float(hi)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                # Fall back to a symmetric interval around the guess.
                span = max(abs(val) * 0.1, 1e-3)
                lo, hi = val - span, val + span
            clean_b.append((lo, hi))
            clean_x.append(float(np.clip(val, lo + 1e-10, hi - 1e-10)))
        return clean_b, clean_x

    # =================================================================
    # Internal helpers
    # =================================================================

    def _make_transit_model(self, params, t_array):
        full = {
            'rp': 0.1, 'inc': 90.0, 'a': 10.0, 't0': self.t0,
            'ecc': ECCENTRICITY, 'w': ARG_PERI,
            'u1': LIMB_DARKENING_COEFFS[0], 'u2': LIMB_DARKENING_COEFFS[1],
            'per': self.period
        }
        for name, val in zip(self.fitted_params, params):
            full[name] = val
        tp = batman.TransitParams()
        tp.limb_dark = LIMB_DARKENING_MODEL
        tp.t0 = full['t0'];  tp.per = full['per'];  tp.rp = full['rp']
        tp.a = full['a'];    tp.inc = full['inc'];   tp.ecc = full['ecc']
        tp.w = full['w'];    tp.u = [full['u1'], full['u2']]
        return batman.TransitModel(tp, t_array).light_curve(tp)

    def _log_prob(self, params):
        """Instance-level log-probability (prior + likelihood)."""
        lp = _log_prior(params, self.fitted_params, self.bounds, self.ld_prior)
        if not np.isfinite(lp):
            return -np.inf
        ll = _log_likelihood_core(
            params, self.time, self.flux, self.flux_err,
            self.period, self.t0, self.fitted_params)
        return lp + ll if np.isfinite(ll) else -np.inf

    def _neg_loglike(self, params):
        """Negative log-LIKELIHOOD (not posterior) for the optimiser.

        Previously this returned ``-self._log_prob(params)``, i.e. the
        negative log-*posterior*, including the soft Gaussian prior on
        (u1, u2).  That meant DE / L-BFGS-B were tugged toward whatever
        ``ld_prior`` was set to (the G-dwarf default by default), biasing
        the MAP for any host that is not roughly G-type.  We now use the
        pure likelihood plus the *hard* feasibility cuts (bounds, Kipping
        envelope, transit-geometry cut), so the optimiser respects
        physical impossibility but is no longer pulled by the LD prior.
        """
        # Hard bound check (prevents wasted batman calls outside box).
        for v, (lo, hi) in zip(params, self.bounds):
            if not (lo <= v <= hi):
                return 1e25

        # Hard physical cuts borrowed from `_log_prior`: transit must
        # actually occur, Kipping envelope must hold.
        full = {'rp': 0.1, 'inc': 90.0, 'a': 10.0,
                'u1': LIMB_DARKENING_COEFFS[0],
                'u2': LIMB_DARKENING_COEFFS[1]}
        for name, val in zip(self.fitted_params, params):
            if name in full:
                full[name] = val
        b_impact = full['a'] * np.cos(np.radians(full['inc']))
        if b_impact > 1 + full['rp']:
            return 1e25
        u1, u2 = full['u1'], full['u2']
        if (u1 + u2) > 1.0 or u1 < 0.0 or (u1 + 2.0 * u2) < 0.0:
            return 1e25

        ll = _log_likelihood_core(
            params, self.time, self.flux, self.flux_err,
            self.period, self.t0, self.fitted_params)
        if not np.isfinite(ll):
            return 1e25
        return -ll

    # =================================================================
    # Pre-optimisation (multi-stage for robust convergence)
    # =================================================================

    def _x0_alternatives(self):
        """
        Build a short list of physically self-consistent alternative
        starting points to escape the limb-darkening / impact-parameter
        degeneracy.

        For a fixed (k, T14, P), Seager & Mallén-Ornelas (2003, eq. 8)
        gives a one-parameter family of (a/Rs, inc) pairs indexed by
        the impact parameter b.  All of them produce the same total
        duration; only the *shape* (ingress slope, depth profile)
        distinguishes them — and that shape is what LD also affects.
        Trying multiple b values turns this hidden ambiguity into an
        explicit set of optimiser starts.
        """
        base = np.array(self.x0, dtype=float)
        candidates = [("x0", base.copy())]

        geom = self._transit_geometry
        if not geom or 'a' not in self.fitted_params or 'inc' not in self.fitted_params:
            return candidates

        k = geom['k']
        t14 = geom['t14']
        period = geom['period']
        sT14 = np.sin(np.pi * t14 / period)
        if sT14 <= 0:
            return candidates

        i_a = self.fitted_params.index('a')
        i_inc = self.fitted_params.index('inc')
        a_lo, a_hi = self.bounds[i_a]
        inc_lo, inc_hi = self.bounds[i_inc]

        # Span the b-axis: central, mildly grazing, near-grazing.  Skip
        # the grazing seeds whenever the geometry inversion has any kind
        # of confidence in a central transit — otherwise we hand the
        # optimiser the very local minimum we are trying to avoid
        # (WASP-121 b symptom: a ``b≈0.65`` polish can win on noisy data
        # even when the truth is ``b≈0.06``).  Only the explicitly
        # ``low``-confidence branch keeps the grazing seeds.
        confidence = self._transit_geometry.get('confidence', 'standard')
        if confidence == 'high':
            seed_bs = [("b≈0.05", 0.05), ("b≈0.20", 0.20)]
        elif confidence == 'low':
            seed_bs = [("b≈0.05", 0.05), ("b≈0.35", 0.35),
                       ("b≈0.65", 0.65), ("b≈0.85", 0.85)]
        else:
            seed_bs = [("b≈0.05", 0.05), ("b≈0.20", 0.20),
                       ("b≈0.35", 0.35)]
        for tag, b_try in seed_bs:
            b_try = min(b_try, 0.9 * (1.0 + k))
            a_try = float(np.sqrt(((1.0 + k) ** 2 - b_try ** 2) / sT14 ** 2
                                  + b_try ** 2))
            if not np.isfinite(a_try) or a_try <= 0:
                continue
            inc_try = float(np.degrees(np.arccos(
                np.clip(b_try / a_try, 0.0, 0.999))))

            cand = base.copy()
            cand[i_a] = float(np.clip(a_try, a_lo + 1e-6, a_hi - 1e-6))
            cand[i_inc] = float(np.clip(inc_try, inc_lo + 1e-6, inc_hi - 1e-6))
            candidates.append((tag, cand))

        return candidates

    def optimize_initial_guess(self):
        """
        Multi-stage optimisation to reliably find the *global* minimum:

        1. Polish the user-supplied ``self.x0`` *first* with L-BFGS-B
           and put it on the candidate list as a privileged start.  A
           carefully chosen physical guess should never be silently
           demoted because two random DE seeds happened to score better
           on a noisy posterior.
        2. L-BFGS-B from each physics-based alternative in
           ``_x0_alternatives()`` (different impact parameters at the
           same total duration) to break the LD ↔ impact-parameter
           degeneracy on near-central transits.
        3. Differential Evolution for a coarse global search, then
           L-BFGS-B polish from each DE result.  DE is run *after* the
           physical seeds rather than before so that we have a
           well-defined "user prior" baseline to compare against.

        Returns the candidate with the highest log-likelihood.
        """
        candidates = []

        # Stage 1+2: physics-based seeds (user x0 is the first entry of
        # ``_x0_alternatives``).  Polish each with L-BFGS-B.
        for tag, start in self._x0_alternatives():
            try:
                res = minimize(self._neg_loglike, np.asarray(start, float),
                               bounds=self.bounds, method="L-BFGS-B",
                               options={"maxiter": 5000, "ftol": 1e-14})
                candidates.append((tag + "-polish", res.x, -res.fun))
            except Exception as e:
                logger.warning("%s failed: %s", tag, e)

        # Stage 3: multi-seed DE.  A single DE run can land in the
        # grazing local minimum on red-noisy real data where both modes
        # have similar χ².  Running two seeds reliably brackets *both*
        # so that the L-BFGS-B polish step lets us keep the better one.
        for seed in (42, 1234):
            try:
                de_result = differential_evolution(
                    self._neg_loglike, bounds=self.bounds,
                    seed=seed, maxiter=600, tol=1e-7,
                    polish=False, mutation=(0.5, 1.5), recombination=0.9,
                    popsize=20,
                )
                candidates.append((f"DE-s{seed}", de_result.x, -de_result.fun))
                # Polish the DE result with L-BFGS-B too.
                try:
                    res = minimize(self._neg_loglike, de_result.x,
                                   bounds=self.bounds, method="L-BFGS-B",
                                   options={"maxiter": 5000, "ftol": 1e-14})
                    candidates.append((f"DE-s{seed}-polish", res.x, -res.fun))
                except Exception as e:
                    logger.warning("DE-s%s polish failed: %s", seed, e)
            except Exception as e:
                logger.warning("DE seed=%s failed: %s", seed, e)

        if not candidates:
            logger.warning("All optimisers failed; using x0.")
            self.best_fit_params = np.array(self.x0)
            return self.best_fit_params

        # Sort by log-likelihood (descending) for transparent diagnostics.
        candidates.sort(key=lambda c: c[2], reverse=True)
        best_tag, best_x, best_ll = candidates[0]
        runners = ", ".join(f"{t}={ll:.2f}" for t, _, ll in candidates[:5])
        logger.info("Best optimiser: %s  (logL = %.2f) | top: %s",
                    best_tag, best_ll, runners)
        self.best_fit_params = np.array(best_x)

        # Nudge LD coefficients slightly off the Kipping (u1 + u2 = 1)
        # boundary if the optimiser landed there.  Otherwise every
        # walker drawn around best_fit_params with random perturbations
        # has a 50 % chance of falling on the unphysical side, and
        # `_init_walkers` will spend its retry budget shrinking back
        # to the boundary again.
        if 'u1' in self.fitted_params and 'u2' in self.fitted_params:
            i_u1 = self.fitted_params.index('u1')
            i_u2 = self.fitted_params.index('u2')
            u_sum = self.best_fit_params[i_u1] + self.best_fit_params[i_u2]
            if u_sum > 0.99:
                # Pull each LD coefficient symmetrically inward by the
                # smallest amount needed to give the walkers ~3% slack.
                shrink = (0.97 / u_sum) if u_sum > 0 else 1.0
                self.best_fit_params[i_u1] *= shrink
                self.best_fit_params[i_u2] *= shrink
                logger.info(
                    "Pulled LD coefficients off Kipping boundary: "
                    "u1+u2 %.3f → %.3f", u_sum,
                    self.best_fit_params[i_u1] + self.best_fit_params[i_u2],
                )

        return self.best_fit_params

    # =================================================================
    # Walker initialisation
    # =================================================================

    def _init_walkers(self, nwalkers, ndim):
        """
        Scatter walkers in a tiny ball (0.1 % of range) around the
        optimised best-fit.  This ensures all walkers start in the
        high-likelihood region, giving healthy initial acceptance.

        Walkers that violate the prior (e.g. land in the Kipping
        unphysical-LD region with `u1 + u2 > 1`) are re-drawn
        individually with a progressively *shrinking* perturbation
        scale, so they are pulled back toward `best_fit_params` and
        away from the constraint boundary that produced the rejection.
        """
        scales = np.array([(hi - lo) * 1e-3 for lo, hi in self.bounds])
        p0 = self.best_fit_params + scales * np.random.randn(nwalkers, ndim)
        for i, (lo, hi) in enumerate(self.bounds):
            p0[:, i] = np.clip(p0[:, i], lo + 1e-10, hi - 1e-10)

        valid = np.array([np.isfinite(self._log_prob(p)) for p in p0])
        if valid.all():
            return p0

        n_bad_initial = int((~valid).sum())
        logger.warning("Re-drawing %d/%d invalid walkers",
                       n_bad_initial, nwalkers)

        for attempt in range(50):
            bad = ~valid
            n_bad = int(bad.sum())
            if n_bad == 0:
                break

            # Geometric shrink toward best_fit_params each round so we
            # eventually escape any tight prior-constraint corner.
            shrink = 0.5 * (0.85 ** attempt)
            new_draws = (self.best_fit_params
                         + scales * shrink
                         * np.random.randn(n_bad, ndim))
            for i, (lo, hi) in enumerate(self.bounds):
                new_draws[:, i] = np.clip(new_draws[:, i],
                                          lo + 1e-10, hi - 1e-10)
            p0[bad] = new_draws
            valid = np.array([np.isfinite(self._log_prob(p)) for p in p0])

        if not valid.all():
            n_left = int((~valid).sum())
            logger.warning(
                "Could not validate %d walkers after 50 attempts; "
                "falling back to best_fit_params for those slots.",
                n_left,
            )
            p0[~valid] = self.best_fit_params
            for i, (lo, hi) in enumerate(self.bounds):
                p0[:, i] = np.clip(p0[:, i], lo + 1e-10, hi - 1e-10)

        return p0

    # =================================================================
    # Main MCMC routine
    # =================================================================

    def run_mcmc(self, nwalkers=32, nsteps=8000,
                 progress_callback=None, use_multiprocessing=False,
                 n_cores=None, thin_by=1):
        """
        Run the MCMC sampler.

        Defaults are tuned for 4-parameter transit fits:
        * 32 walkers (8× ndim) — enough for ensemble moves, not wasteful
        * 8000 steps — typically yields >50× autocorrelation lengths
        """
        ndim = len(self.fitted_params)
        logger.info("Pre-optimising (%d-D)…", ndim)
        self.optimize_initial_guess()
        p0 = self._init_walkers(nwalkers, ndim)
        logger.info("Starting emcee (%d walkers × %d steps)…", nwalkers, nsteps)

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
                _module_level_log_prob,
                args=(self.time, self.flux, self.flux_err, self.period,
                      self.t0, self.fitted_params, self.bounds,
                      self.ld_prior),
                moves=moves, pool=pool
            )
        else:
            self.sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self._log_prob, moves=moves
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
        self._map_cache = None  # invalidate any prior MAP cache
        self.results_summary = self._compute_summary()
        return self.results_summary

    def _run_sampling(self, p0, nsteps, progress_callback):
        for step, _ in enumerate(self.sampler.sample(
                p0, iterations=nsteps, progress=False)):
            if progress_callback and step % 10 == 0:
                progress_callback(step, nsteps)
        if progress_callback:
            progress_callback(nsteps, nsteps)

    # =================================================================
    # Convergence diagnostics
    # =================================================================

    def _estimate_burnin(self, nsteps):
        """Use integrated autocorrelation time for burn-in."""
        try:
            tau = self.sampler.get_autocorr_time(quiet=True)
            self.autocorr_time = float(np.mean(tau))
            self.burn_in = int(2.5 * self.autocorr_time)
            if self.burn_in >= nsteps:
                raise ValueError("burn-in >= nsteps")
            logger.info("τ ≈ %.1f → burn-in = %d", self.autocorr_time,
                        self.burn_in)
        except Exception:
            self.autocorr_time = None
            self.burn_in = int(nsteps * 0.30)
            logger.warning("Could not estimate τ; using 30%% burn-in (%d).",
                           self.burn_in)

    def get_convergence_info(self):
        """Return a dict with diagnostics for the report / UI."""
        chain = self.sampler.get_chain()
        acc = np.mean(self.sampler.acceptance_fraction)
        return {
            "nwalkers": chain.shape[1],
            "nsteps": chain.shape[0],
            "burn_in": self.burn_in,
            "thin": max(1, int(self.autocorr_time / 2)) if self.autocorr_time else 10,
            "autocorr_time": float(self.autocorr_time) if self.autocorr_time else None,
            "mean_acceptance_fraction": float(acc),
            "n_effective_samples": (self.flat_samples.shape[0]
                                    if self.flat_samples is not None else 0),
        }

    # =================================================================
    # Results
    # =================================================================

    def _compute_summary(self):
        results = {}
        for i in range(self.flat_samples.shape[1]):
            q16, q50, q84 = np.percentile(
                self.flat_samples[:, i], [16, 50, 84])
            results[self.labels[i]] = (q50, q84 - q50, q50 - q16)
        return results

    def get_fit_diagnostics(self):
        """Return a dict comparing the per-axis posterior median, the
        chain MAP and the residual RMS of each.  Useful in the notebook
        to confirm that the MAP-overlaid model is genuinely the better
        fit when the marginals are skewed."""
        if self.flat_samples is None:
            raise ValueError("MCMC has not been run. Call run_mcmc() first.")

        median_params = np.array([self.results_summary[self.labels[i]][0]
                                  for i in range(len(self.fitted_params))])
        map_params, map_logp = self.get_map_params()

        def _rms(params):
            model = self._model_at(params, self.time, t0_override=None)
            if model is None:
                return float('nan')
            return float(np.sqrt(np.mean((self.flux - model) ** 2)))

        return {
            "median": dict(zip(self.fitted_params, median_params.tolist())),
            "map":    dict(zip(self.fitted_params, map_params.tolist())),
            "logp_at_map":     map_logp,
            "logp_at_median":  float(self._log_prob(median_params)),
            "rms_residual_median": _rms(median_params),
            "rms_residual_map":    _rms(map_params),
        }

    # -----------------------------------------------------------------
    # Posterior model utilities
    # -----------------------------------------------------------------

    def _params_to_full_dict(self, params):
        """Map a fitted-parameter vector into the full set used by batman,
        filling unfitted entries with the appropriate defaults."""
        full = {
            'rp': 0.1, 'inc': 90.0, 'a': 10.0, 't0': self.t0,
            'ecc': ECCENTRICITY, 'w': ARG_PERI,
            'u1': LIMB_DARKENING_COEFFS[0], 'u2': LIMB_DARKENING_COEFFS[1],
            'per': self.period,
        }
        for name, val in zip(self.fitted_params, params):
            full[name] = val
        return full

    def _model_at(self, params, smooth_time, t0_override=None):
        """Evaluate the batman model for a single parameter vector at the
        given grid.  Returns *None* on any batman failure."""
        full = self._params_to_full_dict(params)
        tp = batman.TransitParams()
        tp.limb_dark = LIMB_DARKENING_MODEL
        tp.rp = full['rp']; tp.a = full['a']; tp.inc = full['inc']
        tp.ecc = full['ecc']; tp.w = full['w']
        tp.u = [full['u1'], full['u2']]; tp.per = full['per']
        tp.t0 = full['t0'] if t0_override is None else t0_override
        try:
            return batman.TransitModel(tp, smooth_time).light_curve(tp)
        except Exception:
            return None

    def _smooth_grid(self, num_points, phase_folded):
        """Return (smooth_time, t0_override) for the requested overlay."""
        if phase_folded:
            # Use the BLS / input period (the same one the data are
            # folded with in the analyzer) so the model is plotted on
            # the same x-axis as the folded scatter.  Using the MCMC
            # posterior period here would otherwise stretch / shift the
            # model relative to the data.
            return (np.linspace(-self.period / 10, self.period / 10,
                                num_points),
                    0.0)
        return (np.linspace(self.time.min(), self.time.max(), num_points),
                None)

    # -----------------------------------------------------------------
    # MAP (maximum-likelihood) sample retrieval
    # -----------------------------------------------------------------

    def get_map_params(self):
        """Return the parameter vector with the highest log-probability
        in the *post-burn-in* flat chain.  Unlike the per-axis median
        (which is not a likelihood-maximiser when the posterior is
        skewed or correlated), this is the single sample that best
        reproduces the observed transit shape — which is exactly what
        you want to overlay on the folded light-curve.

        Cached on the instance so repeated calls (e.g. one for the
        residuals, one for the credible-band figure, one for the
        report) do not retrigger O(N_samples) batman evaluations."""
        if self.flat_samples is None:
            raise ValueError("MCMC has not been run. Call run_mcmc() first.")
        if getattr(self, "_map_cache", None) is not None:
            return self._map_cache

        # Fast path: emcee already evaluated log-prob at every step.
        # We just need to apply the same burn-in/thin slicing used to
        # build flat_samples, so the indices line up.
        try:
            thin = max(1, int(self.autocorr_time / 2)) if self.autocorr_time \
                else 10
            log_probs = self.sampler.get_log_prob(
                discard=self.burn_in, thin=thin, flat=True)
            if log_probs.shape[0] != self.flat_samples.shape[0]:
                raise ValueError("shape mismatch")  # fall back below
        except Exception:
            log_probs = np.array([self._log_prob(s)
                                  for s in self.flat_samples])

        best_idx = int(np.argmax(log_probs))
        self._map_cache = (self.flat_samples[best_idx],
                           float(log_probs[best_idx]))
        return self._map_cache

    # -----------------------------------------------------------------
    # Public model-curve API
    # -----------------------------------------------------------------

    def get_spaghetti_curves(self, n_draws=50, num_points=1000,
                             phase_folded=True):
        """Draw random posterior samples and compute model curves.

        Returns (smooth_time, list_of_flux_arrays).
        """
        if self.flat_samples is None:
            raise ValueError("MCMC has not been run. Call run_mcmc() first.")

        smooth_time, t0_override = self._smooth_grid(num_points, phase_folded)
        n_draws = min(n_draws, len(self.flat_samples))
        indices = np.random.choice(len(self.flat_samples), size=n_draws,
                                   replace=False)
        curves = []
        for idx in indices:
            curve = self._model_at(self.flat_samples[idx], smooth_time,
                                   t0_override)
            if curve is not None:
                curves.append(curve)
        return smooth_time, curves

    def get_credible_band(self, n_draws=500, num_points=1000,
                          phase_folded=True, lower=16, upper=84,
                          mode='posterior'):
        """Return a per-time credible band around the transit model.

        Parameters
        ----------
        n_draws : int
            Number of posterior draws used to build the envelope.  500
            is enough for a smooth ±1σ band on a 1000-point grid; for a
            full ±2σ band increase to ~2000.
        num_points : int
            Resolution of the model grid.
        phase_folded : bool
            If True, the grid is centred on phase 0 and uses
            ``self.period`` (matching the analyzer's folded axis).
        lower, upper : float
            Posterior percentiles for the band (default 16 / 84 = ±1σ).
        mode : {'posterior', 'predictive'}
            * ``'posterior'`` — pure parameter uncertainty.  Width is
              the spread of model curves drawn from the chain.  For a
              high-SNR fit this can be only a few hundred ppm wide,
              i.e. effectively invisible against the per-point data
              scatter.  Useful when the question is "how well are the
              parameters constrained?".
            * ``'predictive'`` — parameter uncertainty quadratically
              combined with the typical per-point data σ (median of
              ``self.flux_err``).  Width is "where should ~68 % of
              new data points fall around the model?".  Useful for
              visually checking whether the fit is consistent with
              the observed scatter.  Recommended default for plots
              shown to a human eye.

        Returns
        -------
        smooth_time : ndarray
        median_curve : ndarray
        lower_curve  : ndarray
        upper_curve  : ndarray
        """
        if self.flat_samples is None:
            raise ValueError("MCMC has not been run. Call run_mcmc() first.")
        if mode not in ('posterior', 'predictive'):
            raise ValueError(f"Unknown mode={mode!r}")

        smooth_time, t0_override = self._smooth_grid(num_points, phase_folded)
        n_draws = min(n_draws, len(self.flat_samples))
        indices = np.random.choice(len(self.flat_samples), size=n_draws,
                                   replace=False)

        all_curves = np.empty((n_draws, smooth_time.size), dtype=np.float64)
        kept = 0
        for idx in indices:
            curve = self._model_at(self.flat_samples[idx], smooth_time,
                                   t0_override)
            if curve is not None and np.all(np.isfinite(curve)):
                all_curves[kept] = curve
                kept += 1
        all_curves = all_curves[:kept]
        if kept == 0:
            raise RuntimeError("No valid posterior draws produced a model.")

        median_curve = np.percentile(all_curves, 50, axis=0)
        lower_curve = np.percentile(all_curves, lower, axis=0)
        upper_curve = np.percentile(all_curves, upper, axis=0)

        if mode == 'predictive':
            # Symmetric Gaussian widening by the per-point data σ.
            # Combined in quadrature with the (asymmetric) posterior
            # half-widths so the band correctly degenerates to the
            # parameter-only band when σ_data is tiny, and to a flat
            # ±σ_data corridor when the parameters are pinned.  Use
            # the median rather than the mean so a few outlier error
            # bars don't bloat the corridor.
            sigma_data = float(np.median(self.flux_err))
            half_lo = median_curve - lower_curve
            half_hi = upper_curve - median_curve
            lower_curve = median_curve - np.sqrt(half_lo ** 2 + sigma_data ** 2)
            upper_curve = median_curve + np.sqrt(half_hi ** 2 + sigma_data ** 2)

        return smooth_time, median_curve, lower_curve, upper_curve

    def get_best_model_curve(self, num_points=5000, phase_folded=False,
                             folded=None, mode='map'):
        """Return the time grid and best-fit transit-model flux.

        Parameters
        ----------
        phase_folded : bool
            If True, the grid is in **orbital phase** (days from mid-transit),
            matching ``LightCurve.fold``'s time axis.  If False, the grid is
            absolute time (BTJD, BKJD, …), matching an *unfolded* fit.
        folded : bool, optional
            Alias for ``phase_folded`` (whichever is passed last wins if both
            are set — prefer passing only one).
        mode : {'map', 'median', 'optimizer'}
            * ``'map'`` (default) — single sample with the highest
              log-probability in the post-burn-in chain.  Best choice
              for overlay plots; correctly tracks tilted likelihood
              ridges where the per-axis medians do not.
            * ``'median'`` — the per-parameter posterior median.  Kept
              for backwards compatibility but tends to look slightly
              misplaced when the posterior is skewed (e.g. the LD
              parameters in TESS transit fits).
            * ``'optimizer'`` — the pre-MCMC L-BFGS-B / DE best fit
              stored in ``self.best_fit_params``.
        """
        if self.results_summary is None:
            raise ValueError("MCMC has not been run. Call run_mcmc() first.")

        if mode == 'map':
            params, _ = self.get_map_params()
        elif mode == 'median':
            params = np.array([self.results_summary[self.labels[i]][0]
                               for i in range(len(self.fitted_params))])
        elif mode == 'optimizer':
            params = np.asarray(self.best_fit_params, float)
        else:
            raise ValueError(f"Unknown mode={mode!r}")

        if folded is not None:
            phase_folded = bool(folded)
        smooth_time, t0_override = self._smooth_grid(num_points, phase_folded)
        curve = self._model_at(params, smooth_time, t0_override)
        if curve is None:
            raise RuntimeError("batman failed for the requested parameters.")
        return smooth_time, curve
