"""
Lightcurve Processing Utilities
-------------------------------
Provides the `LightCurveAnalyzer` class to search, download, clean,
stitch and fold exoplanetary lightcurve data using the Lightkurve package.

The analyzer supports selecting **multiple** rows from the MAST search
result at once (e.g. individual indices, slices, or mixed
``"0-5,7,9-15"`` range strings).  Every selected observation is
downloaded, individually normalized / σ-clipped, and then stitched
into a single extended ``clean_lc`` so all later stages (BLS, phase
folding, MCMC fit, DVR report) see a contiguous time-series that
carries as many transits as possible.

To keep the raw lightcurve graphic readable when stitching sectors
separated by months of downtime, we also expose a *display-time*
representation where any gap wider than a small threshold is
compressed.  The physically meaningful ``clean_lc.time`` is preserved
unchanged so BLS / Batman / Emcee keep running on the true cadence.

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightkurve as lk
import astropy.units as u


# ---------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------

def _parse_index_spec(spec, n: int) -> list[int]:
    """Normalise an index specification into a sorted list of unique ints.

    Accepted forms
    --------------
    * ``int`` or ``numpy.integer`` — single row
    * ``slice`` — standard Python slice semantics on ``range(n)``
    * ``str`` — comma-separated tokens, each either a single integer
      (``"7"``) or an inclusive range (``"0-5"``, ``"9-15"``)
    * iterable of any of the above, plus 2-tuples interpreted as
      inclusive ``(start, end)`` ranges
    * ``None`` — returns ``[0]`` (default first search result, matches
      the historical single-row behaviour)

    Raises
    ------
    IndexError  if any resolved index is outside ``[0, n-1]``.
    TypeError   if ``spec`` cannot be interpreted.
    """
    if n <= 0:
        raise ValueError("Empty search result — nothing to parse.")

    def _inclusive_range(a, b):
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))

    def _parse_scalar(s):
        if isinstance(s, (int, np.integer)):
            return [int(s)]
        if isinstance(s, slice):
            return list(range(*s.indices(n)))
        if isinstance(s, str):
            out: list[int] = []
            for tok in s.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                # A leading minus should still look like "-3" (single
                # negative int, interpreted Python-style from the end).
                if "-" in tok[1:]:
                    a, b = tok.lstrip("-").split("-", 1)
                    if tok.startswith("-"):
                        a = "-" + a
                    out.extend(_inclusive_range(a, b))
                else:
                    out.append(int(tok))
            return out
        raise TypeError(f"Cannot interpret index token: {s!r}")

    if spec is None:
        resolved = [0]
    elif isinstance(spec, (int, np.integer, slice, str)):
        resolved = _parse_scalar(spec)
    elif hasattr(spec, "__iter__"):
        resolved = []
        for item in spec:
            if isinstance(item, (tuple, list)) and len(item) == 2 \
                    and all(isinstance(v, (int, np.integer)) for v in item):
                resolved.extend(_inclusive_range(*item))
            else:
                resolved.extend(_parse_scalar(item))
    else:
        raise TypeError(f"Cannot parse indices spec: {spec!r}")

    # Wrap Python-style negatives, de-duplicate, sort.
    norm = []
    for i in resolved:
        i = int(i)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError(
                f"Index {i} out of range [0, {n - 1}] for search result "
                f"of length {n}.")
        norm.append(i)
    return sorted(set(norm))


def _format_index_spec(indices: list[int]) -> str:
    """Format a sorted list of ints as a compact human-readable spec.

    ``[0, 1, 2, 5, 7, 8, 9]`` → ``"0–2, 5, 7–9"`` (en-dash).  Purely
    cosmetic: used for figure titles and report annotations.
    """
    if not indices:
        return ""
    indices = sorted(set(int(i) for i in indices))
    ranges: list[tuple[int, int]] = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
        else:
            ranges.append((start, prev))
            start = prev = i
    ranges.append((start, prev))
    parts = [f"{a}" if a == b else f"{a}\u2013{b}" for a, b in ranges]
    return ", ".join(parts)


# ---------------------------------------------------------------------
#  Observation-reference labelling
# ---------------------------------------------------------------------
#  ``mission`` strings from Lightkurve look like ``"TESS Sector 02"``,
#  ``"Kepler Quarter 3"``, ``"K2 Campaign 7"`` — we shorten the cadence
#  identifier to ``S02 / Q3 / C7`` and group consecutive same-author
#  entries so the report advertises the real observations, not
#  opaque row indices.
# ---------------------------------------------------------------------

_CADENCE_PREFIXES = (("Sector ", "S"), ("Quarter ", "Q"), ("Campaign ", "C"))


def _split_mission(mission: str) -> tuple[str, str]:
    """Return ``(mission_root, cadence_tag)`` from a Lightkurve mission
    string.  Unknown formats keep the whole string as the root and
    return an empty cadence tag."""
    mission = (mission or "").strip()
    for keyword, short in _CADENCE_PREFIXES:
        if keyword in mission:
            root, _, ident = mission.partition(keyword)
            return root.strip(), f"{short}{ident.strip()}"
    return mission, ""


def _safe_str(v) -> str:
    """Turn pandas/numpy NaN-ish values into an empty string."""
    try:
        if v is None:
            return ""
        s = str(v).strip()
    except Exception:
        return ""
    return "" if s.lower() in ("nan", "none", "--") else s


def _format_selection_label(details: list[dict]) -> str:
    """Render a list of per-segment dicts as a compact human-readable
    string.  Entries that share the same ``(mission_root, author)`` are
    merged so an explicit ``"0-5, 7"`` selection of seven TESS SPOC
    sectors collapses to ``"TESS S01–S05, S07 (SPOC)"`` rather than
    seven separate entries.
    """
    if not details:
        return ""

    keys_order: list[tuple[str, str]] = []
    buckets: dict[tuple[str, str], list[str]] = {}
    for d in details:
        mission = _safe_str(d.get("mission"))
        author = _safe_str(d.get("author"))
        root, ident = _split_mission(mission)
        key = (root, author)
        if key not in buckets:
            keys_order.append(key)
            buckets[key] = []
        if ident and ident not in buckets[key]:
            buckets[key].append(ident)

    parts: list[str] = []
    for root, author in keys_order:
        idents = buckets[(root, author)]
        # Collapse consecutive cadence numbers into an en-dash range.
        def _collapse(seq: list[str]) -> str:
            nums = []
            for s in seq:
                try:
                    nums.append((s[0], int(s[1:])))
                except ValueError:
                    nums.append((None, None))
            out: list[str] = []
            i = 0
            while i < len(nums):
                prefix, n = nums[i]
                if prefix is None:
                    out.append(seq[i])
                    i += 1
                    continue
                j = i
                while (j + 1 < len(nums)
                       and nums[j + 1][0] == prefix
                       and nums[j + 1][1] == nums[j][1] + 1):
                    j += 1
                if j == i:
                    out.append(seq[i])
                else:
                    out.append(f"{seq[i]}\u2013{seq[j]}")
                i = j + 1
            return ", ".join(out)

        idents_str = _collapse(idents)
        chunks = [c for c in (root, idents_str) if c]
        label = " ".join(chunks)
        if author:
            label = f"{label} ({author})" if label else f"({author})"
        parts.append(label or _safe_str(d.get("mission")) or "observation")
    return " + ".join(parts)


def _compress_time(time: np.ndarray, max_display_gap: float = 1.0):
    """Compress any inter-cadence gap larger than ``max_display_gap``.

    Parameters
    ----------
    time : ndarray
        Original time stamps (days, same unit as ``clean_lc.time.value``).
        Can be unsorted — the mapping is built in sort order and then
        un-sorted back to the original order so the returned array
        aligns element-wise with ``time``.
    max_display_gap : float
        Any native gap larger than this value (in days) is collapsed to
        exactly this value.  Smaller gaps are left untouched so the
        in-sector cadence structure (e.g. TESS 2-minute data downlink
        gaps) is preserved.

    Returns
    -------
    display_time : ndarray
        Same shape as ``time``; monotonic within each original segment.
    segment_edges : ndarray
        Display-time values at the start of each new segment (the
        cadences immediately following a compressed gap).  Useful for
        drawing faint vertical dividers on the raw LC panel.
    """
    time = np.asarray(time, dtype=np.float64)
    if time.size == 0:
        return time.copy(), np.array([])

    order = np.argsort(time)
    t_sorted = time[order]
    if time.size == 1:
        return time.copy(), np.array([])

    dt = np.diff(t_sorted)
    gap_mask = dt > max_display_gap
    # Collapse large gaps to max_display_gap; preserve all others.
    dt_compressed = np.where(gap_mask, max_display_gap, dt)
    display_sorted = np.concatenate(
        [[t_sorted[0]], t_sorted[0] + np.cumsum(dt_compressed)]
    )

    display = np.empty_like(display_sorted)
    display[order] = display_sorted

    seg_start_idx = np.where(gap_mask)[0] + 1  # in sorted order
    segment_edges = display_sorted[seg_start_idx]
    return display, segment_edges


# ---------------------------------------------------------------------
#  Main class
# ---------------------------------------------------------------------

class LightCurveAnalyzer:
    """
    A class used to manage the lifecycle of a lightcurve analysis,
    from searching the MAST archive to folding the data on a specific period.

    Multi-sector workflow
    ---------------------
    ``download_and_clean`` accepts a single index (backwards-compatible
    single-row behaviour), a list of indices, an inclusive range string
    (``"0-5,7,9-15"``), or any mix thereof.  Every selected observation
    is downloaded, individually ``.normalize()``-d and σ-clipped, and
    then stitched (sorted by time) into the attribute ``clean_lc``.

    The physical time-stamps are kept intact so downstream physics
    (BLS, folding, Batman) keeps working transparently across epochs.
    A companion attribute ``display_time`` (+ ``segment_edges``) is
    also exposed so the report's raw lightcurve panel can plot the
    stitched data on a compressed axis without being visually stretched
    by inter-sector down-time.
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
        self.raw_lc = None          # first downloaded LC (meta reference)
        self.raw_lcs: list = []      # all individual per-row LCs (cleaned)
        self.clean_lc = None         # stitched, σ-clipped lightcurve
        self.periodogram = None
        self.folded_lc = None

        # Multi-sector bookkeeping
        self.selected_indices: list[int] = []
        self.selection_label: str = ""             # human-readable obs refs
        self.selection_details: list[dict] = []    # per-segment metadata
        self.display_time: np.ndarray | None = None
        self.segment_edges: np.ndarray | None = None

        # Extracted BLS (Box Least Squares) parameters
        self.best_period = None
        self.best_freq = None
        self.best_power = None
        self.epoch_time = None
        self.transit_time = None
        self.transit_depth = None

    # -----------------------------------------------------------------
    #  SEARCH
    # -----------------------------------------------------------------

    def search(self, radius=None, exptime=None, cadence=None,
               mission=('Kepler', 'K2', 'TESS'), author=None,
               quarter=None, month=None, campaign=None, sector=None,
               limit=None) -> pd.DataFrame:
        """
        Searches the MAST archive for available lightcurves matching the target.

        The returned DataFrame includes an explicit ``index`` column so the
        user can unambiguously reference rows in the subsequent call to
        :meth:`download_and_clean` (single row, list, or range string).

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
            return pd.DataFrame()  # Return empty dataframe if nothing is found

        df = self.search_result.table.to_pandas()
        cols_to_show = ['mission', 'year', 'author', 'exptime', 'target_name', 'distance']
        available_cols = [col for col in cols_to_show if col in df.columns]
        out = df[available_cols].copy()
        out.insert(0, 'index', np.arange(len(out)))
        return out

    # -----------------------------------------------------------------
    #  DOWNLOAD + CLEAN (+ STITCH)
    # -----------------------------------------------------------------

    def download_and_clean(self, indices=None, sigma: float = 5.0,
                           *, index=None, max_display_gap: float = 1.0):
        """
        Download one or more lightcurves, clean each, then stitch them
        into a single extended lightcurve.

        Normalisation (division by the per-sector median flux) is performed
        *before* stitching so that sectors with very different flux
        baselines — different telescopes, apertures, or even different
        missions — combine cleanly around unity.  σ-clipping is also done
        per-sector, which prevents a bright cosmic-ray spike in sector
        *A* from biasing the noise estimate in sector *B*.

        The outlier-clipping step is critical for transit fitting: a single
        cosmic-ray hit or scattered-light spike at the wrong phase will
        otherwise be picked up as the deepest point and corrupt the BLS
        epoch estimate (see :meth:`compute_periodogram`).

        Parameters
        ----------
        indices : int, list, slice, str, or None
            Row(s) from ``self.search_result`` to download.  Accepts any
            spec understood by :func:`_parse_index_spec` — e.g. ``0``,
            ``[0, 1, 2, 7]``, ``slice(0, 4)``, or ``"0-5, 7, 9-15"``.
            Default ``None`` behaves like the legacy single-row call
            (first row only).
        sigma : float
            σ-clipping threshold for outlier removal.
            ``5.0`` is conservative — high enough to keep the in-transit
            points, low enough to remove most thruster firings / cosmic
            rays.  Pass ``sigma=None`` to disable clipping entirely.
        index : int, optional
            **Deprecated** alias for ``indices`` kept for backwards
            compatibility with the single-row API.  Ignored when
            ``indices`` is provided explicitly.
        max_display_gap : float, default 1.0
            Gap length (in days) above which the *display* time axis is
            collapsed.  This has **no** effect on the scientific time-
            stamps stored in ``clean_lc`` — only the compressed axis
            used by the DVR raw-lightcurve panel.

        Returns
        -------
        self : LightCurveAnalyzer
            For fluent chaining.
        """
        if self.search_result is None or len(self.search_result) == 0:
            raise ValueError("No search results available. Call search() first.")

        # Back-compat: allow ``download_and_clean(index=0)`` to still work.
        if indices is None and index is not None:
            indices = index
        idx_list = _parse_index_spec(indices, len(self.search_result))

        # ---- download + clean each segment individually ---------------
        segments: list = []
        for i in idx_list:
            lc = self.search_result[i].download()
            if lc is None:
                continue
            try:
                lc = lc.normalize().remove_nans()
            except Exception:
                # Some exotic datasets fail to normalise (flux units).
                # Fall back to raw + NaN-removal; the rest of the pipeline
                # tolerates un-normalised data.
                lc = lc.remove_nans()
            if sigma is not None:
                try:
                    lc = lc.remove_outliers(sigma=sigma)
                except Exception:
                    pass
            segments.append(lc)

        if not segments:
            raise RuntimeError(
                f"No lightcurves could be downloaded for indices {idx_list}. "
                "All selected rows failed to download or returned empty data.")

        # ---- stitch (concatenate + sort by time) ---------------------
        if len(segments) == 1:
            stitched = segments[0]
        else:
            stitched = segments[0].copy()
            for seg in segments[1:]:
                try:
                    stitched = stitched.append(seg)
                except Exception:
                    # Defensive: if .append fails because of schema
                    # mismatches, fall back to a bare (time, flux,
                    # flux_err) re-build — losing the aux columns but
                    # keeping the science.
                    from astropy.time import Time
                    t_vals = np.concatenate(
                        [np.asarray(s.time.value, float) for s in segments])
                    f_vals = np.concatenate(
                        [np.asarray(s.flux.value, float) for s in segments])
                    e_vals = np.concatenate([
                        (np.asarray(s.flux_err.value, float)
                         if s.flux_err is not None
                         else np.full(s.flux.shape,
                                      float(np.nanmedian(s.flux.value)) * 0.01))
                        for s in segments
                    ])
                    ref_t = segments[0].time
                    t_obj = Time(t_vals, format=ref_t.format,
                                 scale=ref_t.scale)
                    stitched = lk.LightCurve(
                        time=t_obj, flux=f_vals, flux_err=e_vals)
                    try:
                        stitched.meta.update(segments[0].meta)
                    except Exception:
                        pass
                    break

            # Stable sort by time (ascending) so BLS / fold see a
            # monotonic series.
            order = np.argsort(np.asarray(stitched.time.value, float))
            stitched = stitched[order]

        # ---- expose everything the pipeline needs --------------------
        self.selected_indices = idx_list

        # Build per-segment metadata (mission / author / exptime / year)
        # from the search table so the report can reference the *actual*
        # observations rather than opaque row indices.
        details: list[dict] = []
        try:
            sr_df = self.search_result.table.to_pandas()
        except Exception:
            sr_df = None
        for i in idx_list:
            rec: dict = {"index": int(i)}
            if sr_df is not None and 0 <= i < len(sr_df):
                row = sr_df.iloc[i]
                for col in ("mission", "author", "exptime",
                            "year", "target_name"):
                    if col in sr_df.columns:
                        rec[col] = row[col]
            details.append(rec)
        self.selection_details = details
        self.selection_label = (
            _format_selection_label(details) or _format_index_spec(idx_list))
        self.raw_lcs = segments
        self.raw_lc = segments[0]
        self.clean_lc = stitched

        t_arr = np.asarray(stitched.time.value, dtype=np.float64)
        self.display_time, self.segment_edges = _compress_time(
            t_arr, max_display_gap=max_display_gap)

        self.periodogram = None
        self.folded_lc = None
        self.best_period = None
        self.best_freq = None
        self.best_power = None
        self.epoch_time = None

        return self

    # -----------------------------------------------------------------
    #  Utility: map an absolute-time value to the compressed display
    #  axis used by the DVR raw-LC panel.  Returns the input unchanged
    #  when no multi-sector stitching is in effect.
    # -----------------------------------------------------------------

    def to_display_time(self, t):
        """Map absolute-time values to the compressed display axis.

        The function is monotonic-within-segment and nearest-neighbour
        safe: points that fall inside an original sector are mapped
        exactly; points inside a compressed inter-sector gap land on
        the gap's edge.  Returns a plain numpy array; NaNs are produced
        for samples outside the covered range.

        When no ``display_time`` has been computed (single-row workflow),
        the input is simply returned as-is.
        """
        t = np.asarray(t, dtype=np.float64)
        if self.display_time is None or self.clean_lc is None:
            return t
        t_orig = np.asarray(self.clean_lc.time.value, dtype=np.float64)
        order = np.argsort(t_orig)
        return np.interp(t,
                         t_orig[order], self.display_time[order],
                         left=np.nan, right=np.nan)

    # -----------------------------------------------------------------
    #  BLS PERIODOGRAM
    # -----------------------------------------------------------------

    def compute_periodogram(self, minimum_period: float | None = None,
                            maximum_period: float | None = None,
                            frequency_factor: float | None = None,
                            **bls_kwargs):
        """
        Computes the Box Least Squares (BLS) periodogram to find the most
        likely orbital period of the transiting exoplanet, and uses the
        BLS fit itself to derive a robust mid-transit epoch.

        Parameters
        ----------
        minimum_period, maximum_period : float, optional
            Period search bounds (days).  By default Lightkurve picks
            ``max ≈ baseline / 2`` and a very short minimum; for stitched
            multi-sector data the baseline can span *years*, which makes
            the default BLS grid explode past Lightkurve's 1e7-point
            safety cap.  Pass your own bounds when you already have a
            rough prior (e.g. ``minimum_period=0.5, maximum_period=20``).
        frequency_factor : float, optional
            BLS oversampling factor (Lightkurve default = 10).  Higher
            values → coarser grid → fewer evaluation points.  When
            stitching many sectors we auto-coarsen the grid so the call
            never hits the 1e7 cap.
        **bls_kwargs
            Extra keyword args forwarded to
            ``lightkurve.LightCurve.to_periodogram``.
        """
        if self.clean_lc is None:
            raise ValueError("No clean lightcurve available. Call download_and_clean() first.")

        kwargs: dict = dict(bls_kwargs)

        # --- period-range defaults --------------------------------------
        t_arr = np.asarray(self.clean_lc.time.value, dtype=np.float64)
        baseline = float(np.nanmax(t_arr) - np.nanmin(t_arr))

        pmin = float(minimum_period) if minimum_period is not None else 0.5
        if maximum_period is not None:
            pmax = float(maximum_period)
        else:
            pmax = min(30.0, max(1.0, baseline / 3.0))
        kwargs["minimum_period"] = pmin
        kwargs["maximum_period"] = pmax

        kwargs.setdefault("duration", [0.05, 0.10, 0.20])

        # --- BLS grid oversampling --------------------------------------
        if frequency_factor is not None:
            ff = float(frequency_factor)
        else:
            min_dur = float(min(kwargs["duration"]))
            target_npts = 2.0e5
            freq_range = 1.0 / pmin - 1.0 / max(pmax, pmin + 1e-6)
            ff = max(
                10.0,
                freq_range * baseline * baseline
                / (target_npts * max(min_dur, 1e-3))
            )
        kwargs["frequency_factor"] = ff

        last_err: Exception | None = None
        for _ in range(6):  # up to ff = 10 * 5^6 ≈ 1.5e5
            try:
                self.periodogram = self.clean_lc.to_periodogram(
                    method="bls", **kwargs)
                last_err = None
                break
            except ValueError as exc:
                msg = str(exc)
                if "too large to evaluate" not in msg:
                    raise
                last_err = exc
                ff *= 5.0
                kwargs["frequency_factor"] = ff
        else:
            raise RuntimeError(
                "BLS periodogram grid remained too large even after "
                f"coarsening frequency_factor to {ff:g}.  Pass explicit "
                "minimum_period / maximum_period bounds."
            ) from last_err

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
        # stacked transits — much more reliable.  On stitched multi-
        # sector data BLS naturally benefits from the extra transits
        # and returns an even more robust epoch, provided the segments
        # share a common time system (TESS→TESS BTJD, Kepler→Kepler
        # BKJD).  Cross-mission stitching is still valid but may widen
        # the BLS peak.
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

        Multi-sector compatibility: folding is a pure modulo operation
        on the absolute time column, so stitched data from several
        epochs map onto the *same* phase axis automatically — no extra
        per-segment offset tracking is needed.  The only requirement
        is that ``period`` and ``epoch_time`` are expressed in the same
        time system as ``clean_lc.time.value`` (which they are:
        ``epoch_time`` comes from BLS, ``period`` is refined by the
        posterior in the original time system).

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

        Works transparently on both single-sector and stitched multi-
        sector lightcurves: the absolute time column carries every
        cadence from every selected observation, so Batman's model
        evaluation naturally spans all epochs.  There is no per-epoch
        ``t0`` offset — the transit model is periodic, so a single
        reference ``t0`` (in the same time system as ``time``) plus the
        posterior period ``per`` define *every* transit mid-time via
        ``t0 + n × per`` for integer ``n``.

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
