"""
Exoplanet Dataset Filtering Utilities
-------------------------------------
Provides a single entry-point function `apply_filters()` for filtering confirmed exoplanets;

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot
"""

import pandas as pd
from typing import Optional


# ------------------------------------- HELPER -------------------------------------


# -------- VALIDATE RANGE --------
def _validate(min_val, max_val, name_min: str, name_max: str):
    if min_val is not None and max_val is not None and min_val > max_val:
        raise ValueError(f"{name_min} ({min_val}) cannot be greater than {name_max} ({max_val})")


# --------- APPLY RANGE ----------
def _apply_range(df, col, min_val, max_val) -> pd.Series:
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    mask = df[col].notna()
    if min_val is not None:
        mask &= df[col] >= min_val
    if max_val is not None:
        mask &= df[col] <= max_val
    return mask


# ---------- ERROR MASK ----------
def _snr_mask(df, col, err1, err2, min_snr) -> pd.Series:
    mask = df[col].notna() & df[err1].notna() & df[err2].notna()
    errs = df.loc[mask, [err1, err2]].abs().max(axis=1)
    snr = df.loc[mask, col] / errs
    result = pd.Series(False, index=df.index)
    result.loc[mask] = snr >= min_snr
    return result


# --------- FULTON MASK ----------
def _fulton_mask(df: pd.DataFrame) -> pd.Series:
    mask_valid = df['st_teff'].notna() & df['st_rad'].notna()
    threshold = 10 ** (0.00025 * (df.loc[mask_valid, 'st_teff'] - 5500) + 0.20)
    pass_fulton = df.loc[mask_valid, 'st_rad'] > threshold
    result = pd.Series(True, index=df.index)
    result.loc[mask_valid] = pass_fulton
    return result




# ------------------------------------- MAIN -------------------------------------

def apply_filters(
    df: pd.DataFrame,

    # --------- Discovery filters ---------
    mission: Optional[str] = None,
    discovery_method: Optional[str] = None,
    date_min: Optional[int] = None,
    date_max: Optional[int] = None,
    kp_max: Optional[float] = None,

    # --------- Stellar filters ----------
    st_type: Optional[str] = None,
    teff_min: Optional[float] = None,
    teff_max: Optional[float] = None,
    lum_min: Optional[float] = None,
    lum_max: Optional[float] = None,
    metallicity_min: Optional[float] = None,
    metallicity_max: Optional[float] = None,
    age_min: Optional[float] = None,
    age_max: Optional[float] = None,
    st_rad_min: Optional[float] = None,
    st_rad_max: Optional[float] = None,
    st_rad_err: Optional[float] = None,
    use_fulton_filter: bool = False,

    # --------- Planetary filters ---------
    rade_min: Optional[float] = None,
    rade_max: Optional[float] = None,
    rade_err: Optional[float] = None,
    mass_min: Optional[float] = None,
    mass_max: Optional[float] = None,
    mass_err: Optional[float] = None,
    density_min: Optional[float] = None,
    density_max: Optional[float] = None,
    distance_min: Optional[float] = None,
    distance_max: Optional[float] = None,
    eccentricity_max: Optional[float] = None,
    transit_depth_min: Optional[float] = None,
    transit_depth_max: Optional[float] = None,
    eqt_min: Optional[float] = None,
    eqt_max: Optional[float] = None,
    period_max: Optional[float] = None,
    impact_param_max: Optional[float] = None,

    # ---------- System filters ----------
    multiplicity_min: Optional[int] = None,
    multiplicity_max: Optional[int] = None,

) -> pd.DataFrame:

    for min_, max_, name_min, name_max in [
        (date_min, date_max, "date_min", "date_max"),
        (teff_min, teff_max, "teff_min", "teff_max"),
        (lum_min, lum_max, "lum_min", "lum_max"),
        (metallicity_min, metallicity_max, "metallicity_min", "metallicity_max"),
        (age_min, age_max, "age_min", "age_max"),
        (st_rad_min, st_rad_max, "st_rad_min", "st_rad_max"),
        (rade_min, rade_max, "rade_min", "rade_max"),
        (mass_min, mass_max, "mass_min", "mass_max"),
        (density_min, density_max, "density_min", "density_max"),
        (transit_depth_min, transit_depth_max, "transit_depth_min", "transit_depth_max"),
        (eqt_min, eqt_max, "eqt_min", "eqt_max"),
        (multiplicity_min, multiplicity_max, "multiplicity_min", "multiplicity_max"),
    ]:
        _validate(min_, max_, name_min, name_max)

    mask = pd.Series(True, index=df.index)


    # ---------------------------------- Discovery filters ----------------------------------
    if mission:
        mask &= df['disc_facility'].notna() & (df['disc_facility'] == mission)
    if discovery_method:
        mask &= df['discoverymethod'].notna() & (df['discoverymethod'] == discovery_method)
    if date_min is not None:
        mask &= df['disc_year'].notna() & (df['disc_year'] >= date_min)
    if date_max is not None:
        mask &= df['disc_year'].notna() & (df['disc_year'] <= date_max)
    if kp_max is not None:
        mask &= df['sy_kepmag'].notna() & (df['sy_kepmag'] <= kp_max)



    # ----------------------------------- Stellar filters -----------------------------------
    if st_type:
        mask &= df['st_spectype'].notna() & df['st_spectype'].str.upper().str.startswith(st_type.upper())
    mask &= _apply_range(df, 'st_teff', teff_min, teff_max)
    mask &= _apply_range(df, 'st_lum', lum_min, lum_max)
    mask &= _apply_range(df, 'st_met', metallicity_min, metallicity_max)
    mask &= _apply_range(df, 'st_age', age_min, age_max)
    mask &= _apply_range(df, 'st_rad', st_rad_min, st_rad_max)
    if st_rad_err is not None:
        mask &= _snr_mask(df, 'st_rad', 'st_raderr1', 'st_raderr2', st_rad_err)
    if use_fulton_filter:
        mask &= _fulton_mask(df)



    # ----------------------------------- Planet filters -----------------------------------
    mask &= _apply_range(df, 'pl_rade', rade_min, rade_max)
    if rade_err is not None:
        mask &= _snr_mask(df, 'pl_rade', 'pl_radeerr1', 'pl_radeerr2', rade_err)
    mask &= _apply_range(df, 'pl_bmasse', mass_min, mass_max)
    if mass_err is not None:
        mask &= _snr_mask(df, 'pl_bmasse', 'pl_bmasseerr1', 'pl_bmasseerr2', mass_err)
    mask &= _apply_range(df, 'pl_orbsmax', distance_min, distance_max)
    mask &= _apply_range(df, 'pl_dens', density_min, density_max)
    if eccentricity_max is not None:
        mask &= df['pl_orbeccen'].notna() & (df['pl_orbeccen'] <= eccentricity_max)
    mask &= _apply_range(df, 'pl_trandep', transit_depth_min, transit_depth_max)
    mask &= _apply_range(df, 'pl_eqt', eqt_min, eqt_max)
    if period_max is not None:
        mask &= df['pl_orbper'].notna() & (df['pl_orbper'] <= period_max)
    if impact_param_max is not None:
        mask &= df['pl_imppar'].notna() & (df['pl_imppar'] <= impact_param_max)
    


    # ----------------------------------- System filters -----------------------------------
    mask &= _apply_range(df, 'sy_pnum', multiplicity_min, multiplicity_max)


    return df.loc[mask].reset_index(drop=True)
