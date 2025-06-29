"""
Planetary and Stellar Dataset Filtering Module
-----------------------------------------------
This module provides a single entry-point function `apply_filters()` for filtering confirmed exoplanets from the NASA Exoplanet Archive dataset.

Users can apply discovery, stellar, planetary, and system filters in any combination to explore subsets of the exoplanet catalog.

Usage:
    from utils.filters import apply_filters
    df_filtered = apply_filters(
        df,
        mission='Kepler', Teff_min=4000, rade_max=2, density_min=5
    )

Author: S.WITTMANN & V.RAGNER
Repository: https://github.com/SimonWtmn/Stage_CEA_Exoplanet
"""

import pandas as pd
from typing import Optional



def apply_filters(
    df: pd.DataFrame,

    # ---------------------------------- Discovery filters ----------------------------------
    mission: Optional[str] = None,
    discovery_method: Optional[str] = None,
    date_min: Optional[int] = None,
    date_max: Optional[int] = None,
    kp_max: Optional[float] = None,

    # ----------------------------------  Stellar filters ----------------------------------
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

    # ----------------------------------  Planetary filters ----------------------------------
    rade_min: Optional[float] = None,
    rade_max: Optional[float] = None,
    rade_err: Optional[float] = None,
    mass_min: Optional[float] = None,
    mass_max: Optional[float] = None,
    mass_err: Optional[float] = None,
    density_min: Optional[float] = None,
    density_max: Optional[float] = None,
    eccentricity_max: Optional[float] = None,
    transit_depth_min: Optional[float] = None,
    transit_depth_max: Optional[float] = None,
    eqt_min: Optional[float] = None,
    eqt_max: Optional[float] = None,
    period_max: Optional[float] = None,
    impact_param_max: Optional[float] = None,

    # ----------------------------------  System filters ----------------------------------
    multiplicity_min: Optional[int] = None,
    multiplicity_max: Optional[int] = None,

) -> pd.DataFrame:

    # validate ranges
    _validate(date_min, date_max, 'date_min', 'date_max')
    _validate(teff_min, teff_max, 'teff_min', 'teff_max')
    _validate(lum_min, lum_max, 'lum_min', 'lum_max')
    _validate(metallicity_min, metallicity_max, 'metallicity_min', 'metallicity_max')
    _validate(age_min, age_max, 'age_min', 'age_max')
    _validate(st_rad_min, st_rad_max, 'st_rad_min', 'st_rad_max')
    _validate(rade_min, rade_max, 'rade_min', 'rade_max')
    _validate(mass_min, mass_max, 'mass_min', 'mass_max')
    _validate(density_min, density_max, 'density_min', 'density_max')
    _validate(transit_depth_min, transit_depth_max, 'transit_depth_min', 'transit_depth_max')
    _validate(eqt_min, eqt_max, 'eqt_min', 'eqt_max')
    _validate(multiplicity_min, multiplicity_max, 'multiplicity_min', 'multiplicity_max')



    mask = pd.Series(True, index=df.index)

    # ---------------------------------- Discovery filters ----------------------------------
    if mission is not None:
        mask &= df['disc_facility'].notna() & (df['disc_facility'] == mission)
    if discovery_method is not None:
        mask &= df['discoverymethod'].notna() & (df['discoverymethod'] == discovery_method)
    if date_min is not None:
        mask &= df['disc_year'].notna() & (df['disc_year'] >= date_min)
    if date_max is not None:
        mask &= df['disc_year'].notna() & (df['disc_year'] <= date_max)
    if kp_max is not None:
        mask &= df['sy_kepmag'].notna() & (df['sy_kepmag'] <= kp_max)



    # ---------------------------------- Stellar filters ----------------------------------
    if st_type is not None:
        mask &= df['st_spectype'].notna() & df['st_spectype'].str.upper().str.startswith(st_type.upper())

    if teff_min is not None:
        mask &= df['st_teff'].notna() & (df['st_teff'] >= teff_min)
    if teff_max is not None:
        mask &= df['st_teff'].notna() & (df['st_teff'] <= teff_max)

    if lum_min is not None:
        mask &= df['st_lum'].notna() & (df['st_lum'] >= lum_min)
    if lum_max is not None:
        mask &= df['st_lum'].notna() & (df['st_lum'] <= lum_max)

    if metallicity_min is not None:
        mask &= df['st_met'].notna() & (df['st_met'] >= metallicity_min)
    if metallicity_max is not None:
        mask &= df['st_met'].notna() & (df['st_met'] <= metallicity_max)

    if age_min is not None:
        mask &= df['st_age'].notna() & (df['st_age'] >= age_min)
    if age_max is not None:
        mask &= df['st_age'].notna() & (df['st_age'] <= age_max)

    if st_rad_min is not None:
        mask &= df['st_rad'].notna() & (df['st_rad'] >= st_rad_min)
    if st_rad_max is not None:
        mask &= df['st_rad'].notna() & (df['st_rad'] <= st_rad_max)
    if st_rad_err is not None:
        mask &= _snr_mask(df, 'st_rad', 'st_raderr1', 'st_raderr2', st_rad_err)

    if use_fulton_filter:
        mask &= _fulton_mask(df)



    # ---------------------------------- Planetary filters ----------------------------------
    if rade_min is not None:
        mask &= df['pl_rade'].notna() & (df['pl_rade'] >= rade_min)
    if rade_max is not None:
        mask &= df['pl_rade'].notna() & (df['pl_rade'] <= rade_max)
    if rade_err is not None:
        mask &= _snr_mask(df, 'pl_rade', 'pl_radeerr1', 'pl_radeerr2', rade_err)

    if mass_min is not None:
        mask &= df['pl_bmasse'].notna() & (df['pl_bmasse'] >= mass_min)
    if mass_max is not None:
        mask &= df['pl_bmasse'].notna() & (df['pl_bmasse'] <= mass_max)
    if mass_err is not None:
        mask &= _snr_mask(df, 'pl_bmasse', 'pl_bmasseerr1', 'pl_bmasseerr2', mass_err)

    if density_min is not None:
        mask &= df['pl_dens'].notna() & (df['pl_dens'] >= density_min)
    if density_max is not None:
        mask &= df['pl_dens'].notna() & (df['pl_dens'] <= density_max)

    if eccentricity_max is not None:
        mask &= df['pl_orbeccen'].notna() & (df['pl_orbeccen'] <= eccentricity_max)

    if transit_depth_min is not None:
        mask &= df['pl_trandep'].notna() & (df['pl_trandep'] >= transit_depth_min)
    if transit_depth_max is not None:
        mask &= df['pl_trandep'].notna() & (df['pl_trandep'] <= transit_depth_max)

    if eqt_min is not None:
        mask &= df['pl_eqt'].notna() & (df['pl_eqt'] >= eqt_min)
    if eqt_max is not None:
        mask &= df['pl_eqt'].notna() & (df['pl_eqt'] <= eqt_max)

    if period_max is not None:
        mask &= df['pl_orbper'].notna() & (df['pl_orbper'] <= period_max)

    if impact_param_max is not None:
        mask &= df['pl_imppar'].notna() & (df['pl_imppar'] <= impact_param_max)



    # ---------------------------------- System filters ----------------------------------
    if multiplicity_min is not None:
        mask &= df['sy_pnum'].notna() & (df['sy_pnum'] >= multiplicity_min)
    if multiplicity_max is not None:
        mask &= df['sy_pnum'].notna() & (df['sy_pnum'] <= multiplicity_max)


    return df.loc[mask].reset_index(drop=True)





# ---------------------------------- Range validater ----------------------------------
def _validate(min_val, max_val, name_min: str, name_max: str):
    if min_val is not None and max_val is not None and min_val > max_val:
        raise ValueError(f"{name_min} ({min_val}) cannot be greater than {name_max} ({max_val})")





# ---------------------------------- SNR Mask Helper ----------------------------------
def _snr_mask(df, col, err1, err2, min_snr) -> pd.Series:
    mask = df[col].notna() & df[err1].notna() & df[err2].notna()
    errs = df.loc[mask, [err1, err2]].abs().max(axis=1)
    snr = df.loc[mask, col] / errs
    result = pd.Series(False, index=df.index)
    result.loc[mask] = snr >= min_snr
    return result





# ---------------------------------- Fulton Mask Helper ----------------------------------
def _fulton_mask(df: pd.DataFrame) -> pd.Series:
    mask_valid = df['st_teff'].notna() & df['st_rad'].notna()
    threshold = 10 ** (0.00025 * (df.loc[mask_valid, 'st_teff'] - 5500) + 0.20)
    pass_fulton = df.loc[mask_valid, 'st_rad'] > threshold
    result = pd.Series(True, index=df.index)
    result.loc[mask_valid] = pass_fulton
    return result

