"""
Exoplanet Dataset Filtering Utilities
-------------------------------------
Provides a single entry-point function `apply_filters()` for filtering confirmed exoplanets;

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot
"""

import pandas as pd


# ------------------------------------- HELPER -------------------------------------


# --------- APPLY RANGE ----------
def _apply_range(df, col, min_val, max_val):
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    if min_val is not None and max_val is not None and min_val > max_val:
        raise ValueError(f"Min value ({min_val}) cannot be greater than max value ({max_val})")
    mask = df[col].notna()
    if min_val is not None:
        mask &= df[col] >= min_val
    if max_val is not None:
        mask &= df[col] <= max_val
    return mask


# ---------- ERROR MASK ----------
def _snr_mask(df, col, min_snr):
    mask = df[col].notna() & (df[f"{col}err1"].notna() | df[f"{col}err2"].notna())
    errs = df.loc[mask, [f"{col}err1", f"{col}err2"]].abs().max(axis=1)
    snr = df.loc[mask, col] / errs
    result = pd.Series(False, index=df.index)
    result.loc[mask] = snr >= min_snr
    return result


# --------- FULTON MASK ----------
def _fulton_mask(df: pd.DataFrame) -> pd.Series:
    mask_valid = df['st_teff'].notna() & df['st_rad'].notna()
    threshold = 10 ** (0.00025 * (df.loc[mask_valid, 'st_teff'] - 5500) + 0.20)
    pass_fulton = df.loc[mask_valid, 'st_rad'] < threshold
    result = pd.Series(True, index=df.index)
    result.loc[mask_valid] = pass_fulton
    return result




# ------------------------------------- MAIN -------------------------------------

def apply_filters(
    df: pd.DataFrame,

    # --------- Discovery filters ---------
    mission=None,
    discovery_method=None,
    date_min=None,
    date_max=None,
    kp_max=None,

    # --------- Stellar filters ----------
    st_type=None,
    teff_min=None,
    teff_max=None,
    lum_min=None,
    lum_max=None,
    metallicity_min=None,
    metallicity_max=None,
    age_min=None,
    age_max=None,
    st_rad_min=None,
    st_rad_max=None,
    st_rad_err=None,
    use_fulton_filter=False,

    # --------- Planetary filters ---------
    rade_min=None,
    rade_max=None,
    rade_err=None,
    mass_min=None,
    mass_max=None,
    mass_err=None,
    density_min=None,
    density_max=None,
    distance_min=None,
    distance_max=None,
    eccentricity_max=None,
    transit_depth_min=None,
    transit_depth_max=None,
    eqt_min=None,
    eqt_max=None,
    period_max=None,
    impact_param_max=None,
    tdyson_min=None,
    tdyson_max=None,

    # ---------- System filters ----------
    multiplicity_min=None,
    multiplicity_max=None,
):
    
    mask = pd.Series(True, index=df.index)


    # ---------------------------------- Discovery filters ----------------------------------
    if mission:
        mask &= df['disc_facility'].notna() & (df['disc_facility'] == mission)

    if discovery_method:
        mask &= df['discoverymethod'].notna() & (df['discoverymethod'] == discovery_method)

    mask &= _apply_range(df, 'disc_year', date_min, date_max)

    mask &= _apply_range(df, 'sy_kepmag', None, kp_max)



    # ----------------------------------- Stellar filters -----------------------------------
    if st_type:
        mask &= df['st_spectype'].notna() & df['st_spectype'].str.upper().str.startswith(st_type.upper())

    mask &= _apply_range(df, 'st_teff', teff_min, teff_max)
    mask &= _apply_range(df, 'st_lum', lum_min, lum_max)
    mask &= _apply_range(df, 'st_met', metallicity_min, metallicity_max)
    mask &= _apply_range(df, 'st_age', age_min, age_max)
    mask &= _apply_range(df, 'st_rad', st_rad_min, st_rad_max)

    if st_rad_err is not None:
        mask &= _snr_mask(df, 'st_rad', st_rad_err)

    if use_fulton_filter:
        mask &= _fulton_mask(df)



    # ----------------------------------- Planet filters -----------------------------------
    mask &= _apply_range(df, 'pl_rade', rade_min, rade_max)
    if rade_err is not None:
        mask &= _snr_mask(df, 'pl_rade', rade_err)

    mask &= _apply_range(df, 'pl_bmasse', mass_min, mass_max)
    if mass_err is not None:
        mask &= _snr_mask(df, 'pl_bmasse', mass_err)

    mask &= _apply_range(df, 'pl_orbsmax', distance_min, distance_max)

    mask &= _apply_range(df, 'pl_dens', density_min, density_max)


    mask &= _apply_range(df, 'pl_orbeccen', None, eccentricity_max)

    mask &= _apply_range(df, 'pl_trandep', transit_depth_min, transit_depth_max)

    mask &= _apply_range(df, 'pl_eqt', eqt_min, eqt_max)

    mask &= _apply_range(df, 'pl_orbper', None, period_max)

    mask &= _apply_range(df, 'pl_imppar', None, impact_param_max)

    if tdyson_min is not None or tdyson_max is not None:
        df['pl_tdyson'] = df['st_teff'] * ((df['st_rad'] * 0.00465) / df['pl_orbsmax'])**0.5
        mask &= _apply_range(df, 'pl_tdyson', tdyson_min, tdyson_max)
    


    # ----------------------------------- System filters -----------------------------------
    mask &= _apply_range(df, 'sy_pnum', multiplicity_min, multiplicity_max)


    return df.loc[mask].reset_index(drop=True)
