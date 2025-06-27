"""
Planetary and Stellar Dataset Filtering Module
------------------------------------------------
This module provides a single entry-point function `apply_filters()`
for filtering confirmed exoplanet from NEA data using a range of stellar,
planetary, and system parameters.

Usage:
    See sample

Author: S.WITTMANN & V.RAGNER
Repository: https://github.com/SimonWtmn/Stage_CEA_Exoplanet
"""

import numpy as np
import pandas as pd

def apply_filters(
    df,

    # Discovery filters
    mission=None, date_min=None, date_max=None,
    kp=None, discovery_method=None,

    # Stellar filters
    st_type=None,
    Teff_min=None, Teff_max=None,
    metallicity_min=None, metallicity_max=None,
    age_min=None, age_max=None,
    st_rad_min=None, st_rad_max=None, st_rad_err=None,
    Fulton_2017=False,

    # Planetary filters
    rade_min=None, rade_max=None, rade_err=None,
    mass_min=None, mass_max=None, mass_err=None,
    density_min=None, density_max=None,
    eccentricity_max=None,
    transit_depth_min=None, transit_depth_max=None,
    eqt_min=None, eqt_max=None,
    P=None, b=None,

    # System filters
    multiplicity_min=None, multiplicity_max=None
    ):


    """Apply a combination of filters to an exoplanet dataset."""
    df_filtered = df.copy()



    # ------------------------ Discovery filters ------------------------
    if mission is not None:
        df_filtered = df_filtered[
            df_filtered['disc_facility'].notna() &
            (df_filtered['disc_facility'] == mission)
        ]

    if discovery_method is not None:
        df_filtered = df_filtered[
            df_filtered['discoverymethod'].notna() &
            (df_filtered['discoverymethod'] == discovery_method)
        ]

    if date_min is not None:
        df_filtered = df_filtered[df_filtered['disc_year'] > date_min]

    if date_max is not None:
        df_filtered = df_filtered[df_filtered['disc_year'] < date_max]

    if kp is not None:
        df_filtered = df_filtered[df_filtered['sy_kepmag'] < kp]




    # ------------------------ Stellar filters ------------------------
    if st_type is not None:
        df_filtered = df_filtered[
            df_filtered['st_spectype'].fillna('').str.upper().str.strip().str.startswith(st_type.upper())
        ]

    if Teff_min is not None:
        df_filtered = df_filtered[df_filtered['st_teff'] > Teff_min]

    if Teff_max is not None:
        df_filtered = df_filtered[df_filtered['st_teff'] < Teff_max]

    if metallicity_min is not None:
        df_filtered = df_filtered[df_filtered['st_met'] > metallicity_min]

    if metallicity_max is not None:
        df_filtered = df_filtered[df_filtered['st_met'] < metallicity_max]

    if age_min is not None:
        df_filtered = df_filtered[df_filtered['st_age'] > age_min]

    if age_max is not None:
        df_filtered = df_filtered[df_filtered['st_age'] < age_max]

    if st_rad_min is not None:
        df_filtered = df_filtered[df_filtered['st_rad'] > st_rad_min]

    if st_rad_max is not None:
        df_filtered = df_filtered[df_filtered['st_rad'] < st_rad_max]

    if st_rad_err is not None:
        mask = df_filtered['st_rad'].notna() & df_filtered['st_raderr1'].notna() & df_filtered['st_raderr2'].notna()
        err = df_filtered[['st_raderr1', 'st_raderr2']].max(axis=1)
        snr = err / df_filtered['st_rad']
        df_filtered = df_filtered[mask & (snr < st_rad_err)]

    if Fulton_2017:
        mask = df_filtered['st_teff'].notna() & df_filtered['st_rad'].notna()
        teff = df_filtered.loc[mask, 'st_teff']
        threshold = 10 ** (0.00025 * ( teff / ( 5000 - 5500) + 0.20) )
        valid = df_filtered.loc[mask, 'st_rad'] > threshold
        df_filtered = df_filtered.loc[mask].loc[valid]



    # ------------------------ Planetary filters ------------------------
    if rade_min is not None:
        df_filtered = df_filtered[df_filtered['pl_rade'] > rade_min]

    if rade_max is not None:
        df_filtered = df_filtered[df_filtered['pl_rade'] < rade_max]

    if rade_err is not None:
        mask = df_filtered['pl_rade'].notna() & df_filtered['pl_radeerr1'].notna() & df_filtered['pl_radeerr2'].notna()
        err = df_filtered[['pl_radeerr1', 'pl_radeerr2']].max(axis=1)
        snr = err / df_filtered['pl_rade']
        df_filtered = df_filtered[mask & (snr < rade_err)]

    if mass_min is not None:
        df_filtered = df_filtered[df_filtered['pl_bmasse'] > mass_min]

    if mass_max is not None:
        df_filtered = df_filtered[df_filtered['pl_bmasse'] < mass_max]

    if mass_err is not None:
        mask = df_filtered['pl_bmasse'].notna() & df_filtered['pl_bmasseerr1'].notna() & df_filtered['pl_bmasseerr2'].notna()
        err = df_filtered[['pl_bmasseerr1', 'pl_bmasseerr2']].max(axis=1)
        snr = err / df_filtered['pl_bmasse']
        df_filtered = df_filtered[mask & (snr < mass_err)]

    if density_min is not None:
        df_filtered = df_filtered[df_filtered['pl_dens'] > density_min]

    if density_max is not None:
        df_filtered = df_filtered[df_filtered['pl_dens'] < density_max]

    if eccentricity_max is not None:
        df_filtered = df_filtered[df_filtered['pl_orbeccen'] < eccentricity_max]

    if transit_depth_min is not None:
        df_filtered = df_filtered[df_filtered['pl_trandep'] > transit_depth_min]

    if transit_depth_max is not None:
        df_filtered = df_filtered[df_filtered['pl_trandep'] < transit_depth_max]

    if eqt_min is not None:
        df_filtered = df_filtered[df_filtered['pl_eqt'] > eqt_min]

    if eqt_max is not None:
        df_filtered = df_filtered[df_filtered['pl_eqt'] < eqt_max]

    if P is not None:
        df_filtered = df_filtered[df_filtered['pl_orbper'] < P]

    if b is not None:
        df_filtered = df_filtered[df_filtered['pl_imppar'] < b]



    # ------------------------ System filters ------------------------
    if multiplicity_min is not None:
        df_filtered = df_filtered[df_filtered['sy_pnum'] >= multiplicity_min]

    if multiplicity_max is not None:
        df_filtered = df_filtered[df_filtered['sy_pnum'] <= multiplicity_max]

    return df_filtered

