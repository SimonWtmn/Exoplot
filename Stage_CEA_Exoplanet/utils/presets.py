"""
Presets for Filtering Exoplanet Dataset
----------------------------------------

This module provides predefined filtering presets for common exoplanet research use-cases.

Usage:
    from utils.presets import load_data, STELLAR_PRESETS, MISSION_PRESETS, LIT_PRESETS, HZ_PRESETS, PLANET_PRESETS
    df = load_data("path/to/confirmed.csv")
    df_g = STELLAR_PRESETS['G-type Stars'](df)
    df_hz = HZ_PRESETS['Conservative HZ (Kopparapu+)'](df)
    df_se = PLANET_PRESETS['Super-Earths'](df)

Author: S.WITTMANN & V.RAGNER
Repository: https://github.com/SimonWtmn/Stage_CEA_Exoplanet
"""

import pandas as pd
from pathlib import Path
from typing import Union, Callable, Dict
from .filters import apply_filters



def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the confirmed exoplanet dataset from CSV.
    """
    p = Path(path)
    df = pd.read_csv(p, comment='#')
    df.columns = df.columns.str.strip()
    return df





# ------------------------ Stellar Spectral Type Presets ------------------------
STELLAR_PRESETS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    f"{t}-type Stars": (lambda df, t=t: apply_filters(df, st_type=t))
    for t in ['O', 'B', 'A', 'F', 'G', 'K', 'M']
}





# ------------------------ Survey Mission Presets ------------------------
MISSION_PRESETS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    'Kepler Mission':    (lambda df: apply_filters(df, mission='Kepler')),
    'K2 Campaign':       (lambda df: apply_filters(df, mission='K2')),
    'TESS Survey':       (lambda df: apply_filters(df, mission='Transiting Exoplanet Survey Satellite (TESS)')),
    'CoRoT Survey':      (lambda df: apply_filters(df, mission='CoRoT')),
    'CHEOPS Mission':    (lambda df: apply_filters(df, mission='CHaracterising ExOPlanets Satellite (CHEOPS)')),
    'JWST Observations': (lambda df: apply_filters(df, mission='James Webb Space Telescope (JWST)')),
    'Spitzer Archive':   (lambda df: apply_filters(df, mission='Spitzer Space Telescope')),
    'Hubble Archive':    (lambda df: apply_filters(df, mission='Hubble Space Telescope')),
    'Gaia Crossmatch':   (lambda df: apply_filters(df, mission='European Space Agency (ESA) Gaia Satellite')),
    'WISE Survey':       (lambda df: apply_filters(df, mission='Wide-field Infrared Survey Explorer (WISE)'))
}





# ------------------------ Literature Sample Presets ------------------------

def fulton2017_full(df: pd.DataFrame) -> pd.DataFrame:
    return apply_filters(
        df,
        mission='Kepler',
        date_max=2017,
        teff_min=4700,
        teff_max=6500
    )


def fulton2017_gap(df: pd.DataFrame) -> pd.DataFrame:
    return apply_filters(
        df,
        mission='Kepler',
        date_max=2017,
        kp_max=14.2,
        teff_min=4700,
        teff_max=6500,
        use_fulton_filter=True,
        st_rad_max=2,
        impact_param_max=0.7,
        rade_max=5
    )


def luque_palle2022_m_dwarfs(df: pd.DataFrame) -> pd.DataFrame:
    return apply_filters(
        df,
        date_max=2022,
        st_type='M',
        rade_max=4,
        rade_err=0.08,
        mass_max=20,
        mass_err=0.25
    )



LIT_PRESETS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    'Fulton et al. (2017) Full Sample': fulton2017_full,
    'Fulton et al. (2017) Radius Gap':  fulton2017_gap,
    'Luque & Pallé (2022) M-dwarfs':   luque_palle2022_m_dwarfs,
}





# ------------------------ Habitable Zone Presets ------------------------

def conservative_hz(df: pd.DataFrame) -> pd.DataFrame:
    subset = apply_filters(df, rade_max=1.5)
    return subset[subset['pl_insol'].between(0.356, 1.11)].reset_index(drop=True)


def optimistic_hz(df: pd.DataFrame) -> pd.DataFrame:
    subset = apply_filters(df, rade_max=2.0)
    return subset[subset['pl_insol'].between(0.22, 1.90)].reset_index(drop=True)



HZ_PRESETS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    'Conservative HZ (Kopparapu+)': conservative_hz,
    'Optimistic HZ (Kopparapu+)':  optimistic_hz,
}





# ------------------------ Planetary Category Presets ------------------------

def super_earths(df: pd.DataFrame) -> pd.DataFrame:
    return apply_filters(df, rade_min=1.0, rade_max=2.0)


def mini_neptunes(df: pd.DataFrame) -> pd.DataFrame:
    return apply_filters(df, rade_min=2.0, rade_max=4.0)

def gas_giants(df: pd.DataFrame) -> pd.DataFrame:
    return apply_filters(df, rade_min=4.0)

def hot_jupiters(df: pd.DataFrame) -> pd.DataFrame:
    return apply_filters(df, rade_min=6.0, period_max=10)

def multi_planet_systems(df: pd.DataFrame) -> pd.DataFrame:
    return apply_filters(df, multiplicity_min=2)

def high_density_planets(df: pd.DataFrame) -> pd.DataFrame:
    return apply_filters(df, density_min=5.0)

def low_density_planets(df: pd.DataFrame) -> pd.DataFrame:

    return apply_filters(df, density_max=1.0)



PLANET_PRESETS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    'Super-Earths':           super_earths,
    'Mini-Neptunes':          mini_neptunes,
    'Gas Giants':             gas_giants,
    'Hot Jupiters':           hot_jupiters,
    'Multi-planet Systems':   multi_planet_systems,
    'High-Density Planets':   high_density_planets,
    'Low-Density Planets':    low_density_planets,
}





# ------------------------ Custom User Preset ------------------------

def custom_user_preset(
    df: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Create a bespoke filter by passing any arguments through to `apply_filters()`.
    E.g., `custom_user_preset(df, mission='TESS', rade_max=2.5)`.
    """
    return apply_filters(df, **kwargs)

CUSTOM_PRESETS: Dict[str, Callable[..., pd.DataFrame]] = {
    'Custom Selection': custom_user_preset
}
