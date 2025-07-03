"""
Presets for Filtering Exoplanet Dataset
----------------------------------------
Provides predefined filtering presets for common exoplanet research use-cases.

Author: S.WITTMANN
Repository: https://github.com/SimonWtmn/Exoplot
"""

import pandas as pd
from pathlib import Path

from .filters import apply_filters



# ---------------------------------------- HELPER ----------------------------------------
def load_data(path):
    p = Path(path)
    df = pd.read_csv(p, comment='#')
    df.columns = df.columns.str.strip()
    return df



# ------------------------ All Data Presets ------------------------
DATA_PATHS = {
    'NEA': Path(r"C:\Users\simon\OneDrive\Bureau\Exoplot\data\NEA.csv"),
    'TOI': Path(r"C:\Users\simon\OneDrive\Bureau\Exoplot\data\TOI.csv")
}

ALL_DATA = {
    name: load_data(path) for name, path in DATA_PATHS.items() if Path(path).exists()
}




# ------------------------ Stellar Spectral Type Presets ------------------------
SPECTRAL_TYPE_TEMPERATURES = {
    'O': (30000, None),
    'B': (10000, 30000),
    'A': (7500, 10000),
    'F': (6100, 7500),
    'G': (5300, 6100),
    'K': (3800, 5300),
    'M': (2650, 3800),
    'L': (1450, 2650),
    'T': (700, 1450)
}


def stellar_type_or_teff(df, t):
    if 'st_type' in df.columns:
        return apply_filters(df, st_type=t)
    else:
        teff_min, teff_max = SPECTRAL_TYPE_TEMPERATURES[t]
        return apply_filters(df, teff_min=teff_min, teff_max=teff_max)


STELLAR_PRESETS = {
    f"{t}-type": (lambda df, t=t: stellar_type_or_teff(df, t)) for t in SPECTRAL_TYPE_TEMPERATURES
}




# ------------------------ Survey Mission Presets ------------------------
MISSION_PRESETS = {
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

LIT_PRESETS = {
    'Fulton et al. (2017) Radius Gap':  fulton2017_gap,
    'Luque & Pallé (2022) M-dwarfs':   luque_palle2022_m_dwarfs,
}



# ------------------------ Habitable Zone Presets (Kopparapu et al.(2014) -> https://personal.ems.psu.edu/~jfk4/ruk15/planets/hzcalc.py)------------------------
HZ_NAMES = [
    'Recent Venus',
    'Runaway Greenhouse',
    'Maximum Greenhouse',
    'Early Mars',
    '5ME Runaway Greenhouse',
    '0.1ME Runaway Greenhouse'
]

HZ_SEFF_SUN  = [1.776, 1.107, 0.356, 0.320, 1.188, 0.99]
HZ_A = [2.136e-4, 1.332e-4, 6.171e-5, 5.547e-5, 1.433e-4, 1.209e-4]
HZ_B = [2.533e-8, 1.580e-8, 1.698e-9, 1.526e-9, 1.707e-8, 1.404e-8]
HZ_C = [-1.332e-11, -8.308e-12, -3.198e-12, -2.874e-12, -8.968e-12, -7.418e-12]
HZ_D = [-3.097e-15, -1.931e-15, -5.575e-16, -5.011e-16, -2.084e-15, -1.713e-15]


def hz_boundaries(teff, luminosity):
    t = teff - 5780.0
    hz_au = {}
    if teff < 2600 or teff > 7200:
        for name in HZ_NAMES:
            hz_au[name] = float('nan')
        return hz_au
    for i, name in enumerate(HZ_NAMES):
        seff = (HZ_SEFF_SUN[i] + HZ_A[i] * t + HZ_B[i] * t**2 + HZ_C[i] * t**3 + HZ_D[i] * t**4)
        if seff <= 0:
            hz_au[name] = float('nan')
            continue
        a = (luminosity / seff) ** 0.5
        hz_au[name] = a
    return hz_au


def _hz_filter(df, inner_name, outer_name):
    def is_in_hz(row):
        hz = hz_boundaries(row['st_teff'], row['st_lum'])
        inner = hz[inner_name]
        outer = hz[outer_name]
        if pd.isna(inner) or pd.isna(outer) or pd.isna(row['pl_orbsmax']):
            return False
        return inner <= row['pl_orbsmax'] <= outer

    return df[df.apply(is_in_hz, axis=1)]


def _hz_near(df, boundary_name, rel_tol=0.1):
    """
    Select planets near a given HZ boundary within ±rel_tol (fractional) range.
    E.g., rel_tol=0.1 means ±10%
    """
    def is_near(row):
        hz = hz_boundaries(row['st_teff'], row['st_lum'])
        center = hz[boundary_name]
        if pd.isna(center) or pd.isna(row['pl_orbsmax']):
            return False
        lower = center * (1 - rel_tol)
        upper = center * (1 + rel_tol)
        return lower <= row['pl_orbsmax'] <= upper

    return df[df.apply(is_near, axis=1)]


HZ_PRESETS = {
    'Optimistic HZ (Recent Venus - Early Mars)': lambda df: _hz_filter(df, 'Recent Venus', 'Early Mars'),
    'Conservative HZ (Runaway GH - Max GH)': lambda df: _hz_filter(df, 'Runaway Greenhouse', 'Maximum Greenhouse'),
    'Recent Venus': lambda df: _hz_near(df, 'Recent Venus', rel_tol=0.1),
    'Runaway Greenhouse': lambda df: _hz_near(df, 'Runaway Greenhouse', rel_tol=0.1),
    'Maximum Greenhouse': lambda df: _hz_near(df, 'Maximum Greenhouse', rel_tol=0.1),
    'Early Mars': lambda df: _hz_near(df, 'Early Mars', rel_tol=0.1),
    '5ME Runaway Greenhouse': lambda df: _hz_near(df, '5ME Runaway Greenhouse', rel_tol=0.1),
    '0.1ME Runaway Greenhouse': lambda df: _hz_near(df, '0.1ME Runaway Greenhouse', rel_tol=0.1),
}





# ------------------------ Planetary Category Presets (E. Plávalová & A. Rosaev (2024) -> https://arxiv.org/pdf/2409.09666) ------------------------
PLANET_PRESETS = {
    'Mercury':                (lambda df: apply_filters(df, mass_max=0.22)),
    'Earth':                  (lambda df: apply_filters(df, mass_min=0.22, mass_max=2.2)),
    'Sub Neptune':            (lambda df: apply_filters(df, mass_min=2.2, mass_max=22)),
    'Jupiter':                (lambda df: apply_filters(df, mass_min=127, mass_max=4450)),
    'Dwarf':                  (lambda df: apply_filters(df, masse_min=4450)),

    'Frozen (Temperature)':   (lambda df: apply_filters(df, tdyson_max=250)),
    'Water (Temperature)':    (lambda df: apply_filters(df, tdyson_min=250, tdyson_max=450)),
    'Gaseous (Temperature)':  (lambda df: apply_filters(df, tdyson_min=450, tdyson_max=1000)),
    'Roaster (Temperature)':  (lambda df: apply_filters(df, tdyson_min=1000)),

    'Gaseous (Density)':      (lambda df: apply_filters(df, density_max=0.25)),
    'Water (Density)':        (lambda df: apply_filters(df, density_min=0.25, density_max=2)),
    'Terrestrial (Density)':  (lambda df: apply_filters(df, density_min=2, density_max=6)),
    'Iron (Density)':         (lambda df: apply_filters(df, density_min=6, density_max=13)),
    'Super Dense':            (lambda df: apply_filters(df, density_min=13)),

}



# ------------------------ Custom User Preset ------------------------
def custom_user_preset(
    df: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    return apply_filters(df, **kwargs)

CUSTOM_PRESETS = {
    'Custom Selection': custom_user_preset
}