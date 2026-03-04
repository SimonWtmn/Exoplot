"""
Constants and Configurations
--------------------------
Provides all static variables, mappings, catalogs, and configuration parameters
used across the Exoplot application.

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot_ENS
"""

from pathlib import Path


# ===========================================================
# Directory Paths
# ===========================================================
# We use pathlib to dynamically find the paths relative to this specific file.
# This prevents "file not found" errors when running Flask from different directories.
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "modules" / "theoretical_models"

# Define the absolute or relative paths for the datasets you use.
DATA_PATHS = {
    'NEA': DATA_DIR / "NEA.csv",
    'TOI': DATA_DIR / "TOI.csv"
}


# ===========================================================
# Label Mapping for UI and Plotting
# ===========================================================
# This dictionary translates raw dataset column names into human-readable HTML-formatted labels for Plotly axes and hover tooltips.
LABEL_MAP = {
    # Planetary properties
    'pl_name': "Planet Name",
    'pl_rade': "Planet Radius [R<sub>⊕</sub>]",
    'pl_radj': "Planet Radius [R<sub>J</sub>]",
    'pl_bmasse': "Planet Mass [M<sub>⊕</sub>]",
    'pl_bmassj': "Planet Mass [M<sub>J</sub>]",
    'pl_bmassprov': "Mass Provenance",
    'pl_dens': "Planet Density [g/cm³]",
    'pl_eqt': "Equilibrium Temperature [K]",
    'pl_insol': "Insolation Flux [S<sub>⊕</sub>]",
    'pl_trandep': "Transit Depth [%]",
    'pl_trandur': "Transit Duration [hours]",
    'pl_occdep': "Occultation Depth [%]",

    # Orbital properties
    'pl_orbper': "Orbital Period [days]",
    'pl_orbsmax': "Semi-Major Axis [AU]",
    'pl_orbeccen': "Eccentricity",
    'pl_orbincl': "Inclination [deg]",
    'pl_imppar': "Impact Parameter",

    # Stellar properties
    'hostname': "Host Star Name",
    'st_spectype': "Spectral Type",
    'st_teff': "Effective Temperature [K]",
    'st_rad': "Stellar Radius [R<sub>⊙</sub>]",
    'st_mass': "Stellar Mass [M<sub>⊙</sub>]",
    'st_met': "Metallicity [dex]",
    'st_metratio': "Metallicity Ratio",
    "st_lum": "Luminosity [log(L<sub>⊙</sub>)]",
    'st_logg': "Surface Gravity [cm/s²]",
    'st_age': "Age [Gyr]",
    'st_dens': "Stellar Density [g/cm³]",
    'st_vsin': "Rotational Velocity [km/s]",
    'st_rotp': "Rotational Period [days]",
    'st_radv': "Radial Velocity [km/s]",

    # System properties
    'sy_snum': "Number of Stars",
    'sy_pnum': "Number of Planets",
    'sy_dist': "Distance [pc]",
    'sy_vmag': "V-band Magnitude",
    'sy_kmag': "Ks Magnitude",
    'sy_gaiamag': "Gaia Magnitude",
    'sy_tmag': "TESS Magnitude",
    'sy_kepmag': "Kepler Magnitude",

    # Discovery & observation
    'discoverymethod': "Discovery Method",
    'disc_year': "Discovery Year",
    'disc_facility': "Discovery Facility",
    'disc_telescope': "Discovery Telescope",
    'disc_instrument': "Discovery Instrument",
    'pl_controv_flag': "Controversial Flag",
    'ttv_flag': "Transit Timing Variations",

    # Coordinates
    'rastr': "RA (sexagesimal)",
    'ra': "RA [deg]",
    'decstr': "Dec (sexagesimal)",
    'dec': "Dec [deg]"
}


# ===========================================================
# Stellar Parameters
# ===========================================================
# Temperature ranges (Min, Max) in Kelvin for each spectral classification.
SPECTRAL_TYPE_TEMPERATURES = {
    'O': (30000, None),     'B': (10000, 30000),     'A': (7500, 10000),
    'F': (6100, 7500),      'G': (5300, 6100),       'K': (3800, 5300),
    'M': (2500, 3800),      'L': (1450, 2500),       'T': (700, 1450)
}


# ===========================================================
# Habitable Zone (HZ) Coefficients
# ===========================================================
# Constants used to compute the polynomial boundaries of the Habitable Zone.
HZ_NAMES = [
    'Recent Venus', 'Runaway Greenhouse', 'Maximum Greenhouse',
    'Early Mars', '5ME Runaway Greenhouse', '0.1ME Runaway Greenhouse'
]

HZ_SEFF_SUN  = [1.776, 1.107, 0.356, 0.320, 1.188, 0.99]
HZ_A = [2.136e-4, 1.332e-4, 6.171e-5, 5.547e-5, 1.433e-4, 1.209e-4]
HZ_B = [2.533e-8, 1.580e-8, 1.698e-9, 1.526e-9, 1.707e-8, 1.404e-8]
HZ_C = [-1.332e-11, -8.308e-12, -3.198e-12, -2.874e-12, -8.968e-12, -7.418e-12]
HZ_D = [-3.097e-15, -1.931e-15, -5.575e-16, -5.011e-16, -2.084e-15, -1.713e-15]


# ===========================================================
# Theoretical Mass-Radius Models
# ===========================================================
# Dictionary mapping an internal key to a tuple containing:
# 1. The exact text filename of the model data
# 2. The human-readable label for the plot legend
MODEL_CATALOG = {
    "zeng_rocky": ("zeng_2019_pure_rock", "Zeng+2019: Pure Rock"),
    "zeng_iron": ("zeng_2019_pure_iron", "Zeng+2019: Pure Iron"),
    "zeng_earth": ("zeng_2019_earth_like", "Zeng+2019: Earth-like"),
    "zeng_2016_20fe": ("zeng_2016_20_Fe", "Zeng+2016: 20% Iron"),
    "Water World": ("MR-Water20_650K_DORN.txt", "Water World: 650K"),
    "marcus_collision": ("marcus_2010_maximum_collision_stripping", "Marcus+2010: Collision"),
}

# Dynamically generating entries for Zeng 2019 Hydrogen envelope models
MODEL_CATALOG.update({
    f"zeng_{pct}h2_{temp}K": (
        f"zeng_2019_{pct}_H2_onto_earth_like_{temp}K",
        f"Zeng+2019: {pct}% H₂ @ {temp}K"
    )
    for pct in [0.1, 0.3, 1, 2, 5]
    for temp in [300, 500, 700, 1000, 2000]
})

# Dynamically generating entries for Zeng 2019 Water world models
MODEL_CATALOG.update({
    f"zeng_{pct}h2o_{temp}K": (
        f"zeng_2019_{pct}_H2O_{temp}K",
        f"Zeng+2019: {pct}% H₂O @ {temp}K"
    )
    for pct in [50, 100]
    for temp in [300, 500, 700, 1000]
})


# ===========================================================
# MCMC & Batman Transit Constants
# ===========================================================
# Formatting labels for the corner plots
MCMC_LABELS = [r"$R_p / R_s$", r"Inclination (deg)", r"$a/R_s$", r"$t_0$"]

# Physical constraints/bounds for the MCMC walkers
# Order: (rp/rs, inclination, a/rs, t0)
MCMC_BOUNDS = [
    (0.001, 0.2),      # Minimum and maximum planetary radius ratio
    (83.0, 89.9),      # Inclination angle in degrees
    (5.0, 15.0),       # Semi-major axis to stellar radius ratio
    (-0.04, 0.04),     # Mid-transit time offset limits
]

# Initial guess vector for optimization before starting MCMC
MCMC_X0 = [0.28, 87.0, 8.0, 0.0]

# Assumptions for the Batman transit model
LIMB_DARKENING_COEFFS = [0.1, 0.3]
LIMB_DARKENING_MODEL = "quadratic"
ECCENTRICITY = 0.0
ARG_PERI = 90.0