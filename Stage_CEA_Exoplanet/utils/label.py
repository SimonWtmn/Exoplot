"""
Label Mapping for NEA Exoplanet Dataset
---------------------------------------
Provides a global dictionary `label_map` for translating raw column names.

Author: S.WITTMANN
Repository: https://github.com/SimonWtmn/Stage_CEA_Exoplanet
"""

label_map = {
    # -------- Planetary Parameters --------
    "pl_rade":       "Planet Radius (R<sub>⊕</sub>)",
    "pl_bmasse":     "Planet Mass (M<sub>⊕</sub>)",
    "pl_orbper":     "Orbital Period (days)",
    "pl_eqt":        "Equilibrium Temperature (K)",
    "pl_dens":       "Planet Density (g.cm<sup>-3</sup>)",

    # -------- Stellar Parameters --------
    "st_teff":       "Stellar Effective Temperature (K)",
    "st_rad":        "Stellar Radius (R<sub>☉</sub>)",
    "st_mass":       "Stellar Mass (M<sub>☉</sub>)",
    "st_lum":        "Stellar Luminosity (L<sub>☉</sub>)",
    "st_age":        "Stellar Age (Gyr)",
    "st_dens":       "Stellar Density (g.cm<sup>-3</sup>)",
    "st_logg":       "Surface Gravity (log g)",
    "st_spectype":   "Spectral Type",

    # -------- System Parameters --------
    "sy_dist":       "System Distance (pc)",
    "sy_vmag":       "Apparent Magnitude (V)",
    "sy_kmag":       "Apparent Magnitude (K)",
    "sy_gaiamag":    "Gaia G Magnitude",
    "sy_tmag":       "TESS Magnitude",
    "sy_kepmag":     "Kepler Magnitude"
}

