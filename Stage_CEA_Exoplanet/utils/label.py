"""
Label Mapping for NEA Exoplanet Dataset
---------------------------------------
This module provides a global dictionary `label_map` for translating 
raw column names from the NASA Exoplanet Archive dataset into 
human-readable axis labels for visualization and interface use.

This enables consistent labeling across all plots and UI elements 
throughout the project.

Usage:
    from label_map import label_map
    ax.set_xlabel(label_map.get(x_axis, x_axis))

Author: S.WITTMANN & V.RAGNER
Repository: https://github.com/SimonWtmn/Stage_CEA_Exoplanet
"""

label_map = {
    # -------- Planetary Parameters --------
    "pl_rade": "Planet Radius ($R_{\\oplus}$)",
    "pl_bmasse": "Planet Mass ($M_{\\oplus}$)",
    "pl_orbper": "Orbital Period (days)",
    "pl_eqt": "Equilibrium Temperature (K)",
    "pl_dens": "Planet density ($\\mathrm{g\\,cm^{-3}}$)",

    # -------- Stellar Parameters --------
    "st_teff": "Stellar Effective Temperature (K)",
    "st_rad": "Stellar Radius ($R_{\\odot}$)",
    "st_mass": "Stellar Mass ($M_{\\odot}$)",
    "st_lum": "Stellar Luminosity ($L_{\\odot}$)",
    "st_age": "Stellar Age (Gyr)",
    "st_dens": "Stellar Density (g/cm³)",
    "st_logg": "Surface Gravity (log g)",
    "st_spectype": "Spectral Type",

    # -------- System Parameters --------
    "sy_dist": "System Distance (pc)",
    "sy_vmag": "Apparent Magnitude (V)",
    "sy_kmag": "Apparent Magnitude (K)",
    "sy_gaiamag": "Gaia G Magnitude",
    "sy_tmag": "TESS Magnitude",
    "sy_kepmag": "Kepler Magnitude"
}