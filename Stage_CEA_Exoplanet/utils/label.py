"""
Label Mapping for NEA Exoplanet Dataset
---------------------------------------
This module provides a global dictionary `label_map` for translating
raw column names from the NASA Exoplanet Archive dataset into
human-readable axis labels for visualization and interface use.

This enables consistent labeling across all plots and UI elements
throughout the project.

Usage:
    from utils.label import label_map
    ax.set_xlabel(label_map.get(x_axis, x_axis))

Author: S.WITTMANN & V.RAGNER
Repository: https://github.com/SimonWtmn/Stage_CEA_Exoplanet
"""

label_map = {
    # -------- Planetary Parameters --------
    "pl_rade": r"$\text{Planet Radius }(R_{\oplus})$",
    "pl_bmasse": r"$\text{Planet Mass }(M_{\oplus})$",
    "pl_orbper": r"$\text{Orbital Period (days)}$",
    "pl_eqt": r"$\text{Equilibrium Temperature (K)}$",
    "pl_dens": r"$\text{Planet Density }(\mathrm{g\,cm^{-3}})$",

    # -------- Stellar Parameters --------
    "st_teff": r"$\text{Stellar Effective Temperature (K)}$",
    "st_rad": r"$\text{Stellar Radius }(R_{\odot})$",
    "st_mass": r"$\text{Stellar Mass }(M_{\odot})$",
    "st_lum": r"$\text{Stellar Luminosity }(L_{\odot})$",
    "st_age": r"$\text{Stellar Age (Gyr)}$",
    "st_dens": r"$\text{Stellar Density }(\mathrm{g\,cm^{-3}})$",
    "st_logg": r"$\text{Surface Gravity (log g)}$",
    "st_spectype": r"$\text{Spectral Type}$",

    # -------- System Parameters --------
    "sy_dist": r"$\text{System Distance (pc)}$",
    "sy_vmag": r"$\text{Apparent Magnitude (V)}$",
    "sy_kmag": r"$\text{Apparent Magnitude (K)}$",
    "sy_gaiamag": r"$\text{Gaia G Magnitude}$",
    "sy_tmag": r"$\text{TESS Magnitude}$",
    "sy_kepmag": r"$\text{Kepler Magnitude}$"
}
