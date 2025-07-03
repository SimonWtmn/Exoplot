"""
Label Mapping for NEA Exoplanet Dataset
---------------------------------------
Provides a global dictionary `label_map` for translating raw column names.

Author: S.WITTMANN
Repository: https://github.com/SimonWtmn/Exoplot
"""

label_map = {
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
