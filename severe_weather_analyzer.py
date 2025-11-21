#!/usr/bin/env python
# severe_weather_analyzer.py

"""
Severe Weather Analysis Tool (v81 - Domain Options)
Author: Gemini
Date: November 3, 2025

v80: 1. Fixed 'imageio.v3' has no attribute 'mimsave' error.
     2. Reverted import to use standard 'import imageio'
        to match the 'imageio.mimsave()' v2 API call.
v81: 1. Ported domain selection from mrms_analyzer.py (v24).
     2. Added REGIONAL_DOMAIN and PADUCAH_CWA_DOMAIN.
     3. Added get_domain_selection() function.
"""

import os
import sys
# v65: Removed argparse
import warnings
from datetime import datetime, timedelta
# v78: Added multiprocessing
import multiprocessing
from functools import partial
import glob

import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# v78: Re-added tqdm
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: 'tqdm' package not found. Progress bar will be disabled.", file=sys.stderr)
    print("  To install: conda install -c conda-forge tqdm", file=sys.stderr)

# v75: Added imageio for GIF creation
# --- v80: Revert to standard v2 import ---
try:
    import imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    # We will warn the user later if they try to make a GIF
# --- End v80 FIX ---

# Import the new, powerful tool
try:
    from herbie import Herbie
except ImportError:
    print("-------------------------------------------------------------------", file=sys.stderr)
    print("FATAL ERROR: `herbie-data` package not found.", file=sys.stderr)
    print("Please install it in your conda environment:", file=sys.stderr)
    print("  conda install -c conda-forge herbie-data", file=sys.stderr)
    print("-------------------------------------------------------------------", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---

# v67: Removed prompt, this is now the locked-in base directory
BASE_OUTPUT_DIR = "./output_plots"

# v35: Recalculated domain to be centered on Paducah (-88.6, 37.08)
#      and "zoomed out" by 20% (approx 3.6 deg lat, 4.4 deg lon)
# v81: Ported domain options from mrms_analyzer.py (v24)
REGIONAL_DOMAIN = {
    'name': 'Regional (Default)',
    'west': -92.2,
    'east': -85.0,
    'south': 34.68,
    'north': 39.48
}

PADUCAH_CWA_DOMAIN = {
    'name': 'NWS Paducah CWA (Zoomed)',
    'west': -92.0,
    'east': -86.2,
    'south': 35.7,
    'north': 39.0
}

ALL_DOMAINS = [REGIONAL_DOMAIN, PADUCAH_CWA_DOMAIN]

# --- Variable Definitions ---
PLOT_VARS_CONFIG = {
    # --- v52: Removed MLCAPE and MLCIN ---
    'SBCAPE': {
        'long_name': 'Surface-Based CAPE',
        'units': 'J/kg',
        'herbie_search_str': ':CAPE:surface',
        'grib_name': ['cape'],  # Generic Name
        'product': 'sfc',
        'grib_level': {'surface': 0},
        'levels': np.arange(500, 4001, 250),
        'cmap': 'hot_r',
        'extend': 'max',
        'plot_type': 'scalar'  # v74: Added type
    },
    'SBCIN': {
        'long_name': 'Surface-Based CIN',
        'units': 'J/kg',
        'herbie_search_str': ':CIN:surface',
        'grib_name': ['cin'],  # Generic Name
        'product': 'sfc',
        'grib_level': {'surface': 0},
        'levels': np.arange(-200, 1, 10),
        'cmap': 'Blues_r',
        'extend': 'both',
        'plot_type': 'scalar'  # v74: Added type
    },
    'SRH01': {
        'long_name': '0-1 km Storm Relative Helicity',
        'units': 'm^2/s^2',
        'herbie_search_str': ':HLCY:1000-0 m above ground',
        'grib_name': ['hlcy'],
        'product': 'sfc',
        'grib_level': {'heightAboveGroundLayer': 1000.0},
        'levels': np.arange(100, 601, 50),
        'cmap': 'BuGn',
        'extend': 'max',
        'plot_type': 'scalar'  # v74: Added type
    },
    'SRH03': {
        'long_name': '0-3 km Storm Relative Helicity',
        'units': 'm^2/s^2',
        'herbie_search_str': ':HLCY:3000-0 m above ground',
        'grib_name': ['hlcy'],
        'product': 'sfc',
        'grib_level': {'heightAboveGroundLayer': 3000.0},
        'levels': np.arange(150, 751, 50),
        'cmap': 'BuGn',
        'extend': 'max',
        'plot_type': 'scalar'  # v74: Added type
    },
    'TEMP2M': {
        'long_name': '2-m Temperature',
        'units': '°F',
        'herbie_search_str': ':TMP:2 m',
        'grib_name': ['t2m', 't'],
        'product': 'sfc',
        'grib_level': {'heightAboveGround': 2.0},
        'levels': np.arange(0, 101, 5),
        'cmap': 'coolwarm',
        'extend': 'both',
        'plot_type': 'scalar'  # v74: Added type
    },
    'DEW2M': {
        'long_name': '2-m Dewpoint',
        'units': '°F',
        'herbie_search_str': ':DPT:2 m',
        'grib_name': ['d2m', 'dpt'],
        'product': 'sfc',
        'grib_level': {'heightAboveGround': 2.0},
        'levels': np.arange(0, 81, 2),
        'cmap': 'Greens',
        'extend': 'both',
        'plot_type': 'scalar'  # v74: Added type
    },
    'MSLP': {
        'long_name': 'Mean Sea Level Pressure',
        'units': 'hPa',
        'herbie_search_str': ':PRMSL:|:MSLET:|:MSLMA:',
        'grib_name': ['prmsl', 'msl', 'mslet', 'mslma'],
        'product': 'sfc',
        'grib_level': {'meanSea': 0},
        'levels': np.arange(980, 1041, 2),
        'cmap': 'RdYlBu_r',
        'extend': 'both',
        'plot_type': 'scalar'  # v74: Added type
    },
    # --- Computed Variables (No 'product' key) ---
    'SHR06': {
        'long_name': '0-6 km Bulk Wind Shear',
        'units': 'kts',
        'grib_name': ['SHR06'],
        'levels': np.arange(30, 81, 5),
        'cmap': 'Oranges',
        'extend': 'max',
        'plot_type': 'scalar'  # v74: Added type
    },
    'SCP': {
        'long_name': 'Supercell Composite Parameter (Effective)',
        'units': '',
        'grib_name': ['SCP'],
        'levels': np.arange(1, 21, 1),
        'cmap': 'magma',
        'extend': 'max',
        'plot_type': 'scalar'  # v74: Added type
    },
    'STP': {
        'long_name': 'Significant Tornado Parameter (Fixed Layer)',
        'units': '',
        'grib_name': ['STP'],
        'levels': np.arange(0.5, 8.1, 0.5),
        'cmap': 'cividis',
        'extend': 'max',
        'plot_type': 'scalar'  # v74: Added type
    },
    # --- v61: Updated Shear Vector Plots ---
    'SHR01_VEC': {
        'long_name': '0-1 km Bulk Shear Vectors (850mb-10m proxy)',
        'units': 'kts',
        'grib_name': ['U_SHR01', 'V_SHR01'],
        'plot_type': 'vector'  # v74: Type updated
    },
    'SHR03_VEC': {
        'long_name': '0-3 km Bulk Shear Vectors (700mb-10m proxy)',
        'units': 'kts',
        'grib_name': ['U_SHR03', 'V_SHR03'],
        'plot_type': 'vector'  # v74: Type updated
    },
    # --- v70: New Wind Plots ---
    'WIND_10M': {
        'long_name': '10-m Wind',
        'units': 'kts',
        'grib_name': ['u10', 'v10'],
        'plot_type': 'vector'  # v74: Type updated
    },
    'WIND_COMBO_LOW': {
        'long_name': 'Low-Level Winds (925/850/700mb)',
        'units': 'kts',
        'grib_name': ['u', 'v'],
        'plot_type': 'multi_vector',  # v74: Type updated
        'levels': [925, 850, 700],
        'colors': ['#0000FF', '#008000', '#FF0000']
    },
}

# Internal variables needed *only* for computation
_INTERNAL_VARS_CONFIG = {
    # --- SFC (Surface) file ---
    '_U_10M': {'herbie_search_str': ':UGRD:10 m', 'grib_name': ['u10'], 'product': 'sfc'},
    '_V_10M': {'herbie_search_str': ':VGRD:10 m', 'grib_name': ['v10'], 'product': 'sfc'},
    '_SP': {'herbie_search_str': ':PRES:surface|:SP:surface', 'grib_name': ['pres', 'sp'], 'product': 'sfc'},
    # v27: More robust

    # --- PRS (Pressure) file ---
    '_T_PL': {'herbie_search_str': r':TMP:\d+ mb', 'grib_name': ['t'], 'product': 'prs'},
    '_DPT_PL': {'herbie_search_str': r':DPT:\d+ mb', 'grib_name': ['dpt'], 'product': 'prs'},
    '_U_PL': {'herbie_search_str': r':UGRD:\d+ mb', 'grib_name': ['u'], 'product': 'prs'},
    '_V_PL': {'herbie_search_str': r':VGRD:\d+ mb', 'grib_name': ['v'], 'product': 'prs'},

    # --- v61: Removed AGL wind config ---
}


# --- Core Functions ---

def _load_herbie_product(dt, product, search_regex, output_dir):
    """
    Helper function to load one product (sfc or prs) from Herbie.
    v31: Added retry logic for corrupt GRIB files.
    """
    H = None
    for attempt in range(2):  # Try twice
        try:
            H = Herbie(
                dt,
                model='hrrr',
                product=product,
                fxx=0,
                save_dir=os.path.join(output_dir, 'herbie_cache'),  # v65: Uses event-specific dir
                priority=['aws', 'nomads', 'google', 'azure']
            )
        except Exception as e:
            print(f"Error: Herbie (HRRR) could not find {product} file for {dt}. {e}", file=sys.stderr)
            return None

        # v78: Be less verbose inside a parallel loop
        # print(f"Herbie is searching HRRR {product} file for: {search_regex}")
        try:
            # v78: Turn off verbose logging
            ds = H.xarray(search_regex, verbose=False)

            if isinstance(ds, list):
                # print(f"Herbie returned {len(ds)} datasets for {product}. Merging them...")
                ds = xr.merge(ds, compat='override')

            return ds  # Success

        except Exception as e:
            # --- v31: Retry Logic ---
            if "End of resource reached" in str(e):
                print(f"Warning: Corrupt HRRR GRIB file detected for {product} on attempt {attempt + 1}.",
                      file=sys.stderr)

                # --- v38 FIX ---
                grib_path = H.grib
                idx_path = H.idx
                # --- END FIX ---

                if grib_path and os.path.exists(grib_path):
                    print(f"Deleting corrupt GRIB: {grib_path}", file=sys.stderr)
                    os.remove(grib_path)
                if idx_path and os.path.exists(idx_path):
                    print(f"Deleting corrupt IDX: {idx_path}", file=sys.stderr)
                    os.remove(idx_path)

                if attempt == 0:
                    print(f"Retrying download for {dt}...", file=sys.stderr)
                    continue  # Go to the next loop iteration to retry
            # --- End v31 Logic ---

            if "Found 0 grib messages" in str(e):
                print(f"Warning: Herbie found 0 messages for '{search_regex}' in HRRR {product} file for {dt}.",
                      file=sys.stderr)
                return None

            print(f"Error: Herbie's .xarray() method failed for HRRR {product} at {dt}.", file=sys.stderr)
            print(f"Details: {e}", file=sys.stderr)
            return None  # Failed after all attempts or for other reason

    print(f"Error: Failed to load HRRR {product} data for {dt} after 2 attempts.", file=sys.stderr)
    return None


def _find_var_in_ds(ds, grib_names_list):
    """Helper to find the first matching grib_name in the dataset."""
    for name in grib_names_list:
        if name in ds:
            return name
    return None


def load_data_with_herbie(dt, vars_to_load_list, output_dir):  # v74: param changed
    """
    Uses Herbie to find, download, and load HRRR data for a specific time.
    v74: Changed to accept a simple list of vars to load.
    """
    # v78: Quieter logging for parallel runs
    # print(f"Initializing Herbie for {dt}...")

    # 1. Build our "shopping lists"
    sfc_searches = []
    prs_searches = []

    # Combine plot vars and internal vars configs
    full_config = {**PLOT_VARS_CONFIG, **_INTERNAL_VARS_CONFIG}

    # v74: Use the pre-compiled list of variables to load
    vars_to_load = set(vars_to_load_list)

    # --- v66: Add plot-variable dependencies ---
    if 'SCP' in vars_to_load:
        vars_to_load.update(['SBCAPE', 'SRH03', 'SHR06'])

    if 'STP' in vars_to_load:
        vars_to_load.update(['SBCAPE', 'SBCIN', 'SRH01', 'SHR06', 'TEMP2M', 'DEW2M'])

    # --- v70: Add new dependencies ---
    if 'WIND_10M' in vars_to_load:
        vars_to_load.update(['_U_10M', '_V_10M'])

    if 'WIND_COMBO_LOW' in vars_to_load:
        vars_to_load.update(['_U_PL', '_V_PL'])
    # --- End v70 ---

    # v66: We must always check comp_vars *after* resolving dependencies
    comp_vars = [v for v in vars_to_load if v in ['SHR06', 'SCP', 'STP', 'SHR01_VEC', 'SHR03_VEC']]
    if comp_vars:
        vars_to_load.update(_INTERNAL_VARS_CONFIG.keys())

    # Build the regex strings
    for var in vars_to_load:
        if var in full_config:
            config = full_config[var]

            if 'product' in config:
                if config['product'] == 'sfc':
                    sfc_searches.append(config['herbie_search_str'])
                elif config['product'] == 'prs':
                    prs_searches.append(config['herbie_search_str'])

    sfc_searches = list(set(sfc_searches))
    prs_searches = list(set(prs_searches))
    datasets_to_merge = []

    # 2. Load MAIN SFC (Surface) variables
    if sfc_searches:
        sfc_regex = "|".join(sfc_searches)
        ds_sfc = _load_herbie_product(dt, 'sfc', sfc_regex, output_dir)
        if ds_sfc:
            datasets_to_merge.append(ds_sfc)
        else:
            print(f"Warning: Failed to load HRRR 'sfc' data for {dt}.", file=sys.stderr)

    # 3. Load PRS (Pressure) variables
    if prs_searches:
        prs_regex = "|".join(prs_searches)
        ds_prs = _load_herbie_product(dt, 'prs', prs_regex, output_dir)
        if ds_prs:
            datasets_to_merge.append(ds_prs)
        else:
            print(f"Warning: Failed to load HRRR 'prs' data for {dt}. Derived variables may fail.", file=sys.stderr)

    # 4. v61: Removed separate AGL loading block

    # 5. Merge all datasets
    if not datasets_to_merge:
        print(f"Error: No HRRR data was loaded from Herbie for {dt}.", file=sys.stderr)
        return None

    try:
        # v78: Quieter logging
        # print("Merging all HRRR datasets...")
        ds_final = xr.merge(datasets_to_merge, compat='override')
        # print("Herbie successfully loaded and merged all HRRR data.")
        return ds_final
    except Exception as e:
        print(f"Error merging HRRR datasets for {dt}: {e}", file=sys.stderr)
        return None


# --- v52: Removed _calc_mlcape_cin_profile helper function ---


def compute_derived_vars(ds):
    """
    Computes derived meteorological variables using MetPy.
    Adds new variables to the xarray Dataset.
    """
    # v78: Quieter logging
    # print("Computing derived variables...")
    try:
        # Get data with units
        with warnings.catch_warnings():
            # Suppress warnings from MetPy about unit conversion
            warnings.simplefilter("ignore", category=UserWarning)

            ds = ds.metpy.parse_cf()  # Parse CRS, add units

            # --- Find required variables ---
            # v26: Use helper to check for multiple names
            t_name = _find_var_in_ds(ds, _INTERNAL_VARS_CONFIG['_T_PL']['grib_name'])
            dpt_name = _find_var_in_ds(ds, _INTERNAL_VARS_CONFIG['_DPT_PL']['grib_name'])
            u_pl_name = _find_var_in_ds(ds, _INTERNAL_VARS_CONFIG['_U_PL']['grib_name'])
            v_pl_name = _find_var_in_ds(ds, _INTERNAL_VARS_CONFIG['_V_PL']['grib_name'])
            sp_name = _find_var_in_ds(ds, _INTERNAL_VARS_CONFIG['_SP']['grib_name'])
            cape_name = _find_var_in_ds(ds, PLOT_VARS_CONFIG['SBCAPE']['grib_name'])
            cin_name = _find_var_in_ds(ds, PLOT_VARS_CONFIG['SBCIN']['grib_name'])
            hlcy_name = _find_var_in_ds(ds, PLOT_VARS_CONFIG['SRH01']['grib_name'])
            u10_name = _find_var_in_ds(ds, _INTERNAL_VARS_CONFIG['_U_10M']['grib_name'])
            v10_name = _find_var_in_ds(ds, _INTERNAL_VARS_CONFIG['_V_10M']['grib_name'])

            # v61: Removed AGL name finders

            # v29: Use robust logic for t2m/d2m
            t2m_name = _find_var_in_ds(ds, PLOT_VARS_CONFIG['TEMP2M']['grib_name'])
            d2m_name = _find_var_in_ds(ds, PLOT_VARS_CONFIG['DEW2M']['grib_name'])

            # Check for 3D variables
            # v70: Allow this to be missing if not needed
            has_3d_data = all([t_name, dpt_name, u_pl_name, v_pl_name])
            if not has_3d_data:
                print(
                    f"Warning: Missing generic 3D Temp, Dewpt, or Wind variables for {ds.time.values}. Dependent calculations will be skipped.",
                    file=sys.stderr)

            if not sp_name:
                print(f"Error: Missing Surface Pressure for {ds.time.values}. Cannot compute derived vars.",
                      file=sys.stderr)
                return ds

            # --- Get Data Arrays ---
            sp = ds[sp_name]

            # --- SELECT 3D variables (if they exist) ---
            if has_3d_data:
                p = ds.isobaricInhPa
                T = ds[t_name].metpy.sel(isobaricInhPa=slice(1000, 100))
                Td = ds[dpt_name].metpy.sel(isobaricInhPa=slice(1000, 100))
                u_pl = ds[u_pl_name].metpy.sel(isobaricInhPa=slice(1000, 100))
                v_pl = ds[v_pl_name].metpy.sel(isobaricInhPa=slice(1000, 100))

                # Ensure pressure is in descending order for MetPy
                if p[0] < p[-1]:
                    p = p[::-1]
                    T = T.isel(isobaricInhPa=slice(None, None, -1))
                    Td = Td.isel(isobaricInhPa=slice(None, None, -1))
                    u_pl = u_pl.isel(isobaricInhPa=slice(None, None, -1))
                    v_pl = v_pl.isel(isobaricInhPa=slice(None, None, -1))
            else:
                # Set to None so downstream calcs know to skip
                u_pl = None
                v_pl = None

            # 1. 0-6 km Bulk Shear (SHR06)
            if u10_name and v10_name and u_pl is not None:
                # print("Calculating 0-6 km Bulk Shear (500mb-10m proxy)...")
                u_10m_shr6 = ds[u10_name].metpy.unit_array
                v_10m_shr6 = ds[v10_name].metpy.unit_array

                u_500 = u_pl.metpy.sel(isobaricInhPa=500).metpy.unit_array
                v_500 = v_pl.metpy.sel(isobaricInhPa=500).metpy.unit_array

                u_shear = u_500 - u_10m_shr6
                v_shear = v_500 - v_10m_shr6

                shr06_mag = mpcalc.wind_speed(u_shear, v_shear).to('kts')

                ds['SHR06'] = (('y', 'x'), shr06_mag.magnitude)
                ds['SHR06'].attrs = {'long_name': '0-6 km Bulk Shear (500mb-10m proxy)', 'units': 'knots'}
            else:
                if 'SHR06' not in ds:
                    print(f"Skipping SHR06 calculation for {ds.time.values} (missing 10m or 3D wind).", file=sys.stderr)

            # --- v52: Removed MLCAPE/MLCIN calculation block ---

            # --- v61: New Shear Vector Calculation (using 850/700mb) ---
            # print("Calculating 0-1km Shear Vectors (850mb-10m proxy)...")
            if not all([u10_name, v10_name]) or u_pl is None or v_pl is None:
                if 'SHR01_VEC' not in ds:  # v78: Check if it was requested
                    print(
                        f"Error: Missing variables for 0-1km shear vector calculation for {ds.time.values}. Skipping.",
                        file=sys.stderr)
            else:
                try:
                    u_10m = ds[u10_name].metpy.unit_array.to('kts')
                    v_10m = ds[v10_name].metpy.unit_array.to('kts')

                    # Select 850mb wind from the 3D pressure-level data
                    u_850 = u_pl.metpy.sel(isobaricInhPa=850).metpy.unit_array.to('kts')
                    v_850 = v_pl.metpy.sel(isobaricInhPa=850).metpy.unit_array.to('kts')

                    u_shr01 = u_850 - u_10m
                    v_shr01 = v_850 - v_10m

                    ds['U_SHR01'] = (('y', 'x'), u_shr01.magnitude)
                    ds['U_SHR01'].attrs = {'long_name': '0-1 km Shear U-Component (850mb-10m)', 'units': 'knots'}
                    ds['V_SHR01'] = (('y', 'x'), v_shr01.magnitude)
                    ds['V_SHR01'].attrs = {'long_name': '0-1 km Shear V-Component (850mb-10m)', 'units': 'knots'}
                    # print("0-1km shear vector components calculated.")
                except Exception as e:
                    print(f"Error calculating 0-1km shear vectors for {ds.time.values}: {e}", file=sys.stderr)

            # print("Calculating 0-3km Shear Vectors (700mb-10m proxy)...")
            if not all([u10_name, v10_name]) or u_pl is None or v_pl is None:
                if 'SHR03_VEC' not in ds:  # v78: Check if it was requested
                    print(
                        f"Error: Missing variables for 0-3km shear vector calculation for {ds.time.values}. Skipping.",
                        file=sys.stderr)
            else:
                try:
                    u_10m = ds[u10_name].metpy.unit_array.to('kts')
                    v_10m = ds[v10_name].metpy.unit_array.to('kts')

                    # Select 700mb wind from the 3D pressure-level data
                    u_700 = u_pl.metpy.sel(isobaricInhPa=700).metpy.unit_array.to('kts')
                    v_700 = v_pl.metpy.sel(isobaricInhPa=700).metpy.unit_array.to('kts')

                    u_shr03 = u_700 - u_10m
                    v_shr03 = v_700 - v_10m

                    ds['U_SHR03'] = (('y', 'x'), u_shr03.magnitude)
                    ds['U_SHR03'].attrs = {'long_name': '0-3 km Shear U-Component (700mb-10m)', 'units': 'knots'}
                    ds['V_SHR03'] = (('y', 'x'), v_shr03.magnitude)
                    ds['V_SHR03'].attrs = {'long_name': '0-3 km Shear V-Component (700mb-10m)', 'units': 'knots'}
                    # print("0-3km shear vector components calculated.")
                except Exception as e:
                    print(f"Error calculating 0-3km shear vectors for {ds.time.values}: {e}", file=sys.stderr)
            # --- End v61 ---

            # 2. Supercell Composite (SCP)
            if not all([cape_name, hlcy_name, 'SHR06' in ds]):  # v66: Check for SHR06
                if 'SCP' not in ds:  # v78: Check if it was requested
                    print(f"Skipping SCP calculation for {ds.time.values} due to missing CAPE, HLCY, or SHR06.",
                          file=sys.stderr)
            else:
                # print("Calculating Supercell Composite Parameter...")
                srh03_var = PLOT_VARS_CONFIG['SRH03']
                sbcape = ds[cape_name].where(ds.surface == 0, drop=True).squeeze()

                if 'heightAboveGroundLayer' not in ds[hlcy_name].coords:
                    print(
                        f"Error: 'heightAboveGroundLayer' coord not found on 'hlcy' var for {ds.time.values}. Skipping SCP/STP.",
                        file=sys.stderr)
                    return ds  # bail out

                srh03 = ds[hlcy_name].where(ds.heightAboveGroundLayer == 3000.0, drop=True).squeeze()
                shr06_da = ds.SHR06.metpy.convert_units('m/s')

                sbcape_term = sbcape / 1000.0
                srh03_term = srh03 / 50.0
                shr06_term = shr06_da / 20.0

                sbcape_term = sbcape_term.where(sbcape_term > 0, 0)
                srh03_term = srh03_term.where(srh03_term > 0, 0)
                shr06_term = shr06_term.where(shr06_term > 0, 0)

                scp = sbcape_term * srh03_term * shr06_term

                ds['SCP'] = (('y', 'x'), scp.values)
                ds['SCP'].attrs = {'long_name': 'Supercell Composite Parameter', 'units': ''}

            # 3. Significant Tornado Parameter (STP)
            if not all([cape_name, cin_name, hlcy_name, t2m_name, d2m_name, 'SHR06' in ds]):  # v66: Check for SHR06
                if 'STP' not in ds:  # v78: Check if it was requested
                    print(f"Skipping STP calculation for {ds.time.values} due to missing variables.", file=sys.stderr)
            else:
                # print("Calculating Significant Tornado Parameter...")
                srh01_var = PLOT_VARS_CONFIG['SRH01']

                srh01 = ds[hlcy_name].where(ds.heightAboveGroundLayer == 1000.0, drop=True).squeeze()
                sbcin = ds[cin_name].where(ds.surface == 0, drop=True).squeeze()

                t2m_da = ds[t2m_name]
                d2m_da = ds[d2m_name]

                # v30: Check if 't2m'/'d2m' is 3D (from 't'/'dpt') or 2D
                if 'heightAboveGround' in t2m_da.coords and 'heightAboveGround' in t2m_da.dims:
                    t2m_da = t2m_da.sel(heightAboveGround=2.0, method='nearest')
                if 'heightAboveGround' in d2m_da.coords and 'heightAboveGround' in d2m_da.dims:
                    d2m_da = d2m_da.sel(heightAboveGround=2.0, method='nearest')

                t2m = t2m_da.metpy.unit_array.to('degC')
                td2m = d2m_da.metpy.unit_array.to('degC')
                sp_ua = sp.metpy.unit_array

                lcl_pressure, lcl_temp = mpcalc.lcl(sp_ua, t2m, td2m)

                lcl_hgt_m = (sp_ua - lcl_pressure).to('hPa').magnitude * 8.3

                lcl_term = (2000.0 - lcl_hgt_m) / 1000.0
                lcl_term = xr.DataArray(lcl_term, coords=sbcape.coords, dims=["y", "x"])
                lcl_term = lcl_term.where(lcl_term > 0, 0)
                lcl_term = lcl_term.where(lcl_term < 1, 1)

                cin_term = (sbcin + 200) / 150
                cin_term = cin_term.where(cin_term > 0, 0)
                cin_term = cin_term.where(cin_term < 1, 1)

                sbcape_term_stp = sbcape / 1500.0
                srh01_term_stp = srh01 / 150.0

                sbcape_term_stp = sbcape_term_stp.where(sbcape_term_stp > 0, 0)
                srh01_term_stp = srh01_term_stp.where(srh01_term_stp > 0, 0)

                # Note: shr06_term is already defined from the SCP calculation
                shr06_term = ds.SHR06.metpy.convert_units('m/s') / 20.0
                shr06_term = shr06_term.where(shr06_term > 0, 0)

                stp = sbcape_term_stp * lcl_term * srh01_term_stp * shr06_term * cin_term

                ds['STP'] = (('y', 'x'), stp.values)
                ds['STP'].attrs = {'long_name': 'Significant Tornado Parameter', 'units': ''}

            # print("Successfully computed derived variables.")
            return ds

    except Exception as e:
        print(f"Error computing derived variables for {ds.time.values}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        print("This may be due to missing input variables in the GRIB file.", file=sys.stderr)
        return ds  # Return the partially processed dataset


# --- v74: Refactored plot_variable ---
def plot_variable(ds, base_var_name, overlay_var_name, domain, output_dir, dt, plot_name):
    """
    Plots a scalar variable and (optionally) a vector overlay on a map.
    v74: Refactored to accept base and overlay vars.
    """

    # --- 1. Get Base Plot Config ---
    if base_var_name not in PLOT_VARS_CONFIG:
        print(f"Warning: Plotting config for base var '{base_var_name}' not found. Skipping.", file=sys.stderr)
        return

    base_config = PLOT_VARS_CONFIG[base_var_name]
    base_plot_type = base_config.get('plot_type', 'scalar')

    # --- 2. Get Overlay Plot Config (if it exists) ---
    overlay_config = None
    overlay_plot_type = None
    if overlay_var_name:
        if overlay_var_name not in PLOT_VARS_CONFIG:
            print(f"Warning: Plotting config for overlay var '{overlay_var_name}' not found. Skipping overlay.",
                  file=sys.stderr)
            overlay_var_name = None  # Disable overlay
        else:
            overlay_config = PLOT_VARS_CONFIG[overlay_var_name]
            overlay_plot_type = overlay_config.get('plot_type', 'vector')

    # --- 3. Set up Figure and Map ---
    # v35: Centered on Paducah
    projection = ccrs.LambertConformal(central_longitude=-88.6, central_latitude=37.08)
    data_transform = ccrs.PlateCarree()

    fig, ax = plt.subplots(
        figsize=(12, 9),
        subplot_kw={'projection': projection}
    )
    ax.set_extent(
        [domain['west'], domain['east'], domain['south'], domain['north']],
        crs=data_transform
    )

    # Add map features
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':', edgecolor='black', zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5, zorder=2)
    # v39: Added county lines
    try:
        counties = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_2_counties',  # v39: Corrected name
            scale='10m',
            facecolor='none',
            edgecolor='gray'
        )
        ax.add_feature(counties, linestyle=':', linewidth=0.5, zorder=1.5)
    except Exception as e:
        print(f"Warning: Could not load 10m county lines. {e}", file=sys.stderr)

    # Get lat/lon from dataset
    lats = ds.latitude.values
    lons = ds.longitude.values

    # --- 4. Plot Base Layer (Scalar) ---
    try:
        if base_plot_type == 'scalar':
            grib_name = _find_var_in_ds(ds, base_config['grib_name'])
            if not grib_name:
                print(f"Warning: No variable found for base plot {base_var_name}. Skipping plot.", file=sys.stderr)
                plt.close(fig)
                return

            # v78: Quieter
            # print(f"Plotting Base: {base_var_name} (using grib_name: '{grib_name}') for {dt}...")
            data = ds[grib_name]

            # Select the specific level
            if 'product' in base_config:
                level_dict = base_config['grib_level']
                level_dim = list(level_dict.keys())[0]
                level_val = level_dict[level_dim]

                # v42: Stricter check
                if level_dim not in data.coords:
                    print(f"Warning: Level coord '{level_dim}' not found for '{base_var_name}'. Skipping plot.",
                          file=sys.stderr)
                    plt.close(fig)
                    return

                if level_dim in ds.dims:
                    data = data.sel(**level_dict, method='nearest')
                else:
                    # v30: Fix for 2m T/D
                    if grib_name in ['t2m', 'd2m'] and level_dim not in data.dims:
                        pass  # This is a 2D var, no level selection needed
                    else:
                        data = data.where(ds[level_dim] == level_val, drop=True)

            # Ensure 2D
            if len(data.dims) > 2:
                data = data.squeeze(drop=True)

            # Handle units
            if base_config['units'] == '°F' and data.metpy.units == units.kelvin:
                data = data.metpy.convert_units('degF')
            if base_var_name == 'MSLP' and data.metpy.units == units.pascal:
                # print("Converting MSLP from Pa to hPa...") # v78: Quieter
                data = data.metpy.convert_units('hPa')

            # Get levels and cmap
            levels = base_config['levels']
            cmap = plt.get_cmap(base_config['cmap'])
            norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            extend = base_config.get('extend', 'max')

            # Plot filled contours
            c = ax.contourf(
                lons, lats, data,
                levels=levels,
                cmap=cmap,
                norm=norm,
                transform=data_transform,
                extend=extend
            )

            # v64: Add contour lines
            line_color = 'black'
            line_width = 0.4
            line_alpha = 0.6

            if base_var_name == 'MSLP':
                line_width = 0.6
                line_alpha = 0.8

            cl = ax.contour(
                lons, lats, data,
                levels=levels,
                colors=line_color,
                linewidths=line_width,
                alpha=line_alpha,
                transform=data_transform
            )

            # v64: Add labels, but not for MSLP (too crowded)
            if base_var_name != 'MSLP':
                try:
                    ax.clabel(cl, inline=True, fontsize=8, fmt='%1.0f')
                except Exception as e:
                    # This can fail if contours are all off-map, etc.
                    print(f"Warning: Could not add clabel for {base_var_name}: {e}", file=sys.stderr)

            # Add colorbar
            plt.colorbar(c, ax=ax, label=f"{base_config['long_name']} ({base_config['units']})",
                         orientation='vertical', pad=0.02, shrink=0.8)

        elif base_plot_type in ['vector', 'multi_vector']:
            # This handles the case where user chose a vector as their "base" plot
            # print(f"Plotting Base: {base_var_name}...") # v78: Quieter
            _plot_vector_layer(ax, ds, lons, lats, base_var_name, base_config, data_transform)

    except Exception as e:
        print(f"Error plotting BASE variable '{base_var_name}' for {dt}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        plt.close(fig)
        return  # Don't try to plot overlay

    # --- 5. Plot Overlay Layer (Vector) ---
    try:
        if overlay_var_name:
            # print(f"Plotting Overlay: {overlay_var_name}...") # v78: Quieter
            _plot_vector_layer(ax, ds, lons, lats, overlay_var_name, overlay_config, data_transform)

    except Exception as e:
        print(f"Error plotting OVERLAY variable '{overlay_var_name}' for {dt}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Continue to save the plot, base layer might be fine

    # --- 6. Set Title and Save ---
    title_long_name = base_config['long_name']
    if overlay_var_name:
        title_long_name += f"\nand {overlay_config['long_name']}"

    title = f"{title_long_name} ({plot_name})\n" \
            f"Valid: {dt.strftime('%Y-%m-%d %H:%M')} UTC"
    ax.set_title(title)

    filename = f"{plot_name}_{dt.strftime('%Y%m%d%H%M')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    # print(f"Saved plot: {filepath}") # v78: Quieter
    plt.close(fig)


# --- v74: New helper function to plot a vector layer ---
def _plot_vector_layer(ax, ds, lons, lats, var_name, var_config, data_transform):
    """Helper function to draw a vector (or multi_vector) layer on an existing axes."""

    plot_type = var_config.get('plot_type', 'vector')

    # Subsample data
    skip = 18  # v71: Increased from 15
    lons_sub = lons[::skip, ::skip]
    lats_sub = lats[::skip, ::skip]

    # Find grib names
    u_grib_name, v_grib_name = None, None
    u_grib_name = _find_var_in_ds(ds, var_config['grib_name'])
    v_grib_name = _find_var_in_ds(ds, var_config['grib_name'][1:])

    if var_name == 'WIND_10M':
        u_grib_name = _find_var_in_ds(ds, _INTERNAL_VARS_CONFIG['_U_10M']['grib_name'])
        v_grib_name = _find_var_in_ds(ds, _INTERNAL_VARS_CONFIG['_V_10M']['grib_name'])

    if not u_grib_name or not v_grib_name:
        print(f"Warning: Variables for '{var_name}' not found. Skipping overlay.", file=sys.stderr)
        return

    if plot_type == 'vector':
        u_data = ds[u_grib_name]
        v_data = ds[v_grib_name]

        # v63: Add .values to pass numpy array, not DataArray
        u_sub = u_data[::skip, ::skip].values
        v_sub = v_data[::skip, ::skip].values

        ax.barbs(
            lons_sub, lats_sub, u_sub, v_sub,
            transform=data_transform,
            length=6,
            zorder=10  # Plot on top of scalar fill
        )

    elif plot_type == 'multi_vector':
        levels = var_config['levels']
        colors = var_config['colors']

        # v78: Find 3D wind names
        u_pl_name = _find_var_in_ds(ds, _INTERNAL_VARS_CONFIG['_U_PL']['grib_name'])
        v_pl_name = _find_var_in_ds(ds, _INTERNAL_VARS_CONFIG['_V_PL']['grib_name'])

        if not u_pl_name or not v_pl_name:
            print("Warning: 3D U/V winds not found, skipping WIND_COMBO_LOW.", file=sys.stderr)
            return

        # print(f"  Plotting {len(levels)} wind levels...") # v78: Quieter
        for i, level in enumerate(levels):
            color = colors[i]

            u_data_level = ds[u_pl_name].metpy.sel(isobaricInhPa=level, method='nearest')
            v_data_level = ds[v_pl_name].metpy.sel(isobaricInhPa=level, method='nearest')

            # v63: Add .values
            u_sub = u_data_level[::skip, ::skip].metpy.convert_units('kts').values
            v_sub = v_data_level[::skip, ::skip].metpy.convert_units('kts').values

            ax.barbs(
                lons_sub, lats_sub, u_sub, v_sub,
                transform=data_transform,
                length=6,  # v71: Reduced from 7
                zorder=10 + i,
                color=color,
                label=f'{level} hPa'
            )

        # v73: Create legend, then set zorder on the object
        leg = ax.legend(loc='upper right', fontsize='medium', facecolor='white', framealpha=1.0)
        leg.set_zorder(20)  # v73: Set zorder on the legend object


# --- v75: New function to create GIFs ---
def create_gif_animation(event_output_dir, job_names):
    """
    Finds all .png files for each job, sorts them,
    and creates an animated GIF.
    """
    if not HAS_IMAGEIO:
        print("Error: 'imageio' package not found. Cannot create GIFs.", file=sys.stderr)
        print("  Please install it: conda install -c conda-forge imageio", file=sys.stderr)
        return

    for job_name in job_names:
        job_dir = os.path.join(event_output_dir, job_name)

        # Find all PNG files in the job directory
        file_pattern = os.path.join(job_dir, "*.png")
        file_paths = sorted(glob.glob(file_pattern))

        if len(file_paths) < 2:
            print(f"  Not enough frames for '{job_name}', skipping GIF.")
            continue

        print(f"  Generating GIF for {job_name}...")
        images = []
        try:
            for filepath in file_paths:
                images.append(imageio.imread(filepath))

            # --- v77 FIX: Save GIF in the job_dir (subfolder) ---
            gif_path = os.path.join(job_dir, f"{job_name}_animation.gif")

            # --- v78 FIX: Use milliseconds for duration ---
            # User wants 2.0 seconds per frame
            duration_ms = 2000

            imageio.mimsave(gif_path, images, duration=duration_ms, loop=0)

            print(f"  Saved GIF: {gif_path}")
        except Exception as e:
            print(f"  Error creating GIF for {job_name}: {e}", file=sys.stderr)


# --- Main Execution ---

# v65: New helper function to get valid datetime
def get_datetime_input(prompt_message):
    """Prompts user for datetime until valid format is entered."""
    while True:
        # v69: Added (UTC) to prompt
        dt_str = input(f"  {prompt_message} (YYYY-MM-DD HH:MM UTC): ").strip()
        try:
            # v69: Fixed typo
            dt_obj = datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
            return dt_obj
        except ValueError:
            print(f"Error: Invalid format. Please use 'YYYY-MM-DD HH:MM'")


# v74: New function to get the plot mode
def get_plot_mode():
    """Asks user for the plot mode."""
    print("\n--- Plot Mode ---")
    print("  1: Plot Single Variables (e.g., just SBCAPE, just WIND_10M)")
    print("  2: Create Layered Plot (e.g., DEW2M contours + WIND_10M barbs)")
    while True:
        choice = input("  Select plot mode (1 or 2): ").strip()
        if choice == '1':
            return 'single'
        elif choice == '2':
            return 'layered'
        else:
            print("Error: Invalid choice. Please enter 1 or 2.")


# v74: New function to get the list of plot jobs
def get_plot_jobs(plot_mode):
    """
    Based on the plot mode, guides user to select variables
    and returns a list of plot jobs.
    """

    # 1. Get lists of scalar and vector variables
    scalar_vars = []
    vector_vars = []
    for var_name, config in PLOT_VARS_CONFIG.items():
        if config.get('plot_type', 'scalar') == 'scalar':
            scalar_vars.append(var_name)
        else:
            vector_vars.append(var_name)

    jobs = []  # This will be a list of dicts

    # --- Mode 1: Select multiple single plots ---
    if plot_mode == 'single':
        print("\n--- Select Single Variables ---")
        var_list = list(PLOT_VARS_CONFIG.keys())  # Show all
        for i, var_name in enumerate(var_list):
            print(f"  {i + 1}: {var_name} ({PLOT_VARS_CONFIG[var_name]['long_name']})")

        while True:
            print("\nEnter numbers (e.g., '1, 3, 7') or 'ALL' to plot all.")
            choice = input("  Which variables do you want to plot? ").strip()

            if choice.strip().upper() == 'ALL':
                selected_vars = var_list
                break  # Exit loop

            selected_vars = []
            try:
                indices = [int(i.strip()) - 1 for i in choice.split(',')]
                valid_selection = True
                for idx in indices:
                    if 0 <= idx < len(var_list):
                        selected_vars.append(var_list[idx])
                    else:
                        print(f"Error: '{idx + 1}' is not a valid number. Please choose from 1 to {len(var_list)}.")
                        valid_selection = False

                if valid_selection and selected_vars:
                    break  # Exit loop
                elif not valid_selection:
                    continue  # Re-prompt
                else:
                    print("Error: No variables selected.")
            except ValueError:
                print(f"Error: Invalid input. Please enter numbers separated by commas or 'ALL'.")

        # Create a job for each selected variable
        for var_name in selected_vars:
            jobs.append({
                'base': var_name,
                'overlay': None,
                'name': var_name  # Folder name is just the var name
            })

    # --- Mode 2: Select one base and one overlay ---
    elif plot_mode == 'layered':
        print("\n--- Create Layered Plot ---")

        # 1. Select Base Scalar
        print("  Step 1: Select a BASE (scalar/filled) layer.")
        for i, var_name in enumerate(scalar_vars):
            print(f"  {i + 1}: {var_name} ({PLOT_VARS_CONFIG[var_name]['long_name']})")

        base_var = None
        while not base_var:
            try:
                choice = input(f"  Select base layer (1-{len(scalar_vars)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(scalar_vars):
                    base_var = scalar_vars[idx]
                else:
                    print("Error: Invalid number.")
            except ValueError:
                print("Error: Invalid input. Please enter a number.")

        # 2. Select Overlay Vector
        print("\n  Step 2: Select an OVERLAY (vector/barb) layer.")
        vector_vars_with_none = vector_vars + ['NONE']  # Add 'NONE' option
        for i, var_name in enumerate(vector_vars_with_none):
            if var_name == 'NONE':
                print(f"  {i + 1}: [Plot without overlay]")
            else:
                print(f"  {i + 1}: {var_name} ({PLOT_VARS_CONFIG[var_name]['long_name']})")

        overlay_var = None
        while True:
            try:
                choice = input(f"  Select overlay layer (1-{len(vector_vars_with_none)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(vector_vars_with_none):
                    selected = vector_vars_with_none[idx]
                    if selected != 'NONE':
                        overlay_var = selected
                    break  # Exit loop
                else:
                    print("Error: Invalid number.")
            except ValueError:
                print("Error: Invalid input. Please enter a number.")

        # 3. Create the single job
        plot_name = base_var
        if overlay_var:
            plot_name = f"{base_var}_with_{overlay_var}"

        jobs.append({
            'base': base_var,
            'overlay': overlay_var,
            'name': plot_name  # Folder name is the combo name
        })

    return jobs


# v81: New function (from mrms_analyzer.py) to select the domain
def get_domain_selection():
    """Asks user which domain to use for plotting."""
    print("\n--- Select Plotting Domain ---")

    for i, domain in enumerate(ALL_DOMAINS):
        print(f"  {i + 1}: {domain['name']}")

    while True:
        choice = input(f"  Enter choice (1-{len(ALL_DOMAINS)}, default is 1): ").strip()

        if not choice:
            return ALL_DOMAINS[0]  # Default to first option

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(ALL_DOMAINS):
                return ALL_DOMAINS[idx]
            else:
                print(f"Error: Please choose from 1 to {len(ALL_DOMAINS)}.")
        except ValueError:
            print("Error: Please enter a valid number.")


# --- v79: New worker function for multiprocessing ---
def process_single_timestep(job_details, dt):
    """
    This function contains the full processing logic
    for a *single* time step. It is designed to be
    called by a multiprocessing pool.

    v79: Arguments swapped to (job_details, dt)
    """

    # Unpack the job details
    vars_to_load = job_details['vars_to_load']
    plot_jobs = job_details['plot_jobs']
    domain = job_details['domain']
    event_output_dir = job_details['event_output_dir']

    print(f"--- Processing {dt} ---")

    # 1. Load Data with Herbie
    ds = load_data_with_herbie(dt, list(vars_to_load), event_output_dir)

    if ds is None:
        print(f"Failed to load data with Herbie for {dt}. Skipping.", file=sys.stderr)
        return f"Failed: {dt} (Load)"

    # 2. Compute Derived Variables
    ds_processed = compute_derived_vars(ds)

    # 3. Normalize Longitudes (for cropping)
    try:
        min_lon = ds_processed.longitude.min().item()
        max_lon = ds_processed.longitude.max().item()

        if min_lon >= 0 and max_lon > 180:
            # print(f"Normalizing longitudes for {dt}...") # v78: Quieter
            ds_processed = ds_processed.assign_coords(
                longitude=(((ds_processed.longitude + 180) % 360) - 180)
            )
    except Exception as e:
        print(f"Warning: Could not check/normalize longitudes for {dt}: {e}", file=sys.stderr)

    # 4. Crop Dataset to Domain
    try:
        lat_mask = (ds_processed.latitude >= domain['south']) & (ds_processed.latitude <= domain['north'])
        lon_mask = (ds_processed.longitude >= domain['west']) & (ds_processed.longitude <= domain['east'])
        combined_mask = lat_mask & lon_mask

        ds_cropped = ds_processed.where(combined_mask, drop=True)

    except Exception as e:
        print(f"Error cropping dataset for {dt}: {e}", file=sys.stderr)
        return f"Failed: {dt} (Crop)"

    if ds_cropped.latitude.size == 0 or ds_cropped.longitude.size == 0:
        print(f"Error: Domain crop resulted in an empty dataset for {dt}.", file=sys.stderr)
        return f"Failed: {dt} (Empty Crop)"

    # 5. Plot each *job*
    plot_count = 0
    for job in plot_jobs:
        plot_name = job['name']
        base_var_name = job['base']
        overlay_var_name = job['overlay']

        plot_output_dir = os.path.join(event_output_dir, plot_name)

        plot_variable(
            ds_cropped,
            base_var_name,
            overlay_var_name,
            domain,
            plot_output_dir,
            dt,
            plot_name
        )
        plot_count += 1

    return f"Success: {dt} ({plot_count} plots)"


# --- End v79 ---


def main():
    """
    Main function to run the interactive severe weather analyzer.
    v78: Refactored to use multiprocessing.
    """
    print("--- Severe Weather Analyzer (v81) ---")
    print("Welcome! Let's set up your analysis.")

    # 1. Base Output Directory is fixed
    user_base_out_dir = BASE_OUTPUT_DIR

    # 2. Get Start/End Times
    start_time = get_datetime_input("Enter START date/time")
    end_time = get_datetime_input("Enter END date/time")

    if end_time < start_time:
        print("Error: End time must be after start time.")
        sys.exit(1)

    # 3. Get Plot Mode
    plot_mode = get_plot_mode()

    # 4. Get Plot Job(s)
    plot_jobs = get_plot_jobs(plot_mode)

    # 5. Get Domain
    domain = get_domain_selection()  # v81: Call new function
    print(f"\nUsing domain '{domain['name']}':")
    print(f"  West: {domain['west']}, East: {domain['east']}, South: {domain['south']}, North: {domain['north']}")

    # 6. Create Event-Specific Output Directory
    event_name = start_time.strftime('%Y%m%d_%H%M_event')
    event_output_dir = os.path.join(user_base_out_dir, event_name)

    if not os.path.exists(event_output_dir):
        try:
            os.makedirs(event_output_dir)
            print(f"Created new event directory: {event_output_dir}")
        except Exception as e:
            print(f"Error: Could not create output directory '{event_output_dir}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Using existing event directory: {event_output_dir}")

    # 7. Create subdirectories for each *job*
    print("Creating variable subdirectories...")
    vars_to_load = set()
    job_names = []
    for job in plot_jobs:
        job_name = job['name']
        job_names.append(job_name)

        var_dir = os.path.join(event_output_dir, job_name)
        if not os.path.exists(var_dir):
            try:
                os.makedirs(var_dir)
            except Exception as e:
                print(f"Warning: Could not create sub-directory '{var_dir}': {e}", file=sys.stderr)

        vars_to_load.add(job['base'])
        if job['overlay']:
            vars_to_load.add(job['overlay'])

    # --- Generate Time Steps ---
    time_steps = []
    current_time = start_time
    while current_time <= end_time:
        time_steps.append(current_time)
        current_time += timedelta(hours=1)

    # --- v75: Ask to create GIFs ---
    make_gifs = False
    if len(time_steps) > 1:
        print("\n--- Animations ---")
        gif_choice = input("  Would you like to create animated GIFs at the end? (y/n): ").strip().lower()
        if gif_choice == 'y':
            if not HAS_IMAGEIO:
                print("Error: 'imageio' package not found. Cannot create GIFs.", file=sys.stderr)
                print("  Please install it: conda install -c conda-forge imageio tqdm", file=sys.stderr)
            else:
                make_gifs = True
                print("  Will create GIFs after processing.")

    # --- Print Summary ---
    print("\n--- Analysis Starting ---")
    print(f"Start: {start_time} UTC")
    print(f"End:   {end_time} UTC")
    print(f"Domain: {domain['name']}")  # v81: Updated to use name
    print(f"Outputting to: {event_output_dir}")
    print(f"Processing {len(plot_jobs)} plot job(s).")
    print(f"Total time steps to process: {len(time_steps)}\n")

    # --- v78: Main Processing Loop (Parallelized) ---

    # 1. Create a "job_details" dict to pass to all workers
    job_details = {
        'vars_to_load': list(vars_to_load),
        'plot_jobs': plot_jobs,
        'domain': domain,
        'event_output_dir': event_output_dir
    }

    # 2. Create a partial function with job_details "baked in"
    # This is needed so pool.map can pass just the 'dt'
    worker_func = partial(process_single_timestep, job_details)

    # 3. Set up and run the pool
    # Use one fewer core than available to leave system responsive
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Starting parallel processing on {num_cores} cores...")

    with multiprocessing.Pool(processes=num_cores) as pool:
        if HAS_TQDM:
            # Use tqdm to create a progress bar
            results = list(tqdm(
                pool.imap(worker_func, time_steps),
                total=len(time_steps),
                desc="Processing Timesteps",
                mininterval=1.0,
                file=sys.stdout
            ))
        else:
            # Run without progress bar
            results = pool.map(worker_func, time_steps)

    print("\n--- Parallel Processing Complete ---")
    print("Results:")
    for res in results:
        print(f"  {res}")
    # --- End v7g ---

    print("\n--- Analysis Complete ---")

    # --- v75: Create GIFs if requested ---
    if make_gifs:
        print("\n--- Creating GIF Animations ---")
        create_gif_animation(event_output_dir, job_names)
        print("GIF creation complete.")
    # --- End v75 ---

    print(f"All output saved in: {event_output_dir}")


if __name__ == "__main__":
    # v78: Required for multiprocessing on Windows
    multiprocessing.freeze_support()

    # Set backend for Matplotlib
    try:
        plt.switch_backend('Agg')
    except Exception as e:
        print(f"Could not switch matplotlib backend: {e}", file=sys.stderr)

    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    main()

