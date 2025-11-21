#!/usr/bin/env python
# mrms_analyzer.py

"""
MRMS Radar Analysis Tool (v30)
Author: Gemini
Date: November 4, 2025

v29: 1. Fixed typo in get_datetime_input's strptime format string
        (e.g., '%Y-m-%d' -> '%Y-%m-%d') to fix ValueError.
v30: 1. Added alpha=0.75 transparency to QPE plots to make map
        boundaries more visible.
     2. Extended QPE color scale levels and colors to a max of 15 inches.
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
import multiprocessing
from functools import partial
import glob
import gzip
import shutil
import requests
import time

import numpy as np
import xarray as xr
# v25: No longer using metpy
# import metpy.calc as mpcalc
# from metpy.units import units
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: 'tqdm' package not found. Progress bar will be disabled.", file=sys.stderr)

# v18: Fix DeprecationWarning by importing v2
try:
    import imageio.v2 as imageio

    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio

        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False

# --- Configuration ---

BASE_OUTPUT_DIR = "./mrms_analysis_plots"

# v21: Renamed to REGIONAL_DOMAIN
REGIONAL_DOMAIN = {
    'name': 'Regional (Default)',
    'west': -92.2,
    'east': -85.0,
    'south': 34.68,
    'north': 39.48
}

# v21: Added new Paducah CWA domain
# v22: Updated coordinates based on user-provided CWA map
# v23: Zoomed out ~10%
# v24: Expanded western boundary and zoomed out slightly more
PADUCAH_CWA_DOMAIN = {
    'name': 'NWS Paducah CWA (Zoomed)',
    'west': -92.0,  # Pushed west to give ~15% more space
    'east': -86.2,  # Pushed east slightly
    'south': 35.7,  # Pushed south slightly
    'north': 39.0  # Pushed north slightly
}

ALL_DOMAINS = [REGIONAL_DOMAIN, PADUCAH_CWA_DOMAIN]

# --- MRMS Radar Variable Definitions ---
MRMS_VARS_CONFIG = {
    'REF': {
        'long_name': 'SeamlessHSR Reflectivity',
        'iem_path': 'SeamlessHSR',
        'iem_product_code': '_00.00_',
        'grib_name': 'refd',  # Our internal standard name
        'units': 'dBZ',
        'interval_minutes': 2,
        'is_time_series': False
    },
    'ROT': {
        'long_name': 'Rotation Track (1440min)',
        'iem_path': 'RotationTrack1440min',
        'iem_product_code': '_00.50_',
        'grib_name': 'rot',  # Our internal standard name
        'units': 's-1',
        'interval_minutes': 30,
        'is_time_series': False
    },
    # v25: Added QPE Products
    'QPE_01H': {
        'long_name': '1-Hour QPE Pass 2',
        'iem_path': 'MultiSensor_QPE_01H_Pass2',
        'iem_product_code': '_00.00_',
        'grib_name': 'qpe01h', # Internal standard name
        'units': 'inches',     # Display units (will convert from mm)
        'interval_minutes': 60,
        'is_time_series': False
    },
    'QPE_24H': {
        'long_name': '24-Hour QPE Pass 2',
        'iem_path': 'MultiSensor_QPE_24H_Pass2',
        'iem_product_code': '_00.00_',
        'grib_name': 'qpe24h',
        'units': 'inches',
        'interval_minutes': 60,
        'is_time_series': False
    },
    'QPE_72H': {
        'long_name': '72-Hour QPE Pass 2',
        'iem_path': 'MultiSensor_QPE_72H_Pass2',
        'iem_product_code': '_00.00_',
        'grib_name': 'qpe72h',
        'units': 'inches',
        'interval_minutes': 60,
        'is_time_series': False
    },
}


# --- Radar Colormaps ---

def _get_radar_colormaps():
    """Returns a dict of custom colormaps for radar data."""
    # NWS Reflectivity Color Scale
    ref_colors = [
        "#FFFFFF",  # 0
        "#04e9e7",  # 5
        "#019ff4",  # 10
        "#0300f4",  # 15
        "#02fd02",  # 20
        "#01c501",  # 25
        "#008e00",  # 30
        "#fdf802",  # 35
        "#e5bc00",  # 40
        "#fd9500",  # 45
        "#fd0000",  # 50
        "#d40000",  # 55
        "#bc0000",  # 60
        "#f800fd",  # 65
        "#9854c6",  # 70
        "#4B0082"  # 75
    ]
    ref_cmap = mcolors.ListedColormap(ref_colors)
    ref_levels = np.arange(0, 80, 5)
    ref_norm = mcolors.BoundaryNorm(ref_levels, ref_cmap.N)

    # v19: New granular Rotation Colormap
    rot_colors_new = [
        '#A9A9A9',  # 0.003 - 0.008 (Gray)
        '#FFFF00',  # 0.008 - 0.009 (Yellow)
        '#FDD000',  # 0.009 - 0.010 (Yellow-Orange)
        '#FDA500',  # 0.010 - 0.011 (Orange)
        '#FD7A00',  # 0.011 - 0.012 (Orange-Red)
        '#FF0000',  # 0.012 - 0.013 (Red)
        '#E60000',  # 0.013 - 0.014 (Dark Red)
        '#CC0000',  # 0.014 - 0.015 (Darker Red)
        '#B30000',  # 0.015 - 0.020 (Darkest Red)
        '#FF00FF'  # > 0.020 (Fuchsia for "extreme")
    ]
    rot_cmap = mcolors.ListedColormap(rot_colors_new)
    rot_levels_new = [
        0.003, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.020, 0.050
    ]
    rot_norm = mcolors.BoundaryNorm(rot_levels_new, rot_cmap.N)

    # v25: New QPE Colormap (levels in inches)
    # v30: Extended levels to 15 inches
    qpe_levels = [
        0.01, 0.10, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
        2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0  # Added 12.0, 15.0
    ]
    # v30: Extended colors to match new levels
    qpe_colors = [
        '#04e9e7',  # 0.01 - 0.10 (light cyan)
        '#019ff4',  # 0.10 - 0.25 (cyan)
        '#0300f4',  # 0.25 - 0.50 (blue)
        '#02fd02',  # 0.50 - 0.75 (light green)
        '#01c501',  # 0.75 - 1.00 (green)
        '#008e00',  # 1.00 - 1.25 (dark green)
        '#fdf802',  # 1.25 - 1.50 (yellow)
        '#e5bc00',  # 1.50 - 1.75 (yellow-orange)
        '#fd9500',  # 1.75 - 2.00 (orange)
        '#fd0000',  # 2.00 - 2.50 (red)
        '#d40000',  # 2.50 - 3.00 (dark red)
        '#bc0000',  # 3.00 - 4.00 (maroon)
        '#f800fd',  # 4.00 - 5.00 (magenta)
        '#9854c6',  # 5.00 - 6.00 (purple)
        '#663399',  # 6.00 - 8.00 (dark purple)
        '#FFFFFF',  # 8.00 - 10.00 (white)
        '#D3D3D3',  # 10.00 - 12.00 (light gray)
        '#A9A9A9'   # 12.00 - 15.00 (dark gray)
    ]
    qpe_cmap = mcolors.ListedColormap(qpe_colors)
    qpe_norm = mcolors.BoundaryNorm(qpe_levels, qpe_cmap.N)

    return {
        'REF': {'cmap': ref_cmap, 'norm': ref_norm, 'levels': ref_levels, 'extend': 'max'},
        'ROT': {'cmap': rot_cmap, 'norm': rot_norm, 'levels': rot_levels_new, 'extend': 'max'},
        'QPE_01H': {'cmap': qpe_cmap, 'norm': qpe_norm, 'levels': qpe_levels, 'extend': 'max'},
        'QPE_24H': {'cmap': qpe_cmap, 'norm': qpe_norm, 'levels': qpe_levels, 'extend': 'max'},
        'QPE_72H': {'cmap': qpe_cmap, 'norm': qpe_norm, 'levels': qpe_levels, 'extend': 'max'}
    }


# --- MRMS Data Loading Functions ---

def _download_and_load_grib(url, grib_path, gz_path, standard_name):
    """
    Handles the download, decompression, and loading of a single GRIB file.
    v17: Gutted all time-handling logic.
    v18: Removed metpy.parse_cf()
    """
    # --- v5: Add file locking for parallel safety ---
    lock_path = gz_path + ".lock"

    # 1. Check if final file is already there.
    if os.path.exists(grib_path):
        pass  # File exists, just load it

    # 2. If final file is NOT there, we need to download/decompress.
    else:
        is_waiting = False
        while os.path.exists(lock_path):
            if not is_waiting:
                # print(f"  Process {os.getpid()} waiting for lock on {os.path.basename(gz_path)}...")
                is_waiting = True
            time.sleep(1)  # Wait 1 second

        # After waiting, check *again*
        if os.path.exists(grib_path):
            pass  # Another process finished it

        # 3. If we are here, lock is free AND file doesn't exist.
        else:
            try:
                with open(lock_path, 'w') as f:
                    f.write(f"locked by {os.getpid()}")

                # print(f"Downloading: {url}") # Quieter
                headers = {'User-Agent': 'Mozilla/5.0'}
                with requests.get(url, stream=True, headers=headers) as r:
                    r.raise_for_status()
                    with open(gz_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

                # print(f"Decompressing {gz_path}...") # Quieter
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(grib_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path)

            finally:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
    # --- End v5 Lock Logic ---

    # v25: Added explicit filter to avoid multi-variable GRIBs
    # The 'unknown' var is what we want, it's the precip data.
    # 'p192m' is some kind of max value var we don't need.
    ds_mrms = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        filter_by_keys={'shortName': 'unknown'}
    )

    var_names = list(ds_mrms.data_vars)
    if not var_names:
        # Fallback if 'unknown' isn't found
        ds_mrms = xr.open_dataset(grib_path, engine="cfgrib")
        var_names = list(ds_mrms.data_vars)
        if not var_names:
            raise ValueError("GRIB file loaded, but no data variables found.")

    # Rename the data variable (e.g., 'unknown' -> 'qpe01h')
    original_name = var_names[0]
    if original_name != standard_name:
        ds_mrms = ds_mrms.rename({original_name: standard_name})

    # v27: Drop the conflicting 'valid_time' coordinate BEFORE concatenation
    # This coordinate is from the GRIB file metadata and causes a MergeError
    if "valid_time" in ds_mrms.coords:
        ds_mrms = ds_mrms.drop_vars("valid_time")

    return ds_mrms


# --- v5: New parallel worker function ---
def _download_worker(task):
    """
    Worker function to download and load a single individual file.
    Returns a tuple of (dt_product, ds_mrms)
    """
    dt_product, mrms_key, mrms_cache_dir = task
    try:
        mrms_config = MRMS_VARS_CONFIG[mrms_key]
        iem_path_name = mrms_config['iem_path']
        product_code = mrms_config['iem_product_code']
        standard_name = mrms_config['grib_name']

        iem_dir = f"{dt_product.strftime('%Y/%m/%d')}/mrms/ncep"
        iem_filename = f"{iem_path_name}{product_code}{dt_product.strftime('%Y%m%d-%H%M00')}.grib2.gz"
        url = f"https://mtarchive.geol.iastate.edu/{iem_dir}/{iem_path_name}/{iem_filename}"

        gz_path = os.path.join(mrms_cache_dir, iem_filename)
        grib_filename = iem_filename.replace(".grib2.gz", f".{dt_product.strftime('%Y%m%d-%H%M%S')}.grib2")
        grib_path = os.path.join(mrms_cache_dir, grib_filename)

        ds_mrms = _download_and_load_grib(url, grib_path, gz_path, standard_name)
        return (dt_product, ds_mrms)
    except Exception as e:
        # print(f"  Error loading file for {mrms_key} at {dt_product}: {e}", file=sys.stderr)
        # Be less verbose on common 404 errors
        if not isinstance(e, requests.exceptions.HTTPError) or e.response.status_code != 404:
            print(f"  Error loading file for {mrms_key} at {dt_product}: {e}", file=sys.stderr)
        return (dt_product, None)  # Return None on failure


# v4: New function to load all data *before* processing
def load_all_data(start_time, end_time, time_step_minutes, products_to_load, domain, event_output_dir):
    """
    Downloads all necessary MRMS data for the entire time range
    and returns a single merged xarray Dataset.
    v6: Crops data *before* concatenating to save memory.
    v15: All products now use the "individual file" logic.
    """
    print(f"\n--- Loading All MRMS Data ---")
    base_url = "https://mtarchive.geol.iastate.edu"
    mrms_cache_dir = os.path.join(event_output_dir, 'mrms_cache')
    os.makedirs(mrms_cache_dir, exist_ok=True)

    # Generate the list of *requested* plot times
    time_steps = []
    current_time = start_time
    while current_time <= end_time:
        time_steps.append(current_time)
        current_time += timedelta(minutes=time_step_minutes)

    all_loaded_data = []

    for mrms_key in products_to_load:
        print(f"Processing product: {mrms_key}")
        mrms_config = MRMS_VARS_CONFIG[mrms_key]
        standard_name = mrms_config['grib_name']
        interval = mrms_config['interval_minutes']

        # v15: Removed 'is_time_series' check. All products use this logic.
        product_data_list = []  # To store DataArrays for this product

        files_to_download = {}
        for dt in time_steps:
            # v25: Use interval to find the correct file time
            # e.g., for interval=60, 10:30 -> 10:00, 10:59 -> 10:00
            minutes_to_subtract = (dt.minute % interval) + dt.second / 60.0 + dt.microsecond / 60e6
            dt_product = dt - timedelta(minutes=minutes_to_subtract)

            # For hourly products, also need to round down the hour if interval is > 2
            if interval >= 60:
                 dt_product = dt_product.replace(minute=0, second=0, microsecond=0)

            if dt_product not in files_to_download:
                files_to_download[dt_product] = []
            files_to_download[dt_product].append(dt)

        print(f"  Found {len(files_to_download)} unique files to download for {len(time_steps)} plot times...")

        tasks = []
        for dt_product in files_to_download.keys():
            tasks.append((dt_product, mrms_key, mrms_cache_dir))

        loaded_files = {}

        num_cores = max(1, multiprocessing.cpu_count() - 1)
        print(f"  Starting parallel download on {num_cores} cores...")
        with multiprocessing.Pool(processes=num_cores) as pool:
            if HAS_TQDM:
                results = list(tqdm(
                    pool.imap(_download_worker, tasks),
                    total=len(tasks),
                    desc=f"  Loading {mrms_key}"
                ))
            else:
                results = pool.map(_download_worker, tasks)

        for dt_product, ds_mrms in results:
            if ds_mrms is not None:
                loaded_files[dt_product] = ds_mrms

        # --- v6: CROP *BEFORE* CONCAT ---
        print("  Cropping and re-assigning plot times...")
        successful_loads = 0
        for dt_product, plot_dts in files_to_download.items():
            if dt_product in loaded_files:
                ds_mrms = loaded_files[dt_product]

                # --- Normalize Longitudes ---
                try:
                    if 'longitude' in ds_mrms.coords:
                        min_lon = ds_mrms.longitude.min().item()
                        if min_lon >= 0:
                            ds_mrms = ds_mrms.assign_coords(
                                longitude=(((ds_mrms.longitude + 180) % 360) - 180)
                            )
                    elif standard_name in ds_mrms and 'longitude' in ds_mrms[standard_name].coords:
                        lon_coord = ds_mrms[standard_name].longitude
                        min_lon = lon_coord.min().item()
                        if min_lon >= 0:
                            normalized_lon = (((lon_coord + 180) % 360) - 180)
                            ds_mrms = ds_mrms.assign_coords(longitude=normalized_lon)
                except Exception as e:
                    print(f"Warning: Could not normalize lon for {dt_product}: {e}", file=sys.stderr)

                # --- Crop Data ---
                try:
                    radar_lats = ds_mrms.latitude
                    radar_lons = ds_mrms.longitude
                    lat_mask_r = (radar_lats >= domain['south']) & (radar_lats <= domain['north'])
                    lon_mask_r = (radar_lons >= domain['west']) & (radar_lons <= domain['east'])
                    ds_cropped = ds_mrms.where(lat_mask_r & lon_mask_r, drop=True)

                    if ds_cropped.latitude.size == 0:
                        print(f"  Warning: Crop resulted in empty dataset for {dt_product}", file=sys.stderr)
                        continue

                    # Add a dimension for *each* plot time it's valid for
                    for dt in plot_dts:
                        da_with_time = ds_cropped[standard_name].expand_dims(time=[dt])
                        product_data_list.append(da_with_time)
                        successful_loads += 1

                except Exception as e:
                    print(f"  Warning: Could not crop file for {dt_product}: {e}", file=sys.stderr)
        # --- End v6 ---

        if product_data_list:
            print(f"  Concatenating {len(product_data_list)} cropped files for {mrms_key}...")
            # v26: FIX - Use coords="minimal" to avoid ValueError with new xarray versions
            combined_da = xr.concat(product_data_list, dim="time", coords="minimal", join="override")
            all_loaded_data.append(combined_da)
        else:
             print(f"  No data successfully loaded for {mrms_key}.")

    if not all_loaded_data:
        print("Error: No MRMS data was loaded at all.", file=sys.stderr)
        return None

    try:
        print("Merging all MRMS products into final dataset...")
        ds_final = xr.merge(all_loaded_data, compat='override')

        print("MRMS data loading complete.")
        return ds_final
    except Exception as e:
        print(f"Error merging MRMS datasets: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


# --- Plotting Function ---

def plot_mrms_data(ds_radar_timestep, product_key, domain, output_dir, dt):
    """
    Plots a single MRMS variable on a map.
    Accepts a dataset that is ALREADY for a single time step.
    v25: No longer uses MetPy for units.
    """

    try:
        config = MRMS_VARS_CONFIG[product_key]
        grib_name = config['grib_name']
        color_maps = _get_radar_colormaps()
        plot_cmap = color_maps[product_key]
    except KeyError as e:
        print(f"Error: Could not find config for variable {e}. Skipping plot.", file=sys.stderr)
        return

    if grib_name not in ds_radar_timestep:
        print(f"Error: Variable '{grib_name}' not in MRMS dataset for {dt}. Skipping plot.", file=sys.stderr)
        return

    da_radar = ds_radar_timestep[grib_name].squeeze()

    # Handle NaNs (e.g., -9999 values)
    # GRIB files often use large negative numbers for missing
    da_radar = da_radar.where(da_radar >= 0)

    # v20: Check if we have ROT data and apply scaling
    if product_key == 'ROT':
        try:
            max_val = da_radar.max().item()
            if max_val > 1.0:
                print(f"  INFO: Scaling ROT data, max value was {max_val:.2f}. Dividing by 1000.")
                da_radar = da_radar / 1000.0
        except Exception as e:
            print(f"  Warning: Could not check/scale ROT data: {e}", file=sys.stderr)

    # v25: Check for QPE data and convert mm -> inches
    if product_key.startswith('QPE_'):
        try:
            # Data is in mm, convert to inches for plotting
            # 1 inch = 25.4 mm
            da_radar = da_radar / 25.4
        except Exception as e:
            print(f"  Warning: Could not convert QPE data to inches: {e}", file=sys.stderr)


    # --- Setup Map Projection ---
    # We can't rely on MetPy/CF parse, so we build it manually
    # MRMS data is on a WGS84 lat/lon grid (PlateCarree)
    data_crs = ccrs.PlateCarree()
    # Plot on a LambertConformal grid centered on our domain
    projection = ccrs.LambertConformal(central_longitude=-88.6, central_latitude=37.08)

    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': projection})
    ax.set_extent([domain['west'], domain['east'], domain['south'], domain['north']], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':', edgecolor='black', zorder=10)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5, zorder=10)
    try:
        counties = cfeature.NaturalEarthFeature(
            category='cultural', name='admin_2_counties', scale='10m',
            facecolor='none', edgecolor='gray')
        ax.add_feature(counties, linestyle=':', linewidth=0.5, zorder=10)
    except Exception:
        pass # Fail gracefully if county lines aren't available

    lats_rad = da_radar.latitude.values
    lons_rad = da_radar.longitude.values

    # v30: Set alpha for transparency on QPE plots
    plot_alpha = 1.0
    if product_key.startswith('QPE_'):
        plot_alpha = 0.75  # 75% opacity

    # Plot the data
    # We use contourf for non-rectilinear grids (which this is)
    c = ax.contourf(
        lons_rad, lats_rad, da_radar,
        levels=plot_cmap['levels'],
        cmap=plot_cmap['cmap'],
        norm=plot_cmap['norm'],
        transform=data_crs,
        zorder=1,
        extend=plot_cmap.get('extend', 'max'), # v25: Add extend, v30: Changed to max
        alpha=plot_alpha  # v30: Add transparency
    )
    plt.colorbar(c, ax=ax, label=f"{config['long_name']} ({config['units']})",
                 orientation='vertical', pad=0.02, shrink=0.8)

    title = f"MRMS {config['long_name']} ({product_key})\n" \
            f"Valid: {dt.strftime('%Y-%m-%d %H:%M')} UTC"
    ax.set_title(title)

    filename = f"{product_key}_{dt.strftime('%Y%m%d%H%M')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


# --- GIF Creation Function ---
def create_gif_animation(event_output_dir, product_keys):
    """
    Finds all .png files for each product, sorts them,
    and creates an animated GIF.
    """
    if not HAS_IMAGEIO:
        print("Error: 'imageio' package not found. Cannot create GIFs.", file=sys.stderr)
        return

    for product_key in product_keys:
        job_dir = os.path.join(event_output_dir, product_key)

        file_pattern = os.path.join(job_dir, "*.png")
        file_paths = sorted(glob.glob(file_pattern))

        if len(file_paths) < 2:
            print(f"  Not enough frames for '{product_key}', skipping GIF.")
            continue

        print(f"  Generating GIF for {product_key}...")
        images = []
        try:
            for filepath in file_paths:
                # v17: Use imageio.v2.imread
                images.append(imageio.imread(filepath))

            gif_path = os.path.join(job_dir, f"{product_key}_animation.gif")

            # Use 700ms (0.7s) duration and infinite loop
            imageio.mimsave(gif_path, images, duration=700, loop=0)

            print(f"  Saved GIF: {gif_path}")
        except Exception as e:
            print(f"  Error creating GIF for {product_key}: {e}", file=sys.stderr)


# --- v4: Removed Parallel Worker Function ---


# --- Main Execution ---

def get_datetime_input(prompt_message):
    """Prompts user for datetime until valid format is entered."""
    while True:
        dt_str = input(f"  {prompt_message} (YYYY-MM-DD HH:MM UTC): ").strip()
        try:
            # v29: FIX - Corrected format string
            dt_obj = datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
            return dt_obj
        except ValueError:
            print(f"Error: Invalid format. Please use 'YYYY-MM-DD HH:MM'")


def get_product_selection():
    """Asks user which MRMS products to plot."""
    print("\n--- Select MRMS Products ---")

    var_list = list(MRMS_VARS_CONFIG.keys())
    for i, var_name in enumerate(var_list):
        print(f"  {i + 1}: {var_name} ({MRMS_VARS_CONFIG[var_name]['long_name']})")

    while True:
        # v23: Removed 'ALL' option
        print("\nEnter numbers (e.g., '1' or '1, 2').")
        choice = input("  Which products do you want to plot? ").strip()

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
                return selected_vars
            elif not valid_selection:
                continue
            else:
                print("Error: No variables selected.")
        except ValueError:
            print(f"Error: Invalid input. Please enter numbers separated by commas.")


# v21: New function to select the domain
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


def get_time_step():
    """Asks user for the time step in minutes."""
    while True:
        try:
            choice = input("\n  Enter the time step between images (in minutes): ").strip()
            minutes = int(choice)
            if minutes > 0:
                return minutes
            else:
                print("Error: Please enter a positive number.")
        except ValueError:
            print("Error: Please enter a valid number.")


def main():
    """
    Main function to run the interactive MRMS analyzer.
    v4: Refactored to load all data first, then plot.
    """
    print("--- MRMS Radar Analyzer (v29) ---")
    print("Welcome! Let's set up your analysis.")

    # 1. Get Start/End Times
    start_time = get_datetime_input("Enter START date/time")
    end_time = get_datetime_input("Enter END date/time")

    # 2. Get Time Step
    time_step_minutes = 10  # Default
    if start_time == end_time:
        print("  Processing a single time step.")
        time_steps = [start_time]
    else:
        time_step_minutes = get_time_step()
        time_steps = []
        current_time = start_time
        while current_time <= end_time:
            time_steps.append(current_time)
            current_time += timedelta(minutes=time_step_minutes)

    if end_time < start_time:
        print("Error: End time must be after start time.")
        sys.exit(1)

    # 3. Get Product Selection
    products_to_plot = get_product_selection()

    # 4. Get Domain
    domain = get_domain_selection()  # v21: Call new function
    print(f"\nUsing domain '{domain['name']}':")
    print(f"  West: {domain['west']}, East: {domain['east']}, South: {domain['south']}, North: {domain['north']}")

    # 5. Create Event-Specific Output Directory
    event_name = start_time.strftime(f'%Y%m%d_%H%M_mrms_') + "_to_" + end_time.strftime(f'%Y%m%d_%H%M')
    event_output_dir = os.path.join(BASE_OUTPUT_DIR, event_name)

    if not os.path.exists(event_output_dir):
        try:
            os.makedirs(event_output_dir)
            print(f"Created new event directory: {event_output_dir}")
        except Exception as e:
            print(f"Error: Could not create output directory '{event_output_dir}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Using existing event directory: {event_output_dir}")

    # 6. Create subdirectories for each product
    print("Creating product subdirectories...")
    for product_key in products_to_plot:
        var_dir = os.path.join(event_output_dir, product_key)
        if not os.path.exists(var_dir):
            try:
                os.makedirs(var_dir)
            except Exception as e:
                print(f"Warning: Could not create sub-directory '{var_dir}': {e}", file=sys.stderr)

    # 7. Ask to create GIFs
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
    print(f"Step:  {time_step_minutes} minutes")
    print(f"Domain: {domain['name']}")
    print(f"Outputting to: {event_output_dir}")
    print(f"Processing {len(products_to_plot)} product(s): {', '.join(products_to_plot)}")
    print(f"Total time steps to process: {len(time_steps)}\n")

    # --- v4: Main Processing Loop (Refactored) ---

    # 1. Load ALL data first
    try:
        ds_radar_full = load_all_data(
            start_time,
            end_time,
            time_step_minutes,
            products_to_plot,
            domain,  # v6: Pass domain
            event_output_dir
        )
    except KeyboardInterrupt:
        print("\n*** User interrupted (CTRL-C) during download ***")
        print("Processing stopped.")
        sys.exit(1)

    if ds_radar_full is None:
        print("Fatal Error: Could not load any MRMS data. Exiting.", file=sys.stderr)
        sys.exit(1)

    # v25: Removed MetPy parse
    ds_radar_cropped = ds_radar_full  # It's already loaded and cropped

    # 4. Loop through timesteps and plot (this is now fast)
    print(f"\n--- Plotting {len(time_steps)} Timesteps ---")

    time_iterator = tqdm(time_steps, desc="Plotting Timesteps") if HAS_TQDM else time_steps

    for dt in time_iterator:
        try:
            # --- v7 FIX: Use standard .sel() ---
            # Use 'nearest' to find the closest *available* time we loaded
            ds_timestep = ds_radar_cropped.sel(time=dt, method='nearest')

            # Check if the found time is too far from the requested time
            time_diff = abs(ds_timestep.time.values - np.datetime64(dt))
            # Set a threshold, e.g., 2x the time step, or 62 mins for hourly data
            threshold_minutes = max(time_step_minutes * 2, 62)
            if time_diff > np.timedelta64(threshold_minutes, 'm'):
                print(f"  Skipping plot for {dt}: No data found within {threshold_minutes} minutes.")
                continue

            # Plot each product for this time step
            for product_key in products_to_plot:
                plot_output_dir = os.path.join(event_output_dir, product_key)

                # --- v28 FIX: Check for the grib_name, not the product_key ---
                config = MRMS_VARS_CONFIG[product_key]
                grib_name = config['grib_name']

                # Check if data exists for this product at this timestep
                if grib_name not in ds_timestep or ds_timestep[grib_name].isnull().all():
                    # print(f"  No data for {product_key} at {dt}, skipping plot.")
                    continue

                plot_mrms_data(
                    ds_timestep,
                    product_key,
                    domain,
                    plot_output_dir,
                    dt
                )
        except Exception as e:
            print(f"Error plotting data for time {dt}: {e}", file=sys.stderr)

    print("\n--- Plotting Complete ---")

    # 5. Create GIFs
    if make_gifs:
        print("\n--- Creating GIF Animations ---")
        create_gif_animation(event_output_dir, products_to_plot)
        print("GIF creation complete.")

    print(f"\nAll output saved in: {event_output_dir}")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    try:
        plt.switch_backend('Agg')
    except Exception as e:
        print(f"Could not switch matplotlib backend: {e}", file=sys.stderr)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # v25: Suppress specific xarray warning about dropping coordinates
    warnings.filterwarnings("ignore", message="Converting non-nanosecond precision datetime values to nanosecond precision")

    main()


