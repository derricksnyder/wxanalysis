#!/usr/bin/env python
# sounding_analyzer.py

"""
HRRR Sounding Analysis Tool (v72)
Author: Gemini
Date: November 10, 2025

v71: 1. Per user request, removed the "zoomed-in" plot for
        winter mode entirely.
     2. Winter mode now uses the same y-axis (1050-100 hPa)
        and x-axis (-30-50 C) as the convective mode.
     3. This standardizes the plots and definitively fixes the
        v70 bug, as the axis limits are set consistently
        before `plot_barbs` is called.
v72: 1. User log showed a `TypeError`, proving the `barb_increments`
        kwarg is not supported by their older MetPy version.
     2. NEW FIX: Removed the `barb_increments` kwarg to fix the crash.
     3. Re-implemented manual 50 hPa slicing, but with a more robust
        floating-point-safe check (e.g., `(p_mod_50 < 1) | (p_mod_50 > 49)`)
        to correctly find levels like 1000, 950, 900, etc.
"""

import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import metpy.plots as mpplots
import matplotlib.pyplot as plt
# v14: Added for monospace font
from matplotlib.font_manager import FontProperties

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
BASE_OUTPUT_DIR = "./sounding_plots"


# --- Helper Functions ---

def get_datetime_input(prompt_message):
    """Prompts user for datetime until valid format is entered."""
    while True:
        dt_str = input(f"  {prompt_message} (YYYY-MM-DD HH:MM UTC): ").strip()
        try:
            # v45 FIX: Corrected format string 'm' to '%m'
            dt_obj = datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
            # For HRRR, we must round to the nearest hour
            if dt_obj.minute != 0:
                dt_rounded = dt_obj.replace(minute=0, second=0, microsecond=0)
                if dt_obj.minute >= 30:
                    dt_rounded = dt_rounded + timedelta(hours=1)
                print(f"  Note: Rounding {dt_str} to the nearest hour: {dt_rounded.strftime('%Y-%m-%d %H:%M')}")
                dt_obj = dt_rounded
            return dt_obj
        except ValueError:
            print(f"Error: Invalid format. Please use 'YYYY-MM-DD HH:MM'")


def get_location_input():
    """Prompts user for a latitude and longitude."""
    while True:
        try:
            lat_str = input("  Enter Latitude (e.g., 37.08 for Paducah): ").strip()
            lat = float(lat_str)
            if not -90 <= lat <= 90:
                print("Error: Latitude must be between -90 and 90.")
                continue

            lon_str = input("  Enter Longitude (e.g., -88.6 for Paducah): ").strip()
            lon = float(lon_str)
            if not -180 <= lon <= 180:
                print("Error: Longitude must be between -180 and 180.")
                continue

            return lat, lon
        except ValueError:
            print("Error: Invalid input. Please enter a number (e.g., 37.08).")


# v2: New function to select analysis mode
def get_analysis_mode():
    """Prompts user to select the analysis mode."""
    print("\n--- Select Analysis Mode ---")
    print("  1: Convective (Default)")
    print("  2: Winter")
    while True:
        choice = input("  Enter choice (1 or 2, default is 1): ").strip()
        if not choice or choice == '1':
            return 'convective'
        elif choice == '2':
            return 'winter'
        else:
            print("Error: Invalid choice. Please enter 1 or 2.")


def _find_var_in_ds(ds, grib_names_list):
    """
    Helper to find the first matching grib_name in the dataset's variables.
    v42: Made case-insensitive.
    """
    ds_vars_lower = {v.lower(): v for v in ds.data_vars}
    for name in grib_names_list:
        if name.lower() in ds_vars_lower:
            return ds_vars_lower[name.lower()]  # Return original case name
    return None


# v5: New helper function to manually find coordinate names
def _find_coord_name(ds, potential_names):
    """Helper to find the first matching name in the dataset's coordinates."""
    for name in potential_names:
        if name in ds.coords:
            return name
    return None


def load_sounding_data(dt, output_dir):
    """
    Uses Herbie to find, download, and load HRRR pressure and surface
    data for a specific time.
    """
    print(f"Initializing Herbie for {dt}...")

    # We need both pressure-level and surface data
    # v69: Use full unambiguous names for 2m T/Td
    sfc_search_regex = ":(PRES|SP|TMP:2 m above ground|DPT:2 m above ground):surface"
    # v39: Added VVEL (Omega)
    # v40: Added 'omega'
    prs_search_regex = ":(TMP|DPT|UGRD|VGRD|HGT|VVEL|omega):.+ mb"

    datasets_to_merge = []

    # 1. Load SFC (Surface) variables
    try:
        H_sfc = Herbie(
            dt,
            model='hrrr',
            product='sfc',
            fxx=0,
            save_dir=os.path.join(output_dir, 'herbie_cache'),
            priority=['aws', 'nomads', 'google', 'azure']
        )
        print(f"  Fetching HRRR 'sfc' file for surface T/Td/P...")
        ds_sfc = H_sfc.xarray(sfc_search_regex, verbose=False)
        if isinstance(ds_sfc, list):
            # v67: Use xr.merge to combine the list of datasets
            print(f"  Note: Merging {len(ds_sfc)} surface 'hypercubes'...")
            ds_sfc = xr.merge(ds_sfc, compat='override')
        datasets_to_merge.append(ds_sfc)
    except Exception as e:
        print(f"Warning: Herbie (HRRR) 'sfc' file failed for {dt}. {e}", file=sys.stderr)

    # 2. Load PRS (Pressure) variables
    try:
        H_prs = Herbie(
            dt,
            model='hrrr',
            product='prs',
            fxx=0,
            save_dir=os.path.join(output_dir, 'herbie_cache'),
            priority=['aws', 'nomads', 'google', 'azure']
        )
        print(f"  Fetching HRRR 'prs' file for vertical profile...")
        ds_prs = H_prs.xarray(prs_search_regex, verbose=False)
        if isinstance(ds_prs, list):
            # v67: Use xr.merge to combine the list of datasets
            print(f"  Note: Merging {len(ds_prs)} pressure 'hypercubes'...")
            ds_prs = xr.merge(ds_prs, compat='override')
        datasets_to_merge.append(ds_prs)
    except Exception as e:
        print(f"Warning: Herbie (HRRR) 'prs' file failed for {dt}. {e}", file=sys.stderr)
        if not datasets_to_merge:
            return None

    # 3. Merge all datasets
    if not datasets_to_merge:
        print(f"Error: No HRRR data was loaded from Herbie for {dt}.", file=sys.stderr)
        return None

    try:
        print("  Merging sfc and prs datasets...")
        # v66: Keep 'override' here, as v65 logic handles sfc/prs conflict
        ds_final = xr.merge(datasets_to_merge, compat='override')
        print("  Herbie data load complete.")
        return ds_final
    except Exception as e:
        print(f"Error merging HRRR datasets for {dt}: {e}", file=sys.stderr)
        return None


# --- v14: New Helper Functions for Metrics ---
def _format_value(value, unit_str, precision=0):
    """Helper to safely format a MetPy quantity or number."""
    if isinstance(value, units.Quantity):
        if np.isnan(value.magnitude):
            return "N/A"
        return f"{value.magnitude:.{precision}f} {unit_str}"
    if isinstance(value, (float, int)) and not np.isnan(value):
        return f"{value:.{precision}f} {unit_str}"
    return "N/A"


# v31: New helper for manual interpolation
def _interp_to_p(target_p, p_profile, data_profile):
    """Interpolates a data_profile value at a target_p pressure."""
    # np.interp needs pressure to be increasing, but MetPy profiles
    # are descending (1000 -> 100). We must reverse them.
    # Ensure data is numpy array magnitude for interp
    p_profile_asc = p_profile.to('hPa').magnitude[::-1]
    data_profile_asc = data_profile.magnitude[::-1]

    interp_val = np.interp(target_p.to('hPa').magnitude, p_profile_asc, data_profile_asc)
    return interp_val * data_profile.units


# v46: New helper to find the *first* (bottom-up) pressure at a given value
def _find_bottom_up_crossing(target_val, data_profile_desc, p_profile_desc):
    """
    Finds the first (lowest-altitude) pressure crossing from the surface up.
    Assumes profiles are sorted descending by pressure (surface first).
    v47: Replaced mpcalc.log_interpolate_1d with np.interp on log(p).
    """
    # Find all indices where the sign of (data - target) changes
    cross_indices = np.where(np.diff(np.sign(data_profile_desc - target_val)))[0]

    if len(cross_indices) == 0:
        # No crossing found
        return np.nan * units.hPa

    # Get the first crossing index (lowest altitude)
    idx = cross_indices[0]

    # Get the data and pressure values just above and below the crossing
    p1 = p_profile_desc[idx]
    p2 = p_profile_desc[idx + 1]
    d1 = data_profile_desc[idx]
    d2 = data_profile_desc[idx + 1]

    # v47 FIX: Use np.interp on log(pressure) for compatibility
    try:
        # We must interpolate on log(p) for meteorological accuracy
        d_vals = np.array([d1.magnitude, d2.magnitude])
        log_p_vals = np.log(np.array([p1.magnitude, p2.magnitude]))

        # np.interp needs increasing x-values (which data is)
        if d_vals[0] > d_vals[1]:
            d_vals = d_vals[::-1]
            log_p_vals = log_p_vals[::-1]

        log_p_crossing = np.interp(target_val.magnitude, d_vals, log_p_vals)
        p_crossing_mag = np.exp(log_p_crossing)

        return p_crossing_mag * p1.units  # Return with units

    except Exception as e:
        print(f"  Warning: _find_bottom_up_crossing failed: {e}", file=sys.stderr)
        return np.nan * units.hPa


# --- End v56 Restore ---


# v26: Relocated text tables to the right side
# v27: Added `hodo` object to signature to allow plotting RM/LM
# v39: Pass `lowest_wb_zero_p`
# v46: Pass `frz_hgt_agl` and `wbz_hgt_agl`
# v51: Refactored winter mode into 3 tables
# v52: Removed FRZ/WBZ annotations, moved tables down
def _calculate_and_plot_metrics(fig, hodo, p, t, td, u, v, z_agl, z_msl, mode,
                                sfc_p, sfc_t, sfc_td,
                                frz_hgt_agl, wbz_hgt_agl):
    """
    Calculates and plots key sounding metrics in a consolidated table.
    All inputs are expected to have units. (sfc_t/td are in degC)
    """
    try:
        # Get monospace font
        mono_font = FontProperties(family='monospace')

        # --- WINTER MODE ---
        if mode == 'winter':
            # v51: Create 3 text columns
            thermo_text = ["[--- Thermo / Moisture ---]"]
            levels_text = ["[--- Key Levels (AGL) ---]"]
            shear_text = ["[--- Shear / Helicity ---]"]

            # v36: Add Surface T/Td
            sfc_t_f = sfc_t.to('degF')
            sfc_td_f = sfc_td.to('degF')
            thermo_text.append(f"Sfc T:    {_format_value(sfc_t, 'C', 1)} ({_format_value(sfc_t_f, 'F', 1)})")
            thermo_text.append(f"Sfc Td:   {_format_value(sfc_td, 'C', 1)} ({_format_value(sfc_td_f, 'F', 1)})")

            # 1. Precipitable Water (PW)
            try:
                pw = mpcalc.precipitable_water(p, td)
                thermo_text.append(f"PW:       {_format_value(pw.to('in'), 'in', 2)}")
            except Exception:
                thermo_text.append("PW:       N/A")

            # 2. Thickness 1000-850 hPa
            try:
                # v35: Use manual slicing for old MetPy
                # v50: Re-apply v35 fix
                mask_1000_850 = (p <= 1000 * units.hPa) & (p >= 850 * units.hPa)
                thick_1000_850 = mpcalc.thickness_hydrostatic(p[mask_1000_850], t[mask_1000_850])
                thermo_text.append(f"1000-850m: {_format_value(thick_1000_850.to('m'), 'm')}")
            except Exception as e:
                print(f"  Warning: Could not calculate 1000-850m thickness: {e}", file=sys.stderr)
                thermo_text.append("1000-850m: N/A")

            # 3. Thickness 1000-500 hPa
            try:
                # v35: Use manual slicing for old MetPy
                # v50: Re-apply v35 fix
                mask_1000_500 = (p <= 1000 * units.hPa) & (p >= 500 * units.hPa)
                thick_1000_500 = mpcalc.thickness_hydrostatic(p[mask_1000_500], t[mask_1000_500])
                thermo_text.append(f"1000-500m: {_format_value(thick_1000_500.to('m'), 'm')}")
            except Exception as e:
                print(f"  Warning: Could not calculate 1000-500m thickness: {e}", file=sys.stderr)
                thermo_text.append("1000-500m: N/A")

            # --- Column 2: Key Levels ---
            levels_text.append(f"Sfc Pres: {_format_value(sfc_p.to('hPa'), 'hPa', 1)}")

            # 4. Wet-Bulb Zero Level
            levels_text.append(f"Frz Lvl:  {_format_value(frz_hgt_agl.to('m'), 'm')}")
            levels_text.append(f"WB Zero:  {_format_value(wbz_hgt_agl.to('m'), 'm')}")

            # 5. DGZ Level (-12C to -18C)
            try:
                t_asc = t.magnitude[::-1]
                p_asc = p.to('hPa').magnitude[::-1]
                z_agl_asc = z_agl.to('m').magnitude[::-1]

                p_at_minus_12 = np.interp(-12.0, t_asc, p_asc)
                p_at_minus_18 = np.interp(-18.0, t_asc, p_asc)

                z_at_minus_12 = np.interp(-12.0, t_asc, z_agl_asc)
                z_at_minus_18 = np.interp(-18.0, t_asc, z_agl_asc)

                levels_text.append(f"DGZ Base: {_format_value(z_at_minus_12, 'm')} (-12C)")
                levels_text.append(f"DGZ Top:  {_format_value(z_at_minus_18, 'm')} (-18C)")

            except Exception as e:
                print(f"  Warning: Could not calculate DGZ heights: {e}", file=sys.stderr)
                levels_text.append("DGZ Base: N/A")
                levels_text.append("DGZ Top:  N/A")

            # --- Column 3: Shear / Helicity ---
            try:
                # v19: Pass z_agl for MetPy 1.6.3
                rm, lm, motion = mpcalc.bunkers_storm_motion(p, u, v, z_agl)
                mean_spd = mpcalc.wind_speed(motion[0], motion[1]).to('kts')
                mean_dir = mpcalc.wind_direction(motion[0], motion[1])
                motion_str = f"{mean_dir.magnitude:.0f}°/{mean_spd.magnitude:.0f} kt"
                shear_text.append(f"Mean Wind: {motion_str}")
            except Exception as e:
                print(f"  Warning: Could not calculate Bunkers motion: {e}", file=sys.stderr)
                shear_text.append("Mean Wind: N/A")

            try:
                srh_1km = mpcalc.storm_relative_helicity(z_agl, u, v, depth=1000 * units.m)
                shear_text.append(f"0-1km SRH: {_format_value(srh_1km[0], 'm2/s2')}")
            except Exception as e:
                print(f"  Warning: Could not calculate 0-1km SRH: {e}", file=sys.stderr)
                shear_text.append("0-1km SRH: N/A")

            try:
                srh_3km = mpcalc.storm_relative_helicity(z_agl, u, v, depth=3000 * units.m)
                shear_text.append(f"0-3km SRH: {_format_value(srh_3km[0], 'm2/s2')}")
            except Exception:
                shear_text.append("0-3km SRH: N/A")

            try:
                z_3km_mask = (z_agl >= 0 * units.m) & (z_agl <= 3000 * units.m)
                p_3km, u_3km, v_3km = p[z_3km_mask], u[z_3km_mask], v[z_3km_mask]
                u_shr_3, v_shr_3 = mpcalc.bulk_shear(p_3km, u_3km, v_3km)
                shear_3km_mag = mpcalc.wind_speed(u_shr_3, v_shr_3).to('kts')
                shear_text.append(f"0-3km Shear: {_format_value(shear_3km_mag, 'kts')}")
            except Exception as e:
                print(f"  Warning: Could not calculate 0-3km Shear: {e}", file=sys.stderr)
                shear_text.append("0-3km Shear: N/A")

            try:
                z_6km_mask = (z_agl >= 0 * units.m) & (z_agl <= 6000 * units.m)
                p_6km, u_6km, v_6km = p[z_6km_mask], u[z_6km_mask], v[z_6km_mask]
                u_shr_6, v_shr_6 = mpcalc.bulk_shear(p_6km, u_6km, v_6km)
                shear_6km_mag = mpcalc.wind_speed(u_shr_6, v_shr_6).to('kts')
                shear_text.append(f"0-6km Shear: {_format_value(shear_6km_mag, 'kts')}")
            except Exception as e:
                print(f"  Warning: Could not calculate 0-6km Shear: {e}", file=sys.stderr)
                shear_text.append("0-6km Shear: N/A")

            # v26: Plot the winter text block on the right side
            table_bbox = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray')
            # v52: Shifted all Y-positions down
            fig.text(0.72, 0.53, "\n".join(thermo_text), fontproperties=mono_font,
                     ha='left', va='top', fontsize=9, bbox=table_bbox)

            fig.text(0.72, 0.35, "\n".join(levels_text), fontproperties=mono_font,
                     ha='left', va='top', fontsize=9, bbox=table_bbox)

            fig.text(0.72, 0.18, "\n".join(shear_text), fontproperties=mono_font,
                     ha='left', va='top', fontsize=9, bbox=table_bbox)

        # --- CONVECTIVE MODE ---
        elif mode == 'convective':
            # v22: Create 3 text columns
            instability_text = ["[--- Instability ---]"]
            shear_text = ["[--- Shear / Helicity ---]"]
            composite_text = ["[--- Composite ---]"]  # New column

            # --- Calculate Surface Height (sfc_z_msl) for AGL calcs ---
            try:
                sfc_p_val = sfc_p.to('hPa').magnitude
                p_vals_asc = p.to('hPa').magnitude[::-1]
                z_msl_vals_asc = z_msl.to('m').magnitude[::-1]
                sfc_z_msl_val = np.interp(sfc_p_val, p_vals_asc, z_msl_vals_asc)
                sfc_z_msl = sfc_z_msl_val * units.m
            except Exception:
                sfc_z_msl = z_msl[0]  # Fallback to lowest level

            # v36: Add Surface T/Td
            sfc_t_f = sfc_t.to('degF')
            sfc_td_f = sfc_td.to('degF')
            instability_text.append(f"Sfc T:    {_format_value(sfc_t, 'C', 1)} ({_format_value(sfc_t_f, 'F', 1)})")
            instability_text.append(f"Sfc Td:   {_format_value(sfc_td, 'C', 1)} ({_format_value(sfc_td_f, 'F', 1)})")

            # --- Column 1: Instability ---
            sbcape, sbcin, lcl_p, lfc_p, el_p = (np.nan * units.dimensionless,) * 5
            sb_prof = None
            try:
                sb_prof = mpcalc.parcel_profile(p, sfc_t, sfc_td).to('degC')
                sbcape, sbcin = mpcalc.cape_cin(p, t, td, sb_prof)
                instability_text.append(f"SBCAPE:  {_format_value(sbcape, 'J/kg')}")
                instability_text.append(f"SBCIN:   {_format_value(sbcin, 'J/kg')}")
            except Exception as e:
                instability_text.append("SBCAPE:  N/A")
                instability_text.append("SBCIN:   N/A")
                print(f"  Warning: Could not calculate SBCAPE/SBCIN: {e}", file=sys.stderr)
                sbcape = np.nan * units('J/kg')  # Ensure it's NaN for STP
                sbcin = np.nan * units('J/kg')

            try:
                pw = mpcalc.precipitable_water(p, td)
                instability_text.append(f"PW:      {_format_value(pw.to('in'), 'in', 2)}")
            except Exception:
                instability_text.append("PW:      N/A")

            # v22: LCL, LFC, EL (AGL)
            lcl_hgt_agl = np.nan * units.m  # Initialize as nan
            try:
                lcl_p, _ = mpcalc.lcl(sfc_p, sfc_t, sfc_td)
                # v23 FIX: Removed `parcel_T_profile` argument
                lfc_p, _ = mpcalc.lfc(p, t, td, sb_prof)
                el_p, _ = mpcalc.el(p, t, td, sb_prof)

                lcl_hgt_msl = mpcalc.pressure_to_height_std(lcl_p)
                lfc_hgt_msl = mpcalc.pressure_to_height_std(lfc_p)
                el_hgt_msl = mpcalc.pressure_to_height_std(el_p)

                lcl_hgt_agl = lcl_hgt_msl - sfc_z_msl
                lfc_hgt_agl = lfc_hgt_msl - sfc_z_msl
                el_hgt_agl = el_hgt_msl - sfc_z_msl

                instability_text.append(f"LCL (AGL): {_format_value(lcl_hgt_agl.to('m'), 'm')}")
                instability_text.append(f"LFC (AGL): {_format_value(lfc_hgt_agl.to('m'), 'm')}")
                instability_text.append(f"EL (AGL):  {_format_value(el_hgt_agl.to('m'), 'm')}")

            except Exception as e:
                print(f"  Warning: Could not calculate LCL/LFC/EL: {e}", file=sys.stderr)
                instability_text.append("LCL (AGL): N/A")
                instability_text.append("LFC (AGL): N/A")
                instability_text.append("EL (AGL):  N/A")
                # lcl_hgt_agl is already nan

            # v31: Manual Lapse Rate calculation (old MetPy compatible)
            try:
                # Interpolate T and Z at standard levels
                t_850 = _interp_to_p(850 * units.hPa, p, t)
                t_700 = _interp_to_p(700 * units.hPa, p, t)
                t_500 = _interp_to_p(500 * units.hPa, p, t)

                z_850 = _interp_to_p(850 * units.hPa, p, z_msl)
                z_700 = _interp_to_p(700 * units.hPa, p, z_msl)
                z_500 = _interp_to_p(500 * units.hPa, p, z_msl)

                # Calculate lapse rates manually: (T_top - T_bot) / (Z_top - Z_bot)
                lr_700_500 = (t_500 - t_700) / (z_500 - z_700)
                # v33 FIX: Use 'delta_degC/km' to avoid 'Coulomb' error
                instability_text.append(f"700-500 LR: {_format_value(lr_700_500.to('delta_degC/km'), 'C/km', 1)}")

                lr_850_500 = (t_500 - t_850) / (z_500 - z_850)
                instability_text.append(f"850-500 LR: {_format_value(lr_850_500.to('delta_degC/km'), 'C/km', 1)}")

            except Exception as e:
                print(f"  Warning: Could not calculate Lapse Rates: {e}", file=sys.stderr)
                instability_text.append("700-500 LR: N/A")
                instability_text.append("850-500 LR: N/A")

            # v30: 0-3km CAPE fix (Old MetPy compatible)
            try:
                # Find pressure at 3km AGL by interpolating
                p_at_3km_val = np.interp(3000, z_agl.to('m').magnitude, p.to('hPa').magnitude)
                p_at_3km = p_at_3km_val * units.hPa

                # Manually slice the profile from surface to 3km
                sfc_to_3km_mask = (p <= sfc_p) & (p >= p_at_3km)

                p_3km = p[sfc_to_3km_mask]
                t_3km = t[sfc_to_3km_mask]
                td_3km = td[sfc_to_3km_mask]

                # We also must slice the parcel profile
                sb_prof_3km = mpcalc.parcel_profile(p_3km, sfc_t, sfc_td).to('degC')

                # Calculate CAPE on the sliced profile
                cape_3km, _ = mpcalc.cape_cin(p_3km, t_3km, td_3km, sb_prof_3km)
                instability_text.append(f"0-3km CAPE: {_format_value(cape_3km, 'J/kg')}")
            except Exception as e:
                print(f"  Warning: Could not calculate 0-3km CAPE: {e}", file=sys.stderr)
                instability_text.append("0-3km CAPE: N/A")

            # v32: Removed DCAPE calculation

            # --- Column 2: Shear / Helicity ---
            motion = (np.nan * units.kts,) * 2
            srh_1km = (np.nan * units('m^2/s^2'),)
            shear_6km_mag = np.nan * units.kts
            try:
                # v19: Pass z_agl for MetPy 1.6.3
                rm, lm, motion = mpcalc.bunkers_storm_motion(p, u, v, z_agl)

                # v34: Convert Bunkers motion to Dir/Spd
                mean_spd = mpcalc.wind_speed(motion[0], motion[1]).to('kts')
                mean_dir = mpcalc.wind_direction(motion[0], motion[1])
                motion_str = f"{mean_dir.magnitude:.0f}°/{mean_spd.magnitude:.0f} kt"

                rm_spd = mpcalc.wind_speed(rm[0], rm[1]).to('kts')
                rm_dir = mpcalc.wind_direction(rm[0], rm[1])
                rm_str = f"{rm_dir.magnitude:.0f}°/{rm_spd.magnitude:.0f} kt"

                lm_spd = mpcalc.wind_speed(lm[0], lm[1]).to('kts')
                lm_dir = mpcalc.wind_direction(lm[0], lm[1])
                lm_str = f"{lm_dir.magnitude:.0f}°/{lm_spd.magnitude:.0f} kt"

                shear_text.append(f"Bunkers: {motion_str} (Mean)")
                shear_text.append(f"RM:      {rm_str}")
                shear_text.append(f"LM:      {lm_str}")

                # v27: Plot RM/LM on Hodograph
                try:
                    hodo.ax.plot(rm[0].to('kts').magnitude, rm[1].to('kts').magnitude, 'ro', markersize=8)
                    hodo.ax.text(rm[0].to('kts').magnitude + 2, rm[1].to('kts').magnitude, 'RM', color='red',
                                 fontsize=10, ha='left')
                    hodo.ax.plot(lm[0].to('kts').magnitude, lm[1].to('kts').magnitude, 'bo', markersize=8)
                    hodo.ax.text(lm[0].to('kts').magnitude + 2, lm[1].to('kts').magnitude, 'LM', color='blue',
                                 fontsize=10, ha='left')
                except Exception as hodo_e:
                    print(f"  Warning: Could not plot RM/LM on hodograph: {hodo_e}", file=sys.stderr)

            except Exception as e:
                print(f"  Warning: Could not calculate Bunkers motion: {e}", file=sys.stderr)
                shear_text.append("Bunkers: N/A")
                shear_text.append("RM:      N/A")
                shear_text.append("LM:      N/A")

            try:
                # v21: Corrected argument order (height, u, v)
                srh_1km = mpcalc.storm_relative_helicity(z_agl, u, v, depth=1000 * units.m)
                shear_text.append(f"0-1km SRH: {_format_value(srh_1km[0], 'm2/s2')}")
            except Exception as e:
                print(f"  Warning: Could not calculate 0-1km SRH: {e}", file=sys.stderr)
                shear_text.append("0-1km SRH: N/A")

            try:
                # v21: Corrected argument order (height, u, v)
                srh_3km = mpcalc.storm_relative_helicity(z_agl, u, v, depth=3000 * units.m)
                shear_text.append(f"0-3km SRH: {_format_value(srh_3km[0], 'm2/s2')}")
            except Exception:
                shear_text.append("0-3km SRH: N/A")

            try:
                # v19: Sliced-array method for bulk shear (MetPy 1.6.3)
                z_6km_mask = (z_agl >= 0 * units.m) & (z_agl <= 6000 * units.m)
                p_6km, u_6km, v_6km = p[z_6km_mask], u[z_6km_mask], v[z_6km_mask]
                u_shr, v_shr = mpcalc.bulk_shear(p_6km, u_6km, v_6km)
                shear_6km_mag = mpcalc.wind_speed(u_shr, v_shr).to('kts')
                shear_text.append(f"0-6km Shear: {_format_value(shear_6km_mag, 'kts')}")
            except Exception as e:
                print(f"  Warning: Could not calculate 0-6km Shear: {e}", file=sys.stderr)
                shear_text.append("0-6km Shear: N/A")

            # --- Column 3: Composite ---
            try:
                lcl_hgt_m = lcl_hgt_agl.to('m').magnitude
                sbcape_jkg = sbcape.to('J/kg').magnitude
                srh_1km_m2s2 = srh_1km[0].to('m^2/s^2').magnitude
                shear_6km_ms = shear_6km_mag.to('m/s').magnitude
                sbcin_jkg = sbcin.to('J/kg').magnitude

                if np.isnan(lcl_hgt_m):
                    lcl_term = 0.0
                else:
                    lcl_term = (2000.0 - lcl_hgt_m) / 1000.0
                    lcl_term = max(0.0, min(1.0, lcl_term))

                if np.isnan(sbcin_jkg):
                    cin_term = 0.0
                else:
                    cin_term = (sbcin_jkg + 200.0) / 150.0
                    cin_term = max(0.0, min(1.0, cin_term))

                sbcape_term = sbcape_jkg / 1500.0 if not np.isnan(sbcape_jkg) else 0.0
                srh_1km_term = srh_1km_m2s2 / 150.0 if not np.isnan(srh_1km_m2s2) else 0.0
                shear_6km_term = shear_6km_ms / 20.0 if not np.isnan(shear_6km_ms) else 0.0

                stp = sbcape_term * lcl_term * srh_1km_term * shear_6km_term * cin_term

                composite_text.append(f"STP (Fixed): {_format_value(stp, '', 1)}")
            except Exception as e:
                print(f"  Warning: Could not calculate STP: {e}", file=sys.stderr)
                composite_text.append("STP (Fixed): N/A")

            # v26: Plot the convective text blocks on the right side
            table_bbox = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray')

            # Stack tables vertically under the hodograph
            # v52: Shifted all Y-positions down
            fig.text(0.72, 0.53, "\n".join(instability_text), fontproperties=mono_font,
                     ha='left', va='top', fontsize=9, bbox=table_bbox)

            fig.text(0.72, 0.23, "\n".join(shear_text), fontproperties=mono_font,
                     ha='left', va='top', fontsize=9, bbox=table_bbox)

            fig.text(0.72, 0.08, "\n".join(composite_text), fontproperties=mono_font,
                     ha='left', va='top', fontsize=9, bbox=table_bbox)

    except Exception as e:
        print(f"  Warning: Failed to calculate/plot metrics: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()


# --- End v14-v32 ---

def plot_sounding(ds, lat, lon, output_dir, dt, mode):
    """
    Plots the Skew-T and Hodograph for the selected location.
    v26: Professional "SHARPy-style" Layout.
    v39: Added Omega (Vertical Velocity) profile.
    v39: Added FRZ/WBZ annotations.
    v41: Made Omega profile optional.
    v42: Fixed FRZ/WBZ annotation placement; made Omega search case-insensitive.
    v43: Added AGL height (in feet) to FRZ/WBZ annotations.
    v44: Fixed ImageSize error by *always* drawing omega_ax.
    v46: Fixed bottom-up crossing logic and refactored height calcs.
    v49: Final layout tweak and fixed annotation plotting
    v50: Reverted to v48 layout, fixed annotation logic, fixed N/A winter params.
    v51: Added 3-block winter layout
    v52: Removed FRZ/WBZ annotations from plot, moved tables down.
    v53: Added 10m sfc wind, padded x-axis to 31C.
    v54: Fixed 10m wind GRIB name, fixed barb plotting logic.
    v55: Fixed UnboundLocalError typo (z_msl vs z_msl_gpm).
    v56: Re-inserted missing helper functions.
    v57: Updated 10m wind GRIB search (UGRD:10 m).
    v58: Broadened 10m wind GRIB search, added 1000 hPa wind fallback.
    v59: Fixed wind barb logic (revert to % 50) and fixed z_msl typo.
    v64: Fixed data corruption by finding sfc vars *before* parse_cf().
         Removed all sfc wind logic and sfc data prepending.
         Simplified barb plot logic to 50 hPa increments only.
    v65: Re-engineered data extraction. Run parse_cf() *first*, then
         pull 3D profile, then *separately* pull sfc vars by their
         full GRIB name to finally fix the -64.4 C data bug.
    v66: 1. v65 fixed the data corruption (Sfc T is now correct), but low-level
            barbs are still missing.
         2. This confirms the "multiple hypercubes" log is the root cause:
            the `xr.merge(..., compat='override')` was dropping the
            low-level pressure data cube.
         3. NEW FIX: Changed `xr.merge(ds_list, compat='override')` to
            `xr.combine_by_coords(ds_list)` in `load_sounding_data`. This
            is the correct way to combine data split along a coordinate
            and should restore the missing low-level pressure data.
    v67: 1. The `xr.combine_by_coords` fix in v66 *still* failed to
            produce low-level barbs, indicating the data is still
            being merged incorrectly.
         2. This is a very persistent bug related to Herbie returning
            multiple datasets for the `prs` file.
         3. NEW FIX: Reverting the v66 change. Instead of
            `xr.combine_by_coords`, we will try merging the list
            of datasets using `xr.merge(..., compat='override')`.
            This is a more robust way to combine datasets that may
            have different coordinates or variables.
    v68: 1. v67 fixed data loading/corruption (T/Td data is correct)
            but low-level barbs are *still* missing.
         2. This proves the data is loaded, but the *plotting logic*
            is failing.
         3. NEW FIX: The logic `p.magnitude % 50 == 0` is susceptible
            to floating-point errors (e.g., 1000.0001 % 50 != 0).
            The logic is now:
            p_rounded = np.round(p.magnitude)
            barb_indices = (p_rounded % 50 == 0)
         4. Also reverted sfc_search_regex to use 'TMP:2 m' to try
            and fix the 'Missing 2m T/D' warning.
    v69: 1. The floating-point rounding fix in v68 *still* failed.
         2. This confirms the manual slicing logic is the root problem.
            It is fighting with MetPy's internal plotting/thinning logic.
         3. NEW FIX: Per user suggestion, removed *all* manual slicing
            (p_rounded, barb_indices).
         4. Replaced with the correct, modern MetPy parameter:
            `skew.plot_barbs(p, u, v, barb_increments=50)`
         5. This lets MetPy handle the logic of finding and plotting the
            50 hPa levels, which is far more robust.
         6. Also updated sfc_search_regex to full unambiguous names
            to fix the 'Missing 2m T/D' warning.
    v70: 1. User provided a convective plot that *works*, proving
            data loading and processing are correct.
         2. THE BUG: `plot_barbs` was called *before* the `if/elif`
            block. In 'winter' mode, `set_ylim(1050, 300)` was
            called *after* plotting, which "zoomed" the axis
            and caused a matplotlib redraw bug, hiding the barbs.
         3. NEW FIX: Moved the `skew.plot_barbs` call to *after*
            the `if/elif` block, so it plots onto the
            finalized, zoomed-in axis. This should fix the
            missing low-level barbs.
    v71: 1. Per user request, removed the "zoomed-in" plot for
            winter mode entirely.
         2. Winter mode now uses the same y-axis (1050-100 hPa)
            and x-axis (-30-50 C) as the convective mode.
         3. This standardizes the plots and definitively fixes the
            v70 bug, as the axis limits are set consistently
            before `plot_barbs` is called.
    v72: 1. User log showed a `TypeError`, proving the `barb_increments`
            kwarg is not supported by their older MetPy version.
         2. NEW FIX: Removed the `barb_increments` kwarg to fix the crash.
         3. Re-implemented manual 50 hPa slicing, but with a more robust
            floating-point-safe check (e.g., `(p_mod_50 < 1) | (p_mod_50 > 49)`)
            to correctly find levels like 1000, 950, 900, etc.
    """
    print(f"Generating sounding plot for {lat}, {lon} (Mode: {mode})...")

    # 1. Select the single nearest grid point from the RAW dataset
    try:
        # --- v7 FIX: Manual 2D Coordinate Selection ---
        lat_coord_name = _find_coord_name(ds, ['latitude', 'lat', 'y'])
        lon_coord_name = _find_coord_name(ds, ['longitude', 'lon', 'x'])

        if not lat_coord_name:
            raise ValueError("Could not find a latitude-like coordinate (latitude, lat, y) in the dataset.")
        if not lon_coord_name:
            raise ValueError("Could not find a longitude-like coordinate (longitude, lon, x) in the dataset.")

        # Load the 2D coordinate arrays
        lats_2d = ds[lat_coord_name].values
        lons_2d = ds[lon_coord_name].values
        target_lon = lon  # Keep a copy of the user's -180 to 180 lon

        # HRRR GRIBs often use 0-360 longitude.
        # If so, convert our target longitude to match.
        if np.max(lons_2d) > 180. and target_lon < 0:
            target_lon = target_lon + 360

        # Calculate the squared distance (faster than sqrt)
        dist_sq = (lats_2d - lat) ** 2 + (lons_2d - target_lon) ** 2

        # Find the (y, x) indices of the minimum distance
        min_idx_flat = np.argmin(dist_sq)
        iy, ix = np.unravel_index(min_idx_flat, lats_2d.shape)

        print(f"  Target: ({lat}, {lon}). Found nearest grid point at index ({iy}, {ix}).")

        # Select the single point using integer selection (isel)
        ds_point = ds.isel(y=iy, x=ix)
        # --- End v7T FIX ---

    except Exception as e:
        print(f"Error: Could not select nearest grid point for {lat}, {lon}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        print("  This can happen if the Herbie download failed or coordinates are named unexpectedly.", file=sys.stderr)
        return

    # 2. Extract and prepare 1D data arrays
    try:
        # --- v65: CRITICAL FIX ---
        # Run parse_cf() FIRST to get isobaricInhPa coordinate
        print("  Parsing point data with MetPy...")
        ds_point = ds_point.metpy.parse_cf()

        # --- Extract 3D Profile Data ---
        print("  Extracting 3D profile data...")
        p = ds_point.isobaricInhPa.values * units.hPa

        t_name = _find_var_in_ds(ds_point, ['t', 'tmp'])
        dpt_name = _find_var_in_ds(ds_point, ['dpt', 'd'])
        u_name = _find_var_in_ds(ds_point, ['u', 'ugrd'])
        v_name = _find_var_in_ds(ds_point, ['v', 'vgrd'])
        z_name = _find_var_in_ds(ds_point, ['gh', 'hgt'])
        wz_name = _find_var_in_ds(ds_point, ['wz', 'vvel', 'omega'])

        if not all([t_name, dpt_name, u_name, v_name, z_name]):
            raise ValueError(
                f"Missing one or more required 3D variables (t, dpt, u, v, gh). Found: t={t_name}, dpt={dpt_name}, u={u_name}, v={v_name}, z={z_name}")

        t = ds_point[t_name].values * units.K
        td = ds_point[dpt_name].values * units.K
        u = ds_point[u_name].values * units('m/s')
        v = ds_point[v_name].values * units('m/s')
        z_msl_gpm = ds_point[z_name].values * units.gpm

        omega = None
        if wz_name:
            omega = ds_point[wz_name].values * units('Pa/s')
        else:
            print("  Warning: Vertical velocity (Omega) not found. Skipping Omega plot.")

        # --- Extract Surface Data Separately ---
        # This avoids the name conflict that caused the -64.4 C bug.
        sp_name = _find_var_in_ds(ds_point, ['sp', 'pres'])
        # v69: Use full unambiguous names
        t2m_name = _find_var_in_ds(ds_point, ['tmp:2 m above ground', 't2m'])
        d2m_name = _find_var_in_ds(ds_point, ['dpt:2 m above ground', 'd2m'])

        # Fallback for surface T/Td
        if not t2m_name: t2m_name = t_name  # Use 3D var
        if not d2m_name: d2m_name = dpt_name  # Use 3D var

        if not all([sp_name, t2m_name, d2m_name]):
            raise ValueError("Missing one or more required surface variables (sp, t2m, d2m).")

        sfc_p = ds_point[sp_name].values * units.Pa

        # Handle 2m vs lowest-level fallback
        if t2m_name == t_name:  # Using fallback
            print("  Warning: Missing 2m T/D. Using lowest model level T/D as surface value.")
            # Find index of max pressure (lowest level)
            sfc_idx = np.argmax(p)
            sfc_t = t[sfc_idx]
            sfc_td = td[sfc_idx]
        else:  # Normal case
            sfc_t = ds_point[t2m_name].values * units.K
            sfc_td = ds_point[d2m_name].values * units.K

        # Convert to Celsius for plotting
        sfc_t_c = sfc_t.to('degC')
        sfc_td_c = sfc_td.to('degC')

        print(f"  Surface values: {sfc_p.to('hPa'):.1f}, {sfc_t_c:.1f}, {sfc_td_c:.1f}")
        # --- End v65 FIX ---

        # Convert 3D profile to Celsius
        t_c = t.to('degC')
        td_c = td.to('degC')

        # v41: Add omega to the nan_mask if it exists
        nan_vars = [p, t_c, td_c, u, v, z_msl_gpm]
        if omega is not None:
            nan_vars.append(omega)

        # Create mask by checking isnan on each variable's magnitude
        nan_mask_list = [np.isnan(var.magnitude) for var in nan_vars]
        nan_mask = np.logical_or.reduce(nan_mask_list)
        nan_mask = ~nan_mask  # Invert to get non-NaN mask

        p = p[nan_mask]
        t_c = t_c[nan_mask]
        td_c = td_c[nan_mask]
        u = u[nan_mask]
        v = v[nan_mask]
        z_msl_gpm = z_msl_gpm[nan_mask]
        if omega is not None:
            omega = omega[nan_mask]  # v41

        # Sort by pressure (descending, surface-first)
        sort_indices = np.argsort(p)[::-1]
        p = p[sort_indices]
        t_c = t_c[sort_indices]
        td_c = td_c[sort_indices]
        u = u[sort_indices]
        v = v[sort_indices]
        z_msl_gpm = z_msl_gpm[sort_indices]
        if omega is not None:
            omega = omega[sort_indices]  # v41

        # v15: Get height (z) for hodo and metrics
        z_msl = z_msl_gpm.to('m')  # Convert geopotential meters to meters

        # Find surface height.
        try:
            sfc_p_val = sfc_p.to('hPa').magnitude
            p_vals_asc = p.to('hPa').magnitude[::-1]
            z_msl_vals_asc = z_msl.to('m').magnitude[::-1]

            sfc_z_msl_val = np.interp(sfc_p_val, p_vals_asc, z_msl_vals_asc)
            sfc_z_msl = sfc_z_msl_val * units.m
        except Exception as interp_e:
            print(f"  Warning: Could not interpolate surface height: {interp_e}", file=sys.stderr)
            # v59 FIX: This is the fallback, and it must assign to `sfc_z_msl`
            sfc_z_msl = z_msl[0]

        z_agl = z_msl - sfc_z_msl

        # v64: Remove all 10m wind and surface wind fallback logic
        # v64: Remove all np.concatenate (prepend) logic.
        # The p, t_c, td_c, u, v, z_agl, z_msl arrays now *only*
        # contain the pressure-level data. sfc_* variables are
        # passed separately to functions that need them.

        # v39: Pre-calculate Wet-Bulb Temp
        # v46: Refactored height/pressure calculations
        lowest_wb_zero_p = np.nan * units.hPa
        wbz_hgt_agl = np.nan * units.m
        frz_p = np.nan * units.hPa
        frz_hgt_agl = np.nan * units.m

        try:
            # --- Wet-Bulb Zero (Bottom-Up) ---
            wet_bulb_temp = mpcalc.wet_bulb_temperature(p, t_c, td_c)
            # Find first crossing from surface (descending p)
            lowest_wb_zero_p = _find_bottom_up_crossing(0 * units.degC, wet_bulb_temp, p)

            if not np.isnan(lowest_wb_zero_p):
                # Interpolate z_agl at that pressure
                wbz_hgt_agl = _interp_to_p(lowest_wb_zero_p, p, z_agl)
        except Exception:
            pass  # Keep as NaN

        try:
            # --- Freezing Level (Bottom-Up) ---
            # Find first crossing from surface (descending p)
            frz_p = _find_bottom_up_crossing(0 * units.degC, t_c, p)

            if not np.isnan(frz_p):
                # Interpolate z_agl at that pressure
                frz_hgt_agl = _interp_to_p(frz_p, p, z_agl)
        except Exception:
            pass  # Keep as NaN


    except Exception as e:
        print(f"Error: Failed to process data arrays for MetPy: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return

    # 3. Create the Skew-T plot
    print(f"  Creating Skew-T and Hodograph (Mode: {mode})...")

    # v50: Revert to v48 layout
    fig = plt.figure(figsize=(17, 12))
    skew_rect = [0.05, 0.1, 0.60, 0.8]
    omega_rect = [0.65, 0.1, 0.05, 0.8]
    hodo_rect = [0.72, 0.6, 0.25, 0.3]

    # Initialize the Skew-T
    skew = mpplots.SkewT(fig, rect=skew_rect, rotation=45)

    # v39: Initialize the Omega plot
    omega_ax = fig.add_axes(omega_rect)

    # Initialize the Hodograph
    hodo_ax = fig.add_axes(hodo_rect)
    hodo = mpplots.Hodograph(hodo_ax, component_range=100.)
    hodo.add_grid(increment=20)
    # --- End v50 Layout ---

    # Plot Temperature and Dewpoint (Same for both modes)
    skew.plot(p, t_c, 'r', linewidth=2, label='Temperature')
    skew.plot(p, td_c, 'g', linewidth=2, label='Dewpoint')

    # v69: Use modern MetPy `barb_increments`
    # v70: BUG FIX: Moved this call to *after* the if/elif block
    # v72: REMOVED `barb_increments=50` - it caused a TypeError
    # skew.plot_barbs(p, u, v, barb_increments=50)

    # v44: Always set up the Omega axis Y-limits
    omega_ax.set_ylim(1050, 100)  # Match Skew-T
    omega_ax.set_yticks([])  # No Y-labels

    # v41: Plot Omega Profile only if it exists
    if omega is not None:
        try:
            omega_ax.axvline(0, color='gray', linestyle='--', linewidth=1, zorder=1)
            omega_ax.plot(omega.to('Pa/s'), p, 'b-', linewidth=1.5, zorder=2)
            # v44: Keep Y-axis, just set label
            omega_ax.set_xlabel('Omega (Pa/s)')
            omega_ax.xaxis.set_label_position('top')
            omega_ax.tick_params(axis='x', labelsize=8)
            # Set symmetric X limits
            max_omega_abs = np.nanmax(np.abs(omega.to('Pa/s').magnitude))
            if np.isnan(max_omega_abs) or max_omega_abs == 0:
                max_omega_abs = 1
            omega_ax.set_xlim(-max_omega_abs, max_omega_abs)
        except Exception as omega_e:
            print(f"  Warning: Could not plot Omega profile: {omega_e}", file=sys.stderr)
    else:
        # v50 FIX: Manually set xlim *before* hiding spines
        # This gives the axis a valid coordinate system for annotations.
        omega_ax.set_xlim(-1, 1)
        omega_ax.set_xticks([])
        omega_ax.set_xlabel('')
        for spine in omega_ax.spines.values():
            spine.set_visible(False)

    # Plot Hodograph (Same for both modes)
    hodo.plot_colormapped(u.to('kts'), v.to('kts'), z_agl, cmap='jet')

    # Add standard sounding lines (Same for both modes)
    skew.plot_dry_adiabats(color='gray', linestyle=':', linewidth=0.5)
    skew.plot_moist_adiabats(color='gray', linestyle=':', linewidth=0.5)
    skew.plot_mixing_lines(color='gray', linestyle=':', linewidth=0.5)

    # --- v2: MODE-SPECIFIC PLOTTING ---
    if mode == 'convective':
        # Set axis limits for convective
        skew.ax.set_ylim(1050, 100)
        skew.ax.set_xlim(-30, 50)  # Wide view for CAPE

        # v44: Need to also limit omega axis for convective mode
        omega_ax.set_ylim(1050, 100)

        # Calculate and plot Surface-Based Parcel Path
        try:
            sb_prof = mpcalc.parcel_profile(p, sfc_t_c, sfc_td_c).to('degC')
            skew.plot(p, sb_prof, 'k', linestyle='--', label='SFC Parcel Path')

            # Shade CAPE/CIN
            skew.shade_cin(p, t_c, sb_prof)
            skew.shade_cape(p, t_c, sb_prof)
        except Exception as e:
            print(f"  Warning: Could not calculate parcel profile: {e}", file=sys.stderr)

    elif mode == 'winter':
        # v71: Set axis limits to full view (same as convective)
        skew.ax.set_ylim(1050, 100)
        skew.ax.set_xlim(-30, 50)  # Standard wide view

        # v71: Match omega axis to full view
        omega_ax.set_ylim(1050, 100)

        # Highlight 0C line
        # v71: Extend line to full 100 hPa height
        skew.plot([1050, 100] * units.hPa, [0, 0] * units.degC,
                  color='blue', linestyle='--', linewidth=2)

        # v38: Re-add DGZ shading
        try:
            # Interpolate to find pressure at -12C and -18C
            t_asc = t_c.magnitude[::-1]
            p_asc = p.magnitude[::-1]

            p_at_minus_12 = np.interp(-12.0, t_asc, p_asc)
            p_at_minus_18 = np.interp(-18.0, t_asc, p_asc)

            if not np.isnan(p_at_minus_12) and not np.isnan(p_at_minus_18):
                # Shade the DGZ
                skew.ax.axhspan(
                    p_at_minus_12, p_at_minus_18,
                    facecolor='cyan', alpha=0.3, zorder=0
                )
        except Exception as dgz_e:
            print(f"  Warning: Could not shade DGZ: {dgz_e}", file=sys.stderr)

        # v52: Removed FRZ/WBZ annotations

    # --- End v2 ---

    # v70: BUG FIX - Plot barbs *after* axis limits are set
    # v72: Re-implementing manual 50 hPa slicing.
    # The `barb_increments` kwarg is not compatible with older MetPy.
    # We use a robust modulo check to handle floating-point noise
    # (e.g., 999.9999 % 50 = 49.9999, which is > 49)
    p_mod_50 = p.magnitude % 50
    is_close_to_50_interval = (p_mod_50 < 1) | (p_mod_50 > 49)
    barb_indices = is_close_to_50_interval & (p.magnitude >= 100)
    skew.plot_barbs(p[barb_indices], u[barb_indices], v[barb_indices])

    skew.ax.legend()

    # --- v14: Calculate and Plot Metrics Text ---
    # We pass all data with units
    # v39: Pass lowest_wb_zero_p
    # v46: Pass frz_hgt_agl, wbz_h_agl
    _calculate_and_plot_metrics(
        fig, hodo, p, t_c, td_c, u, v, z_agl, z_msl, mode, sfc_p, sfc_t_c, sfc_td_c,
        frz_hgt_agl, wbz_hgt_agl
    )
    # --- End v14 ---

    # 4. Add Titles and Save
    loc_name = f"({lat:.2f}, {lon:.2f})"
    title = f"HRRR Model Sounding for {loc_name}\n" \
            f"Valid: {dt.strftime('%Y-%m-%d %H:%M')} UTC"
    # v22: Place title in new top margin
    fig.suptitle(title, fontsize=16, y=0.95)

    # Save the file
    # Use a format that is safe for file names (replace . with _)
    lat_str = f"{abs(lat):.2f}N" if lat >= 0 else f"{abs(lat):.2f}S"
    lon_str = f"{abs(lon):.2f}W" if lon <= 0 else f"{abs(lon):.2f}E"
    lat_str = lat_str.replace('.', '_')
    lon_str = lon_str.replace('.', '_')

    # v30: CRITICAL FIX - Removed colon from strftime
    filename = f"HRRR_Sounding_{dt.strftime('%Y%m%d%H%M')}_{lat_str}_{lon_str}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nSuccessfully saved plot:\n  {filepath}")
    plt.close(fig)


# --- Main Execution ---

def main():
    """
    Main function to run the interactive sounding analyzer.
    """
    print("--- HRRR Sounding Analyzer (v72) ---")  # v72
    print("Welcome! Let's plot a Skew-T.")

    # 1. Get Date/Time
    dt = get_datetime_input("Enter Date/Time for the sounding")

    # 2. Get Location
    lat, lon = get_location_input()

    # v2: Get Analysis Mode
    mode = get_analysis_mode()

    # 3. Create Event-Specific Output Directory
    event_name = dt.strftime(f'%Y%m%d_%H%M_sounding')
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

    # 4. Load Data with Herbie
    try:
        ds_full = load_sounding_data(dt, event_output_dir)
    except KeyboardInterrupt:
        print("\n*** User interrupted (CTRL-C) during download ***")
        print("Processing stopped.")
        sys.exit(1)

    if ds_full is None:
        print("Fatal Error: Could not load any HRRR data. Exiting.", file=sys.stderr)
        sys.exit(1)

    # 5. Plot the sounding
    try:
        plot_sounding(ds_full, lat, lon, event_output_dir, dt, mode)
    except Exception as e:
        print(f"\n--- FATAL PLOTTING ERROR ---", file=sys.stderr)
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        print("---------------------------------", file=sys.stderr)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    # Set backend for Matplotlib
    try:
        plt.switch_backend('Agg')
    except Exception as e:
        print(f"Could not switch matplotlib backend: {e}", file=sys.stderr)

    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Slicing is not supported")
    # v14: Suppress new MetPy warning
    warnings.filterwarnings("ignore", message="The unit 'm' has been non-dimensionalized")
    # v15: Suppress interpolation warning
    warnings.filterwarnings("ignore", message="Warning: 'interp_type' was not specified")
    # v46: Suppress log_interpolate_1d warning
    warnings.filterwarnings("ignore", message="Interpolation point out of data range")

    main()