
'''
The goal is to harmonise the gridded dataset with the IAM projections. For this purpose, each grid cell in the region is multiplied by a correction factor
that is calculated as the ratio of the target regional value from the IAM data to the current regional sum from the gridded data.
Input files are
- IAM projections based on IAMC template
- Gridded dataset (rioxarray DataArray or Dataset) including data (e.g. population, GDP) from 2020 until 2100 at 5-10 year intervals
- Region dataset (rioxarray DataArray) with values 1-NrRegions for land regions (a 0 will be added for ocean cells)

Step 1: Data Inventory and Setup

Keep original temporal resolutions: Leave rioxarray datasets at their native 5-10 year intervals, pandas dataframes at annual resolution
Coordinate system check: Ensure all rioxarray datasets share the same x, y coordinate system
Region handling: Confirm region rioxarray maps to values 0-26 (0=ocean, 1-26=land regions)

Step 2: Create Efficient Regional Setup

Ocean mask only: Create one mask for ocean cells: ocean_mask = (region_array == 0)
Land mask (optional): land_mask = (region_array > 0) if you want to exclude ocean areas efficiently
Use region array directly: The region rioxarray itself serves as your spatial lookup - no need for 26 separate boolean masks

Step 3: Prepare grid and model datasets for downscalig

Harmonise grid and projections data with downscalig years (which are defined settings)

Step 4: Calculate Regional Sums (more efficient approach)
For each target year t:

Interpolate gridded data: gridded_data_t = gridded_data.interp(time=t, method='linear')
Group by region and sum: Use xarray's groupby functionality:

python  # Conceptual example
  regional_sums = gridded_data_t.where(land_mask).groupby(region_array).sum()

This gives you current_regional_sum(r,t) for all regions r=1-26 in one operation
Step 5: Extract Target Regional Values

Directly access values from pandas dataframes for year t and region r
Result: target_regional_value(r,t) - no temporal processing needed since you're working with the dataframe's native timeline

Step 6: Calculate Correction Factors
For each region r and year t:

Calculate ratio: correction_factor(r,t) = target_regional_value(r,t) / current_regional_sum(r,t)
Handle edge cases: Division by zero, missing data, etc.

Step 7: Apply Harmonization (updated approach)
For each target year t:

Interpolate original data: original_grid_t = original_gridded_data.interp(time=t)
Create correction factor array: Build a spatial array where each grid cell contains the correction factor for its region

python  # Conceptual: correction_factor_spatial[x,y] = correction_factor[region_array[x,y], t]

Apply corrections: harmonized_grid_t = original_grid_t * correction_factor_spatial
Handle ocean: Set ocean cells appropriately (usually leave unchanged or set to 0)

Step 8: Build Output Dataset

Temporal structure: Decide whether to:

Keep all annual results, or
Subset to specific years of interest, or
Resample to coarser temporal resolution


Coordinate preservation: Maintain original spatial coordinate system and metadata

Step 9: Validation

Regional sum check: Verify sum(harmonized_grid[region_masks[r], t]) = target_regional_value(r,t)
Spatial pattern check: Confirm relative patterns within regions are preserved
Temporal continuity: Check for reasonable temporal evolution

Step 10: Memory and Performance Optimization (if needed)

Process in chunks: Handle subset of years or regions at a time if memory becomes limiting
Caching: Store frequently used interpolated data temporarily
Alternative: If this becomes too slow/memory intensive, switch to pre-interpolation approach
'''


from pathlib import Path
import time
from datetime import datetime
import re
import glob
import time
import logging
import warnings
import importlib
import logging
import gc

from datetime import datetime
from tabulate import tabulate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xarray as xr
import rioxarray as rxr
from rasterio.transform import from_bounds
from rasterio.errors import NotGeoreferencedWarning
from affine import Affine
from osgeo import gdal

from dask.diagnostics import ProgressBar

import settings_downscaling as dss
import functions_downscaling as dsf
import functions_processing_hist_data as pdf
import functions_read_process_IAM_data as rpd
import functions_logging as log

pd.set_option('display.max_rows', 25)

# Set project_dir to one directory above current working directory
current_dir = Path().cwd()
print("Current working directory:", current_dir)
project_dir = Path.cwd().parent
print(f"Current path: {project_dir.name}")

log.init_logging(project_dir)


# Import own modules
# reload modules
importlib.reload(dss)
importlib.reload(dsf)
importlib.reload(pdf)
importlib.reload(rpd)


# INPUT
# Input files for gridded data to be harmonized
SSP_base = "SSP2"
scenario = "SSP2_Default" # "SSP2_Default_26"
model = "IMAGE"

# Downscale settings
start_year = 2020
end_year = 2100
base_year = 2020

# Population (2UP)
source = "2UP"
coarse_factor = 12
varname = "Population"
varname_save = dsf.clean_varname_for_saving(varname)
SE_grid_file=f"{project_dir}/data/processed/{varname_save}_{source}_{SSP_base}_cf_{coarse_factor}.nc"
conversion_factor_IAM_to_grid = rpd.model_unit_conversions[model]["Population"]
years_downscaling = [2020, 2025, 2030, 2035, 2040, 2045, 2050, 2060, 2070, 2080, 2090, 2100]

if not Path(SE_grid_file).is_file():
    raise FileNotFoundError(f"Gridded data file not found: {SE_grid_file}")
else:
    print(f"Using gridded data file: {SE_grid_file}")

# save, show, clean settings
save_netcdf = True
save_figs = True
show_figs = True
check_values = False
clean_memory = False

# Set dask progress bar
pbar = ProgressBar()
pbar.register()
# # Step 1: Data Inventory and Setup

# Keep original temporal resolutions: Leave rioxarray datasets at their native 5-10 year intervals, pandas dataframes at annual resolution
# Coordinate system check: Ensure all rioxarray datasets share the same x, y coordinate system
# Region handling: Confirm region rioxarray maps to values 0-26 (0=ocean, 1-26=land regions)

# Read in IAM model grid data for region definitions

'''
IAM results must be in IAMC format with region codes
The years must cover the downscaling years (flexible, but default is [2020, 2025, 2030, 2035, 2040, 2045, 2050, 2060, 2070, 2080, 2090, 2100])
Abundant years will be deleted, missing years will be (linearly) interpolated
'''

# Input files for IAM regional projections
file_IAM_model_country_region = "data/input/models/IMAGE/country_to_regions.csv" # ISO3, Country name, Region code (matches region code IAMC template)
file_IAM_model_region_numbers = "data/input/models/IMAGE/image_region_numbers.csv" # ISO3, Country name, Region code (matches region code IAMC template)
file_IAM_regions_grid = "data/input/models/IMAGE/GREG.nc" # netcdf file with model regions on 0.5x0.5 grid

# Read in regional IAM IMAGE data
vars_IAM_projection_se_indicator = [varname]
print(f"Variables to be harmonized: {vars_IAM_projection_se_indicator}")

# read IMAGE regional data
df_IAM_projection = rpd.read_IMAGE_regions_data(scenario)

# Process IMAGE data only for CO2 emissions variable
if varname == "Emissions|CO2|Excl. shipping, aviation, AFOLU":
    df_IAM_projection = rpd.process_IMAGE_regions_data(df_IAM_projection)

# Subset to selected variable (exclude World aggregate)
mask_IAM_projection_se_indicator = (
    df_IAM_projection["Variable"].isin(vars_IAM_projection_se_indicator)
    & (df_IAM_projection["Region"] != "World")
)
df_IAM_projection_se_indicator = df_IAM_projection[mask_IAM_projection_se_indicator].copy()

# Ensure the variable exists
if df_IAM_projection_se_indicator.empty:
    raise ValueError(
        f"The variable '{varname}' was not found in IAM data. "
        f"Available variables: {sorted(df_IAM_projection['Variable'].unique())}"
    )

# Determine number of regions present (after excluding World)
nr_regions = df_IAM_projection_se_indicator["Region_number"].nunique()
unit_SE = df_IAM_projection_se_indicator["Unit"].unique()

# Display the filtered dataframe (helpful for notebook inspection)
print(df_IAM_projection_se_indicator["Year"].unique())
df_IAM_projection_se_indicator[df_IAM_projection_se_indicator["Year"].isin([2020, 2030])].round(0)

# Read in SE grid data
# See process_se_indicator_2UP.ipynb
xr_SE_grid = None
if Path(SE_grid_file).exists():
    xr_SE_grid = dsf.read_netcf_rio_file(filename=SE_grid_file)
    xr_SE_grid = xr_SE_grid.chunk({"time": -1, "y": "auto", "x": "auto"})

    # check time steps
    time_steps_se_indicator = np.unique(xr_SE_grid["time"].values)
    logging.info(f"Time steps in gridded se_indicator data: {time_steps_se_indicator}")
    # check size of time step
    time_step1 = np.diff(time_steps_se_indicator).min()
    time_step2 = np.diff(time_steps_se_indicator).max()

# Read in IAM regions grid data
xr_IAM_regions_grid = rpd.read_grid_data_IAM_regions(model=model, filename_region_grid=f"{project_dir}/{file_IAM_regions_grid}")

# reindex IAM regions grid to SE grid

# checks before reindexing
if check_values:
    print("--------------------------------")
    reg_sec, reg_min, reg_deg = dsf.calculate_resolution(xr_IAM_regions_grid)
    print(f"Type: {type(xr_IAM_regions_grid)}")
    print(f"IAM regions grid resolution: {reg_sec:.3f} seconds, {reg_min:.3f} minutes, {reg_deg:.3f} degrees")
    se_sec, se_min, se_deg = dsf.calculate_resolution(xr_SE_grid)
    print(f"Type: {type(xr_SE_grid)}")
    print(f"SE grid resolution: {se_sec:.3f} seconds, {se_min:.3f} minutes, {se_deg:.3f} degrees")
    print("--------------------------------")

# reindex IAM regions grid to SE grid
xr_IAM_regions_grid_downscaling = xr_IAM_regions_grid.reindex_like(xr_SE_grid.sel(time=2020), method="nearest")
dsf.compare_reindexing(xr_IAM_regions_grid_downscaling.region_number, xr_IAM_regions_grid_downscaling.region_number)

# checks after reindexing
if check_values:
    print("--------------------------------")
    reg_sec, reg_min, reg_deg = dsf.calculate_resolution(xr_IAM_regions_grid_downscaling)
    print(f"Type: {type(xr_IAM_regions_grid_downscaling)}")
    print(f"IAM regions grid resolution: {reg_sec:.3f} seconds, {reg_min:.3f} minutes, {reg_deg:.3f} degrees")
    se_sec, se_min, se_deg = dsf.calculate_resolution(xr_SE_grid)
    print(f"Type: {type(xr_SE_grid)}")
    print(f"SE grid resolution: {se_sec:.3f} seconds, {se_min:.3f} minutes, {se_deg:.3f} degrees")
    print("--------------------------------")

# delete old xr_IAM_regions_grid to save memory
if clean_memory: del xr_IAM_regions_grid

# check units grid and IAM data
if check_values:
    df_2020_World = df_IAM_projection_se_indicator.groupby(["Model", "Scenario", "Year", "Variable", "Unit"])["Value"].sum().reset_index()
    df_2020_World = df_2020_World[df_2020_World["Year"] == 2020]
    print(f"Tabulated DataFrame for 2020:\n{tabulate(df_2020_World)}")
    # sum values for the year 2020 for xr_SE_grid
    xr_2020 = xr_SE_grid[varname].sel(time=2020).sum()
    print(f"Sum of gridded data for year 2020: {xr_2020.values:,.2f}, sum of IAM projections for year 2020: {df_2020_World['Value'].values[0]:,.2f} {unit_SE[0]}0")

# Step 3: Prepare grid and model datasets for harmonisation

# Ocean mask only: Create one mask for ocean cells: ocean_mask = (region_array == 0)
# Land mask (optional): land_mask = (region_array > 0) if you want to exclude ocean areas efficiently
# Use region array: The region rioxarray serves as your the lookup


land_mask = (xr_IAM_regions_grid_downscaling["region_number"] > 0)
ocean_mask = (xr_IAM_regions_grid_downscaling["region_number"] == 0)


# Use pandas dataframe time index as master timeline (annual, 2020-2100)
# This becomes the reference timeline for harmonization calculations
# Extract list of target years for processing

# Harmonise IAM model projections with defined downscaling years
df_IAM_projection_se_indicator_downscaling = df_IAM_projection_se_indicator.copy()
if clean_memory: del df_IAM_projection_se_indicator
if clean_memory: gc.collect()

# Use lower case column names to align with gridded dataset
df_IAM_projection_se_indicator_downscaling.columns = df_IAM_projection_se_indicator_downscaling.columns.str.lower()
print(f"Original years:\n {df_IAM_projection_se_indicator_downscaling["year"].unique()}")

# Keep only matching years (no interpolation)
df_IAM_projection_se_indicator_downscaling = df_IAM_projection_se_indicator_downscaling[df_IAM_projection_se_indicator_downscaling["year"].isin(years_downscaling)].copy()
print(f"Only include downscaling years:\n {df_IAM_projection_se_indicator_downscaling["year"].unique()}")

# Interpolate to exactly target_years
# group by identifying columns (everything except Year and Value)
id_cols = ["model", "scenario", "region", "variable", "unit", "region_number"]

# Ensure Year is int
df_IAM_projection_se_indicator_downscaling["year"] = df_IAM_projection_se_indicator_downscaling["year"].astype(int)
df_IAM_projection_se_indicator_downscaling["value"] = df_IAM_projection_se_indicator_downscaling["value"].astype(float)

# Reindex each group to include all downscaling years, then interpolate
def reindex_and_interp(group):
    # keys are the values of the id_cols for this group
    keys = dict(zip(id_cols, group.name if isinstance(group.name, tuple) else (group.name,)))

    return (
        group.set_index("year")
             .reindex(years_downscaling)
             .assign(**keys)
             .assign(value=lambda g: g["value"].interpolate("linear", limit_area="inside"))
             .reset_index())

df_IAM_projection_se_indicator_downscaling = (
    df_IAM_projection_se_indicator_downscaling.groupby(id_cols, group_keys=False)
      .apply(reindex_and_interp)
      .reset_index()
)
df_IAM_projection_se_indicator_downscaling.drop(columns=["index"], inplace=True)
print(f"Add years to align with downscaling years (interpolation):\n {df_IAM_projection_se_indicator_downscaling["year"].unique()}")
df_IAM_projection_se_indicator_downscaling.to_csv(f"{project_dir}/data/check/step3_IAM_projection_{model}_{scenario}_{varname_save}_downscale_years.csv", sep=";", index=False)
df_IAM_projection_se_indicator_downscaling.head(10).style.format({"value": "{:,.2f}"})


# Process IAM projections to harmonize with gridded SE data
df_IAM_projection_se_indicator_downscaling.rename(columns={"region": "region_code"}, inplace=True)

# convert units with conversion factor
df_IAM_projection_se_indicator_downscaling["value"] *= conversion_factor_IAM_to_grid

# add OCEAN as region 0 to IAM projection se_indicator dataframe
model = df_IAM_projection_se_indicator_downscaling["model"].unique()[0]
scenario = df_IAM_projection_se_indicator_downscaling["scenario"].unique()[0]
years = df_IAM_projection_se_indicator_downscaling["year"].unique()
variable = df_IAM_projection_se_indicator_downscaling["variable"].unique()[0]
unit = df_IAM_projection_se_indicator_downscaling["unit"].unique()[0]
extra_rows = pd.DataFrame({"model": model, "scenario": scenario, "region_code":"OCEAN", "variable":variable, "year": years, "unit": unit, "region_number": 0, "value": 0})
df_IAM_projection_se_indicator_downscaling = pd.concat([df_IAM_projection_se_indicator_downscaling, extra_rows], ignore_index=True).sort_values(["year", "region_number"]).reset_index(drop=True)
df_IAM_projection_se_indicator_downscaling.to_csv(f"{project_dir}/data/check/step4_IAM_projection_{model}_{scenario}_{varname_save}_downscale_years.csv", sep=";", index=False)

# check
df_IAM_projection_se_indicator_downscaling.groupby(["year"])["value"].sum().reset_index().style.format({"year": "{:d}", "value": "{:,.0f}"}, thousands=",", decimal=".")
df_IAM_projection_se_indicator_downscaling[df_IAM_projection_se_indicator_downscaling["year"]==2030].style.format({"value": "{:,.2f}"})


# align grids from xr_SE_grid with downscaling years
xr_SE_grid_downscaling = xr_SE_grid.interp(time=years_downscaling, method="linear")
if clean_memory: del xr_SE_grid
gc.collect()
print(f"Years in se_indicator grid aligned with downscaling years: {xr_SE_grid_downscaling.time.values}")

# # Step 4: Calculate Regional grid Sums for gridded data


# For each target year t in dataframe timeline:
# For se_indicator:

# Interpolate gridded data: se_data_t = xr_SE_grid.interp(time=t, method='linear')
# Calculate regional sums: xr_se_regional_sums_t = se_data_t.where(land_mask).groupby(xr_IAM_regions_grid).sum()
# Result: current_regional_sum_se(r,t) for regions r=1-26

# Create grid with regional sums for se_indicator (for emissions, only base year)
xr_se_regional_sums = None
df_output = None

# Calculate regional sums for all downscaling years
# determine region numbers for each grid cell
print("Determine region numbers for each grid cell")
region_numbers = xr_IAM_regions_grid_downscaling.region_number.compute()

# determine regional sums for grid data
print("Determine regional sums for grid data")
xr_se_regional_sums = (xr_SE_grid_downscaling
                        .sel(time=years_downscaling)
                        .where(land_mask)
                        .groupby(region_numbers)
                        .sum())

df_se_regional_sums_compare = None

logging.info(f"Type: {type(xr_se_regional_sums)}")
print(varname)
xr_se_regional_sums_check = xr_se_regional_sums.rename({varname: f"{varname}_grid"})
print(f"Unique region numbers in regional summations: {np.unique(xr_se_regional_sums.region_number.values)}")
print(f"Years in regional summations: {xr_se_regional_sums_check.time.values}")
print(f"Chunks: {xr_se_regional_sums_check.chunks}")

#---------------------
# Compare grid and IAM regional sums for se_indicator (for emissions, only 2020)
# for comparison, different variable names are created
varname_grid = f"{varname}_grid"
varname_grid_IAM = f"{varname}_IAM"
varname_grid_summed = f"{varname}_grid_summed"

# compare se_indicator from xr_se_regional_sums_check with xr_IAM_regions_grid_downscaling and add to csv file
df_se_regional_sums_compare = pd.merge(
    xr_se_regional_sums_check[varname_grid].to_dataframe().reset_index().rename(columns={"region_number": "region_number", "time": "year", varname_grid: varname_grid_summed}),
    df_IAM_projection_se_indicator_downscaling.rename(columns={"value": varname_grid_IAM})[["region_number", "year", varname_grid_IAM]],
    on=["region_number", "year"],
    how="left"
)
df_se_regional_sums_compare = df_se_regional_sums_compare[df_se_regional_sums_compare["year"].isin(years_downscaling)].copy()

df_se_regional_sums_compare.to_csv(f"{project_dir}/data/check/df_{varname_save}_{source}_regional_sums_{scenario}_{model}.csv", sep=";", index=False)
df_se_regional_sums_compare["difference"] = df_se_regional_sums_compare[varname_grid_summed] - df_se_regional_sums_compare[varname_grid_IAM]
df_se_regional_sums_compare["relative_difference_%"] = df_se_regional_sums_compare["difference"] / df_se_regional_sums_compare[varname_grid_IAM] * 100
df_se_regional_sums_compare = df_se_regional_sums_compare.sort_values(["year", "region_number"]).reset_index(drop=True)
df_se_regional_sums_compare["se_indicator_grid_xr_million"] = df_se_regional_sums_compare[varname_grid_summed] * 10**-6
df_se_regional_sums_compare["se_indicator_df_million"] = df_se_regional_sums_compare[varname_grid_IAM] * 10**-6

# save to csv
df_se_regional_sums_compare.to_csv(f"{project_dir}/data/check/step4_{varname_save}_{source}_regional_sums_comparison_{scenario}_{model}.csv", sep=";", index=False)
df_se_regional_sums_compare_World = df_se_regional_sums_compare.groupby(["year"])[[varname_grid_summed,varname_grid_IAM]].sum().reset_index()
df_se_regional_sums_compare_World.to_csv(f"{project_dir}/data/check/{varname_save}_{source}_regional_sums_comparison_totals_World_{scenario}_{model}.csv", sep=";", index=False)

print(f"se_indicator regional sums comparison saved to {project_dir}/data/check/{varname_save}_regional_sums_comparison_{scenario}_{model}.csv")

if df_se_regional_sums_compare is not None:
    df_se_regional_sums_compare = df_se_regional_sums_compare[df_se_regional_sums_compare["year"]==2030]

    df_output = df_se_regional_sums_compare.style.format(precision=0, thousands=",", decimal=".")

# Step 5: Calculate Cell-Specific Redistribution (correction factors)

# Access values directly from pandas dataframes for year t and region r
# For se_indicator: target_se_value(r,t) from df_se_indicator_regional
# For GDP: target_gdp_value(r,t) from df_GDP_regional
# No temporal processing needed since working with dataframe's native timeline


# Calculate correction factors and redistribute se_indicator (for emissions, only base year)
# Different methods in case of memory issues

print("Calculate correction factors and redistribute se_indicator...")
# Prepare harmonised (target) se_indicator values from IAM projections
xr_harmonised_se_indicator = (df_IAM_projection_se_indicator_downscaling
                              .set_index(['year', 'region_number'])['value']
                              .to_xarray()
                              .sel(year=years_downscaling, region_number=xr_se_regional_sums.region_number))

print(f"Unique region numbers in harmonised se_indicator data: {np.unique(xr_harmonised_se_indicator.region_number.values)}")
xr_harmonised_se_indicator = xr_harmonised_se_indicator.rename({'year': 'time'})

print(f"Years in harmonised se_indicator data: {xr_harmonised_se_indicator.time.values}")
# Calculate regional correction factors
xr_correction_factors_regional = (xr_harmonised_se_indicator / xr_se_regional_sums[varname]).fillna(0)

if clean_memory: del xr_harmonised_se_indicator
if clean_memory: gc.collect()

# save to dataframe for checking
xr_correction_factors_regional.name = "regional_correction_factor"
df_correction_factors_regional = xr_correction_factors_regional.to_dataframe().reset_index()
df_correction_factors_regional = df_correction_factors_regional[df_correction_factors_regional["time"].isin(years_downscaling)].copy()
df_correction_factors_regional.to_csv(f"{project_dir}/data/check/step5_correction_factors_regional_{varname_save}_{source}_{scenario}_{model}.csv", sep=";", index=False)

# Map to spatial grid using numpy

region_ids = xr_IAM_regions_grid_downscaling.region_number.values


# Process x time steps at a time instead of all or one
print("Processing correction factors in blocks...")
block_size = 4
correction_factors_chunks = []

for i in range(0, len(xr_correction_factors_regional.time), block_size):
    # Calculate actual block size (last block might be smaller)
    actual_block_size = min(block_size, len(xr_correction_factors_regional.time) - i)
    print(f"Processing block {i // block_size + 1} of {(len(xr_correction_factors_regional.time) + block_size - 1) // block_size}...")

    time_slice = slice(i, i + actual_block_size)

    # Use actual_block_size instead of block_size
    lookup = np.zeros((actual_block_size, nr_regions + 1), dtype='float32')
    lookup[:, xr_correction_factors_regional.region_number.values.astype(int)] = \
        xr_correction_factors_regional.isel(time=time_slice).values

    correction_factors_chunks.append(lookup[:, region_ids])
    del lookup  # Clean up immediately

    # Optional: periodic garbage collection
    if (i // block_size) % 10 == 0:
        gc.collect()

# Concatenate all chunks AFTER the loop
print("Concatenating correction factor chunks...")
correction_factors_array = np.concatenate(correction_factors_chunks, axis=0)

# Delete chunks immediately after concatenation
if clean_memory: del correction_factors_chunks
gc.collect()

# Create the final DataArray with the full concatenated array
print("Creating final DataArray for correction factors...")
xr_correction_factors = xr.DataArray(
    correction_factors_array,
    coords={'time': xr_correction_factors_regional.time,
            'y': xr_SE_grid_downscaling.y,
            'x': xr_SE_grid_downscaling.x},
    dims=['time', 'y', 'x']
)

# Clean up
if clean_memory: del correction_factors_array
# del xr_correction_factors_regional
if clean_memory: del region_ids
if clean_memory: gc.collect()

# print min and max correction factors
if check_values:
    print(f"Min correction factor: {xr_correction_factors.where(xr_correction_factors > 0).min().values:.6f}, Max correction factor: {xr_correction_factors.max().values:.6f}")

# Step 6: Apply Harmonization

# Apply correction factor
#
# Merge grid se_indicator and correction factors into one Dataset
# 1) Make sure time coords match (important!)
if not np.array_equal(xr_SE_grid_downscaling.time.values, xr_correction_factors.time.values):
    print("Aligning time coordinates between se_indicator grid and correction factors...")
    xr_correction_factors = xr_correction_factors.interp(time=xr_SE_grid_downscaling.time)
# 2) (optional) mask ocean so zeros don’t skew colors
xr_correction_factors_masked = xr_correction_factors.where(xr_IAM_regions_grid_downscaling.region_number > 0)
# 3) Put the factors into the se_indicator Dataset
if isinstance(xr_SE_grid_downscaling, xr.Dataset):
    xr_SE_grid_correction = xr_SE_grid_downscaling.assign(correction_factor=xr_correction_factors_masked)
else:
    # if xr_SE_grid is a DataArray named 'se_indicator'
    xr_SE_grid_correction = xr.Dataset(
        data_vars=dict(
            se_indicator=xr_SE_grid,
            correction_factor=xr_correction_factors_masked
        )
    )
del xr_correction_factors_masked #
# add region number to Dataset
# 2-D region numbers (y, x), int and aligned to SE grid
region2d = xr_IAM_regions_grid_downscaling.region_number.astype("int8")
#del xr_IAM_regions_grid_downscaling # CHECK!
gc.collect()

# attach as a coordinate; stays time-invariant
xr_SE_grid_correction = xr_SE_grid_correction.assign_coords(region_number=(("y", "x"), region2d.values))

# optional: helpful attrs
xr_SE_grid_correction.coords["region_number"].attrs.update(
    long_name=f"{model} region number (0=ocean, 1–{nr_regions}=land regions)"
)

# Apply correction factor to se_indicator variable
varname_corrected = f"{varname}_corrected"
xr_SE_grid_correction[varname_corrected] = xr_SE_grid_correction[varname] * xr_SE_grid_correction["correction_factor"]
# xr_SE_grid_correction = xr_SE_grid_correction.drop_vars(varname)
# xr_SE_grid_correction = xr_SE_grid_correction.rename({varname_corrected: varname})

# Step 7: Checks

if check_values:
    dsf.check_values(xr_SE_grid_correction[varname_corrected])


# Check harmonisation inputs by calculating regional sums from corrected se_indicator grid and comparing to IAM projections
df_se_grid_indicator = pd.DataFrame()
mask_df = df_se_grid_indicator.notnull()

# Create a dataframe that sums the original and corrected se_indicator from the grid for each region and for all years and compare is to IAM model projections
region_numbers = df_IAM_projection["Region_number"].to_list()
convert_IMAGE_regions_to_names = rpd.convert_IMAGE_regions(region_numbers, from_type="number")

df_se_grid_indicator = pd.DataFrame(columns=["region_number"])
if check_values:
    print("Calculating se_indicator regional sums from grid...")
    xr_SE_grid_correction.compute
    df_se_grid_indicator = xr_SE_grid_correction[[varname, varname_corrected]].to_dataframe().reset_index()
    df_se_grid_indicator = (df_se_grid_indicator
                       .groupby(["region_number", "time"])
                       .sum()
                       .reset_index()
                       .drop(columns=["x", "y", "spatial_ref"], errors="ignore")
                       .sort_values(["time", "region_number"]))
    print(f"Merging se_indicator regional sums from grid with IAM model projections...")
    df_se_grid_indicator = pd.merge(
        df_se_grid_indicator,
        df_IAM_projection_se_indicator_downscaling.rename(columns={"value": "value_IAM", "year": "time"})[["region_number", "time", "value_IAM"]],
        on=["region_number", "time"],
        how="left"
    )
    print(f"Dropping columns and saving result to csv file...")
    #df_se_grid_indicator.drop(columns=["x", "y", "spatial_ref"], inplace=True)
    df_se_grid_indicator.to_csv(f"{project_dir}/data/check/{varname_save}_{source}_regional_sums_comparison_harmonised_{scenario}_cf_{coarse_factor}_{model}.csv", sep=";", index=False)

    df_se_grid_indicator = df_se_grid_indicator[(df_se_grid_indicator["region_number"] > 0)]
    df_se_grid_indicator["region_code"] = df_se_grid_indicator["region_number"].map(convert_IMAGE_regions_to_names)
    df_se_grid_indicator.sort_values(["time", "region_code"], inplace=True)
    mask_df = (df_se_grid_indicator["time"].isin([2020, 2030])) & (df_se_grid_indicator["region_number"] > 0)
df_se_grid_indicator[mask_df].style.format(precision=10, thousands=",", decimal=".")


print(type(xr_SE_grid_correction))
for a in xr_SE_grid_correction.attrs:
    print(a, ":", xr_SE_grid_correction.attrs[a])


# Check values
if check_values:
    print("data availability by year:")
    for year in xr_SE_grid_correction[varname_corrected].time.values:
        non_nan_count = (~np.isnan(xr_SE_grid_correction[varname_corrected].sel(time=year).values)).sum()
        print(f"{year}: {non_nan_count} non-NaN values")


# save new se_indicator file to netcdf
xr_SE_grid_corrected = None
if save_netcdf:
    file_path = f"{project_dir}/data/processed/{varname_save}_{source}_{scenario}_cf_{coarse_factor}_harmonized_{model}.nc"
    print(file_path)
    xr_SE_grid_corrected = xr_SE_grid_correction.copy()
    xr_SE_grid_corrected = xr_SE_grid_corrected.drop_vars(["correction_factor", "region_number", varname])
    xr_SE_grid_corrected = xr_SE_grid_corrected.rename({varname_corrected: varname})
    xr_SE_grid_corrected = xr_SE_grid_corrected.transpose("y", "x", "time")
    xr_SE_grid_corrected.to_netcdf(file_path, encoding={varname: {'zlib': True, 'complevel': 4}})
    xr_SE_grid_corrected.close()

# plot histogram of correction factors
logging.getLogger("matplotlib").setLevel(logging.WARNING)



