import atexit
import time
import logging
import warnings
import json

from pathlib import Path

from dask.distributed import get_client
from dask import base as dask_base

import numpy as np
import pandas as pd
from tabulate import tabulate

import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import distance_transform_edt
import rasterio
import xarray as xr
import rioxarray as rxr
import cartopy.crs as ccrs
import geopandas as gpd
from geocube.api.core import make_geocube

current_dir = Path(__file__).parent
print(f"Current directory: {current_dir}")

from tools.functions_logging import init_logging
from tools.general_functions import PRINT_COLORS, replace_punctuation_in_filenames, round_to_half, is_int_or_half, format_factor, apply_root_json
from tools import convert_GIS

import downscaling.plot_maps as plot_maps
import downscaling.read_process_grid_data as process_grid_data
import downscaling.read_process_IAM_data as process_IAM_data
import downscaling.process_IPAT_factors as process_IPAT_factors
import downscaling.settings_models as settings_models
import downscaling.upload_results_ee as upload_results_ee
import downscaling.settings_downscaling as settings
from downscaling.settings_downscaling import SOURCE_PROFILES

project_dir = Path(__file__).parent.parent
log_path = f"{project_dir}/log/downscaling"
debug_tmp_log, _ = init_logging(f"log_tmp", log_path)

def cleanup_empty_logs(log_path):

    # Close all file handlers so Windows releases the file locks
    for logger_name in ['debug', 'results', 'py.warnings']:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    # delete all .log files in the log directory that are empty
    log_dir = Path(log_path)
    for log_file in log_dir.glob("*.log"):
        if log_file.stat().st_size == 0:
            print(f"Deleting empty log file: {log_file}")
            log_file.unlink()


def process_datasets(project_dir:Path, SSP_base="SSP2"):
    # settings
    #SSP_base="SSP2"
    data_sources = [
        #{"varname": "Population", "source": "2UP", "version": "GHSL_2024_M1"},
        {"varname": "Population", "source": "2UP", "version": "GHSL_2024_M3"},
        #{"varname": "Population", "source": "2UP", "version": "M1"},
        #{"varname": "Population", "source": "2UP", "version": "M3"},
        #{"varname": "Population", "source": "Wang", "version": "version_2"},
        #{"varname": "Population", "source": "Wang", "version": "version_3"},
        #{"varname": "Population", "source": "Zhuang", "version": "version_1"},
        #{"varname": "Population", "source": "COMPASS", "version": "version_2"},
        #{"varname": "Population", "source": "Murakami", "version": "version_3"},
        #{"varname": "GDP|PPP", "source": "Wang", "version": "version_7"},
        #{"varname": "GDP|PPP", "source": "Murakami", "version": "version_2021_1"},
        #{"varname": "GDP|PPP", "source": "COMPASS", "version": "version_2"},
        #{"varname": "Emissions_CO2_Excl_shipping_aviation_AFOLU", "source": "EDGAR", "version": "2024"}
        #{"varname": "Emissions_CO2_Excl_shipping_aviation_AFOLU", "source": "CEDS_CMIP7", "version": "2025_04_18"}
    ]

    log_path = f"{project_dir}/log/reading_processing_data"
    debug_log, _ = init_logging(f"log_reading_processing_data", log_path)

    print("-----------------------------")
    print("Processing datasets...")
    print(data_sources)
    print("-----------------------------")

    for data_source in data_sources:
        print(data_source)
        print(f"Processing data for variable: {data_source["varname"]}, source: {data_source["source"]}, version: {data_source["version"]}")
        if data_source["varname"] in ["Population", "GDP|PPP"]:
            process_grid_data.pre_process_data_socioeconomic(varname=data_source["varname"],
                                        source=data_source["source"],
                                        version=data_source["version"],
                                        SSP_base=SSP_base,
                                        log=debug_log)
        elif data_source["varname"] == "Emissions_CO2_Excl_shipping_aviation_AFOLU":
            process_grid_data.pre_process_data_emissions(varname=data_source["varname"],
                                        source=data_source["source"],
                                        version=data_source["version"],
                                        log=debug_log)
        else:
            print(f"{PRINT_COLORS["red"]}Variable {data_source["varname"]} not recognized for processing.{PRINT_COLORS["end"]}")

def determine_regions_file(project_dir:Path,
                           res_min_POP:int|float|None, res_min_GDP:int|float|None, res_min_EM:int|float|None,
                           model:str, log:logging.Logger) -> Path:

    # determine region grid file
    lowest_resolution_minutes = max(res_min_POP or 0, res_min_GDP or 0, res_min_EM or 0)
    lowest_resolution_minutes = round_to_half(lowest_resolution_minutes)
    if is_int_or_half(lowest_resolution_minutes):
        if isinstance(lowest_resolution_minutes, int):
            lowest_resolution_minutes = str(int(lowest_resolution_minutes)) + "_00"
        else:
            lowest_resolution_minutes = str(lowest_resolution_minutes).replace(".", "_") + "0"
    else:
        lowest_resolution_minutes = str(lowest_resolution_minutes).replace(".", "_") + "0"
    log.info(f"Lowest resolution among datasets: {lowest_resolution_minutes} minutes")
    file_regions_stem = Path(settings_models.models[model]["file_model_grid_regions"]).stem
    file_regions_suffix = Path(settings_models.models[model]["file_model_grid_regions"]).suffix
    file_path_file_model_grid_regions = project_dir / f"data/input/models/{model}/{file_regions_stem}_{lowest_resolution_minutes}_arcmin{file_regions_suffix}"
    log.info(f"Looking for model grid regions file at: {file_path_file_model_grid_regions}")
    if not Path(file_path_file_model_grid_regions).exists():
        # TO DO --> coarsen existing regions grid file
        log.warning(f"Model grid regions file not found at {file_path_file_model_grid_regions}. Please create the file with create_GADM_region_raster for the appropriate resolution.")
        log.info(f"Model grid regions file not found at {file_path_file_model_grid_regions}. Please create the file with create_GADM_region_raster for the appropriate resolution.")
        exit()

    return file_path_file_model_grid_regions

def reindex_and_interp(group, id_cols, years_downscaling):
    # keys are the values of the id_cols for this group
    keys = dict(zip(id_cols, group.name if isinstance(group.name, tuple) else (group.name,)))

    return (
        group.set_index("year")
            .reindex(years_downscaling)
            .assign(**keys)
            .assign(value=lambda g: g["value"].interpolate("linear", limit_area="inside"))
            .reset_index())

def aggregate_urban_emissions(xr_emissions: xr.Dataset, gdf_urban_classification: gpd.GeoDataFrame,
                              emissions_varname: str = "Emissions_CO2_Excl_shipping_aviation_AFOLU",
                              region_varname: str = "region_number",
                              final_year: int = 2100,
                              log: logging.Logger=debug_tmp_log) -> tuple[xr.Dataset, pd.DataFrame, pd.DataFrame]:
    '''
    Aggregate emissions per region and year, based on urban classification.
    '''
    if not xr_emissions.chunks:
        xr_emissions = xr_emissions.chunk({"time": 1, "y": 2048, "x": 2048})

    # Derive available classification years from gdf_urban_classification's column names
    cluster_years = sorted(int(col.replace("cluster_", ""))
                        for col in gdf_urban_classification.columns
                        if col.startswith("cluster_"))

    # Keep only years present in both gdf_urban_classification and xr_emissions
    log.info(f"Available classification years in gdf_urban_classification: {cluster_years}")
    emissions_years = set(int(y) for y in xr_emissions["time"].values)
    common_years = sorted(y for y in (set(cluster_years) & emissions_years) if y <= final_year)
    if not common_years:
        raise ValueError("No overlapping years found between gdf_urban_classification and xr_emissions.")
    log.info(f"Using {len(common_years)} common years: {common_years}")

    # Build one (y, x) urban grid per common year, tagged with a "time" coord
    log.info(f"Building urban classification grids for {len(common_years)} common years...")
    urban_slices = []
    for year in common_years:
        log.info(year)
        col_name = f"cluster_{year}"
        year_gdf = gdf_urban_classification[["geometry", col_name]]
        year_grid = make_geocube(vector_data=year_gdf, like=xr_emissions, measurements=[col_name])
        da = year_grid[col_name].rename("urban")
        da = da.expand_dims(time=[year])
        urban_slices.append(da)

    log.info(f"Concatenating {len(urban_slices)} urban classification grids along the time dimension...")
    urban_by_year = xr.concat(urban_slices, dim="time")
    urban_by_year = urban_by_year.assign_coords(x=xr_emissions["x"], y=xr_emissions["y"])
    combined = xr.merge([xr_emissions.sel(time=common_years), urban_by_year.to_dataset()], join="exact", compat="no_conflicts")
    log.info(combined)

    # --- Aggregate emissions per region_number, per year ---
    log.info(f"Aggregating emissions per {region_varname} for {len(common_years)} years...")
    da_emissions = combined[emissions_varname]
    da_region = combined[region_varname].load()
    da_urban = combined["urban"]
    # print unique values for da_urban
    log.info(f"Unique urban values (first year): {np.unique(da_urban.isel(time=0).values)}")
    urban_emissions = da_emissions.where(da_urban == 1)
    regional_totals = urban_emissions.groupby(da_region).sum()
    df_urban_emissions = regional_totals.to_dataframe(name=emissions_varname).reset_index()
    rural_emissions = da_emissions.where(da_urban == 0)
    regional_totals = rural_emissions.groupby(da_region).sum()
    df_rural_emissions = regional_totals.to_dataframe(name=emissions_varname).reset_index()

    df_rural_emissions = regional_totals.to_dataframe(name=emissions_varname).reset_index()
    return combined, df_urban_emissions, df_rural_emissions

def downscale_SE_data(project_dir:Path, variable_SE: str, scenario: str, model: str = "IMAGE", align_resolution: bool = True, profile: str = "default"):
    """Downscale socio-economic (population or GDP) gridded data to match IAM regional projections."""
    # align_resolution determines whether to use the same grid resolution for SE and EM (i.e. the lowest resolution among POP, GDP, EM)
    # or to keep the original SE grid resolution (which is typically higher than EM).
    # If True, the same grid and region mapping will be used for both SE and EM downscaling,
    #   which ensures that the same correction factors are applied to both variables and that they are perfectly aligned.
    # If False, the original higher-resolution SE grid will be used,
    #   which may lead to some misalignment between SE and EM grids and potentially less accurate downscaling results for SE, but preserves the original SE data resolution.

    print(f"\n{PRINT_COLORS['cyan']}Starting downscaling of {variable_SE} data for scenario '{scenario}' and model '{model}' with profile '{profile}'...{PRINT_COLORS['end']}")

    # start timing
    start_time = time.time()

    if profile not in settings.SOURCE_PROFILES:
        available = list(settings.SOURCE_PROFILES.keys())
        raise ValueError(f"Unknown source profile '{profile}'. Available: {available}")
    else:
        sources = settings.SOURCE_PROFILES[profile]

    source_POP = sources["source_POP"]
    version_POP = sources["version_POP"]
    source_GDP = sources["source_GDP"]
    version_GDP = sources["version_GDP"]
    source_EM = sources["source_EM"]
    version_EM = sources["version_EM"]

    coarse_factor_POP, coarse_factor_GDP, coarse_factor_EM, \
    res_min_POP, res_min_GDP, res_min_EM = process_grid_data.get_coarsening_factors(
                                                            population_source=sources["source_POP"],
                                                            gdp_source=sources["source_GDP"],
                                                            emissions_source=sources["source_EM"])
    if not align_resolution:
        res_min_POP = 0.5 if variable_SE == "Population" else None
        res_min_GDP = 0.5 if variable_SE in ("GDP|PPP", "GDP_PPP", "GDP") else None
        res_min_EM = None
        coarse_factor_SE = 1
    elif variable_SE == "Population":
        coarse_factor_SE = coarse_factor_POP
    elif variable_SE in ("GDP|PPP", "GDP_PPP", "GDP"):
        coarse_factor_SE = coarse_factor_GDP
    else:
        raise ValueError(f"Variable '{variable_SE}' not recognized. Supported variables: 'Population', 'GDP|PPP'.")

    base_year = settings.base_year
    years_downscaling = settings.years_downscaling
    convergence_year = settings.convergence_year
    method_extension = settings.method_extension
    vars_downscaling = settings.vars_downscaling
    process_flags = settings.process_flags
    check_flags = settings.check_flags

    varname_GDP = settings.varname_GDP
    varname_POP = settings.varname_POP
    varname_EM = settings.varname_EM
    varname_gdp_per_pop = settings.varname_gdp_per_pop
    varname_em_per_gdp_ppp = settings.varname_em_per_gdp_ppp

    unit_POP = settings.unit_POP
    unit_GDP = settings.unit_GDP_PPP
    unit_EM = settings.unit_EM

    SSP_base = settings.SSP_base

    if variable_SE == "Population":
        varname_SE = varname_POP
        source_SE = source_POP
        version_SE = version_POP
        unit_SE = unit_POP
    elif variable_SE in ("GDP|PPP", "GDP_PPP", "GDP"):
        varname_SE = varname_GDP
        source_SE = source_GDP
        version_SE = version_GDP
        unit_SE = unit_GDP
    else:
        raise ValueError(f"Variable '{variable_SE}' not recognized. Supported variables: 'Population', 'GDP|PPP'.")

    # create output and processed directories
    if align_resolution:
        source_version_grid = f"{source_POP}_{version_POP}_{source_GDP}_{version_GDP}_{source_EM}_{version_EM}"
    else:
        source_version_grid = f"{source_SE}_{version_SE}"
    model_scenario = f"{model}_{scenario}"
    dir_output = project_dir / "data" / "output" / source_version_grid / model_scenario
    dir_processed = project_dir / "data" / "processed" / source_version_grid / model_scenario
    print(f"Project directory: {project_dir}")
    print(f"Output directory: {dir_output}")
    print(f"Processed data directory: {dir_processed}")
    dir_output.mkdir(parents=True, exist_ok=True)
    dir_processed.mkdir(parents=True, exist_ok=True)

    log_path = f"{project_dir}/log/downscaling"
    debug_log, results_log = init_logging(f"log_downscaling_se_{source_version_grid}_{model_scenario}", log_path)
    debug_log.info(SOURCE_PROFILES[profile])
    results_log.info(SOURCE_PROFILES[profile])

    debug_log.info(f"Project directory: {project_dir}")
    debug_log.info(f"Output directory: {dir_output}")
    debug_log.info(f"Processed data directory: {dir_processed}")
    debug_log.info(f"\n{PRINT_COLORS["green"]}coarse_factor_SE: {coarse_factor_SE:.2f}{PRINT_COLORS["end"]}")
    res_min_POP_str = f"{res_min_POP:.2f}" if res_min_POP is not None else "None"
    debug_log.info(f"\n{PRINT_COLORS["green"]}res_min_POP: {res_min_POP_str}{PRINT_COLORS["end"]}")
    res_min_GDP_str = f"{res_min_GDP:.2f}" if res_min_GDP is not None else "None"
    debug_log.info(f"\n{PRINT_COLORS["green"]}res_min_GDP: {res_min_GDP_str}{PRINT_COLORS["end"]}")

    se_downscaling_file = dir_processed / f"{replace_punctuation_in_filenames(varname_SE)}_downscaling_{source_SE}_{version_SE}_{SSP_base}_cf_{coarse_factor_SE}.nc"
    se_harmonised_file = dir_processed / f"{replace_punctuation_in_filenames(varname_SE)}_harmonised_{source_SE}_{version_SE}_{SSP_base}_cf_{coarse_factor_SE}.nc"

    # with open("downscaling/settings_models.json", "r") as f:
    #     data = json.load(f)
    #    conversion_factor_IAM_to_grid = data[model]["model_unit_conversions"][varname_SE]
    conversion_factor_IAM_to_grid = settings_models.models[model]["model_unit_conversions"][varname_SE]
    debug_log.info(f"Conversion factor from IAM to grid units for {varname_SE}: {conversion_factor_IAM_to_grid}")

    # override to fixed downscaling years for SE (covers full century)
    years_downscaling = [2020, 2025, 2030, 2035, 2040, 2045, 2050, 2060, 2070, 2080, 2090, 2100]

    # Step 1: Data inventory and setup
    # ----------------------------------------------------------------------------------------------------------
    debug_log.info(f"\n{PRINT_COLORS["green"]}(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario})Step 1: Data inventory and setup...{PRINT_COLORS["end"]}")

    #file_IAM_model_region_numbers = settings.file_IAM_model_region_numbers
    file_IAM_model_region_numbers = settings_models.models[model]["file_IAM_model_region_numbers"]
    file_path_file_model_grid_regions = determine_regions_file(project_dir, res_min_POP, res_min_GDP, res_min_EM, model, debug_log)

    # Read IAM regional projection data
    df_IAM_projection = process_IAM_data.read_process_IAM_data(project_dir, scenario, model, file_IAM_model_region_numbers, [varname_SE])
    mask_IAM_projection_se = (df_IAM_projection["variable"].isin([varname_SE]) & (df_IAM_projection["region_number"] != "World"))
    df_IAM_projection_se = df_IAM_projection[mask_IAM_projection_se].copy()

    if df_IAM_projection_se.empty:
        raise ValueError(f"The variable '{varname_SE}' was not found in IAM data. "
                         f"Available variables: {sorted(df_IAM_projection['Variable'].unique())}")

    nr_regions = df_IAM_projection_se["region_number"].nunique()
    debug_log.info(f"Number of regions in IAM projection (excluding World): {nr_regions}")
    debug_log.info(f"Units for {variable_SE} in IAM projection: {unit_SE}")
    debug_log.info(df_IAM_projection_se[df_IAM_projection_se["year"].isin([2020, 2030])].round(0))

    # Read gridded SE data
    xr_se, f_se = process_grid_data.read_process_grid_data_socioeconomic(dir_processed=dir_processed, varname=varname_SE, source=source_SE,
                                                                         version=version_SE, SSP_base=SSP_base, coarse_factor=coarse_factor_SE,
                                                                         unit=unit_SE, save=False, check=False, log=debug_log)
    xr_se = xr_se.astype("float32")
    xr_se = xr_se.chunk({"time": 1, "x": "auto", "y": "auto"})
    xr_se = xr_se.sortby("y", ascending=False)  # north-to-south
    xr_se = xr_se.sortby("x", ascending=True)   # west-to-east
    results_log.info(f"Time steps in gridded {variable_SE} data: {np.unique(xr_se['time'].values)}")

    # Read IAM regions grid and align with SE grid
    xr_IAM_regions_grid = xr.open_dataset(file_path_file_model_grid_regions)
    xr_IAM_regions_grid = xr_IAM_regions_grid.drop_vars("band", errors="ignore")
    xr_IAM_regions_grid = xr_IAM_regions_grid.sortby("y", ascending=False)  # north-to-south
    xr_IAM_regions_grid = xr_IAM_regions_grid.sortby("x", ascending=True)   # west-to-east
    xr_IAM_regions_grid_downscaling = xr_IAM_regions_grid.reindex_like(xr_se.sel(time=base_year), method="nearest", tolerance=1e-5)
    xr_IAM_regions_grid.close()

    debug_log.info(f"{PRINT_COLORS["yellow"]}Region numbers: {np.unique(xr_IAM_regions_grid_downscaling['region_number'].values)}{PRINT_COLORS["end"]}")

    # Step 2: Prepare grid and model datasets for harmonisation
    # ----------------------------------------------------------------------------------------------------------
    debug_log.info(f"\n{PRINT_COLORS["green"]}(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario})Step 2: Prepare grid and model datasets for harmonisation...{PRINT_COLORS["end"]}")

    #land_mask = (xr_IAM_regions_grid_downscaling["region_number"] > 0)

    # Align IAM projection DataFrame with downscaling years (filter, then interpolate missing years)
    df_IAM_projection_se_downscaling = df_IAM_projection_se.copy()
    df_IAM_projection_se_downscaling.columns = df_IAM_projection_se_downscaling.columns.str.lower()
    df_IAM_projection_se_downscaling = df_IAM_projection_se_downscaling[df_IAM_projection_se_downscaling["year"].isin(years_downscaling)].copy()

    id_cols = ["model", "scenario", "region_code", "variable", "unit", "region_number"]
    df_IAM_projection_se_downscaling["year"] = df_IAM_projection_se_downscaling["year"].astype(int)
    df_IAM_projection_se_downscaling["value"] = df_IAM_projection_se_downscaling["value"].astype(float)
    df_IAM_projection_se_downscaling = (df_IAM_projection_se_downscaling
                                        .groupby(id_cols, group_keys=False)
                                        .apply(lambda g: reindex_and_interp(g, id_cols, years_downscaling))
                                        .reset_index(drop=True))
    debug_log.info(f"Downscaling years in IAM projection: {df_IAM_projection_se_downscaling['year'].unique()}")

    # Convert IAM units to grid units
    df_IAM_projection_se_downscaling["value"] *= conversion_factor_IAM_to_grid

    # Add ocean (region 0) with zero values so all region IDs in the grid are covered
    years = df_IAM_projection_se_downscaling["year"].unique()
    variable = df_IAM_projection_se_downscaling["variable"].unique()[0]
    unit = df_IAM_projection_se_downscaling["unit"].unique()[0]
    extra_rows = pd.DataFrame({"model": model, "scenario": scenario, "region_code": "OCEAN",
                               "variable": variable, "year": years, "unit": unit, "region_number": 0, "value": 0})
    df_IAM_projection_se_downscaling = (pd.concat([df_IAM_projection_se_downscaling, extra_rows], ignore_index=True)
                                        .sort_values(["year", "region_number"])
                                        .reset_index(drop=True))

    # Align gridded SE with downscaling years by linear interpolation
    # xr_se_downscaling = xr_se.interp(time=years_downscaling, method="linear")
    # xr_se.close(
    if process_flags["process_SE"] or not se_downscaling_file.is_file():
        debug_log.info(f"Aligning gridded SE data with downscaling years {years_downscaling} by linear interpolation...")
        results = []
        for year in years_downscaling:
            debug_log.info(f"{year}")
            xr_se_year = xr_se.interp(time=year, method="linear").compute()
            results.append(xr_se_year)
        xr_se.close()
        xr_se_downscaling = xr.concat(results, dim="time")
        del results
        debug_log.info(f"Saving aligned gridded SE data to {se_downscaling_file}...")
        xr_se_downscaling.to_netcdf(se_downscaling_file, mode="w", engine="netcdf4")
        del xr_se_downscaling # re-read again to free up memory
    else:
        debug_log.info(f"Gridded SE data already aligned with downscaling years and saved at {se_downscaling_file}, skipping interpolation.")
    xr_se_downscaling = xr.open_dataset(se_downscaling_file)
    debug_log.info(f"Years in gridded {variable_SE} after alignment: {xr_se_downscaling.time.values}")

    # Step 3: Calculate regional sums for gridded data
    # ----------------------------------------------------------------------------------------------------------
    debug_log.info(f"\n{PRINT_COLORS["green"]}(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario})Step 3: Calculate regional grid sums for gridded data...{PRINT_COLORS["end"]}")

    df_se_regional_sums_compare, xr_se_regional_sums = process_IPAT_factors.calc_regional_values(xr_se_downscaling, varname_SE,
                                                                                                 xr_IAM_regions_grid_downscaling, df_IAM_projection_se_downscaling,
                                                                                                 years_downscaling, debug_log)
    df_se_regional_sums_compare.to_csv(f"{project_dir}/data/check/step3_{varname_SE}_{source_SE}_regional_sums_comparison_{scenario}_{model}.csv", sep=";", index=False)
    debug_log.info(f"Regional sums for gridded {variable_SE} calculated and comparison saved to {project_dir}/data/check/step3_{varname_SE}_{source_SE}_regional_sums_comparison_{scenario}_{model}.csv")

    # Step 4: Calculate cell-specific correction factors
    # ----------------------------------------------------------------------------------------------------------
    debug_log.info(f"\n{PRINT_COLORS["green"]}(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario})Step 4: Calculate cell-specific correction factors for harmonisation...{PRINT_COLORS["end"]}")

    # Build target regional values as an xarray (time × region_number)
    debug_log.info("4.1 building target regional values from IAM projections...")
    xr_harmonised_se = (df_IAM_projection_se_downscaling
                        .set_index(["year", "region_number"])["value"]
                        .to_xarray()
                        .sel(year=years_downscaling, region_number=xr_se_regional_sums["region_number"])
                        .rename({"year": "time"}))

    denom = xr_se_regional_sums[varname_SE]
    xr_correction_factors_regional = (xr_harmonised_se / denom.where(denom > 0)).fillna(0)
    xr_correction_factors_regional = xr_correction_factors_regional.transpose("time", "region_number")
    debug_log.info(f"Unique region numbers in correction factors: {np.unique(xr_correction_factors_regional.region_number.values)}")

    # Map regional correction factors to the full spatial grid via vectorized xarray indexing.
    # xr_correction_factors_regional has dims (time, region_number); region2d has dims (y, x).
    # sel() broadcasts to produce a (time, y, x) DataArray without any explicit loop.
    debug_log.info("4.2 mapping regional correction factors to spatial grid...")
    region2d = xr_IAM_regions_grid_downscaling["region_number"].astype("int16")
    xr_correction_factors = (xr_correction_factors_regional
                                .sel(region_number=region2d)
                                .drop_vars("region_number")).compute()

    if check_flags["check_SE_correction_factors"]:
        debug_log.info("Checking correction factors...")
        cf_min = float(xr_correction_factors.where(xr_correction_factors > 0).min())
        cf_max = float(xr_correction_factors.max())
        debug_log.info(f"Min correction factor (>0): {cf_min:.6f}, Max correction factor: {cf_max:.6f}")
        csv_file_cf = dir_processed / f"correction_factors_regional_{varname_SE}_{source_SE}_{scenario}_{model}.csv"
        xr_correction_factors_regional.name = "regional_correction_factor"
        xr_correction_factors_regional.to_dataframe().reset_index().to_csv(csv_file_cf, sep=";", index=False)

    # Step 5: Apply harmonisation
    # ----------------------------------------------------------------------------------------------------------
    debug_log.info(f"\n{PRINT_COLORS["green"]}(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario})Step 5: Applying correction factors to grid data...{PRINT_COLORS["end"]}")

    land_mask_2d = xr_IAM_regions_grid_downscaling["region_number"] > 0
    results = []
    for year in years_downscaling:
        debug_log.info(f"  Applying correction factors for {year}...")
        se_year = xr_se_downscaling[varname_SE].sel(time=year).astype("float32").compute()
        se_year = xr_se_downscaling[varname_SE].sel(time=year).compute()
        cf_year = xr_correction_factors.sel(time=year).where(land_mask_2d).compute()
        harmonised_year = (se_year * cf_year).expand_dims(time=[year])
        results.append(harmonised_year)
        del se_year, cf_year
    xr_se_harmonised = xr.concat(results, dim="time").to_dataset(name=varname_SE)
    del results
    xr_se_harmonised[varname_SE].attrs["unit"] = unit_SE

    # Attach region number as a 2-D coordinate (mirrors downscale_emissions convention)
    debug_log.info("5.2 Attaching region numbers as 2-D coordinate to harmonised SE data...")
    xr_se_harmonised = xr_se_harmonised.assign_coords(region_number=(("y", "x"), region2d.values))
    xr_se_harmonised.coords["region_number"].attrs.update(long_name=f"{model} region number (0=ocean, 1-{nr_regions}=land regions)")

    arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_se_harmonised[varname_POP])
    debug_log.info(f"resolution POP grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.1f} arc degrees")

    debug_log.info("5.3 Saving harmonised SE data to NetCDF...")
    #xr_se_harmonised = xr_se_harmonised.chunk({"time": 1, "x": "auto", "y": "auto"})

    t0_compute = time.time()
    debug_log.info("Computing harmonised SE data (this may take some time depending on the dataset size and chunking strategy)...")
    xr_se_harmonised = xr_se_harmonised.compute()
    debug_log.info(f"Computation took {time.time() - t0_compute:.1f} seconds")

    t0_save_ = time.time()
    xr_se_harmonised.to_netcdf(se_harmonised_file, mode="w", engine="netcdf4")
    debug_log.info(f"Writing took {time.time() - t0_save_:.1f} seconds")
    debug_log.info(f"Harmonised {variable_SE} saved to {se_harmonised_file}")

    if process_flags["save_tiffs_results"]:
        debug_log.info(f"5.4 Saving harmonised {variable_SE} to GeoTIFF files in {dir_processed}...")
        plot_maps.save_to_grid_tiff(dir_processed, xr_se_harmonised, varname_SE, "_harmonised", years_downscaling, model, scenario)

    # Step 6: Post-harmonisation checks
    # ----------------------------------------------------------------------------------------------------------
    debug_log.info(f"\n{PRINT_COLORS["green"]}(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario})Step 6: Performing checks on harmonised data...{PRINT_COLORS["end"]}")

    if check_flags["check_SE_harmonised"]:
        process_grid_data.check_values_xr_dataarray(xr_se_harmonised[varname_SE], None, False, debug_log)
        df_se_corrected_compare, xr_regional_sums_corrected = process_IPAT_factors.calc_regional_values(
            xr_se_harmonised, varname_SE,
            xr_IAM_regions_grid_downscaling, df_IAM_projection_se_downscaling,
            years_downscaling)
        csv_file_check = dir_processed / f"{varname_SE}_{source_SE}_regional_sums_post_harmonisation_{scenario}_{model}.csv"
        df_se_corrected_compare.to_csv(csv_file_check, sep=";", index=False)
        debug_log.info(f"Post-harmonisation regional sums comparison saved to {csv_file_check}")

    xr_se_downscaling.close()
    xr_se_harmonised.close()
    xr_IAM_regions_grid_downscaling.close()

    elapsed_time = time.time() - start_time
    debug_log.info(f"\n{PRINT_COLORS["green"]}Total elapsed time: {elapsed_time:,.2f} seconds or ({elapsed_time/60:.2f} minutes).{PRINT_COLORS["end"]}")

def downscale_emissions(project_dir:Path, scenario:str, model:str="IMAGE", profile:str="default", net_emissions:bool=True):

    # Make sure program stops (and not only give a warning) if divided by zero, invalid value, or overvflow
    # Set up logging and warnings
    warnings.filterwarnings("error", message="divide by zero", category=RuntimeWarning)
    warnings.filterwarnings("error", message="invalid value", category=RuntimeWarning)
    warnings.filterwarnings("error", message="overflow", category=RuntimeWarning)

    # start timing
    start_time = time.time()

    # read profile
    if profile not in settings.SOURCE_PROFILES:
        available = list(settings.SOURCE_PROFILES.keys())
        raise ValueError(f"Unknown source profile '{profile}'. Available: {available}")
    else:
        sources = settings.SOURCE_PROFILES[profile]

    # settings
    #coarse_factor_SE:int = 12 # 12 (2UP)
    #coarse_factor_GDP:float = 1.2 # 12 (Wang), 1.2 (Murakam version_2021_1)
    #coarse_factor_EM:int =  1 # EDGAR
    coarse_factor_POP, coarse_factor_GDP, coarse_factor_EM, \
    res_min_POP, res_min_GDP, res_min_EM = process_grid_data.get_coarsening_factors(
                                                                population_source=sources["source_POP"],
                                                                gdp_source=sources["source_GDP"],
                                                                emissions_source=sources["source_EM"])


    coarse_factor_POP_str = f"{format_factor(coarse_factor_POP)}"
    coarse_factor_GDP_str = f"{format_factor(coarse_factor_GDP)}"
    coarse_factor_EM_str = f"{format_factor(coarse_factor_EM)}"
    print(f"Coarsening factors as string- Population: {coarse_factor_POP_str}, GDP: {coarse_factor_GDP_str}, Emissions: {coarse_factor_EM_str}")

    source_POP = sources["source_POP"]
    version_POP = sources["version_POP"]
    source_GDP = sources["source_GDP"]
    version_GDP = sources["version_GDP"]
    source_EM = sources["source_EM"]
    version_EM = sources["version_EM"]

    base_year = settings.base_year
    years_downscaling = settings.years_downscaling
    convergence_year = settings.convergence_year
    method_extension = settings.method_extension
    vars_downscaling = settings.vars_downscaling
    process_flags = settings.process_flags
    check_flags   = settings.check_flags

    varname_GDP = settings.varname_GDP
    varname_POP = settings.varname_POP
    varname_EM = settings.varname_EM
    varname_gdp_per_pop = settings.varname_gdp_per_pop
    varname_em_per_gdp_ppp = settings.varname_em_per_gdp_ppp

    unit_GDP_PPP = settings.unit_GDP_PPP
    unit_POP = settings.unit_POP
    unit_EM = settings.unit_EM

    SSP_base = settings.SSP_base

    # coarse_factor_POP, coarse_factor_GDP, coarse_factor_EM = process_grid_data.get_coarsening_factors(population_source="2UP",gdp_source="Murakami",emissions_source="EDGAR")
    # print(f"Coarsening factors - Population: {coarse_factor_POP}, GDP: {coarse_factor_GDP}, Emissions: {coarse_factor_EM}")

    # create output and processed directories
    print(f"Project directory: {project_dir}")
    gross_net = "net" if net_emissions else "gross"
    source_version_grid = f"{source_POP}_{version_POP}_{source_GDP}_{version_GDP}_{source_EM}_{version_EM}_{gross_net}"
    model_scenario = f"{model}_{scenario}"
    dir_output = project_dir / "data" / "output"
    dir_processed = project_dir / "data" / "processed" / profile / source_version_grid / model_scenario
    print(f"Output directory: {dir_output}")
    print(f"Processed data directory: {dir_processed}")
    dir_output.mkdir(parents=True, exist_ok=True)
    dir_processed.mkdir(parents=True, exist_ok=True)
    dir_urban = project_dir / "data" / "processed" / "DLL"

    log_path = f"{project_dir}/log/downscaling"
    debug_log, results_log = init_logging(f"downscaling_emissions_{profile}_{source_version_grid}_{model_scenario}", log_path)
    debug_log.info(f"\n\n{PRINT_COLORS['purple']}{'|'*100}{PRINT_COLORS['end']}")
    debug_log.info(f"{PRINT_COLORS['purple']}Logging started{PRINT_COLORS['end']}")
    debug_log.info(f"{PRINT_COLORS['purple']}{SOURCE_PROFILES[profile]}{PRINT_COLORS['end']}")
    #results_log.info(SOURCE_PROFILES[profile])

    #-------------------------------------------------------------------------------------------------------------------------
    debug_log.info(f"\n0. Init {"-"*25}")
    debug_log.info(f"Project directory: {project_dir}")
    debug_log.info(f"Output directory: {dir_output}")
    debug_log.info(f"Processed data directory: {dir_processed}")
    debug_log.info(f"\n{PRINT_COLORS["yellow"]}{"net emissions" if net_emissions else "gross emissions"}{PRINT_COLORS["end"]}")
    debug_log.info(f"\n{PRINT_COLORS["green"]}coarse_factor_EM: {coarse_factor_EM:.2f}{PRINT_COLORS["end"]}")
    res_min_POP_str = f"{res_min_POP:.2f}" if res_min_POP is not None else "None"
    debug_log.info(f"\n{PRINT_COLORS["green"]}res_min_POP: {res_min_POP_str}{PRINT_COLORS["end"]}")
    res_min_GDP_str = f"{res_min_GDP:.2f}" if res_min_GDP is not None else "None"
    debug_log.info(f"\n{PRINT_COLORS["green"]}res_min_GDP: {res_min_GDP_str}{PRINT_COLORS["end"]}")
    res_min_EM_str = f"{res_min_EM:.2f}" if res_min_EM is not None else "None"
    debug_log.info(f"\n{PRINT_COLORS["green"]}res_min_EM: {res_min_EM_str}{PRINT_COLORS["end"]}")

    file_path_file_model_grid_regions = determine_regions_file(project_dir, res_min_POP, res_min_GDP, res_min_EM, model, debug_log)
    file_IAM_model_region_numbers = settings_models.models[model]["file_IAM_model_region_numbers"]

    pop_file = dir_processed / f"Population_{source_POP}_{version_POP}_{SSP_base}_cf_{coarse_factor_POP_str}.nc"
    gdp_ppp_file = dir_processed / f"GDP_PPP_{source_GDP}_{version_GDP}_{SSP_base}_cf_{coarse_factor_GDP_str}.nc"
    em_file = dir_processed / f"{replace_punctuation_in_filenames(varname_EM)}_hist_{source_EM}_{version_EM}_{SSP_base}_cf_{coarse_factor_EM_str}.nc"
    pop_processed_file = dir_processed / f"Population_processed_{source_POP}_{version_POP}_{SSP_base}_cf_{coarse_factor_POP_str}.nc"
    gdp_ppp_processed_file = dir_processed / f"GDP_PPP_processed_{source_GDP}_{version_GDP}_{SSP_base}_cf_{coarse_factor_GDP_str}.nc"
    gdp_ppp_per_pop_file = dir_processed / f"GDP_PPP_per_pop_{source_GDP}_{version_GDP}_{source_POP}_{version_POP}_{SSP_base}.nc"
    em_per_gdp_ppp_file = dir_processed / f"{replace_punctuation_in_filenames(varname_EM)}_per_gdp_ppp_{source_EM}_{version_EM}_{source_GDP}_{version_GDP}_{SSP_base}.nc"
    em_unharmonised_file = dir_processed / f"{replace_punctuation_in_filenames(varname_EM)}_unharmonised_{SSP_base}.nc"
    em_harmonised_file = dir_processed / f"{replace_punctuation_in_filenames(varname_EM)}_harmonised_{SSP_base}.nc"

    # 1. Read and process gridded data
    debug_log.info(f"\n\n1. Read and process gridded data {"-"*25}")

    # 1.1 Read in IAM regions grid
    debug_log.info(f"\n\n1.1. Read and process in IAM data {"-"*25}")
    debug_log.info(f"\n\n(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}: {PRINT_COLORS["green"]}Reading (and processing) IAM population, gdp, and emissionsdata...{PRINT_COLORS["end"]}")
    xr_IAM_regions_grid = xr.open_dataset(file_path_file_model_grid_regions)
    xr_IAM_regions_grid = xr_IAM_regions_grid.drop_vars("band", errors="ignore")
    xr_IAM_regions_grid = xr_IAM_regions_grid.sortby("y", ascending=False)  # north-to-south
    xr_IAM_regions_grid = xr_IAM_regions_grid.sortby("x", ascending=True)  # west-to-east
    debug_log.info(f"Variable: {xr_IAM_regions_grid.data_vars}")
    arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_IAM_regions_grid["region_number"])
    debug_log.info(f"resolution EM grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.1f} arc degrees")

    debug_log.info(f"{PRINT_COLORS["yellow"]}region numbers: {np.unique(xr_IAM_regions_grid["region_number"].values)}{PRINT_COLORS["end"]}")
    if process_flags["save_tiffs_intermediate"]:
        xr_IAM_regions_grid_save = (xr_IAM_regions_grid
                                    .assign_coords(time=2020)
                                    .expand_dims("time"))
        xr_IAM_regions_grid_save = xr_IAM_regions_grid_save.rio.set_spatial_dims(x_dim="x",  y_dim="y")
        xr_IAM_regions_grid_save = xr_IAM_regions_grid_save.rio.write_crs("EPSG:4326")
        xr_IAM_regions_grid_save = xr_IAM_regions_grid_save.rio.write_transform()
        plot_maps.save_to_grid_tiff(dir_processed, xr_IAM_regions_grid_save, "region_number", "", [2020], model, scenario)

    # 1.2 Read and process in IAM data
    debug_log.info(f"\n\n1.2. Read and process in IAM data {"-"*25}")
    if process_flags["read_process_IAM"]:
        debug_log.info(f"\n{PRINT_COLORS["green"]}(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net} Reading and processing IAM data...{PRINT_COLORS["end"]}")
        df_IAM = process_IAM_data.read_process_IAM_data(project_dir, scenario, model, file_IAM_model_region_numbers, vars_downscaling)
        file_path = dir_processed / f"IAM_{model}_{scenario}_processed.csv"
        df_IAM.to_csv(file_path, sep=";", index=False)
    else:
        file_path = dir_processed / f"IAM_{model}_{scenario}_processed.csv"
        df_IAM = pd.read_csv(file_path, sep=";")

    # TO DO --> reindex populatoin, GDP, and emissions with xr_IAM_regions_grid to ensure alignment with the grid, especially if the grid has been modified or reprojected.
    # Now reprojections (for GDP) givea a small error of 0.67%
    #       CHECK: regions_grid needs to have a similar or higher resolution.
    # tolerance = max(new_res_x, new_res_y) / 2
    # xr_IAM_regions_coarse = xr_IAM_regions_grid.reindex_like(da_coarsened, method="nearest", tolerance=tolerance)

    # 1.3 read POP data
    debug_log.info(f"\n\n1.3. Read POP data {"-"*25}")
    df_population = None
    xr_population = None
    debug_log.info(f"(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}: {PRINT_COLORS["green"]}Reading (and processing) population data...{PRINT_COLORS["end"]}")
    if process_flags["read_process_POP"] or pop_file.is_file()==False:
        save_POP = True
        # read population data
        xr_population, f_population = process_grid_data.read_process_grid_data_socioeconomic(dir_processed=dir_processed, varname=varname_POP, source=source_POP, version=version_POP, SSP_base=SSP_base,
                                                                                             coarse_factor=coarse_factor_POP, unit=unit_POP, save=False, check=check_flags["check_POP_data"], log=debug_log)
        debug_log.info(f"{PRINT_COLORS["yellow"]}xr_population - [{xr_population.x.min().item()}, {xr_population.x.max().item()}{PRINT_COLORS["end"]}]")
        if check_flags["check_POP_data"]:
            locs_test = process_grid_data.check_data_locations(xr_population[varname_POP],2020)
            for l in locs_test:
                debug_log.info(f"Locations with non-zero population in 2020: {l}")
        xr_population = xr_population.sortby("y", ascending=False)  # north-to-south
        xr_population = xr_population.sortby("x", ascending=True)  # west-to-east
        xr_population.to_netcdf(pop_file, mode="w", engine="netcdf4")
        debug_log.info(f"Variable: {xr_population.data_vars})")
        arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_population[varname_POP])
        debug_log.info(f"resolution POP grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.1f} arc degrees")

        debug_log.info(f"{PRINT_COLORS["blue"]}Population data nodata, CRS and transform after read/process:{PRINT_COLORS["end"]}")
        debug_log.info(f"nodata: {PRINT_COLORS["blue"]}{xr_population[varname_POP].rio.nodata}{PRINT_COLORS["end"]}")
        debug_log.info(f"_FillValue: {PRINT_COLORS["blue"]}{xr_population.encoding.get('_FillValue')}{PRINT_COLORS["end"]}")
        debug_log.info(f"crs: {PRINT_COLORS["blue"]}{xr_population.rio.crs}{PRINT_COLORS["end"]}")
        debug_log.info(f"transform:\n{PRINT_COLORS["blue"]}{xr_population.rio.transform()}{PRINT_COLORS["end"]}")
        debug_log.info(f"unit: {PRINT_COLORS["blue"]}{xr_population[varname_POP].attrs["unit"]}{PRINT_COLORS["end"]}")
    else:
        if not pop_file.is_file():
            debug_log.info(f"{PRINT_COLORS["red"]}Population file not found at {pop_file}, cannot read data. Please set process_POP to True to process and save the data.{PRINT_COLORS["end"]}")
            exit()
        xr_population = xr.open_dataset(pop_file)
    if process_flags["save_tiffs_intermediate"]:
        plot_maps.save_to_grid_tiff(dir_processed, xr_population, varname_POP, "", [2020, 2030, 2050], model, scenario, False)

    debug_log.info("--------------------------------")

    # 1.4 read GDP (PPP) data
    debug_log.info(f"\n\n1.4. Read GDP (PPP) data {"-"*25}")
    f_gdp_ppp = None
    xr_gdp_ppp = None
    debug_log.info(f"(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}: {PRINT_COLORS["green"]}Reading (and processing) GDP (PPP) data...{PRINT_COLORS["end"]}")
    #if source_GDP=="Murakami" and version_GDP=="version_2021_1":
    #        xr_gdp_ppp = xr.open_dataset("Z:/cold_data_storage/users/roelfsemam/data_downscaling/GDP_PPP/processed\Murakami\processed_version_2021_1/SSP2/GDP_PPP_Murakami_SSP2_cf_12.nc")
    if process_flags["read_process_GDP_PPP"] or gdp_ppp_file.is_file()==False:
        save_GDP = True
        # read GDP data
        xr_gdp_ppp, f_gdp_ppp = process_grid_data.read_process_grid_data_socioeconomic(dir_processed=dir_processed, varname=varname_GDP, source=source_GDP, version=version_GDP, SSP_base=SSP_base,
                                                                                       coarse_factor=coarse_factor_GDP, unit=unit_GDP_PPP, save=False, check=check_flags["check_GDP_data"], log=debug_log)
        debug_log.info(f"{PRINT_COLORS["yellow"]}xr_gdp_ppp - [{xr_gdp_ppp.x.min().item()}, {xr_gdp_ppp.x.max().item()}{PRINT_COLORS["end"]}]")
        xr_gdp_ppp = xr_gdp_ppp.sortby("y", ascending=False)  # north-to-south
        xr_gdp_ppp = xr_gdp_ppp.sortby("x", ascending=True)  # west-to-east
        xr_gdp_ppp.to_netcdf(gdp_ppp_file, mode="w", engine="netcdf4")
        debug_log.info("--------------------------------")
        debug_log.info("process_grid_data.read_process_grid_data_socioeconomic")
        debug_log.info(f"{PRINT_COLORS["blue"]}GDP (PPP) data nodata, CRS and transform after read/process:{PRINT_COLORS["end"]}")
        debug_log.info(f"nodata: {PRINT_COLORS["blue"]}{xr_gdp_ppp[varname_GDP].rio.nodata}{PRINT_COLORS["end"]}")
        debug_log.info(f"_FillValue: {PRINT_COLORS["blue"]}{xr_gdp_ppp.encoding.get('_FillValue')}{PRINT_COLORS["end"]}")
        debug_log.info(f"crs: {PRINT_COLORS["blue"]}{xr_gdp_ppp.rio.crs}{PRINT_COLORS["end"]}")
        debug_log.info(f"transform:\n{PRINT_COLORS["blue"]}{xr_gdp_ppp.rio.transform()}{PRINT_COLORS["end"]}")
        debug_log.info(f"unit: {PRINT_COLORS["blue"]}{xr_gdp_ppp[varname_GDP].attrs["unit"]}{PRINT_COLORS["end"]}")
    else:
        if not gdp_ppp_file.is_file():
            debug_log.info(f"{PRINT_COLORS["red"]}GDP (PPP) file not found at {gdp_ppp_file}, cannot read data. Please set process_GDP to True to process and save the data.{PRINT_COLORS["end"]}")
            exit()
        xr_gdp_ppp = xr.open_dataset(gdp_ppp_file)
    if process_flags["save_tiffs_intermediate"]:
            plot_maps.save_to_grid_tiff(dir_processed, xr_gdp_ppp, varname_GDP, "", [2020, 2030, 2050], model, scenario, False)

    debug_log.info("--------------------------------")

    # 1.5 Read CO2 emissions data
    debug_log.info(f"\n\n1.5. Read CO2 emissions data {"-"*25}")
    f_emissions = None
    xr_emissions = None
    debug_log.info(f"\n(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}: {PRINT_COLORS["green"]}Reading (and processing) emissions data...{PRINT_COLORS["end"]}")
    if process_flags["read_process_EM"] or em_file.is_file()==False:
        save_EM = True
        xr_emissions, f_emissions = process_grid_data.read_process_grid_data_EM(dir_processed, varname=varname_EM, unit=unit_EM, source=source_EM, version=version_EM,
                                                                                base_year=base_year, coarse_factor=coarse_factor_EM, save=False, log=debug_log)
        debug_log.info(f"{PRINT_COLORS["yellow"]}xr_emissions - [{xr_emissions.x.min().item()}, {xr_emissions.x.max().item()}{PRINT_COLORS["end"]}]")
        xr_emissions = xr_emissions.sortby("y", ascending=False)  # north-to-south
        xr_emissions = xr_emissions.sortby("x", ascending=True)  # west-to-east
        xr_IAM_regions_grid = xr_IAM_regions_grid.reindex_like(xr_emissions.sel(time=base_year), method="nearest", tolerance=arc_minutes/60/2) # reindex to population grid (nearest neighbor with tolerance of half a grid cell)
        xr_emissions.to_netcdf(em_file, mode="w", engine="netcdf4")
        debug_log.info(f"unit: {PRINT_COLORS["blue"]}{xr_emissions[varname_EM].attrs["unit"]}{PRINT_COLORS["end"]}")
    else:
        if not em_file.is_file():
            debug_log.info(f"{PRINT_COLORS["red"]}Emissions file not found at {em_file}, cannot read data. Please set process_EM to True to process and save the data.{PRINT_COLORS["end"]}")
            exit()
        xr_emissions = xr.open_dataset(em_file)
    unit_EM = xr_emissions[varname_EM].attrs["unit"]
    debug_log.info("--------------------------------")
    if process_flags["save_tiffs_intermediate"]:
        plot_maps.save_to_grid_tiff(dir_processed, xr_emissions, varname_EM, "", [2020], model, scenario, False)

    # Checks
    # Check if data is read in successfully
    if xr_population is None or xr_gdp_ppp is None or xr_emissions is None:
        debug_log.info("Population, GDP (PPP) or emissions data not available. Cannot proceed further.")
        exit()
    # compare resolution
    debug_log.info(f"Variable: {xr_population.data_vars})")
    arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_population[varname_POP])
    debug_log.info(f"resolution POP grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.1f} arc degrees")
    debug_log.info(f"Variable: {xr_gdp_ppp.data_vars})")
    arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_gdp_ppp[varname_GDP])
    debug_log.info(f"resolution GDP grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.1f} arc degrees")
    debug_log.info(f"Variable: {xr_emissions.data_vars})")
    arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_emissions[varname_EM])
    debug_log.info(f"resolution EM grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.1f} arc degrees")

    # 1.6 Read in urban classification
    debug_log.info(f"\n\n1.6. Read urban classification data {"-"*25}")
    debug_log.info(f"{PRINT_COLORS["green"]}Reading urban classification data from: {dir_urban / 'urban_classification_years.parquet'}{PRINT_COLORS["end"]}")
    path_urban = dir_urban / "urban_classification_years.parquet"
    gdf_urban = gpd.read_parquet(path_urban)

    #----------------------------------------------------------------------------------------------------------------------------------------
    # 2. Process data
    debug_log.info(f"\n\n2. Process data {"-"*25}")

    # 2.1 Calculate GDP_PPP per capita
    debug_log.info(f"\n\n2.1. Calculate GDP_PPP per capita {"-"*25}")

    # 2.1.1 Grid data
    debug_log.info(f"\n\n2.1.1 Grid data{"-"*25}")
    debug_log.info(f"\n(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}: {PRINT_COLORS["green"]}Processing GDP and population data for downscaling...{PRINT_COLORS["end"]}")
    if check_flags["check_IAM_grid_data"]:
        plot_maps.plot_factors_GDP_POP(project_dir, xr_population, xr_gdp_ppp, None, year=2020, coarsen=12)
    # align, downscale, and set pop to 1 where gdp>0
    if process_flags["process_GDP_POP_grid"]:
        debug_log.info(f"{PRINT_COLORS["yellow"]}CHECK:{PRINT_COLORS["end"]}")
        debug_log.info(f"Unit population: {xr_population[varname_POP].attrs.get("unit", "N/A")}")
        debug_log.info(f"Unit GDP (PPP): {xr_gdp_ppp[varname_GDP].attrs.get("unit", "N/A")}")
        xr_population_processed, xr_gdp_ppp_processed = process_IPAT_factors.process_factors_GDP_POP(xr_population, xr_gdp_ppp, xr_emissions,
                                                                                                     varname_POP, varname_GDP,
                                                                                                     unit_POP, unit_GDP_PPP,
                                                                                                     years_downscaling, check_flags["check_GDP_POP"])

        debug_log.info(f"{PRINT_COLORS["yellow"]}xr_population_processed - [{xr_population_processed.x.min().item()}, {xr_population_processed.x.max().item()}{PRINT_COLORS["end"]}]")
        debug_log.info(f"{PRINT_COLORS["yellow"]}xr_gdp_ppp_processed - [{xr_gdp_ppp_processed.x.min().item()}, {xr_gdp_ppp_processed.x.max().item()}{PRINT_COLORS["end"]}]")
        process_IPAT_factors.check_POP_GDP_alignment(dir_processed, xr_population_processed, xr_gdp_ppp_processed, varname_POP, varname_GDP)
        xr_population_processed = xr_population_processed.reindex_like(xr_emissions.sel(time=base_year), method="nearest", tolerance=1e-5)
        xr_gdp_ppp_processed = xr_gdp_ppp_processed.reindex_like(xr_emissions.sel(time=base_year), method="nearest", tolerance=1e-5)
        xr_population_processed.to_netcdf(pop_processed_file, mode="w", engine="netcdf4")
        xr_gdp_ppp_processed.to_netcdf(gdp_ppp_processed_file, mode="w", engine="netcdf4")
        debug_log.info(f"time steps pop: {xr_population_processed[varname_POP].time.values}")
        debug_log.info(f"time steps gdp_per_pop: {xr_gdp_ppp_processed[varname_GDP].time.values}")
        xr_gdp_ppp_per_population = process_IPAT_factors.calculate_gdp_per_pop(xr_population_processed, xr_gdp_ppp_processed,
                                                                               varname_POP, varname_GDP, varname_gdp_per_pop,
                                                                               unit_POP, unit_GDP_PPP)
        xr_gdp_ppp_per_population.to_netcdf(gdp_ppp_per_pop_file, mode="w", engine="netcdf4")
        ds_check = xr.open_dataset(gdp_ppp_per_pop_file, decode_coords="all")
    else:
        xr_population_processed = xr.open_dataset(pop_processed_file)
        xr_gdp_ppp_processed = xr.open_dataset(gdp_ppp_processed_file)
        xr_gdp_ppp_per_population = xr.open_dataset(gdp_ppp_per_pop_file, decode_coords="all")
    if process_flags["save_tiffs_intermediate"]:
        plot_maps.save_to_grid_tiff(dir_processed, xr_population_processed, varname_POP, "_processed", [2020, 2030, 2050], model, scenario)
        plot_maps.save_to_grid_tiff(dir_processed, xr_gdp_ppp_processed, varname_GDP, "_processed", [2020, 2030, 2050], model, scenario)
        plot_maps.save_to_grid_tiff(dir_processed, xr_gdp_ppp_per_population, varname_gdp_per_pop, "", [2020, 2030, 2050], model, scenario)

    del xr_population, xr_gdp_ppp

    if check_flags["check_grid_GDP_per_pop"]:
        debug_log.info("--------------------------------")
        debug_log.info("xr_gdp_ppp_per_population")
        debug_log.info(f"varname: {varname_gdp_per_pop}")
        debug_log.info(f"Type: {type(varname_gdp_per_pop)}")
        debug_log.info(xr_gdp_ppp_per_population)
        process_grid_data.count_values_rio_xarray(xr_gdp_ppp_per_population, varname_gdp_per_pop, 2020, debug_log)
        process_IPAT_factors.check_location_for_GDP_per_pop_calculation(xr_gdp_ppp_per_population, varname_gdp_per_pop)

    # 2.1.2 IAM data
    debug_log.info(f"\n\n2.1.2 IAM data{"-"*25}")
    # reindex IAM regions grid with population and GDP grid
    debug_log.info(f"\n(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}: {PRINT_COLORS["green"]}Harmonising IAM regions grid with population and GDP grid...{PRINT_COLORS["end"]}")

    xr_IAM_regions_grid_downscaling = xr_IAM_regions_grid.reindex_like(xr_emissions, method="nearest", tolerance=1e-5)

    # calculate model GDP per capita
    csv_file = dir_processed / f"IAM_{model}_{scenario}_projection_gdp_per_pop.csv"
    if process_flags["process_GDP_per_POP"]:
        debug_log.info(f"(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}: {PRINT_COLORS["green"]}Calculating GDP per capita for IAM data...{PRINT_COLORS["end"]}")
        df_IAM_projection_pop = df_IAM[df_IAM["variable"] == varname_POP]
        df_IAM_projection_gpd_ppp = df_IAM[df_IAM["variable"] == varname_GDP]
        df_IAM_projection_gdp_ppp_per_population = pd.concat([df_IAM_projection_pop, df_IAM_projection_gpd_ppp], axis=0)
        df_IAM_projection_gdp_ppp_per_population = df_IAM_projection_gdp_ppp_per_population.pivot(index=["model", "scenario", "region_code", "region_number", "year"], columns="variable", values="value").reset_index()
        df_IAM_projection_gdp_ppp_per_population["value"] = df_IAM_projection_gdp_ppp_per_population[varname_GDP] / df_IAM_projection_gdp_ppp_per_population[varname_POP]
        df_IAM_projection_gdp_ppp_per_population.drop([varname_POP, varname_GDP], axis=1, inplace=True)
        df_IAM_projection_gdp_ppp_per_population["variable"] = varname_gdp_per_pop
        df_IAM_projection_gdp_ppp_per_population["unit"] = "USD_2005/yr/person"
        df_IAM_projection_gdp_ppp_per_population.to_csv(csv_file, sep=";")
    else:
        df_IAM_projection_gdp_ppp_per_population = pd.read_csv(csv_file, sep=";")

    if check_flags["check_IAM_GDP_per_pop"]:
        # 2.1.3 compare IAM and grid data for GDP per capita
        df_grid, df_compare = process_IPAT_factors.compare_IAM_grid_regions_GDP_per_capita(xr_gdp_ppp_per_population, varname_gdp_per_pop,
                                                                            xr_population_processed, varname_POP,
                                                                            df_IAM_projection_gdp_ppp_per_population,
                                                                            xr_IAM_regions_grid_downscaling)
        debug_log.info(df_grid.to_string(index=False))
        debug_log.info(df_compare.to_string())
        csv_file_grid = dir_processed / f"selection_grid_gdp_per_pop.csv"
        csv_file_compare = dir_processed / f"compare_IAM_grid_gdp_per_pop.csv"
        df_grid.to_csv(csv_file_grid, sep=";", index=False)
        df_compare.to_csv(csv_file_compare, sep=";", index=True)

    # 2.2 Calculate EM per GDP (PPP)
    debug_log.info(f"\n\n2.2. Calculate EM per GDP (PPP) {"-"*25}")
    debug_log.info(f"(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}: {PRINT_COLORS["green"]}Calculating EM per GDP (PPP) for IAM data...{PRINT_COLORS["end"]}")

    # 2.2.1 process grid data
    debug_log.info(f"\n\n2.2.1 Process grid data {"-"*25}")
    debug_log.info(xr_gdp_ppp_processed)
    debug_log.info(xr_emissions)

    # 2.2.2 process IAM (including the column names which are made lowercase)
    debug_log.info(f"\n\n2.2.2 Process grid data {"-"*25}")
    df_IAM_GDP = pd.DataFrame(df_IAM[df_IAM["variable"]==varname_GDP])
    df_IAM_EM = process_IAM_data.process_EM_regions_data(df_IAM, years_downscaling, varname_EM, vars_downscaling, net_emissions, model, debug_log)
    df_IAM_EM.to_csv(dir_processed / f"IAM_{model}_{scenario}_emissions_processed.csv", index=False, sep=";")

    one_unit_IMAGE_GDP_PPP = process_IAM_data.model_unit_conversions[model]["GDP|PPP"]
    one_unit_IMAGE_em = process_IAM_data.model_unit_conversions[model]["Emissions|CO2"]
    df_IAM_GDP = process_IAM_data.extrapolate_IAM_values_to_convergence_year(dir_processed, df_IAM_GDP, one_unit_IMAGE_GDP_PPP, convergence_year, method_extension, debug_log)
    df_IAM_EM = process_IAM_data.extrapolate_IAM_values_to_convergence_year(dir_processed, df_IAM_EM, one_unit_IMAGE_em, convergence_year, method_extension, debug_log)
    csv_file_GDP = dir_processed / f"IAM_{model}_{scenario}_gdp_ppp_downscaling_extended.csv"
    csv_file_EM = dir_processed / f"IAM_{model}_{scenario}_em_downscaling_extended.csv"
    df_IAM_GDP.to_csv(csv_file_GDP, index=False, sep=";")
    df_IAM_EM.to_csv(csv_file_EM, index=False, sep=";")

    csv_file_em_per_gdp_ppp = dir_processed / f"IAM_{model}_{scenario}_em_per_gdp_ppp.csv"
    if process_flags["process_df_EM_per_GDP"]:
        # unit_GDP_PPP = df_IAM_GDP["unit"].unique()[0]
        # unit_em = df_IAM_EM["unit"].unique()[0]
        df_IAM_projection_em_per_gdp_ppp = pd.concat([df_IAM_EM, df_IAM_GDP], axis=0)
        df_IAM_projection_em_per_gdp_ppp = df_IAM_projection_em_per_gdp_ppp.pivot(index=["model", "scenario", "region_code", "region_number", "year"], columns="variable", values="value").reset_index()
        df_IAM_projection_em_per_gdp_ppp["value"] = df_IAM_projection_em_per_gdp_ppp[varname_EM] / df_IAM_projection_em_per_gdp_ppp[varname_GDP]
        df_IAM_projection_em_per_gdp_ppp["variable"] = varname_EM + "_per_" + varname_GDP
        df_IAM_projection_em_per_gdp_ppp["unit"] = "tCO2/USD_2005/yr"
        df_IAM_projection_em_per_gdp_ppp.drop([varname_EM, varname_GDP], axis=1, inplace=True)
        df_IAM_projection_em_per_gdp_ppp.to_csv(csv_file_em_per_gdp_ppp, index=False, sep=";")
    else:
        df_IAM_projection_em_per_gdp_ppp = pd.read_csv(csv_file_em_per_gdp_ppp, sep=";")

    # 2.2.3 Pocess grid emissions per GDP (PPP)
    debug_log.info(f"\n\n2.2.3 Process grid emissions per GDP (PPP) {"-"*25}")
    xr_gdp_ppp_by = xr_gdp_ppp_processed.sel(time=base_year)
    xr_gdp_ppp_by["name"] = varname_GDP
    xr_gdp_ppp_by = xr_gdp_ppp_by.chunk({"x": "auto", "y": "auto"})
    xr_em_by = xr_emissions.sel(time=base_year)
    xr_em_by = xr_em_by.chunk({"x": "auto", "y": "auto"})

    # check resoltuion
    debug_log.info(f"Variable: {xr_em_by.data_vars})")
    arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_em_by[varname_EM])
    debug_log.info(f"resolution EM grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.2f} arc degrees")
    debug_log.info(f"Variable: {xr_gdp_ppp_by.data_vars})")
    arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_gdp_ppp_by[varname_GDP])
    debug_log.info(f"resolution GDP (PPP) grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.2f} arc degrees")
    debug_log.info(f"Variable: {xr_IAM_regions_grid_downscaling.data_vars})")
    arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_IAM_regions_grid_downscaling["region_number"])
    debug_log.info(f"resolution region grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.2f} arc degrees")

    # calculate CO2/GDP (PPP) for base year
    xr_em_per_gdp_ppp_by_downscaling = xr_em_by[varname_EM] / xr_gdp_ppp_by[varname_GDP].where(xr_gdp_ppp_by[varname_GDP] != 0)  # Avoid division by zero

    # # check
    # import dask.config
    # with dask.config.set(scheduler="single-threaded"):
    #     _ = xr_em_per_gdp_ppp_by_downscaling.compute()

    debug_log.info(xr_em_by[varname_EM].attrs["unit"])
    debug_log.info(xr_gdp_ppp_by[varname_GDP].attrs["unit"])
    xr_em_per_gdp_ppp_by_downscaling.attrs["unit"] = xr_em_by[varname_EM].attrs["unit"] + "/" + xr_gdp_ppp_by[varname_GDP].attrs["unit"]
    xr_em_per_gdp_ppp_by_downscaling["name"] = varname_em_per_gdp_ppp
    debug_log.info(f"Type xr_em_per_gdp_ppp_by_downscaling: {type(xr_em_per_gdp_ppp_by_downscaling)}")
    xr_em_per_gdp_ppp_by_downscaling.attrs["unit"] = df_IAM_projection_em_per_gdp_ppp["unit"].iloc[0]  # Update unit attribute

    # 2.3 Calculate CO2 emissions grid
    debug_log.info(f"\n\n2.3 Calculate CO2 emissions grid {"-"*25}")
    if process_flags["process_grid_EM_per_GDP"]:
        debug_log.info(f"(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}: {PRINT_COLORS["green"]} Calculating CO2 grid emissions...{PRINT_COLORS["end"]}")
        # calculate emissions per GDP (PPP) for years after base year
        debug_log.info("Calculating scaling factors...")
        xr_scaling_factor_by, regions, x_coords, y_coords = process_IPAT_factors.calc_scaling_factors_EM_per_GDP(xr_IAM_regions_grid_downscaling, base_year,
                                                                                                                 xr_gdp_ppp_by[varname_GDP], xr_em_per_gdp_ppp_by_downscaling,
                                                                                                                 df_IAM_projection_em_per_gdp_ppp)
        # downscale emissions per GDP (PPP) for years after base year
        # Extend years to include target year if not present
        years_downscaling_extended = sorted(list(set(years_downscaling + [convergence_year])))
        debug_log.info("Downscaling emissions per GDP (PPP) for years after base year...")
        xr_em_per_gdp_ppp =  process_IPAT_factors.downscale_em_per_gdp(xr_scaling_factor_by, varname_em_per_gdp_ppp,
                                                                        xr_IAM_regions_grid_downscaling,
                                                                        df_IAM_projection_em_per_gdp_ppp,
                                                                        years_downscaling_extended, base_year, convergence_year,
                                                                        regions, x_coords, y_coords)
        xr_em_per_gdp_ppp.to_netcdf(em_per_gdp_ppp_file, mode="w", engine="netcdf4")
        if process_flags["save_tiffs_intermediate"]:
            plot_maps.save_to_grid_tiff(dir_processed, xr_em_per_gdp_ppp, varname_em_per_gdp_ppp, "", [2020, 2030, 2050], model, scenario)

        # 2.3.1 Calculate grid emissions by applying IPAT factors to population and GDP per capita grids
        debug_log.info(f"\n\n2.3.1 Calculate grid emissions by applying IPAT factors to population and GDP per capita grids {"-"*25}")
        xr_gdp_ppp_per_population_processed = xr_gdp_ppp_per_population.copy()
        xr_em = (xr_population_processed[varname_POP] * xr_gdp_ppp_per_population_processed[varname_gdp_per_pop] * xr_em_per_gdp_ppp[varname_em_per_gdp_ppp])
        xr_em = xr_em.to_dataset(name=varname_EM)
        xr_em[varname_EM].attrs["unit"] = unit_EM
        xr_em.to_netcdf(em_unharmonised_file, mode="w", engine="netcdf4")
    else:
        xr_em = xr.open_dataset(em_unharmonised_file)
    if process_flags["save_tiffs_intermediate"]:
        plot_maps.save_to_grid_tiff(dir_processed, xr_em, varname_EM, "_unharmonised", [2020, 2030, 2050], model, scenario)

    del xr_population_processed, xr_gdp_ppp_processed, xr_gdp_ppp_per_population

    # 2.3.2 harmonise grid emissions per region with IAM emissions per region
    debug_log.info(f"\n\n2.3.2 Harmonise grid emissions per region with IAM emissions per region {"-"*25}")
    debug_log.info(f"(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}:{PRINT_COLORS["green"]}: Harmonising grid emissions per region with IAM emissions per region...{PRINT_COLORS["end"]}")
    years = df_IAM_EM["year"].unique()
    variable = df_IAM_EM["variable"].unique()[0]
    extra_rows = pd.DataFrame({"model": model, "scenario": scenario, "region_code":"OCEAN", "variable":variable, "year": years, "unit": unit_EM, "region_number": 0, "value": 0})
    df_IAM_EM_harm = pd.concat([df_IAM_EM, extra_rows], ignore_index=True).sort_values(["year", "region_number"]).reset_index(drop=True)
    df_IAM_EM_compare, xr_regional_sums = process_IPAT_factors.calc_regional_values(xr_em, varname_EM,
                                                                                    xr_IAM_regions_grid_downscaling, df_IAM_EM_harm,
                                                                                    years_downscaling)
    csv_file_compare = dir_output / f"Emissions_region_{scenario}_{profile}_unharmonised.csv"
    df_IAM_EM_compare.to_csv(csv_file_compare, sep=";", index=False)

    # 2.3.3 Calculate harmonisation factor for grid emissions per region with IAM emissions per region
    debug_log.info(f"\n\n2.3.3 Calculate harmonisation factors for grid emissions per region with IAM emissions per region {"-"*25}")
    xr_em_correction_factors = process_IPAT_factors.calculate_harmonisation_factors_emissions(xr_em, varname_EM, xr_regional_sums,
                                                                                              xr_IAM_regions_grid_downscaling, df_IAM_EM_harm,
                                                                                              years_downscaling)

    # 2.3.4 apply harmonisation factors to grid emissions
    debug_log.info(f"\n\n2.3.4 Apply harmonisation factors to grid emissions {"-"*25}")
    xr_em_grid_correction = process_IPAT_factors.apply_harmonisation_factors_emissions(xr_em_correction_factors,
                                                                                       xr_em, varname_EM,
                                                                                       xr_IAM_regions_grid_downscaling,
                                                                                       model, scenario)
    xr_em_grid_correction[varname_EM].attrs["unit"] = unit_EM
    plot_maps.save_to_grid_tiff(dir_processed, xr_em, varname_EM, "_harmonised", years_downscaling, model, scenario)


    debug_log.info(f"Variable: {xr_em_grid_correction.data_vars})")
    arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_em_grid_correction[varname_EM])
    debug_log.info(f"resolution downscaled EM grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.2f} arc degrees")

    x_min, x_max = float(xr_em_grid_correction[varname_EM].x.min()), float(xr_em_grid_correction[varname_EM].x.max())
    y_min, y_max = float(xr_em_grid_correction[varname_EM].y.min()), float(xr_em_grid_correction[varname_EM].y.max())
    xr_em_grid_correction = xr_em_grid_correction.sortby("y", ascending=False)  # north-to-south
    xr_em_grid_correction = xr_em_grid_correction.sortby("x", ascending=True)  # west-to-east
    xr_em_grid_correction.to_netcdf(em_harmonised_file, mode="w", engine="netcdf4")
    debug_log.info(f"extent downscaled EM grid: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

    debug_log.info(f"\n{PRINT_COLORS["green"]}(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net} Downscaling complete. Processed data saved to {dir_processed} and output to {dir_output}.{PRINT_COLORS["end"]}")

    # calculate sum per region per year for harmonised emissions
    df_IAM_EM_corrected_compare, xr_regional_sums_corrected = process_IPAT_factors.calc_regional_values(xr_em_grid_correction, varname_EM,
                                                                                                        xr_IAM_regions_grid_downscaling, df_IAM_EM_harm,
                                                                                                        years_downscaling)
    csv_file_compare_corrected = dir_output / f"Emissions_region_{scenario}_{profile}_harmonised.csv"
    df_IAM_EM_corrected_compare.to_csv(csv_file_compare_corrected, sep=";", index=False)

    # 2.4 Calculate urban emissions
    debug_log.info(f"\n\n2.4.1 Calculate urban emissions {"-"*25}")

    # 2.4.1 Unharmonised
    debug_log.info(f"\n\n2.4.1 Unharmonised {"-"*25}")
    debug_log.info(f"\n\n(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}: {PRINT_COLORS["green"]}Calculating urban emissions...{PRINT_COLORS["end"]}")
    debug_log.info(f"unharmonised emissions")
    # add regions to xr_em
    xr_em_regions = xr_em.copy()
    xr_em_regions["region_number"] = xr_IAM_regions_grid_downscaling["region_number"]
    xr_em_urban_unharmonised, df_em_urban_unharmonised, df_em_rural_unharmonised = aggregate_urban_emissions(xr_emissions=xr_em_regions, gdf_urban_classification=gdf_urban,
                                                                                   emissions_varname=varname_EM,
                                                                                   region_varname="region_number",
                                                                                   final_year=2050)
    df_em_urban_unharmonised.to_csv(dir_output / f"Emissions_urban_region_{scenario}_{profile}_unharmonised.csv", index=False, sep=";")
    df_em_rural_unharmonised.to_csv(dir_output / f"Emissions_rural_region_{scenario}_{profile}_unharmonised.csv", index=False, sep=";")
    # combine dataframes for urban and rural emissions into one dataset, add the sum of urban and rural as "total", and using "urban", "rural" and "total" as a column names
    df_em_urban_unharmonised["Type"] = "urban"
    df_em_rural_unharmonised["Type"] = "rural"
    df_combined_unharmonised = pd.concat([df_em_urban_unharmonised, df_em_rural_unharmonised], ignore_index=True)
    df_combined_unharmonised.to_csv(dir_output / f"Emissions_combined_region_{scenario}_{profile}_unharmonised.csv", index=False, sep=";")
    # add total = urban + rural
    df_total_unharmonised = df_combined_unharmonised.groupby(["model", "scenario", "region_code", "region_number", "year", "unit"]).agg({"value": "sum"}).reset_index()
    df_total_unharmonised["Type"] = "total"
    df_total_unharmonised.to_csv(dir_output / f"Emissions_total_region_{scenario}_{profile}_unharmonised.csv", index=False, sep=";")

    try:
        xr_em_urban_unharmonised.to_netcdf(dir_output / f"Emissions_urban_region_{scenario}_{profile}_unharmonised.nc", engine="netcdf4")
    except Exception as e:
        debug_log.error(f"Error occurred while saving urban emissions NetCDF file: {e}")

    # 2.4.2 Harmonised
    debug_log.info(f"\n\n2.4.2 Harmonised {"-"*25}")
    xr_em_urban_harmonised, df_em_urban_harmonised, df_em_rural_harmonised = aggregate_urban_emissions(xr_emissions=xr_em_grid_correction, gdf_urban_classification=gdf_urban,
                                                                               emissions_varname=varname_EM,
                                                                               region_varname="region_number",
                                                                               final_year=2050)
    df_em_urban_harmonised.to_csv(dir_output / f"Emissions_urban_classification_region_{scenario}_{profile}_harmonised.csv", index=False, sep=";")
    df_em_rural_harmonised.to_csv(dir_output / f"Emissions_rural_classification_region_{scenario}_{profile}_harmonised.csv", index=False, sep=";")
    # combine dataframes for urban and rural emissions into one dataset, add the sum of urban and rural as "total", and using "urban", "rural" and "total" as a column names
    df_em_urban_harmonised["Type"] = "urban"
    df_em_rural_harmonised["Type"] = "rural"
    df_combined_harmonised = pd.concat([df_em_urban_harmonised, df_em_rural_harmonised], ignore_index=True)
    df_combined_harmonised.to_csv(dir_output / f"Emissions_combined_region_{scenario}_{profile}_harmonised.csv", index=False, sep=";")
    # add total = urban + rural
    df_total_harmonised = df_combined_harmonised.groupby(["model", "scenario", "region_code", "region_number", "year", "unit"]).agg({"value": "sum"}).reset_index()
    df_total_harmonised["Type"] = "total"
    df_total_harmonised.to_csv(dir_output / f"Emissions_total_region_{scenario}_{profile}_harmonised.csv", index=False, sep=";")
    try:
        xr_em_urban_harmonised.to_netcdf(dir_output / f"Emissions_urban_classification_region_{scenario}_{profile}_harmonised.nc", engine="netcdf4")
    except Exception as e:
        debug_log.error(f"Error occurred while saving urban emissions NetCDF file: {e}")

    # 2.5 End code
    debug_log.info(f"\n\n2.5 End code {"-"*25}")

    elapsed_time = time.time() - start_time
    # diviede elapsed time into hours, minutes, and seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    debug_log.info(f"\n{PRINT_COLORS["green"]}{profile}-{scenario}-{gross_net}) Total elapsed time: {hours:,.2f} hours, {minutes:,.2f} minutes, {seconds:,.2f} seconds{PRINT_COLORS["end"]}")

    # cleanup temporary log files if they are empty
    cleanup_empty_logs(log_path)

    # exit code
    try:
        client = get_client()
        client.close()
    except Exception:
        pass

def plot_results(scenario:str = "ELV-SSP2-CP", model:str="IMAGE", profile:str = "default", net_emissions:bool=True, global_min:float|None=None, global_max:float|None=None):
    from shapely.ops import unary_union  # add alongside the other imports
    debug_log, results_log = init_logging(f"log_downscaling_{profile}_{model}_{scenario}", "log/plotting")

    if profile not in settings.SOURCE_PROFILES:
        available = list(settings.SOURCE_PROFILES.keys())
        raise ValueError(f"Unknown source profile '{profile}'. Available: {available}")
    else:
        sources = settings.SOURCE_PROFILES[profile]

    source_POP = sources["source_POP"]
    version_POP = sources["version_POP"]
    source_GDP = sources["source_GDP"]
    version_GDP = sources["version_GDP"]
    source_EM = sources["source_EM"]
    version_EM = sources["version_EM"]

    varname_GDP = settings.varname_GDP
    varname_POP = settings.varname_POP
    varname_EM = settings.varname_EM

    SSP_base = settings.SSP_base

    vars_downscaling = settings.vars_downscaling

    file_model_grid_regions = settings_models.models[model]["file_model_grid_regions"]
    file_IAM_model_region_numbers = settings_models.models[model]["file_IAM_model_region_numbers"]

    project_dir = Path(__file__).parent.parent
    print(f"\nProject directory: {project_dir}")
    gross_net = "net" if net_emissions else "gross"
    source_version_grid = f"{source_POP}_{version_POP}_{source_GDP}_{version_GDP}_{source_EM}_{version_EM}_{gross_net}"
    model_scenario = f"{model}_{scenario}"
    #dir_output = project_dir / "data" / "output"
    dir_processed = project_dir / "data" / "processed" / profile / source_version_grid / model_scenario
    #print(f"Output directory: {dir_output}")
    print(f"Processed data directory: {dir_processed}")

    coarse_factor_POP, coarse_factor_GDP, coarse_factor_EM, res_min_POP, res_min_GDP, res_min_EM = process_grid_data.get_coarsening_factors(population_source="2UP",gdp_source="Murakami",emissions_source="EDGAR")
    print(f"Coarsening factors - Population: {coarse_factor_POP}, GDP: {coarse_factor_GDP}, Emissions: {coarse_factor_EM}")

    # files for processed grid data
    pop_file = dir_processed / f"Population_{source_POP}_{version_POP}_{SSP_base}_cf_{coarse_factor_POP}.nc"
    gdp_ppp_file = dir_processed / f"GDP_PPP_{source_GDP}_{version_GDP}_{SSP_base}_cf_{coarse_factor_GDP}.nc"
    em_file = dir_processed / f"{replace_punctuation_in_filenames(varname_EM)}_hist_{source_EM}_{version_EM}_{SSP_base}_cf_{coarse_factor_EM}.nc"
    pop_processed_file = dir_processed / f"Population_processed_{source_POP}_{version_POP}_{SSP_base}_cf_{coarse_factor_POP}.nc"
    gdp_ppp_processed_file = dir_processed / f"GDP_PPP_processed_{source_GDP}_{version_GDP}_{SSP_base}_cf_{coarse_factor_GDP}.nc"
    em_harmonised_file = dir_processed / f"{replace_punctuation_in_filenames(varname_EM)}_harmonised_{SSP_base}.nc"
    #file_path_file_model_grid_regions = project_dir / f"data/input/models/{model}/{file_model_grid_regions}"
    file_path_file_model_grid_regions = determine_regions_file(project_dir, res_min_POP, res_min_GDP, res_min_EM, model, debug_log)

    figures_dir = dir_processed / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    xr_population_hist = xr.open_dataset(pop_file)
    xr_gdp_ppp_hist = xr.open_dataset(gdp_ppp_file)
    xr_emissions_hist = xr.open_dataset(em_file)
    xr_population_proj = xr.open_dataset(pop_processed_file)
    xr_gdp_ppp_proj = xr.open_dataset(gdp_ppp_processed_file)
    xr_emissions_proj = xr.open_dataset(em_harmonised_file)
    xr_IAM_regions_grid = xr.open_dataset(file_path_file_model_grid_regions)

    arc_seconds_pop, arc_minutes_pop, arc_degrees_pop = process_grid_data.calculate_resolution(xr_population_proj[varname_POP])
    print(f"{PRINT_COLORS["green"]}Resolution: degrees-{arc_degrees_pop:.2f},  minutes-{arc_minutes_pop:.2f}, seconds-{arc_seconds_pop:.2f}{PRINT_COLORS["end"]}")
    arc_seconds_gdp_ppp, arc_minutes_gdp_ppp, arc_degrees_gdp_ppp = process_grid_data.calculate_resolution(xr_gdp_ppp_proj[varname_GDP])
    print(f"{PRINT_COLORS["green"]}Resolution: degrees-{arc_degrees_gdp_ppp:.2f},  minutes-{arc_minutes_gdp_ppp:.2f}, seconds-{arc_seconds_gdp_ppp:.2f}{PRINT_COLORS["end"]}")
    arc_seconds_em, arc_minutes_em, arc_degrees_em = process_grid_data.calculate_resolution(xr_emissions_proj[varname_EM])
    print(f"{PRINT_COLORS["green"]}Resolution: degrees-{arc_degrees_em:.2f},  minutes-{arc_minutes_em}, seconds-{arc_seconds_em:.2f}{PRINT_COLORS["end"]}")

    # 1. compare IAM and grid data for population, GDP, and emissions for historical and projected data
    years_downscaling = [2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]

    df_IAM = process_IAM_data.read_process_IAM_data(project_dir, scenario, model, file_IAM_model_region_numbers, vars_downscaling)

    df_IAM_POP = df_IAM[df_IAM["variable"]==varname_POP]
    df_IAM_GDP = df_IAM[df_IAM["variable"]==varname_GDP]
    df_IAM_EM = process_IAM_data.process_EM_regions_data(df_IAM, years_downscaling, varname_EM, vars_downscaling, net_emissions)

    years_EM = df_IAM_EM["year"].unique()
    variable_EM = df_IAM_EM["variable"].unique()[0]
    unit_EM = xr_emissions_proj[varname_EM].attrs.get("unit", "N/A")
    extra_rows = pd.DataFrame({"model": model, "scenario": scenario, "region_code":"OCEAN", "variable":variable_EM, "year": years_EM, "unit": unit_EM, "region_number": 0, "value": 0})
    df_IAM_EM_harm = pd.concat([df_IAM_EM, extra_rows], ignore_index=True).sort_values(["year", "region_number"]).reset_index(drop=True)
    xr_emissions_proj[varname_EM] = xr_emissions_proj[varname_EM] * 10**-6

    # plot_maps.plot_comparison_IAM_grid(project_dir, dir_processed, False,
    #                                     df_IAM_EM, xr_emissions_proj, xr_IAM_regions_grid, varname_EM,
    #                                     model, scenario, years_downscaling,
    #                                     file_IAM_model_region_numbers)

    # # 2. plot maps for population, GDP, and emissions for historical and projected data    print(f"directory: {project_dir.resolve()}")
    # plot = plot_maps.plot_IPAT_summary(dir_processed, f"{scenario}",
    #                   xr_population_hist, xr_gdp_ppp_hist, xr_emissions_hist,
    #                   xr_population_proj, xr_gdp_ppp_proj, xr_emissions_proj,
    #                   varname_POP, varname_GDP, varname_EM,
    #                   2020, [2030, 2050],1)

    # # 3. Plot histograms for emissions projections
    # for y in [2020, 2030, 2050]:
    #     plot_maps.plot_hist(dir_processed, xr_emissions_proj, scenario, varname_EM, "", y)

    # plot_maps.plot_boxplot_per_region(project_dir, dir_processed, file_IAM_model_region_numbers,
    #                                   xr_emissions_proj, varname_EM,
    #                                   model, scenario, [2020, 2050])

    # plot_maps.plot_hist_map(dir_processed, xr_population_hist, scenario, varname_POP, 2020)
    # plot_maps.plot_hist_map(dir_processed, xr_gdp_ppp_hist, scenario, varname_GDP, 2020)
    # plot_maps.plot_hist_map(dir_processed, xr_emissions_proj, scenario, varname_EM, 2020)
    # plot_maps.plot_hist_map(dir_processed, xr_emissions_proj, scenario, varname_EM, 2030)
    # plot_maps.plot_hist_map(dir_processed, xr_emissions_proj, scenario, varname_EM, 2050)

    fig_2020, ax, pm = plot_maps.plot_Mercator_projection(xr_emissions_proj[varname_EM].sel(time=2020),
                                                            ax=None, coarsen=12, transform="linear", show=False, title=None,
                                                            cbar_shrink=0.6, cbar_aspect=20, cbar_pad=0.05)
    fig_2020.savefig(f"{figures_dir}/map_{varname_EM}_{scenario}_2020.jpg", dpi=150, bbox_inches="tight")
    fig_2030, ax, pm = plot_maps.plot_Mercator_projection(xr_emissions_proj[varname_EM].sel(time=2030),
                                                            ax=None, coarsen=12, transform="linear", show=False, title=None,
                                                            cbar_shrink=0.6, cbar_aspect=20, cbar_pad=0.05)
    fig_2030.savefig(f"{figures_dir}/map_{varname_EM}_{scenario}_2030.jpg", dpi=150, bbox_inches="tight")
    fig_2050, ax, pm = plot_maps.plot_Mercator_projection(xr_emissions_proj[varname_EM].sel(time=2050),
                                                            ax=None, coarsen=12, transform="linear", show=False, title=None,
                                                            cbar_shrink=0.6, cbar_aspect=20, cbar_pad=0.05)
    fig_2050.savefig(f"{figures_dir}/map_{varname_EM}_{scenario}_2050.jpg", dpi=150, bbox_inches="tight")

    # Plot specific cities/towns
    cities_towns = ["Amsterdam", "Lima", "Raleigh", "New York"]
    coord_Amsterdam = [3.0, 7.0, 51.0, 54.0]
    coord_Lima = [-79.03, -75.03, -14.05, -10.05]
    coord_Raleigh = [-80.64, -76.64, 33.77, 37.77]
    coord_NewYork = [-76.01, -72.01, 38.71, 42.71]
    coords = [coord_Amsterdam, coord_Lima, coord_Raleigh, coord_NewYork]

    # cities_config = [{"name": "Amsterdam", "within_US": False, "iso3": "NLD", "coords": coord_Amsterdam},
    #                  {"name": "Lima",      "within_US": False, "iso3": "PER", "coords": coord_Lima},
    #                  {"name": "Raleigh",  "within_US": True,  "iso3": None,  "coords": coord_Raleigh},
    #                  {"name": "New York",  "within_US": True,  "iso3": None,  "coords": coord_NewYork}]
    cities_config = [{"name": "Amsterdam", "within_US": False, "iso3": "NLD", "coords": coord_Amsterdam,
                    "sub_cities": ["Amsterdam", "Rotterdam", "'s-Gravenhage", "Utrecht"]},
                    {"name": "Lima",     "within_US": False, "iso3": "PER", "coords": coord_Lima},
                    {"name": "Raleigh",  "within_US": True,  "iso3": None,  "coords": coord_Raleigh},
                    {"name": "New York", "within_US": True,  "iso3": None,  "coords": coord_NewYork}]

    settings_file = project_dir / "downscaling" / "settings_data_locations.json"
    with open(settings_file, "r") as f:
        data_files = json.load(f)
    data_files = apply_root_json(data_files, data_files["data_root"])
    dir_GADM_geopackage = Path(data_files["GADM"]["dir_GADM_geopackage"])
    dir_US_Census_Tiger = Path(data_files["US_Census"]["dir_US_Census_TIGER"])
    gadm_gpkg_path = dir_GADM_geopackage / "gadm_410-levels.gpkg"
    dir_polygons = Path(f"{dir_processed}/polygons")

    # for city in cities_config:
    #     try:
    #         if city["within_US"]:
    #             city["polygon"] = convert_GIS.get_us_city_polygon(tiger_dir=dir_US_Census_Tiger, city_name=city["name"], output_dir=dir_polygons)
    #         else:
    #             city["polygon"] = convert_GIS.get_city_polygon(gpkg_path=gadm_gpkg_path, iso3=city["iso3"], city_name=city["name"], output_dir=dir_polygons)
    #     except (ValueError, FileNotFoundError) as e:
    #         print(f"Warning: Could not load polygon for '{city['name']}': {e}")
    #         city["polygon"] = None
    for city in cities_config:
        names = city.get("sub_cities", [city["name"]])
        polys = []
        for nm in names:
            try:
                if city["within_US"]:
                    poly = convert_GIS.get_us_city_polygon(tiger_dir=dir_US_Census_Tiger, city_name=nm, output_dir=dir_polygons)
                else:
                    poly = convert_GIS.get_city_polygon(gpkg_path=gadm_gpkg_path, iso3=city["iso3"], city_name=nm, output_dir=dir_polygons)
                if poly is not None:
                    polys.append(poly)
            except (ValueError, FileNotFoundError) as e:
                print(f"Warning: Could not load polygon for '{nm}': {e}")
        city["polygons"] = polys
        city["polygon"] = unary_union(polys) if polys else None

    if global_min is None or (not isinstance(global_min, float)):
        print(f"{PRINT_COLORS["yellow"]}Using 2.5% percentile for global_min emissions{PRINT_COLORS["end"]}")
        global_min = float(xr_emissions_proj[varname_EM].quantile(0.025))
    if global_max is None or (not isinstance(global_max, float)):
        print(f"{PRINT_COLORS["yellow"]}Using 97.5% percentile for global_max emissions{PRINT_COLORS["end"]}")
        global_max = float(xr_emissions_proj[varname_EM].quantile(0.975))
    print(f"{PRINT_COLORS["green"]}min: {global_min:,.6f}, max: {global_max:,.6f}{PRINT_COLORS["end"]}")

    emission_records = []
    years_plot = [2020, 2030, 2040, 2050]
    for city in cities_config:
        for y in years_plot:
            print(f"City/town: {city['name']} at coordinates {city['coords']} for year {y}")
            da_city_town = xr_emissions_proj[varname_EM].sel(time=y).rio.clip_box(minx=city["coords"][0], miny=city["coords"][2], maxx=city["coords"][1], maxy=city["coords"][3])
            # Calculate emissions within polygon
            stats = convert_GIS.calculate_emissions_in_polygon(da_city_town, city["polygon"], city["name"])
            stats["year"] = y
            stats["scenario"] = scenario
            stats["variable"] = varname_EM
            emission_records.append(stats)
            fig_city_town, ax_city_town, pm_city_town = plot_maps.plot_Mercator_projection(da_city_town, coarsen=1, transform="linear", show=False,
                                                                                           title=f"Emissions {scenario} around {city['name']} by year {y}",
                                                                                           vmin=global_min, vmax=global_max, add_polygon=city["polygon"])
            ylabel_text = ax_city_town.get_ylabel()
            ax_city_town.set_ylabel(ylabel_text, labelpad=40)  # increase value until clear of ticks
            ax_city_town.set_extent(city["coords"], crs=ccrs.PlateCarree())
            # Add raster cell boundary lines
            x_dim = "x" if "x" in da_city_town.dims else "lon"; y_dim = "y" if "y" in da_city_town.dims else "lat"
            x_coords = da_city_town[x_dim].values; y_coords = da_city_town[y_dim].values
            res_x = abs(float(x_coords[1] - x_coords[0])); res_y = abs(float(y_coords[1] - y_coords[0]))
            # Cell edges are at centre ± half resolution
            x_edges = np.append(x_coords - res_x / 2, x_coords[-1] + res_x / 2)
            y_edges = np.append(y_coords - res_y / 2, y_coords[-1] + res_y / 2)

            ax_city_town.set_xticks(x_edges, crs=ccrs.PlateCarree())
            ax_city_town.set_yticks(y_edges, crs=ccrs.PlateCarree())
            ax_city_town.xaxis.set_ticklabels([])  # hide tick labels, we only want the grid lines
            ax_city_town.yaxis.set_ticklabels([])
            ax_city_town.grid(True, color="white", linewidth=0.5, alpha=0.5, linestyle="-")
            sum_cells = stats["sum_weighted"]; sum_full = stats["sum_full"]; avg_sum_per_cell = stats["mean_per_m2"]
            ax_city_town.text(0.99, 0.01, f"City: {city["name"]} ({unit_EM})\nsum_cells: {sum_cells:.3f}\nsum_full: {f'{sum_full:.3f}' if sum_full != 0 else 'NA'}\navg_sum_per_cell: {avg_sum_per_cell:.3f}",
                              color="white", bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=3),
                              transform=ax_city_town.transAxes, ha="right", va="bottom", fontsize=8)
            fig_city_town.savefig(f"{figures_dir}/map_{varname_EM}_{scenario}_{city['name']}_{y}.jpg", dpi=150, bbox_inches="tight")
            plt.close(fig_city_town)

    # Save emission statistics to CSV
    df_emissions_table = pd.DataFrame(emission_records)
    cols = ["city", "year", "scenario", "variable", "sum_weighted", "sum_full", "mean_per_m2", "min", "max", "n_pixels_full", "n_pixels_partial", "n_pixels_any"]
    df_emissions_table = df_emissions_table[cols]
    out_csv = Path(f"{figures_dir}/emissions_per_city_table_{scenario}_{varname_EM}.csv")
    df_emissions_table.to_csv(out_csv, sep=";", index=False)
    df_emissions = df_emissions_table.drop(columns="variable").melt(id_vars=["city", "year", "scenario"], var_name="variable", value_name="value")
    df_emissions.to_csv(Path(f"{figures_dir}/emissions_per_city_{scenario}_{varname_EM}.csv"), sep=";", index=False)

    #city	year	scenario	variable	value
    cities = df_emissions["city"].unique()
    colours = cm.tab10(np.linspace(0, 1, len(cities)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for city, colour in zip(cities, colours):
        df_city = df_emissions[(df_emissions["city"] == city) &(df_emissions["year"].isin(years_plot))]
        sum_weighted_vals = df_city[df_city["variable"] == "sum_weighted"].set_index("year")["value"].reindex(years_plot)
        mean_per_m2_vals  = df_city[df_city["variable"] == "mean_per_m2"].set_index("year")["value"].reindex(years_plot)
        ax1.plot(years_plot, sum_weighted_vals, marker="o", color=colour)
        ax1.annotate(city, xy=(2020, sum_weighted_vals[2020]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=colour, va="center")
        ax2.plot(years_plot, mean_per_m2_vals, marker="o", color=colour)
        ax2.annotate(city, xy=(2020, mean_per_m2_vals[2020]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=colour, va="center")

    ax1.set_title("Total weighted emissions")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("sum_weighted")
    ax1.set_xticks(years_plot)

    ax2.set_title("Mean emissions per m²")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("mean_per_m2")
    ax2.set_xticks(years_plot)

    fig.tight_layout()
    plt.savefig(Path(f"{figures_dir}/city_emissions_{scenario}.png"), dpi=150, bbox_inches="tight")

def plot_results_urban_emissions(model: str, scenario: str):

    project_dir = Path(__file__).parent
    print(f"Project directory: {project_dir}")

    # script settings
    import downscaling.settings_downscaling as settings
    from downscaling.settings_downscaling import SOURCE_PROFILES

    df_urban_emissions = pd.DataFrame()
    for profile in SOURCE_PROFILES.keys():
        sources = settings.SOURCE_PROFILES[profile]
        source_POP = sources["source_POP"]
        version_POP = sources["version_POP"]
        source_GDP = sources["source_GDP"]
        version_GDP = sources["version_GDP"]
        source_EM = sources["source_EM"]
        version_EM = sources["version_EM"]

        source_version_grid = f"{source_POP}_{version_POP}_{source_GDP}_{version_GDP}_{source_EM}_{version_EM}"
        model_scenario = f"{model}_{scenario}"
        dir_processed = project_dir / "data" / "processed" / source_version_grid / model_scenario
        print(f"Processed data directory: {dir_processed}")
        if dir_processed.exists():
            csv_file = dir_processed / f"df_urban_emissions_{model}_{scenario}.csv"
            if csv_file.exists():
                df_urban_emissions = pd.read_csv(dir_processed / f"df_urban_emissions_{model}_{scenario}.csv", sep=";")
                df_urban_emissions["profile"] = profile
                df_urban_emissions = pd.concat([df_urban_emissions, df_urban_emissions], axis=0)
        else:
            print(f"{PRINT_COLORS['red']}Processed data directory {dir_processed} does not exist. Please run the downscaling process first to generate the data.{PRINT_COLORS['end']}")

    df_urban_emissions.to_csv(project_dir / "data" / "processed" / f"df_urban_emissions_{model}_{scenario}_all_profiles.csv", sep=";", index=False)

    plot_maps.plot_urban_emissions_per_region(project_dir, df_urban_emissions)

def upload_to_GEE(scenario:str = "ELV-SSP2-CP", model:str="IMAGE", profile:str = "default"):

    #GCP_PROJECT = "phrasal-brand-469215-b2"
    GCP_PROJECT = "unique-nebula-467816-n2"
    #EE_ASSET_FOLDER = "projects/phrasal-brand-469215-b2/assets"
    EE_ASSET_FOLDER = "projects/unique-nebula-467816-n2/assets"

    start_time = time.time()
    # script settings
    import downscaling.settings_downscaling as settings
    from downscaling.settings_downscaling import SOURCE_PROFILES

    if profile not in settings.SOURCE_PROFILES:
        available = list(settings.SOURCE_PROFILES.keys())
        raise ValueError(f"Unknown source profile '{profile}'. Available: {available}")
    else:
        sources = settings.SOURCE_PROFILES[profile]

    source_POP = sources["source_POP"]
    version_POP = sources["version_POP"]
    source_GDP = sources["source_GDP"]
    version_GDP = sources["version_GDP"]
    source_EM = sources["source_EM"]
    version_EM = sources["version_EM"]

    varname_GDP = settings.varname_GDP
    varname_POP = settings.varname_POP
    varname_EM = settings.varname_EM

    SSP_base = settings.SSP_base

    file_model_grid_regions = settings_models.models[model]["file_model_grid_regions"]
    file_IAM_model_region_numbers = settings_models.models[model]["file_IAM_model_region_numbers"]

    project_dir = Path(__file__).parent
    print(f"Project directory: {project_dir}")
    source_version_grid = f"{source_POP}_{version_POP}_{source_GDP}_{version_GDP}_{source_EM}_{version_EM}"
    model_scenario = f"{model}_{scenario}"
    dir_output = project_dir / "data" / "output" / source_version_grid / model_scenario
    dir_processed = project_dir / "data" / "processed" / source_version_grid / model_scenario
    print(f"Output directory: {dir_output}")
    print(f"Processed data directory: {dir_processed}")

    upload_results_ee.ensure_ee_authenticated()
    upload_results_ee.upload_years(scenario, source_version_grid, "all", dir_processed, [2020, 2030, 2050], EE_ASSET_FOLDER)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n{PRINT_COLORS['green']}Upload to Google Earth Engine complete. Total elapsed time: {elapsed_time:,.2f} seconds ({elapsed_time/60:.2f} minutes).{PRINT_COLORS['end']}")

def compare_two_raster_files():

    project_dir = Path(__file__).parent
    root = tk.Tk()
    root.withdraw()  # Hide the root window, only show the dialogs

    messagebox.showinfo("File Selection", "Please select the first raster file.")
    file_path_1 = filedialog.askopenfilename(
        title="Select the first raster file",
        filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
    )

    if not file_path_1:
        print("No file selected for the first raster. Exiting.")
        return

    messagebox.showinfo("File Selection", "Please select the second raster file.")
    file_path_2 = filedialog.askopenfilename(
        title="Select the second raster file",
        filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
    )

    if not file_path_2:
        print("No file selected for the second raster. Exiting.")
        return

    print(f"Comparing '{file_path_1}' with '{file_path_2}'...")

    with rasterio.open(file_path_1) as src1, rasterio.open(file_path_2) as src2:
        process_grid_data.compare_two_raster_files(project_dir, src1, src2)
