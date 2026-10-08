import atexit
import time
import logging
import warnings
import json
from itertools import chain

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

import geopandas as gpd
import rasterio
import xarray as xr
import rioxarray as rxr
import cartopy.crs as ccrs
from geocube.api.core import make_geocube

from tools.functions_logging import init_logging
from tools.general_functions import PRINT_COLORS, replace_punctuation_in_filenames, round_to_half, is_int_or_half, format_factor, apply_root_json
from tools import convert_GIS

import downscaling.plot_maps as plot_maps
import downscaling.read_process_grid_data as process_grid_data
import downscaling.read_process_IAM_data as process_IAM_data
import downscaling.process_IPAT_factors as process_IPAT_factors
import downscaling.process_urban_grid_emissions as process_urban_grid_emissions

import downscaling.settings_models as settings_models
import downscaling.settings_downscaling as settings_downscaling
from downscaling.settings_downscaling import SOURCE_PROFILES
from downscaling.settings_downscaling_cities import coord_Amsterdam, coord_Lima, coord_Raleigh, coord_NewYork
import downscaling.upload_results_ee as upload_results_ee

project_dir = Path(__file__).parent.parent
log_path = f"{project_dir}/log/downscaling"
debug_tmp_log, _ = init_logging(f"log_tmp", log_path)

def cleanup_empty_logs(log_path):

    # Close all file handlers so Windows releases the file locks
    for logger_name in ["debug", "results", "py.warnings"]:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    # delete all empty .log files in the log directory and one level below
    log_dir = Path(log_path)
    for log_file in chain(log_dir.glob("*.log"), log_dir.glob("*/*.log")):
        if log_file.stat().st_size == 0:
            print(f"Deleting empty log file: {log_file}")
            log_file.unlink()

def process_one_dataset(project_dir:Path, driver:str="Population", source:str="2UP", version:str="GHSL_2024_M3", SSP_base:str="SSP2"):
    # driver = "Population", "GDP|PPP", "Emissions"
    log_path = f"{project_dir}/log/reading_processing_data"
    debug_log, results_log = init_logging(f"read_process_data_{driver.replace("|", "_")}_{source}_{version}_{SSP_base}", log_path)

    if driver not in ["Population", "GDP|PPP", "Emissions"]:
        raise ValueError(f"Unknown type '{driver}'. Supported types: 'Population', 'GDP|PPP', 'Emissions_CO2_Excl_shipping_aviation_AFOLU'.")

    # set variable naems
    match driver:
        case "Population":
            varname = settings_downscaling.varname_POP
        case "GDP|PPP":
            varname = settings_downscaling.varname_GDP
        case "Emissions":
            varname = settings.varname_EM
        case _:
            raise ValueError(f"Unknown type '{driver}'. Supported types: 'Population', 'GDP|PPP', 'Emissions_CO2_Excl_shipping_aviation_AFOLU'.")

    debug_log.info(f"{PRINT_COLORS["green"]}Data sources to process for type: {driver}, source: {source}, version: {version}, and SSP baseline {SSP_base}{PRINT_COLORS["end"]}")

    debug_log.info("-----------------------------")
    debug_log.info("Processing datasets...")

    debug_log.info(f"{PRINT_COLORS["yellow"]}-----------------------------{PRINT_COLORS["end"]}")


    if driver in ["Population", "GDP|PPP"]:
        debug_log.info(f"{PRINT_COLORS["yellow"]}Processing data for variable: {varname}, source: {source}, version: {version}, and SSP baseline: {SSP_base}{PRINT_COLORS["end"]}")
        if driver in ["Population", "GDP|PPP"]:
            process_grid_data.pre_process_data_socioeconomic(varname=varname,
                                                                source=source,
                                                                version=version,
                                                                SSP_base=SSP_base,
                                                                base_year=settings_downscaling.base_year,
                                                                log=debug_log)
    elif driver == "Emissions":
        process_grid_data.pre_process_data_emissions(varname=varname,
                                                     source=source,
                                                     version=version,
                                                     log=debug_log)
    else:
        debug_log.info(f"{PRINT_COLORS["red"]}Variable {varname} not recognized for processing.{PRINT_COLORS["end"]}")

    cleanup_empty_logs(log_path)

def process_datasets(project_dir:Path, profile:str, SSP_base="SSP2"):

    log_path = f"{project_dir}/log/reading_processing_data"
    debug_log, results_log = init_logging(f"read_process_data_{profile}_{SSP_base}", log_path)

    # set variable naems
    varname_POP = settings_downscaling.varname_POP
    varname_GDP = settings_downscaling.varname_GDP
    varname_EM = settings_downscaling.varname_EM

    # read profile
    if profile not in settings_downscaling.SOURCE_PROFILES:
        available = list(settings_downscaling.SOURCE_PROFILES.keys())
        raise ValueError(f"Unknown source profile '{profile}'. Available: {available}")
    else:
        sources = settings_downscaling.SOURCE_PROFILES[profile]

    source_POP = sources["source_POP"]
    version_POP = sources["version_POP"]
    source_GDP = sources["source_GDP"]
    version_GDP = sources["version_GDP"]
    source_EM = sources["source_EM"]
    version_EM = sources["version_EM"]

    data_source_population = {"varname": varname_POP, "source": source_POP, "version": version_POP}
    data_source_gdp_ppp = {"varname": varname_GDP, "source": source_GDP, "version": version_GDP}
    data_source_emissions = {"varname": varname_EM, "source": source_EM, "version": version_EM}
    # combine data sources dicts into list
    data_sources = [data_source_population, data_source_gdp_ppp, data_source_emissions]
    debug_log.info(f"{PRINT_COLORS["green"]}Data sources to process for profile {profile} and SSP baseline {SSP_base}: {data_sources}{PRINT_COLORS["end"]}")

    debug_log.info("-----------------------------")
    debug_log.info("Processing datasets...")

    debug_log.info(data_sources)
    debug_log.info(f"{PRINT_COLORS["yellow"]}-----------------------------{PRINT_COLORS["end"]}")

    for data_source in data_sources:
        debug_log.info(data_source)
        debug_log.info(f"{PRINT_COLORS["yellow"]}Processing data for variable: {data_source["varname"]}, source: {data_source["source"]}, version: {data_source["version"]}, and SSP baseline: {SSP_base}{PRINT_COLORS["end"]}")
        if data_source["varname"] in ["Population", "GDP|PPP"]:
            process_grid_data.pre_process_data_socioeconomic(varname=data_source["varname"],
                                        source=data_source["source"],
                                        version=data_source["version"],
                                        SSP_base=SSP_base,
                                        base_year=settings_downscaling.base_year,
                                        log=debug_log)
        elif data_source["varname"] == "Emissions_CO2_Excl_shipping_aviation_AFOLU":
            process_grid_data.pre_process_data_emissions(varname=data_source["varname"],
                                        source=data_source["source"],
                                        version=data_source["version"],
                                        log=debug_log)
        else:
            debug_log.info(f"{PRINT_COLORS["red"]}Variable {data_source["varname"]} not recognized for processing.{PRINT_COLORS["end"]}")

    cleanup_empty_logs(log_path)

def determine_regions_file(project_dir:Path,
                           res_min_POP:int|float|None, res_min_GDP:int|float|None, res_min_EM:int|float|None,
                           model:str, log:logging.Logger) -> Path:

    '''
    Determine the appropriate model grid regions file based on the lowest resolution among the datasets (POP, GDP, EM).
    '''

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

    # determine the appropriate model grid regions file based on the lowest resolution
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

def reindex_and_interp(group:pd.DataFrame, id_cols:list[str], years_downscaling:list[int]) -> pd.DataFrame:
    '''
    Reindex the group DataFrame to include all years in years_downscaling and interpolate missing values.
    '''
    # keys are the values of the id_cols for this group
    keys = dict(zip(id_cols, group.name if isinstance(group.name, tuple) else (group.name,)))

    return (
        group.set_index("year")
            .reindex(years_downscaling)
            .assign(**keys)
            .assign(value=lambda g: g["value"].interpolate("linear", limit_area="inside"))
            .reset_index())

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def plot_floor_comparison(da, title, floor=1e3, n_bins=60, save_path=None):
    """Two log-scale histograms of one 2D slice: left as-is, right with values <= floor removed.
    Pass e.g. xr_gdp_ppp_processed[varname_GDP].sel(time=base_year)."""
    values = np.asarray(da.squeeze().values, dtype="float64").ravel()
    pos = values[np.isfinite(values) & (values > 0)]
    floored = pos[pos > floor]

    bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), n_bins)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    ax_left.hist(pos, bins=bins, color="#4c72b0")
    ax_left.axvline(floor, color="red", linestyle="--", label=f"floor = {floor:g}")
    ax_left.set_title(f"{title}: no floor ({pos.size:,} cells)")
    ax_left.legend()
    ax_right.hist(floored, bins=bins, color="#55a868")
    ax_right.axvline(floor, color="red", linestyle="--")
    ax_right.set_title(f"{title}: floor applied ({floored.size:,} cells)")
    for ax in (ax_left, ax_right):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("value per cell")
    ax_left.set_ylabel("number of cells")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

def downscale_SE_data(project_dir:Path, variable_SE: str, scenario: str, model: str = "IMAGE", align_resolution: bool = True, profile: str = "default", SSP_base:str="SSP2"):
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

    if profile not in settings_downscaling.SOURCE_PROFILES:
        available = list(settings_downscaling.SOURCE_PROFILES.keys())
        raise ValueError(f"Unknown source profile '{profile}'. Available: {available}")
    else:
        sources = settings_downscaling.SOURCE_PROFILES[profile]

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

    base_year = settings_downscaling.base_year
    years_downscaling = settings_downscaling.years_downscaling
    #convergence_year = settings_downscaling.convergence_year
    method_extension = settings_downscaling.method_extension
    #vars_downscaling = settings_downscaling.vars_downscaling
    process_flags = settings_downscaling.process_flags
    check_flags = settings_downscaling.check_flags

    vars_downscaling = settings_models.models[model]["vars_downscaling"]

    varname_GDP = settings_downscaling.varname_GDP
    varname_POP = settings_downscaling.varname_POP
    varname_EM = settings_downscaling.varname_EM
    varname_gdp_per_capita = settings_downscaling.varname_gdp_per_capita
    varname_em_per_gdp_ppp = settings_downscaling.varname_em_per_gdp_ppp

    unit_POP = settings_downscaling.unit_POP
    unit_GDP = settings_downscaling.unit_GDP_PPP
    unit_EM = settings_downscaling.unit_EM

    #SSP_base = settings_downscaling.SSP_base

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
    # if align_resolution:
    #     source_version_grid = f"{source_POP}_{version_POP}_{source_GDP}_{version_GDP}_{source_EM}_{version_EM}"
    # else:
    #     source_version_grid = f"{source_SE}_{version_SE}"
    model_scenario = f"{model}_{scenario}"
    dir_output = project_dir / "data" / "output" / profile / model_scenario
    dir_processed = project_dir / "data" / "processed" / profile / model_scenario
    print(f"Project directory: {project_dir}")
    print(f"Output directory: {dir_output}")
    print(f"Processed data directory: {dir_processed}")
    dir_output.mkdir(parents=True, exist_ok=True)
    dir_processed.mkdir(parents=True, exist_ok=True)

    log_path = f"{project_dir}/log/downscaling"
    debug_log, results_log = init_logging(f"log_downscaling_se_{profile}_{model_scenario}", log_path)
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

def _intermediate_downscale_POP(varname: str, xr_POP: xr.Dataset, df_POP: pd.DataFrame, xr_IAM_grid: xr.Dataset, unit_conversion: float, save_path: Path, log: logging.Logger) -> xr.Dataset:
    '''
    Downscale gridded population data to match IAM regional projections.
    This function is intended for internal use within the downscale_SE_data function.
    '''
    start_time_downascaling = time.time()
    log.info(f"\nStarting intermediate downscaling of {varname} data...")

    # Safety checks
    assert xr_IAM_grid["region_number"].shape == xr_POP[varname].isel(time=0).shape, "Grid and population rasters differ in shape - cannot snap coordinates"
    xr_IAM_grid = xr_IAM_grid.assign_coords(x=xr_POP["x"], y=xr_POP["y"])
    assert not df_POP.duplicated(["year", "region_number"]).any(), "Still multiple rows per (year, region_number) - filter further"

    # pre-check:
    crs_in = xr_POP.rio.crs
    log.info(f"{PRINT_COLORS["yellow"]}Input CRS for {varname}: {crs_in}{PRINT_COLORS["end"]}")

    # add region to xr_POP for grouping
    xr_POP_region = xr_POP.copy()
    xr_POP_region["region_number"] = xr_IAM_grid["region_number"]

    xr_POP_region_total = (xr_POP_region[varname]
                            .groupby(xr_POP_region["region_number"])
                            .sum())
    xr_POP_region_outside = (xr_POP_region[varname]
                            .where(xr_POP_region["region_number"].isnull())
                            .sum())
    df_POP_grid = xr_POP_region_total.to_dataframe().reset_index()
    df_POP_grid.rename(columns={"time": "year", varname: f"{varname}_grid"}, inplace=True)
    df_POP_grid.drop(columns=["spatial_ref"], inplace=True, errors="ignore")
    df_POP_model = df_POP.copy()
    df_POP_model.rename(columns={"value": f"{varname}_IAM"}, inplace=True)
    df_POP_model[f"{varname}_IAM"] = unit_conversion * df_POP_model[f"{varname}_IAM"]
    df_ratio = pd.merge(df_POP_grid, df_POP_model, how="left", on=["year", "region_number"], suffixes=("_grid", "_IAM"))
    df_ratio["ratio_IAM_to_grid"] = df_ratio[f"{varname}_IAM"] / df_ratio[f"{varname}_grid"]
    df_ratio = df_ratio[["year", "region_number", f"{varname}_grid", f"{varname}_IAM", 'ratio_IAM_to_grid', "model", "scenario", "variable"]]
    df_ratio.to_csv(save_path / f"compare_{varname.replace('|', '_')}_regional_sums_grid.csv", sep=";", index=False)

    # Add ratios to xarray -> dims (time, region_number)
    ratio = (df_ratio
              .set_index(["year", "region_number"])["ratio_IAM_to_grid"]
              .to_xarray()
              .rename({"year": "time"}))

    # Align to the grid's region/time set; non-finite (missing / 0-division) -> factor 1
    grid_regions = np.unique(xr_IAM_grid["region_number"].values)
    ratio = ratio.reindex(region_number=grid_regions, time=xr_POP_region_total["time"], fill_value=1.0)
    ratio = ratio.where(np.isfinite(ratio), 1.0)
    ratio_grid = ratio.sel(region_number=xr_IAM_grid["region_number"])

    #xr_POP_out = xr_POP.assign({varname: xr_POP[varname] * ratio_grid})
    with xr.set_options(keep_attrs=True):
        xr_POP_out = xr_POP.assign({varname: xr_POP[varname] * ratio_grid})
    if crs_in is not None:
        xr_POP_out = xr_POP_out.rio.write_crs(crs_in, inplace=False)


    log.info(f"Time taken for downscaling {varname}: {(time.time()-start_time_downascaling)/60:,.1f} minutes")

    return xr_POP_out

def _check_same_grid(da_ref: xr.DataArray, da_other: xr.DataArray, name: str) -> None:
    """Raise if da_other is not on the x/y grid of da_ref (sizes equal, coordinates within half a cell)."""
    for dim in ["x", "y"]:
        if da_ref.sizes[dim] != da_other.sizes[dim]:
            raise ValueError(f"{name}: size of '{dim}' ({da_other.sizes[dim]}) differs from reference ({da_ref.sizes[dim]}), reindex first")
        half_cell = float(np.abs(da_ref[dim].diff(dim)).median()) / 2
        if not np.allclose(da_ref[dim].values, da_other[dim].values, rtol=0, atol=half_cell):
            raise ValueError(f"{name}: '{dim}' coordinates differ by more than half a cell from reference "
                             "(offset or reversed order), reindex first")

def _intermediate_downscale_GDPpc(varname: str, xr_SE_GDPpc: xr.Dataset, df_SE_GDPpc: pd.DataFrame,
                                  xr_IAM_grid: xr.Dataset, xr_weights: xr.Dataset, weight_varname: str,
                                  unit_conversion: float, save_path: Path, log: logging.Logger) -> xr.Dataset:
    '''
    Downscale a gridded INTENSIVE (per-capita) variable to match IAM regional
    projections, preserving the population-weighted regional average.
    Intended for internal use within downscale_SE_GDPpc_data.
    df_SE_GDPpc is :"year"	integer years (Same dtype as xr_SE_GDPpc["time"])
                    "region_number"	Numeric (same as xr_IAM_grid["region_number"])
                    Column with name varname (identical to the data variable name in xr_SE_GDPpc)
    '''
    start_time_downascaling = time.time()
    log.info(f"\nStarting intermediate downscaling of {varname} data (population-weighted)...")

    gdppc = xr_SE_GDPpc[varname]
    # grids must match before the coordinates are snapped
    _check_same_grid(gdppc, xr_IAM_grid["region_number"], "IAM region grid")
    _check_same_grid(gdppc, xr_weights[weight_varname], "weights")
    # NaN region numbers cannot be selected later with ratio.sel(region_number=region_da)
    if xr_IAM_grid["region_number"].isnull().any():
        raise ValueError("IAM region grid contains NaN, fill nodata with a region number first")
    # avoid missing years would silently disappear from the output (inner join on time)
    years_missing = np.setdiff1d(gdppc["time"].values, xr_weights["time"].values)
    if years_missing.size > 0:
        raise ValueError(f"Weights are missing years of {varname}: {years_missing.tolist()}")
    if df_SE_GDPpc.duplicated(["year", "region_number"]).any():
        raise ValueError("Still multiple rows per (year, region_number) - filter further")

    # snap coordinates (only removes floating-point differences, checked above)
    xr_IAM_grid = xr_IAM_grid.assign_coords(x=gdppc["x"], y=gdppc["y"])
    weights = xr_weights[weight_varname].assign_coords(x=gdppc["x"], y=gdppc["y"])
    weights = weights.where(gdppc.notnull())

    crs_in = xr_SE_GDPpc.rio.crs
    log.info(f"{PRINT_COLORS["yellow"]}Input CRS for {varname}: {crs_in}{PRINT_COLORS["end"]}")

    # add region for grouping
    region_da = xr_IAM_grid["region_number"]

    # Population-weighted regional mean of the per-capita grid, per year:
    #   Sum(gdp_pc * pop) / Sum(pop)   -> dims (time, region_number)
    weighted_num = (xr_SE_GDPpc[varname] * weights).groupby(region_da).sum()
    weight_den = weights.groupby(region_da).sum()
    grid_weighted_mean = weighted_num / weight_den.where(weight_den > 0)

    df_SE_GDPpc_grid = grid_weighted_mean.to_dataframe(name=f"{varname}").reset_index()
    df_SE_GDPpc_grid.rename(columns={"time": "year"}, inplace=True)
    df_SE_GDPpc_grid.drop(columns=["spatial_ref"], inplace=True, errors="ignore")

    df_SE_GDPpc_model = df_SE_GDPpc.copy()
    #df_SE_GDPpc_model.rename(columns={varname: f"{varname}_IAM"}, inplace=True)
    df_SE_GDPpc_model[varname] = unit_conversion * df_SE_GDPpc_model[varname]

    df_ratio = pd.merge(df_SE_GDPpc_grid, df_SE_GDPpc_model, how="left", on=["year", "region_number"], suffixes=("_grid", "_IAM"))
    df_ratio["ratio_IAM_to_grid"] = df_ratio[f"{varname}_IAM"] / df_ratio[f"{varname}_grid"]
    df_ratio = df_ratio[["year", "region_number", f"{varname}_grid", f"{varname}_IAM", "ratio_IAM_to_grid", "model", "scenario"]]
    df_ratio.to_csv(save_path / f"compare_{varname.replace('|', '_')}_regional_means.csv", sep=";", index=False)

    # ratios back to xarray -> dims (time, region_number)
    ratio = (df_ratio
              .set_index(["year", "region_number"])["ratio_IAM_to_grid"]
              .to_xarray()
              .rename({"year": "time"}))

    grid_regions = np.unique(region_da.values)
    ratio = ratio.reindex(region_number=grid_regions, time=grid_weighted_mean["time"], fill_value=1.0)
    ratio = ratio.where(np.isfinite(ratio), 1.0)
    ratio_grid = ratio.sel(region_number=region_da)

    with xr.set_options(keep_attrs=True):
        xr_SE_GDPpc_out = xr_SE_GDPpc.assign({varname: xr_SE_GDPpc[varname] * ratio_grid})
    if crs_in is not None:
        xr_SE_GDPpc_out = xr_SE_GDPpc_out.rio.write_crs(crs_in, inplace=False)

    log.info(f"Time taken for downscaling {varname}: {(time.time()-start_time_downascaling)/60:,.1f} minutes")

    return xr_SE_GDPpc_out

def _compare_grid_IAM_global(xr_grid_before: xr.Dataset, xr_grid_after: xr.Dataset, df_IAM: pd.DataFrame, varname: str, log: logging.Logger) -> pd.DataFrame:
    log.info(f"Calculate global sums of Population before and after downscaling for comparison...")
    # 2. Calculate global sums of Population before and after downscaling for comparison
    df_pop_grid_global_before = xr_grid_before[varname].sum(dim=["y", "x"]).to_dataframe().reset_index()
    df_pop_grid_global_before.drop(columns=["spatial_ref"], inplace=True, errors="ignore")
    df_pop_grid_global_after = xr_grid_after[varname].sum(dim=["y", "x"]).to_dataframe().reset_index()
    df_pop_grid_global_after.drop(columns=["spatial_ref"], inplace=True, errors="ignore")
    df_pop_grid_global = pd.merge(df_pop_grid_global_before, df_pop_grid_global_after, on="time", suffixes=("_before", "_after"))
    df_pop_grid_global.rename(columns={"time": "year",f"{varname}_before": f"grid_before", f"{varname}_after": f"grid_after"}, inplace=True)

    # 3. Compare grid and IAM for global population sums
    df_IAM_population_global = df_IAM.groupby(["year"], as_index=False)["value"].sum().reset_index()
    df_IAM_population_global.rename(columns={"value": "IAM"}, inplace=True)

    df_pop_compare_global = pd.merge(df_pop_grid_global, df_IAM_population_global[["year", "IAM"]], on="year", how="left")
    df_pop_compare_global["variable"] = varname

    return df_pop_compare_global

def _compare_ratio_grid_IAM(xr_denom: xr.Dataset, xr_ratio_before: xr.Dataset, xr_ratio_after: xr.Dataset,
                            df_IAM_ratio: pd.DataFrame,
                            varname_nom: str, varname_denom: str, varname_ratio: str,
                            conversion_factor_gdp_per_capita: float) -> pd.DataFrame:
    # 2. Compare global sums of gridded population, after and from IAM model for comparison
    df_grid_ratio_before = process_IPAT_factors.calculate_gdp_per_capita_global(xr_denom, varname_denom, xr_ratio_before, varname_ratio)
    df_grid_ratio_after = process_IPAT_factors.calculate_gdp_per_capita_global(xr_denom, varname_denom, xr_ratio_after, varname_ratio)
    df_grid_gdp_ppp_global = pd.merge(df_grid_ratio_before, df_grid_ratio_after, on="time", suffixes=("_before", "_after"))
    df_grid_gdp_ppp_global.rename(columns={"time": "year",f"{varname_ratio}_before": f"grid_before", f"{varname_ratio}_after": f"grid_after"}, inplace=True)
    # Calculate global gdp per capita from IAM data
    df_IAM_ratio_global = (df_IAM_ratio
                            .groupby("year", as_index=False)[[varname_nom, varname_denom]]
                            .sum())
    df_IAM_ratio_global[varname_ratio] = df_IAM_ratio_global[varname_nom] / df_IAM_ratio_global[varname_denom]
    df_IAM_ratio_global[varname_ratio] *= conversion_factor_gdp_per_capita
    df_IAM_ratio_global.drop([varname_nom, varname_denom], axis=1, inplace=True)
    df_IAM_ratio_global.rename(columns={varname_ratio: "IAM"}, inplace=True)
    # Merge grid and IAM data for comparison
    df_grid_gdp_ppp_global.drop(columns=["spatial_ref"], inplace=True, errors="ignore")
    df_compare_gdp_per_global = pd.merge(df_grid_gdp_ppp_global, df_IAM_ratio_global, on="year", how="left")
    df_compare_gdp_per_global["Variable"] = varname_ratio

    return df_compare_gdp_per_global

def downscale_emissions(project_dir:Path, scenario:str, model:str="IMAGE", profile:str="default", SSP_base:str="SSP2", convergence_year:int=2150, net_emissions:bool=True, downscale_SE:bool=True):
    # pre:
    # - population, GDP, and emissions gridded data must be pre-processed and available for the given sources and versions (run main.py -- process_grid_data)
    # - GADM region raster must be created for the given model and resolution (run main.py -- create_GAMD_region_raster)
    # - urban/region classification data must be available (run main.py -- process_urban_classification)

    # CO2 = POP x (GDP/POP) x (EM/GDP)

    # Make sure program stops (and not only give a warning) if divided by zero, invalid value, or overvflow
    # Set up logging and warnings
    warnings.filterwarnings("error", message="divide by zero", category=RuntimeWarning)
    warnings.filterwarnings("error", message="invalid value", category=RuntimeWarning)
    warnings.filterwarnings("error", message="overflow", category=RuntimeWarning)

    # start timing
    start_time = time.time()

    # read profile
    if profile not in settings_downscaling.SOURCE_PROFILES:
        available = list(settings_downscaling.SOURCE_PROFILES.keys())
        raise ValueError(f"Unknown source profile '{profile}'. Available: {available}")
    else:
        sources = settings_downscaling.SOURCE_PROFILES[profile]

    # settings
    #coarse_factor_SE:int = 12 # 12 (2UP)
    #coarse_factor_GDP:float = 1.2 # 12 (Wang), 1.2 (Murakam version_2021_1)
    #coarse_factor_EM:int =  1 # EDGAR
    coarse_factor_POP, coarse_factor_GDP, coarse_factor_EM, \
    res_min_POP, res_min_GDP, res_min_EM = process_grid_data.get_coarsening_factors(population_source=sources["source_POP"],
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

    base_year = settings_downscaling.base_year
    years_downscaling = settings_downscaling.years_downscaling
    #convergence_year = settings_downscaling.convergence_year
    method_extension = settings_downscaling.method_extension
    #vars_downscaling = settings_downscaling.vars_downscaling
    process_flags = settings_downscaling.process_flags
    check_flags   = settings_downscaling.check_flags

    varname_GDP = settings_downscaling.varname_GDP
    varname_POP = settings_downscaling.varname_POP
    varname_EM = settings_downscaling.varname_EM
    varname_gdp_per_capita = settings_downscaling.varname_gdp_per_capita
    varname_em_per_gdp_ppp = settings_downscaling.varname_em_per_gdp_ppp

    vars_downscaling = settings_models.models[model]["vars_downscaling"]

    unit_GDP_PPP = settings_downscaling.unit_GDP_PPP
    unit_POP = settings_downscaling.unit_POP
    unit_EM = settings_downscaling.unit_EM

    # create output and processed directories
    print(f"Project directory: {project_dir}")
    gross_net = "net" if net_emissions else "gross"
    downscale_se_str = "downscale_SE" if downscale_SE else "no_downscale_SE"
    model_scenario_convergence_year = f"{model}_{scenario}_conv_year_{convergence_year}_{gross_net}"
    profile_model_scenario_convergence_year = f"{profile}_{model_scenario_convergence_year}"
    run_tag = f"{profile}_{downscale_se_str}_{model_scenario_convergence_year}"
    dir_processed = project_dir / "data" / "processed" / f"{profile}_{downscale_se_str}" / model_scenario_convergence_year
    dir_processed.mkdir(parents=True, exist_ok=True)
    dir_check = dir_processed / "check"
    dir_check.mkdir(parents=True, exist_ok=True)
    print(f"Processed data directory: {dir_processed}")

    dir_output = project_dir / "data" / "output"
    dir_output.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {dir_output}")

    dir_urban = project_dir / "data" / "processed" / "DLL"
    dir_tiff_plots = dir_processed.parent / "tiff"
    dir_tiff_plots.mkdir(parents=True, exist_ok=True)
    log_path = dir_processed / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    debug_log, results_log = init_logging(f"downscaling_emissions_{profile_model_scenario_convergence_year}", str(log_path))
    debug_log.info(f"\n\n{PRINT_COLORS['purple']}{'|'*100}{PRINT_COLORS['end']}")
    debug_log.info(f"{PRINT_COLORS['purple']}Logging for profile {profile}, scenario {scenario}, model {model}, convergence year {convergence_year}, emissions {gross_net}, downscale_SE {downscale_SE} started{PRINT_COLORS['end']}")
    debug_log.info(f"{PRINT_COLORS['purple']}{SOURCE_PROFILES[profile]}{PRINT_COLORS['end']}")

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

    # Read in GADM raster file based on used resolution (e.g. IMAGE_GADM_regions_raster_6_00_arcmin.nc), and retrieve the region numbers of the IAM model
    match profile:
        case "test_run_POP":
            file_path_file_model_grid_regions = Path(f"{project_dir}/data/processed/GADM/IMAGE_ScenarioMIP_GADM_regions_raster_6_00_base_run_POP_2UP_GHSL_2024_M3_arcmin_base_run_POP_2UP_GHSL_2024_M3.nc")
        case "test_run_GDP":
            file_path_file_model_grid_regions = Path(f"{project_dir}/data/processed/GADM/IMAGE_ScenarioMIP_GADM_regions_raster_6_00_base_run_GDP_Murakami_version_2021_1_arcmin_base_run_GDP_Murakami_version_2021_1.nc")
        case "test_run_EM":
            file_path_file_model_grid_regions = Path(f"{project_dir}/data/processed/GADM/IMAGE_ScenarioMIP_GADM_regions_raster_6_00_base_run_EM_EDGAR_2024_arcmin_base_run_EM_EDGAR_2024.nc")
        case _:
            #file_path_file_model_grid_regions = determine_regions_file(project_dir, res_min_POP, res_min_GDP, res_min_EM, model, debug_log)
            debug_log.info(f"Determine GADM raster file based on POP, GDP, and EM resolutions, and retrieve the region numbers of the IAM model")
            file_path_file_model_grid_regions = Path(f"{project_dir}/data/processed/GADM/IMAGE_ScenarioMIP_GADM_regions_raster_6_00_arcmin_base_run_POP_GDP_EM.nc")
    file_IAM_model_region_numbers = settings_models.models[model]["file_IAM_model_region_numbers"]

    # output files in parent of processed directory (scenario independent)
    pop_file = dir_processed.parent / f"Population_{source_POP}_{version_POP}_{SSP_base}_cf_{coarse_factor_POP_str}.nc"
    gdp_ppp_file = dir_processed.parent / f"GDP_PPP_{source_GDP}_{version_GDP}_{SSP_base}_cf_{coarse_factor_GDP_str}.nc"
    em_file = dir_processed.parent / f"{replace_punctuation_in_filenames(varname_EM)}_hist_{source_EM}_{version_EM}_{SSP_base}_cf_{coarse_factor_EM_str}.nc"

    # output files in processed directory (scenario dependent)
    pop_downscaled_file = dir_processed / f"Population_downscaled_{source_POP}_{version_POP}_{SSP_base}_cf_{coarse_factor_POP_str}.nc"
    pop_processed_file = dir_processed / f"Population_processed_{source_POP}_{version_POP}_{SSP_base}_cf_{coarse_factor_POP_str}.nc"
    gdp_per_capita_downscaled_file = dir_processed / f"GDP_per_capita_downscaled_{source_GDP}_{version_GDP}_{source_POP}_{version_POP}_{SSP_base}_cf_{coarse_factor_GDP_str}.nc"
    gdp_ppp_processed_file = dir_processed / f"GDP_PPP_processed_{source_GDP}_{version_GDP}_{SSP_base}_cf_{coarse_factor_GDP_str}.nc"
    gdp_ppp_per_capita_file = dir_processed / f"GDP_PPP_per_capita_{source_GDP}_{version_GDP}_{source_POP}_{version_POP}_{SSP_base}.nc"

    # output files in processed scenarios directories
    iam_input_file = dir_processed / f"IAM_{model}_{scenario}_processed.csv"
    iam_gdp_per_capita_file = dir_processed / f"IAM_{model}_{scenario}_projection_gdp_per_capita.csv"
    iam_em_per_gdp_ppp_file = dir_processed / f"IAM_{model}_{scenario}_em_per_gdp_ppp.csv"
    em_per_gdp_ppp_file = dir_processed / f"{replace_punctuation_in_filenames(varname_EM)}_per_gdp_ppp_{source_EM}_{version_EM}_{source_GDP}_{version_GDP}_{SSP_base}.nc"

    csv_file_compare_unharmonised = dir_output / f"Emissions_region_{run_tag}_unharmonised.csv"
    csv_file_compare_harmonised = dir_output / f"Emissions_region_{run_tag}_harmonised.csv"
    em_unharmonised_file = dir_processed / f"{replace_punctuation_in_filenames(varname_EM)}_unharmonised_{SSP_base}.nc"
    em_harmonised_file = dir_processed / f"{replace_punctuation_in_filenames(varname_EM)}_harmonised_{SSP_base}.nc"
    em_unharmonised_urban_file = dir_output / f"Emissions_urban_region_{run_tag}_unharmonised.nc"
    em_harmonised_urban_file = dir_output /  f"Emissions_urban_region_{run_tag}_harmonised.nc"

    # 1. Read and process gridded data
    debug_log.info(f"{PRINT_COLORS['green']}Logging for profile {profile}, scenario {scenario}, model {model}, convergenc year {convergence_year}, downscale_SE {downscale_SE} started{PRINT_COLORS['end']}")
    debug_log.info(f"\n\n1. Read and process gridded data {"-"*25}")

    # 1.1 Read in IAM regions grid
    debug_log.info(f"\n\n1.1. Read and process in IAM data {"-"*25}")
    debug_log.info(f"\n\n(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}-Conv_year: {convergence_year}-Downscale: {downscale_SE}: {PRINT_COLORS["green"]}Reading (and processing) IAM population, gdp, and emissionsdata...{PRINT_COLORS["end"]}")
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
        tiff_file = plot_maps.save_to_grid_tiff(dir_processed, xr_IAM_regions_grid_save, "region_number", "", [2020], model, scenario)

    # 1.2 Read and process in IAM data
    debug_log.info(f"\n\n1.2. Read and process in IAM data {"-"*25}")
    if process_flags["read_process_IAM"] or iam_input_file.is_file()==False:
        debug_log.info(f"\n{PRINT_COLORS["green"]}(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net} Reading and processing IAM data...{PRINT_COLORS["end"]}")
        df_IAM = process_IAM_data.read_process_IAM_data(project_dir, scenario, model, file_IAM_model_region_numbers, vars_downscaling)
        df_IAM.to_csv(iam_input_file, sep=";", index=False)
    else:
        df_IAM = pd.read_csv(iam_input_file, sep=";")

    # 1.3 read POP data
    # (1) POP
    debug_log.info(f"\n\n1.3. Read POP data {"-"*25}")
    df_population = None
    xr_population = None
    debug_log.info(f"(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}-Conv_year: {convergence_year}-Downscale: {downscale_SE}: {PRINT_COLORS["green"]}Reading (and processing) population data...{PRINT_COLORS["end"]}")
    if process_flags["read_process_grid_POP"] or pop_file.is_file()==False:
        xr_population, _, _ = process_grid_data.read_process_grid_data_socioeconomic(dir_processed=dir_processed.parent, varname=varname_POP, source=source_POP, version=version_POP, SSP_base=SSP_base, base_year=base_year,
                                                                                             coarse_factor=coarse_factor_POP, unit=unit_POP, save=False, check=check_flags["check_POP_data"], log=debug_log)
        print(f"{PRINT_COLORS["yellow"]}xr_population - [{xr_population[varname_POP].sum(dim=["y", "x"])}{PRINT_COLORS["end"]}")
        debug_log.info(f"Population years:: {np.unique(xr_population["time"].values)}")
        debug_log.info(f"{PRINT_COLORS["yellow"]}xr_population - [{xr_population.x.min().item()}, {xr_population.x.max().item()}{PRINT_COLORS["end"]}]")
        if check_flags["check_POP_data"]:
            locs_test = process_grid_data.check_data_locations(xr_population[varname_POP],2020)
            for l in locs_test:
                debug_log.info(f"Locations with non-zero population in 2020: {l}")
        # Sort population data to be north-to-south and west-to-east (for consistency with other datasets)
        xr_population = xr_population.sortby("y", ascending=False)  # north-to-south
        xr_population = xr_population.sortby("x", ascending=True)  # west-to-east
        debug_log.info(f"Saving population data after read/process")
        xr_population.to_netcdf(pop_file, mode="w", engine="netcdf4")
        debug_log.info(f"{PRINT_COLORS["yellow"]}Variable: {xr_population.data_vars}{PRINT_COLORS["end"]}")
        arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_population[varname_POP])
        debug_log.info(f"{PRINT_COLORS["yellow"]}resolution POP grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.1f} arc degrees{PRINT_COLORS["end"]}")

        debug_log.info(f"{PRINT_COLORS["blue"]}Population data nodata, CRS and transform after read/process:{PRINT_COLORS["end"]}")
        debug_log.info(f"nodata: {PRINT_COLORS["blue"]}{xr_population[varname_POP].rio.nodata}{PRINT_COLORS["end"]}")
        debug_log.info(f"_FillValue: {PRINT_COLORS["blue"]}{xr_population.encoding.get('_FillValue')}{PRINT_COLORS["end"]}")
        debug_log.info(f"crs: {PRINT_COLORS["blue"]}{xr_population.rio.crs}{PRINT_COLORS["end"]}")
        debug_log.info(f"transform:\n{PRINT_COLORS["blue"]}{xr_population.rio.transform()}{PRINT_COLORS["end"]}")
        debug_log.info(f"unit: {PRINT_COLORS["blue"]}{xr_population[varname_POP].attrs["unit"]}{PRINT_COLORS["end"]}")
    else:
        xr_population = xr.open_dataset(pop_file)
    if process_flags["save_tiffs_intermediate"]:
        plot_maps.save_to_grid_tiff(dir_tiff_plots, xr_population, varname_POP, "", [2020, 2030, 2050], model, scenario, False)

    # downscale population data to match with IAM regions
    # (1) POP downscaling
    # Collect and process IAM population data for downscaling
    conversion_factor_pop = settings_models.models[model]["model_unit_conversions"]["Population"]
    df_IAM_population = df_IAM[df_IAM["variable"]==varname_POP].copy()
    df_IAM_population["value"] = conversion_factor_pop * df_IAM_population["value"]
    if downscale_SE:
        if process_flags["downscale_grid_POP"] or pop_downscaled_file.is_file()==False:
            debug_log.info(f"\n\n1.3.1. Downscale population data to IAM regions {"-"*25}\nPreparing...")
            # 1. Downscale population data to match IAM regions (incl. conversion factor to match IAM units)
            xr_population_downscaled = _intermediate_downscale_POP(varname_POP, xr_population, df_IAM_population, xr_IAM_regions_grid, 1, dir_processed, debug_log)
            xr_population_downscaled = xr_population_downscaled.drop_vars(["region_number", "country_id_GADM"], errors="ignore") #.to_netcdf(pop_processed_file, mode="w", engine="netcdf4")
            xr_population_downscaled.to_netcdf(pop_downscaled_file, mode="w", engine="netcdf4")
            # 2. Compare global sums of gridded population, after and from IAM model for comparison
            df_pop_check_global = _compare_grid_IAM_global(xr_population, xr_population_downscaled, df_IAM_population, varname_POP, debug_log)
            df_pop_check_global.to_csv(dir_check / f"df_population_{profile}_check.csv", index=False, sep=";")
            debug_log.info("--------------------------------")
        else:
            # Population was already downscaled and saved to file, read it from file
            debug_log.info(f"\n\n1.3.1. Reading downscaled population data from file...")
            xr_population_downscaled = xr.open_dataset(pop_downscaled_file, decode_coords="all")
            df_pop_check_global = pd.read_csv(dir_check / f"df_population_{profile}_check.csv", sep=";")
    else:
        # No downscaling of population data --> copy the original population data to the downscaled variable
        xr_population_downscaled = xr_population.copy()
        df_pop_check_global = _compare_grid_IAM_global(xr_population, xr_population_downscaled, df_IAM_population, varname_POP, debug_log)
        df_pop_check_global.to_csv(dir_check / f"df_population_{profile}_check.csv", index=False, sep=";")
        debug_log.info("--------------------------------")

    # 1.4 read GDP (PPP) data
    debug_log.info(f"\n\n1.4. Read GDP (PPP) data {"-"*25}")
    debug_log.info(f"(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}-Conv_year: {convergence_year}-Downscale: {downscale_SE}: {PRINT_COLORS["green"]}Reading (and processing) GDP (PPP) data...{PRINT_COLORS["end"]}")
    if process_flags["read_process_grid_GDP_PPP"] or gdp_ppp_file.is_file()==False:
        debug_log.info(f"\n\n1.4.1. Processing GDP (PPP) data...")
        # read GDP data
        xr_gdp_ppp, _, _ = process_grid_data.read_process_grid_data_socioeconomic(dir_processed=dir_processed.parent, varname=varname_GDP, source=source_GDP, version=version_GDP, SSP_base=SSP_base, base_year=base_year,
                                                                                       coarse_factor=coarse_factor_GDP, unit=unit_GDP_PPP, save=False, check=check_flags["check_GDP_data"], log=debug_log)
        debug_log.info(f"GDP years:: {np.unique(xr_gdp_ppp['time'].values)}")
        debug_log.info(f"{PRINT_COLORS["yellow"]}xr_gdp_ppp - [{xr_gdp_ppp.x.min().item()}, {xr_gdp_ppp.x.max().item()}{PRINT_COLORS["end"]}]")
        xr_gdp_ppp = xr_gdp_ppp.sortby("y", ascending=False)  # north-to-south
        xr_gdp_ppp = xr_gdp_ppp.sortby("x", ascending=True)  # west-to-east
        xr_gdp_ppp.to_netcdf(gdp_ppp_file, mode="w", engine="netcdf4")
        debug_log.info(f"{PRINT_COLORS["cyan"]}CRS for GDP|PPP after processing: {xr_gdp_ppp.rio.crs}{PRINT_COLORS["end"]}")

        debug_log.info("--------------------------------")
        debug_log.info("process_grid_data.read_process_grid_data_socioeconomic")
        debug_log.info(f"{PRINT_COLORS["blue"]}GDP (PPP) data nodata, CRS and transform after read/process:{PRINT_COLORS["end"]}")
        debug_log.info(f"nodata: {PRINT_COLORS["blue"]}{xr_gdp_ppp[varname_GDP].rio.nodata}{PRINT_COLORS["end"]}")
        debug_log.info(f"_FillValue: {PRINT_COLORS["blue"]}{xr_gdp_ppp.encoding.get('_FillValue')}{PRINT_COLORS["end"]}")
        debug_log.info(f"crs: {PRINT_COLORS["blue"]}{xr_gdp_ppp.rio.crs}{PRINT_COLORS["end"]}")
        debug_log.info(f"transform:\n{PRINT_COLORS["blue"]}{xr_gdp_ppp.rio.transform()}{PRINT_COLORS["end"]}")
        debug_log.info(f"unit: {PRINT_COLORS["blue"]}{xr_gdp_ppp[varname_GDP].attrs["unit"]}{PRINT_COLORS["end"]}")
        debug_log.info(f"{PRINT_COLORS["yellow"]}Variable: {xr_gdp_ppp.data_vars}{PRINT_COLORS["end"]}")
        arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_gdp_ppp[varname_GDP])
        debug_log.info(f"{PRINT_COLORS["yellow"]}resolution GDP grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.1f} arc degrees{PRINT_COLORS["end"]}")
        # save GDP (PPP) data to csv for each IAM region
        xr_IAM_regions_grid_aligned_GDP = xr_IAM_regions_grid.assign_coords(x=xr_gdp_ppp["x"], y=xr_gdp_ppp["y"])
        xr_gdp_ppp["region_number"] = xr_IAM_regions_grid_aligned_GDP["region_number"]
        xr_gdp_ppp_region_total = (xr_gdp_ppp[varname_GDP]
                                    .groupby(xr_gdp_ppp["region_number"])
                                    .sum())
        df_gdp_ppp_grid = xr_gdp_ppp_region_total.to_dataframe().reset_index()
        df_gdp_ppp_grid.rename(columns={"time": "year", varname_GDP: f"{varname_GDP}_grid"}, inplace=True)
        df_gdp_ppp_grid.drop(columns=["spatial_ref"], inplace=True, errors="ignore")
        df_gdp_ppp_grid.to_csv(dir_processed / f"compare_{varname_GDP.replace("|", "_")}_regional_sums_grid.csv", index=False, sep=";")
    else:
        debug_log.info(f"\n\n1.4.1. Reading GDP (PPP) data from file...")
        xr_gdp_ppp = xr.open_dataset(gdp_ppp_file, decode_coords="all")
        debug_log.info(f"{PRINT_COLORS["cyan"]}CRS for GDP|PPP after reading from file: {xr_gdp_ppp.rio.crs}{PRINT_COLORS["end"]}")
    if process_flags["save_tiffs_intermediate"]:
            plot_maps.save_to_grid_tiff(dir_tiff_plots, xr_gdp_ppp, varname_GDP, "", [2020, 2030, 2050], model, scenario, False)

    # check sum
    df_grid_gdp_ppp_World = (xr_gdp_ppp[varname_GDP].sum(dim=["y", "x"], skipna=True)
                            .compute()
                            .to_dataframe(name="value")
                            .reset_index())
    df_grid_gdp_ppp_World.rename(columns={"time": "year", "value": "grid"}, inplace=True)
    df_grid_gdp_ppp_World.drop(columns=["spatial_ref"], inplace=True, errors="ignore")
    df_IAM_GDP_World = df_IAM[df_IAM["variable"]==varname_GDP].groupby(["year"])["value"].sum().reset_index()
    conversion_factor_GDP_ppp = settings_models.models[model]["model_unit_conversions"]["GDP|PPP"]
    df_IAM_GDP_World["value"] = conversion_factor_GDP_ppp * df_IAM_GDP_World["value"]
    df_IAM_GDP_World.rename(columns={"value": "IAM"}, inplace=True)
    df_IAM_GDP_World = df_IAM_GDP_World[["year", "IAM"]]
    df_gdp_ppp_check_global = pd.merge(df_grid_gdp_ppp_World, df_IAM_GDP_World, on="year", how="outer", suffixes=("_grid", "_IAM"))
    df_gdp_ppp_check_global["Variable"] = varname_GDP
    df_gdp_ppp_check_global.to_csv(dir_check / f"df_GDP_{profile}_check.csv", index=False, sep=";")
    debug_log.info("--------------------------------")

    # 1.5 Read CO2 emissions data
    debug_log.info(f"\n\n1.5. Read CO2 emissions data {"-"*25}")
    debug_log.info(f"\n(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}-Conv_year: {convergence_year}-Downscale: {downscale_SE}: {PRINT_COLORS["green"]}Reading (and processing) emissions data...{PRINT_COLORS["end"]}")
    if process_flags["read_process_grid_EM"] or em_file.is_file()==False:
        save_EM = True
        xr_emissions, f_emissions = process_grid_data.read_process_grid_data_EM(dir_processed.parent, varname=varname_EM, unit=unit_EM, source=source_EM, version=version_EM,
                                                                                base_year=base_year, coarse_factor=coarse_factor_EM, save=False, log=debug_log)

        debug_log.info(f"Emissions years:: {np.unique(xr_emissions['time'].values)}")
        debug_log.info(f"{PRINT_COLORS["yellow"]}xr_emissions - [{xr_emissions.x.min().item()}, {xr_emissions.x.max().item()}{PRINT_COLORS["end"]}]")
        xr_emissions = xr_emissions.sortby("y", ascending=False)  # north-to-south
        xr_emissions = xr_emissions.sortby("x", ascending=True)  # west-to-east
        xr_emissions.to_netcdf(em_file, mode="w", engine="netcdf4")
        debug_log.info(f"unit: {PRINT_COLORS["blue"]}{xr_emissions[varname_EM].attrs["unit"]}{PRINT_COLORS["end"]}")
        arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_emissions[varname_EM])
        debug_log.info(f"{PRINT_COLORS["yellow"]}resolution emissions grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.1f} arc degrees{PRINT_COLORS["end"]}")
        # save emissions data to csv for each IAM region
        xr_IAM_regions_grid_aligned_EM = xr_IAM_regions_grid.assign_coords(x=xr_emissions["x"], y=xr_emissions["y"])
        xr_emissions["region_number"] = xr_IAM_regions_grid_aligned_EM["region_number"]
        xr_emissions_region_total = (xr_emissions[varname_EM]
                                    .groupby(xr_emissions["region_number"])
                                    .sum())
        df_emissions_grid = xr_emissions_region_total.to_dataframe().reset_index()
        df_emissions_grid.rename(columns={"time": "year", varname_EM: f"{varname_EM}_grid"}, inplace=True)
        df_emissions_grid.drop(columns=["spatial_ref"], inplace=True, errors="ignore")
        df_emissions_grid.to_csv(dir_processed / f"compare_{varname_EM.replace("|", "_")}_regional_sums_grid.csv", index=False, sep=";")
    else:
        xr_emissions = xr.open_dataset(em_file)
    _, arc_minutes_em, __builtins__ = process_grid_data.calculate_resolution(xr_emissions[varname_EM])
    xr_IAM_regions_grid = xr_IAM_regions_grid.reindex_like(xr_emissions.sel(time=base_year), method="nearest", tolerance=arc_minutes_em/60/2) # reindex to population grid (nearest neighbor with tolerance of half a grid cell)
    unit_EM = xr_emissions[varname_EM].attrs["unit"]
    debug_log.info("--------------------------------")
    if process_flags["save_tiffs_intermediate"]:
        plot_maps.save_to_grid_tiff(dir_tiff_plots, xr_emissions, varname_EM, "", [2020], model, scenario, False)

    # Check if data is read in successfully
    if xr_population_downscaled is None or xr_gdp_ppp is None or xr_emissions is None:
        debug_log.info("Population, GDP (PPP) or emissions data not available. Cannot proceed further.")
        exit()
    # compare resolution
    debug_log.info(f"{PRINT_COLORS["yellow"]}Variable: {xr_population_downscaled.data_vars}){PRINT_COLORS["end"]}")
    arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_population_downscaled[varname_POP])
    debug_log.info(f"{PRINT_COLORS["yellow"]}resolution population grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.2f} arc degrees")
    debug_log.info(f"{PRINT_COLORS["yellow"]}Variable: {xr_gdp_ppp.data_vars}){PRINT_COLORS["end"]}")
    arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_gdp_ppp[varname_GDP])
    debug_log.info(f"{PRINT_COLORS["yellow"]}resolution GDP (PPP) grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.2f} arc degrees")
    debug_log.info(f"{PRINT_COLORS["yellow"]}Variable: {xr_emissions.data_vars}){PRINT_COLORS["end"]}")
    arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_emissions[varname_EM])
    debug_log.info(f"{PRINT_COLORS["yellow"]}resolution EM grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.2f} arc degrees{PRINT_COLORS["end"]}")

    # 1.6 Read in urban classification
    debug_log.info(f"\n\n1.6. Read urban classification data {"-"*25}")
    debug_log.info(f"{PRINT_COLORS["green"]}Reading urban classification data from: {dir_urban / 'urban_classification_years.parquet'}{PRINT_COLORS["end"]}")
    path_urban = dir_urban / "urban_classification_years.parquet"
    gdf_urban_classification = gpd.read_parquet(path_urban)

    #----------------------------------------------------------------------------------------------------------------------------------------
    # 2. Process data
    debug_log.info(f"{PRINT_COLORS['green']}Logging for profile {profile}, scenario {scenario}, model {model} started{PRINT_COLORS['end']}")
    debug_log.info(f"\n\n2. Process data {"-"*25}")

    # 2.1 Calculate GDP_PPP per capita
    debug_log.info(f"\n\n2.1. Calculate GDP_PPP per capita {"-"*25}")

    # 2.1.1 Calculate IAM GDP per capita
    debug_log.info(f"\n\n2.1.2 IAM data{"-"*25}")
    debug_log.info(f"\n(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}-Conv_year: {convergence_year}-Downscale: {downscale_SE}: {PRINT_COLORS["green"]}Harmonising IAM regions grid with population and GDP grid...{PRINT_COLORS["end"]}")
    # calculate model GDP per capita
    xr_IAM_regions_grid_downscaling = xr_IAM_regions_grid.reindex_like(xr_emissions, method="nearest", tolerance=1e-5)
    if process_flags["process_IAM_GDP_per_POP"] or iam_gdp_per_capita_file.is_file()==False:
        debug_log.info(f"(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}-Conv_year: {convergence_year}-Downscale: {downscale_SE}: {PRINT_COLORS["green"]}Calculating GDP per capita for IAM data...{PRINT_COLORS["end"]}")
        df_IAM_projection_pop = df_IAM[df_IAM["variable"] == varname_POP]
        df_IAM_projection_gpd_ppp = df_IAM[df_IAM["variable"] == varname_GDP]
        df_IAM_projection_gdp_ppp_per_capita = pd.concat([df_IAM_projection_pop, df_IAM_projection_gpd_ppp], axis=0)
        df_IAM_projection_gdp_ppp_per_capita = df_IAM_projection_gdp_ppp_per_capita.pivot(index=["model", "scenario", "region_code", "region_number", "year"], columns="variable", values="value").reset_index()
        df_IAM_projection_gdp_ppp_per_capita["value"] = df_IAM_projection_gdp_ppp_per_capita[varname_GDP] / df_IAM_projection_gdp_ppp_per_capita[varname_POP]
        df_IAM_projection_gdp_ppp_per_capita.drop([varname_POP, varname_GDP], axis=1, inplace=True)
        df_IAM_projection_gdp_ppp_per_capita["variable"] = varname_gdp_per_capita
        df_IAM_projection_gdp_ppp_per_capita["unit"] = "USD_2005/yr/person"
        df_IAM_projection_gdp_ppp_per_capita.to_csv(iam_gdp_per_capita_file, sep=";")
    else:
        df_IAM_projection_gdp_ppp_per_capita = pd.read_csv(iam_gdp_per_capita_file, sep=";")

    # 2.1.2 Process population and GDP grid data
    debug_log.info(f"\n\n2.1.1 Process population and GDP grid data{"-"*25}")
    debug_log.info(f"\n(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}-Conv_year: {convergence_year}-Downscale: {downscale_SE}: {PRINT_COLORS["green"]}Processing GDP and population data for downscaling...{PRINT_COLORS["end"]}")
    if check_flags["check_IAM_grid_data"]:
        plot_maps.plot_factors_GDP_POP(dir_processed, source_POP, source_GDP, version_POP, version_GDP, xr_population_downscaled, xr_gdp_ppp, None, year=2020, coarsen=12)
    # align, downscale, and set pop to 1 where gdp>0
    if process_flags["process_grid_GDP_POP"] or pop_processed_file.is_file()==False or gdp_ppp_processed_file.is_file()==False:
        debug_log.info(f"{PRINT_COLORS["yellow"]}CHECK:{PRINT_COLORS["end"]}")
        debug_log.info(f"Unit population: {xr_population_downscaled[varname_POP].attrs.get("unit", "N/A")}")
        debug_log.info(f"Unit GDP (PPP): {xr_gdp_ppp[varname_GDP].attrs.get("unit", "N/A")}")
        _, _, deg_em = process_grid_data.calculate_resolution(xr_emissions[varname_EM])
        _, _, deg_proc = process_grid_data.calculate_resolution(xr_population_downscaled[varname_POP])
        tol_em = max(deg_em, deg_proc) / 2
        debug_log.info(f"Resolution of EM grid: {deg_em:.2f} degrees")
        debug_log.info(f"Resolution of population grid: {deg_proc:.2f} degrees")
        debug_log.info(f"Tolerance for reindexing: {tol_em:.2f} degrees")
        debug_log.info(f"Unique time values in population data (before alignment): {np.unique(xr_population_downscaled["time"].values)}")

        # align population and GDP grids to emissions grid
        xr_population_aligned = xr_population_downscaled.reindex(x=xr_emissions["x"], y=xr_emissions["y"], method="nearest", tolerance=tol_em)
        xr_gdp_ppp_aligned = xr_gdp_ppp.reindex(x=xr_emissions["x"], y=xr_emissions["y"], method="nearest", tolerance=tol_em)
        debug_log.info(f"Unique time values in population data (after alignment): {np.unique(xr_population_downscaled["time"].values)}")
        finite_pop = int(np.isfinite(xr_population_aligned[varname_POP].sel(time=2020)).sum())
        if finite_pop == 0:
            raise ValueError("Population is all-NaN after reindex onto the emissions grid: " "tolerance too tight or grids do not overlap")
        finite_gdp = int(np.isfinite(xr_gdp_ppp_aligned[varname_GDP].sel(time=2020)).sum())
        if finite_gdp == 0:
            raise ValueError("GDP is all-NaN after reindex onto the population grid: " "tolerance too tight or grids do not overlap")
        debug_log.info(f"(process_factors_GDP_POP) GDP finite cells after reindex: {finite_gdp:,}")

        # set population to 1 where GDP>0, to avoid division by zero when calculating GDP per capita
        debug_log.info(f"Setting population to 1 where GDP>0 to avoid division by zero when calculating GDP per capita...")
        xr_population_processed, xr_gdp_ppp_processed = process_IPAT_factors.process_factors_GDP_POP(xr_population_aligned, xr_gdp_ppp_aligned,
                                                                                                     varname_POP, varname_GDP,
                                                                                                     unit_POP, unit_GDP_PPP,
                                                                                                     base_year, years_downscaling,
                                                                                                     check_flags["check_GDP_POP"],
                                                                                                     debug_log)
        debug_log.info(f"{PRINT_COLORS["yellow"]}xr_population_processed - [{xr_population_processed.x.min().item()}, {xr_population_processed.x.max().item()}{PRINT_COLORS["end"]}]")
        debug_log.info(f"{PRINT_COLORS["yellow"]}xr_gdp_ppp_processed - [{xr_gdp_ppp_processed.x.min().item()}, {xr_gdp_ppp_processed.x.max().item()}{PRINT_COLORS["end"]}]")

        # check if population and GDP grids are aligned with each other
        process_IPAT_factors.check_POP_GDP_alignment(dir_processed, xr_population_processed, xr_gdp_ppp_processed, varname_POP, varname_GDP, debug_log)
        xr_population_processed = xr_population_processed.compute()
        xr_gdp_ppp_processed = xr_gdp_ppp_processed.compute()
        debug_log.info(f"{PRINT_COLORS["cyan"]}CRS after processing GDP|PPP from file: {xr_gdp_ppp.rio.crs}{PRINT_COLORS["end"]}")
        xr_population_processed.to_netcdf(pop_processed_file, mode="w", engine="netcdf4")
        xr_gdp_ppp_processed.to_netcdf(gdp_ppp_processed_file, mode="w", engine="netcdf4")
        debug_log.info(f"time steps pop: {xr_population_processed[varname_POP].time.values}")
        debug_log.info(f"time steps gdp_per_capita: {xr_gdp_ppp_processed[varname_GDP].time.values}")
    else:
        xr_population_processed = xr.open_dataset(pop_processed_file)
        xr_gdp_ppp_processed = xr.open_dataset(gdp_ppp_processed_file)
        debug_log.info(f"{PRINT_COLORS["cyan"]}CRS after reading processed GDP|PPP from file: {xr_gdp_ppp.rio.crs}{PRINT_COLORS["end"]}")
    if process_flags["save_tiffs_intermediate"]:
        plot_maps.save_to_grid_tiff(dir_processed, xr_population_processed, varname_POP, "_processed", [2020, 2030, 2050], model, scenario)
        plot_maps.save_to_grid_tiff(dir_processed, xr_gdp_ppp_processed, varname_GDP, "_processed", [2020, 2030, 2050], model, scenario)

    # 2.1.3 Calculate grid GDP per capita
    # (2) GDP/POP (GDP per capita)
    debug_log.info(f"\n\n2.1.2 Calculate grid GDP per capita {"-"*25}")
    if process_flags["process_grid_GDP_per_POP"] or gdp_ppp_per_capita_file.is_file()==False:
        xr_gdp_ppp_per_capita = process_IPAT_factors.calculate_gdp_per_capita(xr_population_processed, xr_gdp_ppp_processed,
                                                                               varname_POP, varname_GDP, varname_gdp_per_capita,
                                                                               unit_POP, unit_GDP_PPP,
                                                                               log=debug_log)
        xr_gdp_ppp_per_capita.to_netcdf(gdp_ppp_per_capita_file, mode="w", engine="netcdf4")
    else:
        xr_gdp_ppp_per_capita = xr.open_dataset(gdp_ppp_per_capita_file, decode_coords="all")
    if process_flags["save_tiffs_intermediate"]:
        plot_maps.save_to_grid_tiff(dir_processed, xr_gdp_ppp_per_capita, varname_gdp_per_capita, "", [2020, 2030, 2050], model, scenario)

    # downscale GDP per capita data to match with IAM regions
    # (2) GDP/POP downscaling
    conversion_factor_pop = settings_models.models[model]["model_unit_conversions"]["Population"]
    conversion_factor_gdp_ppp = settings_models.models[model]["model_unit_conversions"]["GDP|PPP"]
    conversion_factor_gdp_per_capita = conversion_factor_gdp_ppp / conversion_factor_pop
    df_IAM_gdp_per_capita = df_IAM[df_IAM["variable"].isin([varname_GDP, varname_POP])].copy()
    df_IAM_gdp_per_capita = df_IAM_gdp_per_capita.pivot(index=["model", "scenario", "region_code", "region_number", "year"], columns="variable", values="value").reset_index()
    df_IAM_gdp_per_capita[varname_gdp_per_capita] = df_IAM_gdp_per_capita[varname_GDP] / df_IAM_gdp_per_capita[varname_POP]
    df_IAM_gdp_per_capita[varname_gdp_per_capita] = conversion_factor_gdp_per_capita * df_IAM_gdp_per_capita[varname_gdp_per_capita]
    if downscale_SE:
        # Collect and process IAM population data for downscaling
        if process_flags["downscale_grid_GDP_per_capita"] or gdp_per_capita_downscaled_file.is_file()==False:
            debug_log.info(f"\n\n1.3.1. Downscale GDP per capita data to IAM regions {"-"*25}\nPreparing...")
            # 1. Downscale GDP per capita data to match IAM regions (incl. conversion factor to match IAM units)
            xr_gdp_ppp_per_capita_downscaled = _intermediate_downscale_GDPpc(varname_gdp_per_capita, xr_gdp_ppp_per_capita, df_IAM_gdp_per_capita, xr_IAM_regions_grid_downscaling, xr_population, varname_POP, 1, dir_processed, debug_log)
            xr_gdp_ppp_per_capita_downscaled = xr_gdp_ppp_per_capita_downscaled.drop_vars(["region_number", "country_id_GADM"], errors="ignore") #.to_netcdf(gdp_per_capita_downscaled_file, mode="w", engine="netcdf4")
            xr_gdp_ppp_per_capita_downscaled.to_netcdf(gdp_per_capita_downscaled_file, mode="w", engine="netcdf4")
            # 2. Compare global sums of gridded GDP per capita, after and from IAM model for comparison
            df_gdp_per_capita_check_global = _compare_ratio_grid_IAM(xr_population, xr_gdp_ppp_per_capita, xr_gdp_ppp_per_capita_downscaled, df_IAM_gdp_per_capita,
                                                                     varname_GDP, varname_POP, varname_gdp_per_capita, conversion_factor_gdp_per_capita)
            cols_to_drop = df_gdp_per_capita_check_global.columns[df_gdp_per_capita_check_global.columns.str.startswith("spatial_ref", na=False)]
            df_gdp_per_capita_check_global = df_gdp_per_capita_check_global.drop(columns=cols_to_drop, errors="ignore")
            df_gdp_per_capita_check_global.rename(columns={f"{varname_gdp_per_capita}_before": "grid_before", f"{varname_gdp_per_capita}_after": "grid_after"})
            df_gdp_per_capita_check_global.to_csv(dir_check / f"df_gdp_per_capita_{profile}_check.csv", index=False, sep=";")
            debug_log.info("--------------------------------")
        else:
            # GDP (PPP) per capita was already downscaled and saved to file, read it from file
            debug_log.info(f"\n\n1.3.1. Reading downscaled GDP per capita data from file...")
            xr_gdp_ppp_per_capita_downscaled = xr.open_dataset(gdp_per_capita_downscaled_file, decode_coords="all")
            df_gdp_per_capita_check_global = pd.read_csv(dir_check / f"df_gdp_per_capita_{profile}_check.csv", sep=";")
    else:
        # No downscaling of GDP (PPP) per capitadata --> copy the original population data to the downscaled variable
        xr_gdp_ppp_per_capita_downscaled = xr_gdp_ppp.copy()
        df_gdp_per_capita_check_global = _compare_ratio_grid_IAM(xr_population, xr_gdp_ppp_per_capita, xr_gdp_ppp_per_capita_downscaled, df_IAM_gdp_per_capita,
                                                                 varname_GDP, varname_POP, varname_gdp_per_capita, conversion_factor_gdp_per_capita)
        cols_to_drop = df_gdp_per_capita_check_global.columns[df_gdp_per_capita_check_global.columns.str.startswith("spatial_ref", na=False)]
        df_gdp_per_capita_check_global = df_gdp_per_capita_check_global.drop(columns=cols_to_drop, errors="ignore")
        df_gdp_per_capita_check_global.to_csv(dir_check / f"df_gdp_per_capita_{profile}_check.csv", index=False, sep=";")
        debug_log.info("--------------------------------")

    if check_flags["check_grid_GDP_per_capita"]:
        debug_log.info("--------------------------------")
        debug_log.info("xr_gdp_ppp_per_capita_downscaled")
        debug_log.info(f"varname: {varname_gdp_per_capita}")
        debug_log.info(f"Type: {type(varname_gdp_per_capita)}")
        debug_log.info(xr_gdp_ppp_per_capita_downscaled)
        process_grid_data.count_values_rio_xarray(dir_processed, xr_gdp_ppp_per_capita_downscaled, varname_gdp_per_capita, 2020, debug_log)
        process_IPAT_factors.check_location_for_GDP_per_capita_calculation(xr_gdp_ppp_per_capita_downscaled, varname_gdp_per_capita)

    if check_flags["check_IAM_GDP_per_capita"]:
        # 2.1.3 compare IAM and grid data for GDP per capita
        df_grid, df_compare = process_IPAT_factors.compare_IAM_grid_regions_GDP_per_capita(xr_gdp_ppp_per_capita_downscaled, varname_gdp_per_capita,
                                                                            xr_population_processed, varname_POP,
                                                                            df_IAM_projection_gdp_ppp_per_capita,
                                                                            xr_IAM_regions_grid_downscaling)
        debug_log.info(df_grid.to_string(index=False))
        debug_log.info(df_compare.to_string())
        csv_file_grid = dir_check / f"selection_grid_gdp_per_capita.csv"
        csv_file_compare = dir_check / f"compare_IAM_grid_gdp_per_capita.csv"
        df_grid.to_csv(csv_file_grid, sep=";", index=False)
        df_compare.to_csv(csv_file_compare, sep=";", index=True)

    # 2.2 Calculate EM per GDP (PPP)
    debug_log.info(f"\n\n2.2. Calculate EM per GDP (PPP) {"-"*25}")
    debug_log.info(f"(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}-Conv_year: {convergence_year}-Downscale: {downscale_SE}: {PRINT_COLORS["green"]}Calculating EM per GDP (PPP) for IAM data...{PRINT_COLORS["end"]}")

    # 2.2.1 process grid data
    debug_log.info(f"\n\n2.2.1 Process grid data {"-"*25}")
    debug_log.info(xr_gdp_ppp_processed)
    debug_log.info(xr_emissions)

    # 2.2.2 process IAM (including the column names which are made lowercase)
    debug_log.info(f"\n\n2.2.2 Process grid data {"-"*25}")
    df_IAM_GDP = pd.DataFrame(df_IAM[df_IAM["variable"]==varname_GDP])
    df_IAM_EM = process_IAM_data.process_EM_regions_data(df_IAM, years_downscaling, varname_EM, vars_downscaling, net_emissions, model, debug_log)
    df_IAM_EM.to_csv(dir_processed / f"IAM_{profile_model_scenario_convergence_year}_emissions_processed.csv", index=False, sep=";")

    one_unit_IAM_model_GDP_PPP = process_IAM_data.model_unit_conversions[model]["GDP|PPP"]
    one_unit_IAM_model_em = process_IAM_data.model_unit_conversions[model]["Emissions|CO2"]
    # check if convergence year is multile of 10
    if convergence_year % 10 != 0:
        debug_log.info(f"{PRINT_COLORS["red"]}Convergence year {convergence_year} is not a multiple of 10. Please choose a convergence year that is a multiple of 10.{PRINT_COLORS["end"]}")
        raise ValueError(f"Convergence year {convergence_year} is not a multiple of 10. Please choose a convergence year that is a multiple of 10.")
    df_IAM_GDP = process_IAM_data.extrapolate_IAM_values_to_convergence_year(dir_processed, df_IAM_GDP, one_unit_IAM_model_GDP_PPP, convergence_year, method_extension, debug_log)
    df_IAM_EM = process_IAM_data.extrapolate_IAM_values_to_convergence_year(dir_processed, df_IAM_EM, one_unit_IAM_model_em, convergence_year, method_extension, debug_log)
    csv_file_GDP = dir_processed / f"IAM_{model}_{scenario}_gdp_ppp_downscaling_extended.csv"
    csv_file_EM = dir_processed / f"IAM_{model}_{scenario}_em_downscaling_extended.csv"
    df_IAM_GDP.to_csv(csv_file_GDP, index=False, sep=";")
    df_IAM_EM.to_csv(csv_file_EM, index=False, sep=";")

    if process_flags["process_IAM_EM_per_GDP"] or iam_em_per_gdp_ppp_file.is_file()==False:
        df_IAM_projection_em_per_gdp_ppp = pd.concat([df_IAM_EM, df_IAM_GDP], axis=0)
        df_IAM_projection_em_per_gdp_ppp = df_IAM_projection_em_per_gdp_ppp.pivot(index=["model", "scenario", "region_code", "region_number", "year"], columns="variable", values="value").reset_index()
        df_IAM_projection_em_per_gdp_ppp["value"] = df_IAM_projection_em_per_gdp_ppp[varname_EM] / df_IAM_projection_em_per_gdp_ppp[varname_GDP]
        df_IAM_projection_em_per_gdp_ppp["variable"] = varname_EM + "_per_" + varname_GDP
        df_IAM_projection_em_per_gdp_ppp["unit"] = "tCO2/USD_2005/yr"
        df_IAM_projection_em_per_gdp_ppp.drop([varname_EM, varname_GDP], axis=1, inplace=True)
        df_IAM_projection_em_per_gdp_ppp.to_csv(iam_em_per_gdp_ppp_file, index=False, sep=";")
    else:
        df_IAM_projection_em_per_gdp_ppp = pd.read_csv(iam_em_per_gdp_ppp_file, sep=";")

    # 2.2.3 Pocess grid emissions per GDP (PPP)
    debug_log.info(f"\n\n2.2.3 Process grid emissions per GDP (PPP) {"-"*25}")
    xr_gdp_ppp_by = xr_gdp_ppp_processed.sel(time=base_year)
    xr_em_by = xr_emissions.sel(time=base_year)
    _, arc_minutes_em, _ = process_grid_data.calculate_resolution(xr_em_by[varname_EM])
    tolerance = arc_minutes_em / 60 / 2
    xr_gdp_ppp_by = xr_gdp_ppp_by.reindex_like(xr_em_by, method="nearest", tolerance=tolerance)

    xr_gdp_ppp_by = xr_gdp_ppp_by.chunk({"x": "auto", "y": "auto"})
    xr_em_by = xr_em_by.chunk({"x": "auto", "y": "auto"})

    # check resolution
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
    gdp_floor = 1
    xr_em_per_gdp_ppp_by_downscaling = xr_em_by[varname_EM] / xr_gdp_ppp_by[varname_GDP].where(xr_gdp_ppp_by[varname_GDP] != 0)  # Avoid division by zero
    save_path=dir_processed / "figures" / f"plot_em_per_gdp_ppp_{base_year}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    plot_floor_comparison(xr_em_per_gdp_ppp_by_downscaling, f"{source_GDP}: GDP (PPP) {base_year}", gdp_floor, save_path=dir_processed / "figures" / f"plot_em_per_gdp_ppp_{base_year}.png")
    xr_em_per_gdp_ppp_by_downscaling = xr_em_per_gdp_ppp_by_downscaling.where(xr_gdp_ppp_by[varname_GDP] >= gdp_floor, other=0).compute() # avoid division by very small GDP values

    unit_grid = xr_em_by[varname_EM].attrs["unit"] + "/" + xr_gdp_ppp_by[varname_GDP].attrs["unit"]
    unit_iam = df_IAM_projection_em_per_gdp_ppp["unit"].iloc[0]
    debug_log.info(f"Unit emissions per GDP: grid {unit_grid}, IAM {unit_iam}")
    if unit_grid != unit_iam:
        debug_log.warning(f"{PRINT_COLORS["red"]}Unit grid ({unit_grid}) differs from unit IAM ({unit_iam}): "
                          f"check that the values are converted before calc_scaling_factors_EM_per_GDP{PRINT_COLORS["end"]}")
    xr_em_per_gdp_ppp_by_downscaling = xr_em_per_gdp_ppp_by_downscaling.rename(varname_em_per_gdp_ppp)  # item 6
    xr_em_per_gdp_ppp_by_downscaling.attrs["unit"] = unit_iam

    # 2.3 Calculate CO2 emissions per capita grid
    # (3) CO2/GDP (CO2 emissions per GDP (PPP))
    debug_log.info(f"\n\n2.3 Calculate CO2 emissions grid {"-"*25}")
    if process_flags["process_grid_EM_per_GDP"] or em_per_gdp_ppp_file.is_file()==False:
        debug_log.info(f"(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}-Conv_year: {convergence_year}-Downscale: {downscale_SE}: {PRINT_COLORS["green"]} Calculating CO2 grid emissions...{PRINT_COLORS["end"]}")
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

    else:
        xr_em_per_gdp_ppp = xr.open_dataset(em_per_gdp_ppp_file)

    # 2.4 Calculate grid emissions by applying IPAT factors to population and GDP per capita grids
    debug_log.info(f"\n\n2.4 Calculate grid emissions by applying IPAT factors to population and GDP per capita grids {"-"*25}")
    #xr_gdp_ppp_per_capita_processed = xr_gdp_ppp_per_capita_downscaled.copy()
    # first check if the time steps of the population and GDP per capita grids match
    if process_flags["process_grid_EM"] or em_unharmonised_file.is_file()==False:
        factors = {"population": xr_population_processed[varname_POP],
            "GDP per capita": xr_gdp_ppp_per_capita_downscaled[varname_gdp_per_capita],
            "emissions per GDP": xr_em_per_gdp_ppp[varname_em_per_gdp_ppp]}
        for name, da in factors.items():
            same_x = np.array_equal(da["x"].values, xr_emissions["x"].values)
            same_y = np.array_equal(da["y"].values, xr_emissions["y"].values)
            if not (same_x and same_y):
                debug_log.info(f"{PRINT_COLORS["red"]}Grid of {name} does not match the emissions grid.{PRINT_COLORS["end"]}")
                raise ValueError(f"Grid of {name} does not match the emissions grid. Please check the data.")

        xr_emissions_unharmonised = (xr_population_processed[varname_POP] * xr_gdp_ppp_per_capita_downscaled[varname_gdp_per_capita] * xr_em_per_gdp_ppp[varname_em_per_gdp_ppp])
        xr_emissions_unharmonised = xr_emissions_unharmonised.to_dataset(name=varname_EM)
        xr_emissions_unharmonised[varname_EM].attrs["unit"] = unit_EM
        xr_emissions_unharmonised.to_netcdf(em_unharmonised_file, mode="w", engine="netcdf4")

        # save GDP (PPP) data to csv for each IAM region
        xr_IAM_regions_grid_aligned_EM = xr_IAM_regions_grid.assign_coords(x=xr_emissions_unharmonised["x"], y=xr_emissions_unharmonised["y"])
        xr_emissions_unharmonised["region_number"] = xr_IAM_regions_grid_aligned_EM["region_number"]
        xr_emissions_harmonised_region_total = (xr_emissions_unharmonised[varname_EM]
                                                .groupby(xr_emissions_unharmonised["region_number"])
                                                .sum())
        df_emissions_unharmonised_grid = xr_emissions_harmonised_region_total.to_dataframe().reset_index()
        df_emissions_unharmonised_grid.rename(columns={"time": "year", varname_EM: f"{varname_EM}_grid"}, inplace=True)
        df_emissions_unharmonised_grid.drop(columns=["spatial_ref"], inplace=True, errors="ignore")
        df_emissions_unharmonised_grid.to_csv(dir_processed / f"compare_{varname_EM}_unharmonised_regional_sums_grid.csv", index=False, sep=";")
        if check_flags["check_grid_EM"]:
            for year in [2020, 2030, 2040, 2050]:
                info = process_grid_data.check_data_locations(xr_emissions_unharmonised[varname_EM], year)
                info_str = "\n".join(str(i) for i in info)
                debug_log.info(f"{PRINT_COLORS["yellow"]}Check grid emissions unharmonised {year}:\n{info_str}{PRINT_COLORS["end"]}")
    else:
        xr_emissions_unharmonised = xr.open_dataset(em_unharmonised_file)
    if process_flags["save_tiffs_intermediate"]:
        plot_maps.save_to_grid_tiff(dir_processed, xr_emissions_unharmonised, varname_EM, "_unharmonised", [2020, 2030, 2050], model, scenario)

    # 2.4.1 harmonise grid emissions per region with IAM emissions per region
    debug_log.info(f"\n\n2.4.1 Harmonise grid emissions per region with IAM emissions per region {"-"*25}")
    debug_log.info(f"(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}-Conv_year: {convergence_year}-Downscale: {downscale_SE}:{PRINT_COLORS["green"]}: Harmonising grid emissions per region with IAM emissions per region...{PRINT_COLORS["end"]}")
    years = df_IAM_EM["year"].unique()
    variable = df_IAM_EM["variable"].unique()[0]
    extra_rows = pd.DataFrame({"model": model, "scenario": scenario, "region_code":"OCEAN", "variable":variable, "year": years, "unit": unit_EM, "region_number": 0, "value": 0})
    df_IAM_EM_add_ocean = pd.concat([df_IAM_EM, extra_rows], ignore_index=True).sort_values(["year", "region_number"]).reset_index(drop=True)
    df_IAM_EM_unharmonised_compare, xr_regional_sums = process_IPAT_factors.calc_regional_values(xr_emissions_unharmonised, varname_EM,
                                                                                                 xr_IAM_regions_grid_downscaling, df_IAM_EM_add_ocean,
                                                                                                 years_downscaling)
    df_IAM_EM_unharmonised_compare.drop(columns=["difference", "relative_difference_%", "indicator_grid_xr_million", "indicator_df_million"], inplace=True, errors="ignore")
    #df_IAM_EM_unharmonised_compare.rename(columns={"Emissions_CO2_Excl_shipping_aviation_AFOLU_IAM": "total_iam", "Emissions_CO2_Excl_shipping_aviation_AFOLU_grid_summed": "total_grid"}, inplace=True)
    df_IAM_EM_unharmonised_compare.rename(columns={f"{varname_EM}_IAM": "total_iam", f"{varname_EM}_grid_summed": "total_grid"}, inplace=True)
    df_IAM_EM_unharmonised_compare = df_IAM_EM_unharmonised_compare[[ "year", "region_number", "total_iam", "total_grid"]]
    df_IAM_EM_unharmonised_compare_World = df_IAM_EM_unharmonised_compare.groupby(["year"]).agg({"total_iam": "sum", "total_grid": "sum"}).reset_index()
    df_IAM_EM_unharmonised_compare_World["region_number"] = 28
    df_IAM_EM_unharmonised_compare = pd.concat([df_IAM_EM_unharmonised_compare, df_IAM_EM_unharmonised_compare_World], ignore_index=True).sort_values(["year", "region_number"]).reset_index(drop=True)
    df_IAM_EM_unharmonised_compare.to_csv(csv_file_compare_unharmonised, sep=";", index=False)

    # 2.4.2 Calculate harmonisation factor for grid emissions per region with IAM emissions per region
    debug_log.info(f"\n\n2.4.2 Calculate harmonisation factors for grid emissions per region with IAM emissions per region {"-"*25}")
    save_dir = dir_processed / "harmonisation_factors"
    save_dir.mkdir(parents=True, exist_ok=True)
    xr_em_correction_factors = process_IPAT_factors.calculate_harmonisation_factors_emissions(xr_emissions_unharmonised, varname_EM, xr_regional_sums,
                                                                                              xr_IAM_regions_grid_downscaling, df_IAM_EM_add_ocean,
                                                                                              years_downscaling, save_dir, debug_log)

    # 2.4.3 apply harmonisation factors to grid emissions
    debug_log.info(f"\n\n2.4.3 Apply harmonisation factors to grid emissions {"-"*25}")
    xr_emissions_harmonised = process_IPAT_factors.apply_harmonisation_factors_emissions(xr_em_correction_factors,
                                                                                         xr_emissions_unharmonised, varname_EM,
                                                                                         xr_IAM_regions_grid_downscaling,
                                                                                         model, scenario, debug_log)
    xr_emissions_harmonised[varname_EM].attrs["unit"] = unit_EM
    plot_maps.save_to_grid_tiff(dir_processed, xr_emissions_harmonised, varname_EM, "_harmonised", years_downscaling, model, scenario)


    debug_log.info(f"Variable: {xr_emissions_harmonised.data_vars}")
    arc_seconds, arc_minutes, arc_degrees = process_grid_data.calculate_resolution(xr_emissions_harmonised[varname_EM])
    debug_log.info(f"resolution downscaled EM grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.2f} arc degrees")

    x_min, x_max = float(xr_emissions_harmonised[varname_EM].x.min()), float(xr_emissions_harmonised[varname_EM].x.max())
    y_min, y_max = float(xr_emissions_harmonised[varname_EM].y.min()), float(xr_emissions_harmonised[varname_EM].y.max())
    xr_emissions_harmonised = xr_emissions_harmonised.sortby("y", ascending=False)  # north-to-south
    xr_emissions_harmonised = xr_emissions_harmonised.sortby("x", ascending=True)  # west-to-east
    xr_emissions_harmonised.to_netcdf(em_harmonised_file, mode="w", engine="netcdf4")
    debug_log.info(f"extent downscaled EM grid: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

    debug_log.info(f"\n{PRINT_COLORS["green"]}(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}-Conv_year: {convergence_year}-Downscale: {downscale_SE}: Downscaling complete. Processed data saved to {dir_processed} and output to {dir_output}.{PRINT_COLORS["end"]}")

    debug_log.info(f"Unique time values in xr_population_processed: {np.unique(xr_population_processed["time"].values)}")
    debug_log.info(f"Unique time values in xr_gdp_ppp_processed: {np.unique(xr_gdp_ppp_processed["time"].values)}")
    debug_log.info(f"Unique time values in xr_emissions_unharmonised: {np.unique(xr_emissions_unharmonised["time"].values)}")
    debug_log.info(f"Unique time values in xr_emissions_harmonised: {np.unique(xr_emissions_harmonised["time"].values)}")
    # Check: calculate sum per region per year for harmonised emissions
    df_IAM_EM_harmonised_compare, xr_regional_sums_corrected = process_IPAT_factors.calc_regional_values(xr_emissions_harmonised, varname_EM,
                                                                                                         xr_IAM_regions_grid_downscaling, df_IAM_EM_add_ocean,
                                                                                                         years_downscaling)
    df_IAM_EM_harmonised_compare.drop(columns=["difference", "relative_difference_%", "indicator_grid_xr_million", "indicator_df_million"], inplace=True, errors="ignore")
    df_IAM_EM_harmonised_compare.rename(columns={f"{varname_EM}_IAM": "total_iam", f"{varname_EM}_grid_summed": "total_grid"}, inplace=True)
    df_IAM_EM_harmonised_compare = df_IAM_EM_harmonised_compare[[ "year", "region_number", "total_iam", "total_grid"]]
    df_IAM_EM_harmonised_compare_World = df_IAM_EM_harmonised_compare.groupby(["year"]).agg({"total_iam": "sum", "total_grid": "sum"}).reset_index()
    df_IAM_EM_harmonised_compare_World["region_number"] = 28
    df_IAM_EM_harmonised_compare = pd.concat([df_IAM_EM_harmonised_compare, df_IAM_EM_harmonised_compare_World], ignore_index=True).sort_values(["year", "region_number"]).reset_index(drop=True)
    df_IAM_EM_harmonised_compare.to_csv(csv_file_compare_harmonised, sep=";", index=False)

    # 3 Calculate urban emissions
    debug_log.info(f"{PRINT_COLORS['green']}Logging for profile {profile}, scenario {scenario}, model {model} started{PRINT_COLORS['end']}")
    debug_log.info(f"\n\n3.1 Calculate urban emissions {"-"*25}")
    # TO DO --> check emissions grids that are not in the polygons, but are in IAM regions (is currently processed in Google Earth Engine, but not in this script)
    # 3.1 Calculate or read unharmonised and harmonised urban emissions
    if process_flags["process_urban_classification_emissions"] or not (em_unharmonised_urban_file.is_file() and em_harmonised_urban_file.is_file()):
        debug_log.info(f"\n\n(({(time.time()-start_time)/60:,.1f} mins): {profile}-{scenario}-{gross_net}-Conv_year: {convergence_year}-Downscale: {downscale_SE}: {PRINT_COLORS["green"]}Calculating urban emissions...{PRINT_COLORS["end"]}")
        # add regions to xr_em
        xr_emissions_regions_unharmonised = xr_emissions_unharmonised.copy()
        xr_emissions_regions_unharmonised["region_number"] = xr_IAM_regions_grid_downscaling["region_number"]
        xr_emissions_regions_harmonised = xr_emissions_harmonised.copy()
        xr_emissions_regions_harmonised["region_number"] = xr_IAM_regions_grid_downscaling["region_number"]

        # Calcualte unharmonised emissions
        debug_log.info(f"\n\n2.4.1 Calculate unharmonised emissions {"-"*25}")
        xr_em_urban_unharmonised = process_urban_grid_emissions.aggregate_urban_values(project_dir,
                                                          save_dir=dir_processed.parent,
                                                          profile=profile,
                                                          add_txt=f"_emissions_unharmonised_{run_tag}",
                                                          xr_dataset=xr_emissions_regions_unharmonised, gdf_urban_classification=gdf_urban_classification,
                                                          varname=varname_EM,
                                                          region_varname="region_number",
                                                          final_year=2050,
                                                          use_saved=True,
                                                          save_tif=True,
                                                          log=debug_log)
        xr_em_urban_unharmonised.to_netcdf(em_unharmonised_urban_file, engine="netcdf4")
        debug_log.info(f"\n\n2.4.2 Calculate harmonised emissions {"-"*25}")
        xr_em_urban_harmonised = process_urban_grid_emissions.aggregate_urban_values(project_dir,
                                                        save_dir=dir_processed.parent,
                                                        profile=profile,
                                                        add_txt=f"_emissions_harmonised_{run_tag}",
                                                        xr_dataset=xr_emissions_regions_harmonised, gdf_urban_classification=gdf_urban_classification,
                                                        varname=varname_EM,
                                                        region_varname="region_number",
                                                        final_year=2050,
                                                        use_saved=True,
                                                        save_tif=False,
                                                        log=debug_log)
        xr_em_urban_harmonised.to_netcdf(em_harmonised_urban_file, engine="netcdf4")
    else:
        xr_em_urban_unharmonised = xr.open_dataset(em_unharmonised_urban_file, decode_coords="all")
        xr_em_urban_harmonised = xr.open_dataset(em_harmonised_urban_file, decode_coords="all")

    # 3.2a Aggregate unharmonised and harmonised urban emissions per region and year
    debug_log.info(f"\n\n3.2a Aggregate unharmonised emissions {"-"*25}")
    df_em_urban_unharmonised, df_em_rural_unharmonised, df_em_ocean_unharmonised = process_urban_grid_emissions.calculate_urban_rural_totals(xr_dataset=xr_em_urban_unharmonised, varname=varname_EM, region_varname="region_number")
    df_em_urban_unharmonised.to_csv(dir_output / f"Emissions_urban_region_{profile_model_scenario_convergence_year}_unharmonised.csv", index=False, sep=";")
    df_em_urban_unharmonised["Type"] = "urban"
    df_em_rural_unharmonised["Type"] = "rural"
    df_em_ocean_unharmonised["Type"] = "ocean"
    # combine dataframes for urban and rural emissions into one dataset, add the sum of urban and rural as "total", and using "urban", "rural" and "total" as a column names
    # also add world and combine with total IAM and grid emissions for comparison
    df_em_combined_unharmonised = pd.concat([df_em_urban_unharmonised, df_em_rural_unharmonised, df_em_ocean_unharmonised], ignore_index=True)
    df_em_combined_unharmonised.drop(columns=["spatial_ref"], inplace=True, errors="ignore")
    df_em_combined_unharmonised["year"] = df_em_combined_unharmonised["year"].astype(int)
    df_em_combined_unharmonised = df_em_combined_unharmonised.pivot(index=["year", "region_number"], columns="Type", values=varname_EM).reset_index()
    df_em_combined_unharmonised["total_urban_rural_ocean"] = df_em_combined_unharmonised["urban"] + df_em_combined_unharmonised["rural"] + df_em_combined_unharmonised["ocean"]
    df_em_combined_unharmonised_World = df_em_combined_unharmonised.groupby("year").agg({"urban": "sum", "rural": "sum", "total_urban_rural_ocean": "sum"}).reset_index()
    df_em_combined_unharmonised_World["region_number"] = 28
    df_em_combined_unharmonised = pd.concat([df_em_combined_unharmonised, df_em_combined_unharmonised_World], ignore_index=True).sort_values(["year", "region_number"]).reset_index(drop=True)
    df_em_combined_unharmonised = pd.merge(df_IAM_EM_unharmonised_compare, df_em_combined_unharmonised, on=["year", "region_number"], how="left", suffixes=("_IAM", "_grid"))
    df_em_combined_unharmonised["ratio_iam_grid"] = df_em_combined_unharmonised["total_iam"]/df_em_combined_unharmonised["total_grid"]
    df_em_combined_unharmonised["ratio_urban_rural_ocean"] = df_em_combined_unharmonised["total_iam"]/df_em_combined_unharmonised["total_urban_rural_ocean"]
    df_em_combined_unharmonised["ratio_ocean"] = df_em_combined_unharmonised["ocean"]/df_em_combined_unharmonised["total_urban_rural_ocean"]
    df_em_combined_unharmonised.to_csv(dir_output / f"Emissions_region_combined_{profile_model_scenario_convergence_year}_unharmonised.csv", index=False, sep=";")

    # 3.2b Harmonised
    debug_log.info(f"\n\n3.2 Aggregate harmonised emissions {"-"*25}")
    df_em_urban_harmonised, df_em_rural_harmonised, df_em_ocean_harmonised = process_urban_grid_emissions.calculate_urban_rural_totals(xr_dataset=xr_em_urban_harmonised, varname=varname_EM, region_varname="region_number")
    df_em_urban_harmonised.to_csv(dir_output / f"Emissions_urban_region_{profile_model_scenario_convergence_year}_harmonised.csv", index=False, sep=";")
    df_em_urban_harmonised["Type"] = "urban"
    df_em_rural_harmonised["Type"] = "rural"
    df_em_ocean_harmonised["Type"] = "ocean"
    # combine dataframes for urban and rural emissions into one dataset
    df_em_combined_harmonised = pd.concat([df_em_urban_harmonised, df_em_rural_harmonised, df_em_ocean_harmonised], ignore_index=True)
    df_em_combined_harmonised.drop(columns=["spatial_ref"], inplace=True, errors="ignore")
    df_em_combined_harmonised["year"] = df_em_combined_harmonised["year"].astype(int)
    df_em_combined_harmonised = df_em_combined_harmonised.pivot(index=["year", "region_number"], columns="Type", values=varname_EM).reset_index()
    df_em_combined_harmonised["total_urban_rural_ocean"] = df_em_combined_harmonised["urban"] + df_em_combined_harmonised["rural"] + df_em_combined_harmonised["ocean"]
    # add World
    df_em_combined_harmonised_World = df_em_combined_harmonised.groupby("year").agg({"urban": "sum", "rural": "sum", "ocean": "sum", "total_urban_rural_ocean": "sum"}).reset_index()
    df_em_combined_harmonised_World["region_number"] = 28
    df_em_combined_harmonised = pd.concat([df_em_combined_harmonised, df_em_combined_harmonised_World], ignore_index=True).sort_values(["year", "region_number"]).reset_index(drop=True)
    # add IAM and grid emissions for comparison
    df_em_combined_harmonised = pd.merge(df_IAM_EM_harmonised_compare, df_em_combined_harmonised, on=["year", "region_number"], how="left", suffixes=("_IAM", "_grid"))
    df_em_combined_harmonised["ratio_iam_grid"] = df_em_combined_harmonised["total_iam"]/df_em_combined_harmonised["total_grid"]
    df_em_combined_harmonised["ratio_urban_rural_ocean"] = df_em_combined_harmonised["total_iam"]/df_em_combined_harmonised["total_urban_rural_ocean"]
    df_em_combined_harmonised["ratio_ocean"] = df_em_combined_harmonised["ocean"]/df_em_combined_harmonised["total_urban_rural_ocean"]
    df_em_combined_harmonised.to_csv(dir_output / f"Emissions_region_combined_{profile_model_scenario_convergence_year}_harmonised.csv", index=False, sep=";")

    # 4. Final checks
    debug_log.info(f"{PRINT_COLORS['green']}Final checks for profile {profile}, scenario {scenario}, model {model} started{PRINT_COLORS['end']}")
    # print global year values for population, GDP, and emissions
    #years = [base_year, 2030, 2040, 2050]
    years_per_ds = []
    for label, ds in [("population", xr_population_processed), ("GDP (PPP)", xr_gdp_ppp_processed), ("emissions", xr_emissions_harmonised)]:
        if ds is not None:
            years_ds = ds["time"].values
            debug_log.info(f"Unique years for {label} variable: {years_ds}")
            years_per_ds.append(set(years_ds.tolist()))
    years_common = sorted(set.intersection(*years_per_ds))
    debug_log.info(f"Common years: {years_common}")
    years = [y for y in years_common if y in [base_year, 2030, 2040, 2050]]

    sum_rows = [(varname_POP, xr_population_downscaled, varname_POP),
                (varname_GDP, xr_gdp_ppp, varname_GDP),
                (varname_EM, xr_emissions, varname_EM),
                (f"{varname_POP}_processed", xr_population_processed, varname_POP),
                (f"{varname_GDP}_processed", xr_gdp_ppp_processed, varname_GDP),
                (f"{varname_EM}_unharmonised", xr_emissions_unharmonised, varname_EM),
                (f"{varname_EM}_harmonised", xr_emissions_harmonised, varname_EM)]

    urban_rows = [(f"{varname_EM}_urban_unharmonised", xr_em_urban_unharmonised, varname_EM),
                (f"{varname_EM}_urban_harmonised", xr_em_urban_harmonised, varname_EM)]

    values = {label: {year: (float(ds[varname].sel(time=year).sum())
                            if ds is not None and year in ds["time"].values else None)
                    for year in years}
            for label, ds, varname in sum_rows}

    values.update({label: {year: (float(ds.sel(time=year)[varname].where(ds.sel(time=year)["urban"] == 1).sum())
                                if ds is not None and year in ds["time"].values else None)
                        for year in years}
                for label, ds, varname in urban_rows})

    values["percent_urban_emissions_unharmonised"] = {
        year: (values[f"{varname_EM}_urban_unharmonised"][year] / values[f"{varname_EM}_unharmonised"][year] * 100
               if values[f"{varname_EM}_urban_unharmonised"][year] is not None and values[f"{varname_EM}_unharmonised"][year]
               else None)
        for year in years}

    values["percent_urban_emissions_harmonised"] = {
        year: (values[f"{varname_EM}_urban_harmonised"][year] / values[f"{varname_EM}_harmonised"][year] * 100
               if values[f"{varname_EM}_urban_harmonised"][year] is not None and values[f"{varname_EM}_harmonised"][year]
               else None)
        for year in years}

    row_order = ([label for label, _, _ in sum_rows] + [label for label, _, _ in urban_rows]
                + ["percent_urban_emissions_unharmonised", "percent_urban_emissions_harmonised"])

    df_comparison = pd.DataFrame({"variable": row_order,
                                **{str(year): [values[label][year] for label in row_order] for year in years}})

    summary_table = tabulate(df_comparison, headers="keys", tablefmt="grid", showindex=False, floatfmt=",.0f", intfmt="")
    debug_log.info(f"{PRINT_COLORS['green']}Summary of population, GDP, and emissions for {profile}-{model}-{scenario}-{convergence_year}-{downscale_SE}{PRINT_COLORS['end']}\n")
    debug_log.info(f"\n\n{PRINT_COLORS['yellow']}{summary_table}{PRINT_COLORS['end']}")
    df_comparison.to_csv(dir_processed / f"Comparison_{profile}_{model}_{scenario}_{downscale_SE}.csv", index=False, sep=";")

    # 5 End code
    debug_log.info(f"{PRINT_COLORS['green']}Logging for profile {profile}, scenario {scenario}, model {model} started{PRINT_COLORS['end']}")
    debug_log.info(f"\n\n2.5 End code {"-"*25}")

    elapsed_time = time.time() - start_time
    # diviede elapsed time into hours, minutes, and seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    debug_log.info(f"\n{PRINT_COLORS["green"]}{profile}-{scenario}-{gross_net}-Conv_year: {convergence_year}-Downscale: {downscale_SE}: Total elapsed time: {hours:,.2f} hours, {minutes:,.2f} minutes, {seconds:,.2f} seconds{PRINT_COLORS["end"]}")

    # cleanup temporary log files if they are empty
    cleanup_empty_logs(log_path)

    # exit code
    try:
        client = get_client()
        client.close()
    except Exception:
        pass

def plot_results(scenario:str = "ELV-SSP2-CP", model:str="IMAGE", profile:str = "default", SSP_base:str="SSP2", convergence_year:int=2050, net_emissions:bool=True, downscale_SE:bool=True, global_min:float|None=None, global_max:float|None=None):
    '''
    Plot results of downscaling for a given scenario, model, and profile.
    1. compare IAM and grid data for population, GDP, and emissions for historical and projected data
       TO DO: (comparison_IAM_grid_{varname}_{profile}_{model}_{scenario}.png)
    2. plot maps for population, GDP, and emissions for historical and projected data print
       (IPAT_summary_{combined_varnames}_{profile}_{model}_{scenario}_cf_{coarse_factor})
       (boxplot_per_region_{varname}_{profile}_{model}_{scenario}.png)
    3. Plot histograms for emissions projections
       (hist_map_{varname_save}_{year}{add_text}.png)
       (map_{varname_EM}_{profile}_{model}_{scenario}_<year>.jpg)
    4a. Plot specific cities/towns
        (map_{varname_EM}_{model}_{profile}_{scenario}_{<city>}_{y}.jpg)
        (city_emissions_{profile}_{model}_{scenario}.png)
    4b. Save emission statistics to CSV
    5. Plot urban and rural emissions for each region and year
    '''

    from shapely.ops import unary_union  # add alongside the other imports

    if profile not in settings_downscaling.SOURCE_PROFILES:
        available = list(settings_downscaling.SOURCE_PROFILES.keys())
        raise ValueError(f"Unknown source profile '{profile}'. Available: {available}")
    else:
        sources = settings_downscaling.SOURCE_PROFILES[profile]

    source_POP = sources["source_POP"]
    version_POP = sources["version_POP"]
    source_GDP = sources["source_GDP"]
    version_GDP = sources["version_GDP"]
    source_EM = sources["source_EM"]
    version_EM = sources["version_EM"]

    varname_GDP = settings_downscaling.varname_GDP
    varname_POP = settings_downscaling.varname_POP
    varname_EM = settings_downscaling.varname_EM
    varname_gdp_per_capita = settings_downscaling.varname_gdp_per_capita

    vars_downscaling = settings_models.models[model]["vars_downscaling"]

    file_model_grid_regions = settings_models.models[model]["file_model_grid_regions"]
    file_IAM_model_region_numbers = settings_models.models[model]["file_IAM_model_region_numbers"]

    project_dir = Path(__file__).parent.parent
    print(f"\nProject directory: {project_dir}")
    gross_net = "net" if net_emissions else "gross"
    model_scenario_convergence_year = f"{model}_{scenario}_conv_year_{convergence_year}_{gross_net}"
    profile_model_scenario_convergence_year = f"{profile}_{model_scenario_convergence_year}"
    downscale_se_str = "downscale_SE" if downscale_SE else "no_downscale_SE"
    dir_output = project_dir / "data" / "output"
    dir_processed = project_dir / "data" / "processed" / f"{profile}_{downscale_se_str}" / model_scenario_convergence_year
    dir_check = dir_processed / "check"
    print(f"Processed data directory: {dir_processed}")
    dir_output = project_dir / "data" / "output"

    log_path = dir_processed / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    debug_log, results_log = init_logging(f"log_plot_{profile}_{model}_{scenario}_{convergence_year}_{downscale_se_str}", str(log_path))

    coarse_factor_POP, coarse_factor_GDP, coarse_factor_EM, res_min_POP, res_min_GDP, res_min_EM = process_grid_data.get_coarsening_factors(population_source=source_POP,gdp_source=source_GDP,emissions_source=source_EM)
    coarse_factor_POP_str = f"{format_factor(coarse_factor_POP)}"
    coarse_factor_GDP_str = f"{format_factor(coarse_factor_GDP)}"
    coarse_factor_EM_str = f"{format_factor(coarse_factor_EM)}"
    print(f"Coarsening factors - Population: {coarse_factor_POP_str}, GDP: {coarse_factor_GDP_str}, Emissions: {coarse_factor_EM_str}")

    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    # READ IN DATA

    # files for processed grid data
    pop_file = dir_processed.parent / f"Population_{source_POP}_{version_POP}_{SSP_base}_cf_{coarse_factor_POP_str}.nc"
    gdp_ppp_file = dir_processed.parent / f"GDP_PPP_{source_GDP}_{version_GDP}_{SSP_base}_cf_{coarse_factor_GDP_str}.nc"
    em_file = dir_processed.parent / f"{replace_punctuation_in_filenames(varname_EM)}_hist_{source_EM}_{version_EM}_{SSP_base}_cf_{coarse_factor_EM_str}.nc"
    pop_processed_file = dir_processed / f"Population_processed_{source_POP}_{version_POP}_{SSP_base}_cf_{coarse_factor_POP_str}.nc"
    gdp_ppp_processed_file = dir_processed / f"GDP_PPP_processed_{source_GDP}_{version_GDP}_{SSP_base}_cf_{coarse_factor_GDP_str}.nc"
    em_harmonised_file = dir_processed / f"{replace_punctuation_in_filenames(varname_EM)}_harmonised_{SSP_base}.nc"
    em_harmonised_urban_file = dir_output /  f"Emissions_urban_region_{scenario}_{profile}_harmonised.nc"
    file_path_file_model_grid_regions = determine_regions_file(project_dir, res_min_POP, res_min_GDP, res_min_EM, model, debug_log)

    figures_dir = dir_processed / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    xr_population_hist = xr.open_dataset(pop_file)
    xr_gdp_ppp_hist = xr.open_dataset(gdp_ppp_file)
    xr_emissions_hist = xr.open_dataset(em_file)
    xr_population_proj = xr.open_dataset(pop_processed_file)
    xr_gdp_ppp_proj = xr.open_dataset(gdp_ppp_processed_file)
    xr_emissions_proj = xr.open_dataset(em_harmonised_file)
    xr_urban_emissions_proj = xr.open_dataset(em_harmonised_urban_file)
    xr_IAM_regions_grid = xr.open_dataset(file_path_file_model_grid_regions)

    arc_seconds_pop, arc_minutes_pop, arc_degrees_pop = process_grid_data.calculate_resolution(xr_population_proj[varname_POP])
    print(f"{PRINT_COLORS["green"]}Resolution: degrees-{arc_degrees_pop:.2f},  minutes-{arc_minutes_pop:.2f}, seconds-{arc_seconds_pop:.2f}{PRINT_COLORS["end"]}")
    arc_seconds_gdp_ppp, arc_minutes_gdp_ppp, arc_degrees_gdp_ppp = process_grid_data.calculate_resolution(xr_gdp_ppp_proj[varname_GDP])
    print(f"{PRINT_COLORS["green"]}Resolution: degrees-{arc_degrees_gdp_ppp:.2f},  minutes-{arc_minutes_gdp_ppp:.2f}, seconds-{arc_seconds_gdp_ppp:.2f}{PRINT_COLORS["end"]}")
    arc_seconds_em, arc_minutes_em, arc_degrees_em = process_grid_data.calculate_resolution(xr_emissions_proj[varname_EM])
    print(f"{PRINT_COLORS["green"]}Resolution: degrees-{arc_degrees_em:.2f},  minutes-{arc_minutes_em:.2f}, seconds-{arc_seconds_em:.2f}{PRINT_COLORS["end"]}")

    dir_urban = project_dir / "data" / "processed" / "DLL"
    path_urban = dir_urban / "urban_classification_years.parquet"
    gdf_urban_classification = gpd.read_parquet(path_urban)

    path_urban_classification = project_dir / "data" / "output" / f"Emissions_urban_region_{scenario}_{profile}_harmonised.nc"
    xr_urban_classification = xr.open_dataset(path_urban_classification, decode_coords="all")
    if "Emissions_CO2_Excl_shipping_aviation_AFOLU" in xr_urban_classification.data_vars:
        xr_urban_classification = xr_urban_classification.drop_vars(varname_EM)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------
    # PLOT

    # 0. Plot comparison global grid and IAM
    conversion_factor_pop = settings_models.models[model]["model_unit_conversions"]["Population"]
    conversion_factor_GDP_PPP = settings_models.models[model]["model_unit_conversions"]["GDP|PPP"]
    conversion_factor_GDP_per_capita = settings_models.models[model]["model_unit_conversions"]["GDP|PPP"]/settings_models.models[model]["model_unit_conversions"]["Population"]
    df_pop_check_global = pd.read_csv(dir_check / f"df_population_{profile}_check.csv", sep=";")
    df_gdp_ppp_check_global = pd.read_csv(dir_check / f"df_gdp_ppp_{profile}_check.csv", sep=";")
    df_gdp_pc_check_global = pd.read_csv(dir_check / f"df_gdp_per_capita_{profile}_check.csv", sep=";")
    #df_pop_check_global["grid_before"] = df_pop_check_global["population_grid"] * conversion_factor_pop
    #df_pop_check_global["grid_after"] = df_pop_check_global["population_grid"] * conversion_factor_pop

    df_check_global = pd.merge(pd.merge(df_pop_check_global, df_gdp_ppp_check_global, on="year"), df_gdp_pc_check_global, on="year", how="outer")
    df_check_global = pd.concat([df_pop_check_global, df_gdp_ppp_check_global, df_gdp_pc_check_global], axis=1)
    df_check_global.to_csv(dir_processed / f"Check_global_{profile}.csv", index=False, sep=";")

    # plot comparison of global population, GDP, and emissions for IAM and grid data
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["tab:blue", "tab:green"]
    axes[0].plot(data=df_check_global[["variable"]==varname_POP], x="year", y=["IAM", "grid_before", "grid_after"], kind="line", color=colors, title="Global Population Comparison")
    axes[1].plot(data=df_check_global[["variable"]==varname_GDP], x="year", y=["IAM", "grid"], kind="line", color=colors, title="Global GDP (PPP) Comparison")
    axes[2].plot(data=df_check_global[["variable"]==varname_gdp_per_capita], x="year", y=["IAM", "grid_before", "grid_after"], kind="line", color=colors, title="Global GDP (PPP) Comparison")
    fig.tight_layout()
    fig.savefig(dir_processed / f"Compare_grid_IAM_global_{profile_model_scenario_convergence_year}.png")

    plot_maps.plot_em_urban_unharmonised_country(project_dir, profile, model, scenario, convergence_year, "JPN",
                                                 xr_urban_emissions_proj, varname_EM, xr_IAM_regions_grid, "country_id_GADM", gdf_urban_classification,
                                                 debug_log)

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

    # 2. plot maps for population, GDP, and emissions for historical and projected data print(f"directory: {project_dir.resolve()}")
    plot = plot_maps.plot_IPAT_summary(dir_processed, f"{profile}_{model}_{scenario}_{convergence_year}_{downscale_se_str}",
                      xr_population_hist, xr_gdp_ppp_hist, xr_emissions_hist,
                      xr_population_proj, xr_gdp_ppp_proj, xr_emissions_proj,
                      varname_POP, varname_GDP, varname_EM,
                      2020, [2030, 2050],1)

    # 3. Plot histograms for emissions projections
    for y in [2020, 2030, 2050]:
        plot_maps.plot_hist(dir_processed, xr_emissions_proj, scenario, varname_EM, "", y)

    plot_maps.plot_boxplot_per_region(project_dir, dir_processed, file_IAM_model_region_numbers,
                                      xr_emissions_proj, varname_EM,
                                      profile, model, scenario, [2020, 2050])

    max_y = 600000
    vmin = 1e-6 # log scale
    vmax = 0.4 # log scale
    plot_maps.plot_hist_map(dir_processed, xr_population_hist, f"_{source_POP}_{scenario}_{downscale_se_str}", varname_POP, 2020,
                            hist_y_max=max_y, vmin=vmin, vmax=vmax)
    plot_maps.plot_hist_map(dir_processed, xr_gdp_ppp_hist, f"_{source_GDP}_{scenario}_{downscale_se_str}", varname_GDP, 2020)
    plot_maps.plot_hist_map(dir_processed, xr_emissions_proj, f"_{profile}_{model}_{scenario}_{convergence_year}_{downscale_se_str}", varname_EM, 2020,
                            hist_y_max=max_y, vmin=vmin, vmax=vmax)
    plot_maps.plot_hist_map(dir_processed, xr_emissions_proj, f"_{profile}_{model}_{scenario}_{convergence_year}_{downscale_se_str}", varname_EM, 2030,
                            hist_y_max=max_y, vmin=vmin, vmax=vmax)
    plot_maps.plot_hist_map(dir_processed, xr_emissions_proj, f"_{profile}_{model}_{scenario}_{convergence_year}_{downscale_se_str}", varname_EM, 2050,
                            hist_y_max=max_y, vmin=vmin, vmax=vmax)

    fig_2020, ax, pm = plot_maps.plot_Mercator_projection(xr_emissions_proj[varname_EM].sel(time=2020),
                                                            ax=None, coarsen=12, transform="linear", show=False, title=None,
                                                            cbar_shrink=0.6, cbar_aspect=20, cbar_pad=0.05)
    fig_2020.savefig(f"{figures_dir}/map_{varname_EM}_{profile}_{model}_{scenario}_2020.jpg", dpi=150, bbox_inches="tight")
    fig_2030, ax, pm = plot_maps.plot_Mercator_projection(xr_emissions_proj[varname_EM].sel(time=2030),
                                                            ax=None, coarsen=12, transform="linear", show=False, title=None,
                                                            cbar_shrink=0.6, cbar_aspect=20, cbar_pad=0.05)
    fig_2030.savefig(f"{figures_dir}/map_{varname_EM}_{profile}_{model}_{scenario}_2030.jpg", dpi=150, bbox_inches="tight")
    fig_2050, ax, pm = plot_maps.plot_Mercator_projection(xr_emissions_proj[varname_EM].sel(time=2050),
                                                            ax=None, coarsen=12, transform="linear", show=False, title=None,
                                                            cbar_shrink=0.6, cbar_aspect=20, cbar_pad=0.05)
    fig_2050.savefig(f"{figures_dir}/map_{varname_EM}_{profile}_{model}_{scenario}_2050.jpg", dpi=150, bbox_inches="tight")

    # 4. Plot specific cities/towns
    cities_towns = ["Amsterdam", "Lima", "Raleigh", "New York"]
    coords = [coord_Amsterdam, coord_Lima, coord_Raleigh, coord_NewYork]

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

    # get city polygon from GADM or US Census TIGER shapefiles
    for city in cities_config:
        names = city.get("sub_cities", [city["name"]])
        polys = []
        for nm in names:
            try:
                if city["within_US"]:
                    poly = convert_GIS.get_us_city_polygon(tiger_dir=dir_US_Census_Tiger, city_name=nm, output_dir=dir_polygons)
                else:
                    poly = convert_GIS.get_city_polygon(GADM_gpkg_path=gadm_gpkg_path, iso3=city["iso3"], city_name=nm, output_dir=dir_polygons)
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
            fig_city_town.savefig(f"{figures_dir}/map_{varname_EM}_{model}_{profile}_{scenario}_{city['name']}_{y}.jpg", dpi=150, bbox_inches="tight")
            plt.close(fig_city_town)

    # 5. Save emission statistics to CSV
    df_emissions_table = pd.DataFrame(emission_records)
    cols = ["city", "year", "scenario", "variable", "sum_weighted", "sum_full", "mean_per_m2", "min", "max", "n_pixels_full", "n_pixels_partial", "n_pixels_any"]
    df_emissions_table = df_emissions_table[cols]
    out_csv = Path(f"{figures_dir}/emissions_per_city_table_{scenario}_{varname_EM}.csv")
    df_emissions_table.to_csv(out_csv, sep=";", index=False)
    df_emissions = df_emissions_table.drop(columns="variable").melt(id_vars=["city", "year", "scenario"], var_name="variable", value_name="value")
    emissions_path = Path(f"{dir_processed}/emissions_per_city_{scenario}_{varname_EM}.csv")
    df_emissions.to_csv(emissions_path, sep=";", index=False)

    # 6. Plot emission trends for each city
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
    plt.savefig(Path(f"{figures_dir}/city_emissions_{profile}_{model}_{scenario}.png"), dpi=150, bbox_inches="tight")

def upload_to_GEE(scenario:str = "ELV-SSP2-CP", model:str="IMAGE", profile:str = "default", SSP_base="SSP2", downscale_SE:bool=True, convergence_year:int=2050, net_emissions:bool=True):

    #GCP_PROJECT = "phrasal-brand-469215-b2"
    GCP_PROJECT = "unique-nebula-467816-n2"
    #EE_ASSET_FOLDER = "projects/phrasal-brand-469215-b2/assets"
    EE_ASSET_FOLDER = "projects/unique-nebula-467816-n2/assets"

    start_time = time.time()
    # script settings
    from downscaling.settings_downscaling import SOURCE_PROFILES

    if profile not in settings_downscaling.SOURCE_PROFILES:
        available = list(settings_downscaling.SOURCE_PROFILES.keys())
        raise ValueError(f"Unknown source profile '{profile}'. Available: {available}")
    else:
        sources = settings_downscaling.SOURCE_PROFILES[profile]

    source_POP = sources["source_POP"]
    version_POP = sources["version_POP"]
    source_GDP = sources["source_GDP"]
    version_GDP = sources["version_GDP"]
    source_EM = sources["source_EM"]
    version_EM = sources["version_EM"]

    varname_GDP = settings_downscaling.varname_GDP
    varname_POP = settings_downscaling.varname_POP
    varname_EM = settings_downscaling.varname_EM

    file_model_grid_regions = settings_models.models[model]["file_model_grid_regions"]
    file_IAM_model_region_numbers = settings_models.models[model]["file_IAM_model_region_numbers"]

    project_dir = Path(__file__).parent
    print(f"Project directory: {project_dir}")
    # source_version_grid = f"{source_POP}_{version_POP}_{source_GDP}_{version_GDP}_{source_EM}_{version_EM}"
    # model_scenario = f"{model}_{scenario}"
    # TO DO: adjust 'source_version_grid', which is not used anymore in the directory names

    gross_net = "net" if net_emissions else "gross"
    downscale_se_str = "downscale_SE" if downscale_SE else "no_downscale_SE"
    model_scenario_convergence_year = f"{model}_{scenario}_conv_year_{convergence_year}_{gross_net}"
    profile_model_scenario_convergence_year = f"{profile}_{downscale_se_str}_{model_scenario_convergence_year}"

    dir_processed = project_dir / "data" / "processed" / f"{profile}_{downscale_se_str}" / model_scenario_convergence_year
    print(f"Processed data directory: {dir_processed}")

    grouping_1 = f"{profile}_{downscale_se_str}"
    grouping_2 = model_scenario_convergence_year

    upload_results_ee.ensure_ee_authenticated()
    upload_results_ee.upload_tifs_years(grouping_1, grouping_2, "all", dir_processed, [2020, 2030, 2050], EE_ASSET_FOLDER)
    upload_results_ee.upload_tifs(grouping_1, dir_processed, EE_ASSET_FOLDER)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n{PRINT_COLORS['green']}Upload to Google Earth Engine complete. Total elapsed time: {elapsed_time:,.2f} seconds ({elapsed_time/60:.2f} minutes).{PRINT_COLORS['end']}")


def compare_scens_raster_files(dirs:list, filename_before_year:str,
                              scenario:str, model:str="IMAGE", profile:str="default", SSP_base:str="SSP2",
                              convergence_year:int=2150, net_emissions:bool=True, downscale_SE:bool=True):

    project_dir = Path(__file__).parent.parent

    gross_net = "net" if net_emissions else "gross"
    downscale_se_str = "downscale_SE" if downscale_SE else "no_downscale_SE"
    model_scenario_convergence_year = f"{model}_{scenario}_conv_year_{convergence_year}_{gross_net}"
    profile_model_scenario_convergence_year = f"{profile}_{model_scenario_convergence_year}"

def _plot_difference_map(dirs:list, filenames:list, varnames: list,year: int, time_dim:str="time",
                         coarsen_factor:int=1, label:str|None=None, vmax:float|None = None, vmax_min: float|None = None, plot_value_type:str="absolute"):

    if plot_value_type not in ("percentage", "absolute"):
        raise ValueError(f"difference_type must be 'percentage' or 'absolute', not '{plot_value_type}'")

    project_dir = Path(__file__).parent.parent

    files = [project_dir / d / filename for d, filename in zip(dirs, filenames)]
    missing = [file for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"These files do not exist: {missing}")

    run_names = [Path(d).parts[Path(d).parts.index("processed") + 1] for d in dirs]

    with xr.open_dataset(files[0]) as ds_1, xr.open_dataset(files[1]) as ds_2:

        # one chunk per time step, so that dask only reads the year that is plotted
        da_1, da_2 = ds_1[varnames[0]].chunk({time_dim: 1}), ds_2[varnames[1]].chunk({time_dim: 1})
        units = da_1.attrs.get("units", "")

        # datetime64 ("M") or cftime objects ("O") have a .dt accessor, otherwise the values are the years themselves
        years_1, years_2 = [da[time_dim].dt.year.values if da[time_dim].dtype.kind in "MO"
                            else da[time_dim].values.astype(int) for da in (da_1, da_2)]

        da_1_year = da_1.isel({time_dim: years_1 == year})
        da_2_year = da_2.isel({time_dim: years_2 == year})
        if da_1_year.sizes[time_dim] == 0 or da_1_year.sizes[time_dim] != da_2_year.sizes[time_dim]:
            raise ValueError(f"Year {year} is not available with the same number of time steps in both files")
        # same time labels, so that the subtraction is done position by position within the year
        da_2_year = da_2_year.assign_coords({time_dim: da_1_year[time_dim].values})

        # one map per file: drop the time dimension, or average if the year has multiple time steps
        if da_1_year.sizes[time_dim] == 1:
            da_1_year, da_2_year = da_1_year.squeeze(time_dim, drop=True), da_2_year.squeeze(time_dim, drop=True)
        else:
            da_1_year, da_2_year = da_1_year.mean(time_dim), da_2_year.mean(time_dim)

        # a coarser grid keeps memory use and plotting time low for global data
        if coarsen_factor > 1:
            window = {dim: coarsen_factor for dim in da_1_year.dims}
            da_1_year = da_1_year.coarsen(window, boundary="trim").mean()
            da_2_year = da_2_year.coarsen(window, boundary="trim").mean()

        if plot_value_type == "percentage":
            # percentage difference relative to file 1, which is not defined where file 1 is zero
            diff = (100 * (da_2_year - da_1_year) / da_1_year.where(da_1_year != 0)).compute()
        else:
            # file 2 minus file 1, in the units of the variable
            diff = (da_2_year - da_1_year).compute()

    var_label = varnames[0] if varnames[0] == varnames[1] else f"{varnames[0]}_vs_{varnames[1]}"

    if plot_value_type == "percentage":
        # the colourbar covers at least -1% to +1%, and more when the differences are larger
        if vmax is None:
            vmax = max(1.0, float(abs(diff).max()))
        cbar_label = f"Difference in {var_label} (% of {run_names[0]})"
        title = f"{label}-{plot_value_type}\n{run_names[1]} relative to {run_names[0]} ({var_label}), {year}"
    else:
        cbar_label = f"Difference in {var_label} ({units})" if units else f"Difference in {var_label}"
        title = f"{label}-{plot_value_type}\n{run_names[1]} minus {run_names[0]} ({var_label}), {year}"

    fig, ax = plt.subplots(figsize=(14, 7))
    # diverging colours centred on zero: red where file 2 is higher, blue where file 2 is lower
    if plot_value_type == "percentage":
        if vmax is None and vmax_min is None:
            diff.plot(ax=ax, cmap="RdBu_r", center=0, cbar_kwargs={"label": cbar_label})
        elif vmax is None and vmax_min is not None:
            diff.plot(ax=ax, cmap="RdBu_r", center=0, vmax=max(vmax_min, float(abs(diff).max())), cbar_kwargs={"label": cbar_label})
        elif vmax is not None and vmax_min is None:
            diff.plot(ax=ax, cmap="RdBu_r", center=0, vmax=vmax, cbar_kwargs={"label": cbar_label})
        elif vmax is not None and vmax_min is not None:
            diff.plot(ax=ax, cmap="RdBu_r", center=0, vmax=max(vmax_min, vmax), cbar_kwargs={"label": cbar_label})
        else:
            print(f"{PRINT_COLORS['yellow']}Warning: unexpected combination of vmax and vmax_min values: vmax={vmax}, vmax_min={vmax_min}. Using default vmax for colourbar.{PRINT_COLORS['end']}")
    else:
        diff.plot(ax=ax, cmap="RdBu_r", center=0, vmax=vmax, cbar_kwargs={"label": cbar_label})
    ax.set_title(title)
    ax.set_aspect("equal")

    save_dir = project_dir / "data" / "check" / "compare"
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f"difference_map_{label}_{plot_value_type}_{run_names[0]}_vs_{run_names[1]}_{var_label}_{year}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

def compare_dirs_netcdf_files(dirs: list, filenames: list, varnames: list, time_dim: str = "time",
                              plot_year=2050, label:str|None=None):

    print(f"{PRINT_COLORS["green"]}Comparing .nc files in directories, label: {label}{PRINT_COLORS["end"]}")
    project_dir = Path(__file__).parent.parent

    dirs_full = [project_dir / d for d in dirs]

    missing = [d for d in dirs_full if not d.is_dir()]
    if missing:
        raise FileNotFoundError(f"These directories do not exist: {missing}")

    files = [d / filename for d, filename in zip(dirs_full, filenames)]
    missing = [file for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"These files do not exist: {missing}")

    run_names = [Path(d).parts[Path(d).parts.index("processed") + 1] for d in dirs]

    list_info = []
    with xr.open_dataset(files[0]) as ds_1, xr.open_dataset(files[1]) as ds_2:

        for file, ds, varname in zip(files, [ds_1, ds_2], varnames):
            if varname not in ds.data_vars:
                raise KeyError(f"Variable '{varname}' does not exist in {file}")
            if time_dim not in ds[varname].dims:
                raise KeyError(f"Dimension '{time_dim}' does not exist for '{varname}' in {file}")

        # one chunk per time step, so that dask only reads the year that is compared
        da_1, da_2 = ds_1[varnames[0]].chunk({time_dim: 1}), ds_2[varnames[1]].chunk({time_dim: 1})

        # the grids have to be identical, otherwise xarray silently aligns on the overlapping coordinates only
        for dim in da_1.dims:
            if dim == time_dim:
                continue
            if dim not in da_2.dims or not da_1[dim].equals(da_2[dim]):
                raise ValueError(f"Dimension '{dim}' is not identical in both files")

        # datetime64 ("M") or cftime objects ("O") have a .dt accessor, otherwise the values are the years themselves
        years_1, years_2 = [da[time_dim].dt.year.values if da[time_dim].dtype.kind in "MO"
                            else da[time_dim].values.astype(int) for da in (da_1, da_2)]

        only_in_0 = sorted(set(years_1.tolist()) - set(years_2.tolist()))
        only_in_1 = sorted(set(years_2.tolist()) - set(years_1.tolist()))
        if only_in_0 or only_in_1:
            print(f"Only in {files[0]}: {only_in_0}")
            print(f"Only in {files[1]}: {only_in_1}")

        for year in sorted(set(years_1.tolist()) & set(years_2.tolist())):
            print(f"{PRINT_COLORS["green"]}year: {year}{PRINT_COLORS["end"]}")
            da_1_year = da_1.isel({time_dim: years_1 == year})
            da_2_year = da_2.isel({time_dim: years_2 == year})
            if da_1_year.sizes[time_dim] != da_2_year.sizes[time_dim]:
                print(f"Different number of time steps for {year}, no comparison result")
                continue
            # same time labels, so that the subtraction is done position by position within the year
            da_2_year = da_2_year.assign_coords({time_dim: da_1_year[time_dim].values})

            valid = da_1_year.notnull() & da_2_year.notnull()
            diff = da_2_year - da_1_year
            stats = xr.Dataset({"mean_1": da_1_year.mean(), "mean_2": da_2_year.mean(),
                                "mean_difference": diff.mean(), "max_difference": diff.max(),
                                "min_difference": diff.min(), "num_differing_pixels": ((diff != 0) & valid).sum(),
                                "total_pixels": valid.sum()})

            # one single compute, so that dask reads every chunk only once for all statistics
            stats = stats.compute()
            info = {key: float(stats[key]) for key in stats.data_vars}
            info["num_differing_pixels"] = int(info["num_differing_pixels"])
            info["total_pixels"] = int(info["total_pixels"])
            if info["total_pixels"] == 0:
                print(f"No comparison result for {year}")
                continue
            info["percentage_differing_pixels"] = 100 * info["num_differing_pixels"] / info["total_pixels"]
            info["year"] = year
            list_info.append(info)

    # save to file in data / check
    var_label = varnames[0] if varnames[0] == varnames[1] else f"{varnames[0]}_vs_{varnames[1]}"
    metrics = [("Mean file 1:", "mean_1", 4), ("Mean file 2:", "mean_2", 4), ("Mean difference:", "mean_difference", 4),
            ("Max difference:", "max_difference", 4), ("Min difference:", "min_difference", 4),
            ("Number of differing pixels:", "num_differing_pixels", 0), ("Total pixels compared:", "total_pixels", 0),
            ("Percentage of differing pixels:", "percentage_differing_pixels", 2)]
    header_label = f"{run_names[0]} vs {run_names[1]} ({var_label})"
    headers = [header_label] + [info["year"] for info in list_info]
    rows = []
    for metric_label, key, decimals in metrics:
        values = [info[key] for info in list_info]
        # zero decimals when the smallest absolute value of the row is higher than 100
        if values and min(abs(value) for value in values) > 100:
            decimals = 0
        rows.append([metric_label] + [format(value, f",.{decimals}f") for value in values])
    table = tabulate(rows, headers=headers, disable_numparse=True, colalign=("left",) + ("right",) * len(list_info))

    save_dir = project_dir / "data" / "check" / "compare"
    save_dir.mkdir(parents=True, exist_ok=True)
    if label is None:
        output_file = save_dir / f"comparison_nc_{run_names[0]}_vs_{run_names[1]}_{var_label}.txt"
    else:
        output_file = save_dir / f"comparison_nc_{run_names[0]}_vs_{run_names[1]}_{var_label}_{label}.txt"
    with output_file.open("w", encoding="utf-8") as f:
        f.write(table + "\n")
    table = tabulate(rows, headers=headers, disable_numparse=True, colalign=("left",) + ("right",) * len(list_info))

    print(table)
    with output_file.open("w", encoding="utf-8") as f:
        f.write(table + "\n")

    _plot_difference_map(dirs=dirs, filenames=filenames, varnames=varnames,
                        year=plot_year, coarsen_factor=10, label=label, vmax=None, vmax_min=1.0, plot_value_type="absolute")
    _plot_difference_map(dirs=dirs, filenames=filenames, varnames=varnames,
                        year=plot_year, coarsen_factor=10, label=label, vmax_min=1.0, plot_value_type="percentage")

def compare_dirs_tif_raster_files(dirs:list, filename_before_year:str, label:str|None=None):

    print(f"{PRINT_COLORS["green"]}Comparing tif files, label: {label}{PRINT_COLORS["end"]}")
    project_dir = Path(__file__).parent.parent

    # iterate through the list of directories and compare the raster files
    dirs_full = [project_dir / d for d in dirs]

    missing = [d for d in dirs_full if not d.is_dir()]
    if missing:
        raise FileNotFoundError(f"These directories do not exist: {missing}")

    pattern = f"{filename_before_year}_[0-9][0-9][0-9][0-9].tif"
    names_0 = {path.name for path in dirs_full[0].glob(pattern)}
    names_1 = {path.name for path in dirs_full[1].glob(pattern)}

    only_in_0 = sorted(names_0 - names_1)
    only_in_1 = sorted(names_1 - names_0)
    if only_in_0 or only_in_1:
        print(f"Only in {dirs_full[0]}: {only_in_0}")
        print(f"Only in {dirs_full[1]}: {only_in_1}")

    list_info = []
    for name in sorted(names_0 & names_1):
        info = compare_two_raster_files([dirs_full[0], dirs_full[1]], name)
        if info is None:
            print(f"No comparison result for {name}")
            continue
        year = Path(name).stem[-4:]
        print(f"{PRINT_COLORS["green"]}year: {year}{PRINT_COLORS["end"]}")
        info["year"] = year
        list_info.append(info)

    # save to file in data / check
    run_names = [Path(d).parts[Path(d).parts.index("processed") + 1] for d in dirs]
    scenario = next((s for s in ("Low", "Medium", "High") if f"_{s} " in filename_before_year), "")
    save_dir = project_dir / "data" / "check" / "compare"
    save_dir.mkdir(parents=True, exist_ok=True)
    if label is None:
        output_file = save_dir / f"comparison_tif_{run_names[0]}_vs_{run_names[1]}_{scenario}_{filename_before_year}.txt"
    else:
        output_file = save_dir / f"comparison_tif_{run_names[0]}_vs_{run_names[1]}_{scenario}_{filename_before_year}_{label}.txt"
    metrics = [("Mean file 1:", "mean_1", ".4f"), ("Mean file 2:", "mean_2", ".4f"),
               ("Mean difference:", "mean_difference", ".4f"), ("Max difference:", "max_difference", ".4f"),
               ("Min difference:", "min_difference", ".4f"),
               ("Number of differing pixels:", "num_differing_pixels", ","),
               ("Total pixels compared:", "total_pixels", ","),
               ("Percentage of differing pixels:", "percentage_differing_pixels", ".2f")]

    header_label = f"{run_names[0]} vs {run_names[1]} ({scenario})"
    headers = [header_label] + [info["year"] for info in list_info]
    rows = [[label] + [format(info[key], fmt) for info in list_info] for label, key, fmt in metrics]
    table = tabulate(rows, headers=headers, disable_numparse=True, colalign=("left",) + ("right",) * len(list_info))

    print(table)
    with output_file.open("w", encoding="utf-8") as f:
        f.write(table + "\n")

def compare_two_raster_files(dirs:list|None=None, filename:Path|None=None):

    project_dir = Path(__file__).parent
    root = tk.Tk()
    root.withdraw()  # Hide the root window, only show the dialogs

    if dirs is None or len(dirs)!=2:
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
            info = process_grid_data.compare_two_raster_files(project_dir, src1, src2)
        list_info = [info]
    elif len(dirs)==2 and filename is not None:
        file_path_1 = Path(dirs[0]) / filename
        file_path_2 = Path(dirs[1]) / filename

        if not file_path_1.exists():
            print(f"File '{file_path_1}' does not exist. Exiting.")
            return
        if not file_path_2.exists():
            print(f"File '{file_path_2}' does not exist. Exiting.")
            return

        print(f"Comparing '{file_path_1}' with '{file_path_2}'...")

        # list_info = []
        # with rasterio.open(file_path_1) as src1, rasterio.open(file_path_2) as src2:
        #     info = process_grid_data.compare_two_raster_files(project_dir, src1, src2)
        #     list_info.append(info)

        with rasterio.open(file_path_1) as src1, rasterio.open(file_path_2) as src2:
            info = process_grid_data.compare_two_raster_files(project_dir, src1, src2)
            total_pixels = src1.width * src1.height * src1.count

        stats = info["difference_stats"]
        return {"file1": str(file_path_1), "file2": str(file_path_2),
                "mean_1": stats["mean_1"], "mean_2": stats["mean_2"],
                "mean_difference": stats["mean"],
                "max_difference": stats["max"], "min_difference": stats["min"],
                "num_differing_pixels": stats["num_different_pixels"], "total_pixels": total_pixels,
                "percentage_differing_pixels": 100 * stats["num_different_pixels"] / total_pixels}


