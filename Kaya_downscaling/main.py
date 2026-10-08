import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

import geopandas as gpd
import xarray as xr

import downscaling.downscaling as downscaling
import downscaling.IAM_spatial_model_maps as IAM_maps
import downscaling.read_process_grid_data as process_grid_data
import downscaling.process_urban_grid_emissions as process_urban_grid_emissions
import downscaling.download_ScenarioMIP as download_ScenarioMIP
from tools.general_functions import PRINT_COLORS

"""
Configure GDAL and PROJ data directories for the active Python environment.

The environment root is derived from `sys.executable`, ensuring that both GDAL and PROJ use data files from the same environment (e.g. Pixi/Conda). The paths
to the GDAL and PROJ data folders are then constructed relative to this root(`Library/share/gdal` and `Library/share/proj`).

The environment variables `GDAL_DATA`, `PROJ_LIB`, and `PROJ_DATA` are set so that the underlying native libraries can locate their required resource files.
In addition, `pyproj.datadir.set_data_dir` is used to explicitly direct PROJ tothe correct data directory at runtime.
"""

if __name__ == "__main__":
    '''
    -Process data ('copy' to run folder or 'no_copy)
    python run main.py --process_grid_data --profile first_round --ssp_baseline SSP2
    python run main.py --process_grid_data --profile second_round --ssp_baseline SSP2

    -Create GADM raster for countries
    python run main.py --create_GADM_raster --model IMAGE --resolution 6.00

    -Compare to raster files
    pixi run python main.py --compare

    -Downscaling emissions to grid level
    **********************************************
    INPUT PROFILE
    **********************************************
    'First round' (2UP, Wang, EDGAR)
    'Second round' (2UP, Murakami, EDGAR)
    'Third round' (2UP, Murakami, CEDS_CMIP7)
    'Fourth round' (Zhuang, Murakami, CEDS_CMIP7)
    'Fifth round' (COMPASS, COMPASS, CEDS_CMIP7)

    --Downscale POPULATION
    pixi run python main.py --downscale_population --scenario ELV-SSP2-CP --model IMAGE --profile %profile% --emissions net
    pixi run python main.py --downscale_population --scenario ELV-SSP2-1150F --model IMAGE --profile %profile% --emissions net

    --Downscale NET EMISSIONS
    pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile %profile% --emissions net
    pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile first_round --emissions net
    pixi run python main.py --downscale_emissions --scenario ELV-SSP2-1150F --model IMAGE --profile  %profile% --emissions net

    --Downscale GROSS EMISSIONS
    pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile %profile% --emissions gross
    pixi run python main.py --downscale_emissions --scenario ELV-SSP2-1150F --model IMAGE --profile %profile% --emissions gross

    -Plot results
    python run main.py --plot --scenario ELV-SSP2-CP --model IMAGE --profile %profile% --emissions net
    python run main.py --plot --scenario ELV-SSP2-CP --model IMAGE --profile %profile% --emissions net --global_min 0 --global_max 100

    -Upload results to Google Earth Engine
    python run main.py --upload --scenario ELV-SSP2-CP --model IMAGE --profile %profile%

    '''
    project_dir = Path(__file__).parent.resolve()
    rounds = {
              "first_round": "2UP_GHSL_2024_M3_Wang_version_7_EDGAR_2024_net",
              "second_round": "2UP_GHSL_2024_M3_Murakami_version_2021_1_EDGAR_2024_net",
              "third_round": "2UP_GHSL_2024_M3_Murakami_version_2021_1_CEDS_CMIP7_2025_04_18_net",
              "fourth_round": "Zhuang_version_1_Murakami_version_2021_1_CEDS_CMIP7_2025_04_18_net",
              "fifth_round": "COMPASS_version_2_COMPASS_version_2_CEDS_CMIP7_2025_04_18_net"
              }

    parser = argparse.ArgumentParser(description="Downscaling emissions to grid level") # add_help=True by default
    parser.add_argument("--process_grid_data_profile", action="store_true", help="process datasets based on profile")
    parser.add_argument("--process_grid_data_source", action="store_true", help="process datasets based on source")
    parser.add_argument("--process_urban_classification", action="store_true", help="process urban classification data")
    parser.add_argument("--ssp_baseline", type=str, help="baseline scenario from SSP")
    parser.add_argument("--convergence_year", type=int, help="year of convergence")
    parser.add_argument("--driver", type=str, help="driver for the data (Population, GDP_PPP, Emissions)")
    parser.add_argument("--source", type=str, help="data source (e.g. 2UP, Murakami, EDGAR)")
    parser.add_argument("--version", type=str, help="data version (e.g. version_7, version_2021_1, 2025_04_18)")

    parser.add_argument("--create_GADM_raster", action="store_true", help="create GADM raster file for countries")
    parser.add_argument("--create_GADM_raster_profile", action="store_true", help="create GADM raster file for countries")
    parser.add_argument("--resolution", type=str, help="Resolution for GADM raster in minutes")

    parser.add_argument("--download_regions", action="store_true", help="Download regions from ScenarioMIP")
    parser.add_argument("--download_emissions", action="store_true", help="Download emissions from ScenarioMIP for the specified model (IMAGE_ScenarioMIP or REMIND_ScenarioMIP)")

    parser.add_argument("--downscale_population", action="store_true", help="downscale population")
    parser.add_argument("--downscale_gdp_ppp", action="store_true", help="downscale GDP (PPP)")
    parser.add_argument("--downscale_emissions", action="store_true", help="downscale emissions")
    parser.add_argument("--scenario", type=str, help="Scenario to downscale for (e.g. ELV-SSP2-CP)")
    parser.add_argument("--model", type=str, help="Model from which scenario input is used (e.g. IMAGE, REMIND")
    parser.add_argument("--profile", type=str, help="Settings for input files")
    parser.add_argument("--emissions", type=str, help="net" or "gross")
    parser.add_argument("--downscale_SE", type=str, help="downscale SE data (Population, GDP|PPP, Emissions)")

    parser.add_argument("--global_min", type=str, help="minimum for plot range emissions")
    parser.add_argument("--global_max", type=str, help="maximum for plot range emissions")

    parser.add_argument("--plot", action="store_true", help="plot results")

    parser.add_argument("--upload", action="store_true", help="Upload results to Google Earth Engine")

    parser.add_argument("--compare_choose", action="store_true", help="Compare two raster files")

    parser.add_argument("--compare_tifs", action="store_true", help="Compare two raster files")
    parser.add_argument("--dir1", type=str, help="First directory containing raster files")
    parser.add_argument("--dir2", type=str, help="Second directory containing raster files")
    parser.add_argument("--filename", type=str, help="Name of the raster file to compare")
    parser.add_argument("--label", type=str, help="Label for the comparison output file")

    parser.add_argument("--compare_ncs", action="store_true", help="Compare two raster files")
    parser.add_argument("--filename1", type=str, help="Name of the raster file to compare")
    parser.add_argument("--filename2", type=str, help="Name of the second raster file to compare")

    parser.add_argument("--run_urban_aggregation", action="store_true", help="Run urban aggregation")

    arguments = parser.parse_args()
    print(f"Arguments provided: {arguments}")
    # process
    # 1. Population, GDP and emissions datasets
    # 2. Create GADM raster for IMAGE regions based on GADM shapefile and IMAGE region numbers
    # 3. Process DLL data on urban areas (combine geopandas dataframe with csv dataframe on GDAM_ID)
    if hasattr(arguments, 'process_grid_data_profile') and arguments.process_grid_data_profile is True:
        if arguments.profile is None or arguments.ssp_baseline is None :
            parser.error("--process_grid_data_profile requires a profile and a SSP baseline scenario to be specified with --ssp_base")
        # 1. Pre-process population, GDP and emissions datasets
        downscaling.process_datasets(project_dir, arguments.profile, arguments.ssp_baseline)
    if hasattr(arguments, "process_grid_data_source") and arguments.process_grid_data_source is True:
        if arguments.source is None or arguments.driver is None or arguments.version is None:
            parser.error("--process_grid_data_source requires a source, driver (Population, GDP|PPP, Emissions), and version to be specified")
        if arguments.driver != "Emissions" and arguments.ssp_baseline is None:
            parser.error("--process_grid_data_source with driver Population or GDP|PPP requires a SSP baseline scenario to be specified with --ssp_base")
        # 1. Pre-process population, GDP and emissions datasets
        downscaling.process_one_dataset(project_dir, arguments.driver.replace("_", "|"), arguments.source, arguments.version, arguments.ssp_baseline)

    # ScenaroMIP data download
    if hasattr(arguments, 'download_regions') and arguments.download_regions is True:
        download_ScenarioMIP.download_IAMC_regions()
    if hasattr(arguments, 'download_emissions') and arguments.download_emissions is True:
        if arguments.model is None:
            raise ValueError("Please provide a model name using --model when downloading emissions data.")
        download_ScenarioMIP.download_emissions(arguments.model)

    # 2. Create raster for IMAGE regions based on GADM shapefile and IMAGE region numbers
    if hasattr(arguments, 'create_GADM_raster') and arguments.create_GADM_raster is True:
        if arguments.model is None or arguments.resolution is None:
            parser.error("--Creating GADM raster requires a model and a resolution to be specified with --model and --resolution")
        IAM_maps.create_GADM_region_raster(project_dir, arguments.model, float(arguments.resolution), True)
    if hasattr(arguments, 'create_GADM_raster_profile') and arguments.create_GADM_raster_profile is True:
        if arguments.profile is None or arguments.model is None or arguments.ssp_baseline is None:
            parser.error("--create_GADM_raster_profile requires a profile, model, and SSP baseline to be specified")
        IAM_maps.create_GADM_region_raster_profile(project_dir, arguments.profile, arguments.model, arguments.ssp_baseline, 2020)
    # 3. Process DLL data on urban areas (combine geopandas dataframe with csv dataframe on GDAM_ID)
    if hasattr(arguments, 'process_urban_classification') and arguments.process_urban_classification is True:
        process_urban_grid_emissions.process_urban_classification_data(Path(project_dir), save_gdf=False, plot=False)

    # downscale population, GDP and emissions datasets
    if hasattr(arguments, 'downscale_population') and arguments.downscale_population is True:
        if arguments.scenario is None or arguments.profile is None:
            parser.error("--scenario requires a scenario to be specified and/or --profile requires a profile to be specified")
        downscaling.downscale_SE_data(project_dir, "Population", arguments.scenario, arguments.model, False, arguments.profile, arguments.ssp_baseline)
    if hasattr(arguments, 'downscale_gdp_ppp') and arguments.downscale_gdp_ppp is True:
        if arguments.scenario is None or arguments.profile is None:
            parser.error("--scenario requires a scenario to be specified and/or --profile requires a profile to be specified")
        downscaling.downscale_SE_data(project_dir, "GDP|PPP", arguments.scenario, arguments.model, True, arguments.profile, arguments.ssp_baseline)
    if hasattr(arguments, 'downscale_emissions') and arguments.downscale_emissions is True:
        if arguments.scenario is None or arguments.profile is None or arguments.emissions is None or arguments.emissions is None:
            parser.error("--scenario requires a scenario to be specified and/or --profile requires a profile to be specified")
        if arguments.ssp_baseline is None:
            parser.error("--downscale_emissions requires a SSP baseline scenario to be specified with --ssp_baseline")
        if arguments.convergence_year is None:
            parser.error("--downscale_emissions requires a convergence year to be specified with --convergence_year")
        if arguments.emissions not in ["net", "gross"]:
            parser.error("--emissions requires a value of 'net' or 'gross'")
        elif arguments.emissions == "net":
            net_emissions = True
        else:
            net_emissions = False
        if arguments.downscale_SE is None:
            parser.error("--downscale_emissions requires a yes/no value to be specified for --downscale_SE")
        match arguments.downscale_SE:
            case "yes":
                downscale_SE = True
            case "no":
                downscale_SE = False
            case _:
                parser.error("--downscale_SE requires a value of 'yes' or 'no'")
        downscaling.downscale_emissions(project_dir, arguments.scenario, arguments.model, arguments.profile, arguments.ssp_baseline, int(arguments.convergence_year), net_emissions, downscale_SE)

    # plot results
    if hasattr(arguments, 'plot') and arguments.plot is True:
        if arguments.model is None or arguments.scenario is None or arguments.profile is None:
            if arguments.model is None:
                parser.error("--model requires a model to be specified")
            if arguments.scenario is None:
                parser.error("--scenario requires a scenario to be specified")
            if arguments.profile is None:
                parser.error("--profile requires a profile to be specified")
            parser.error("--model requires a model to be specified and/or --scenario requires a scenario to be specified and/or --profile requires a profile to be specified")
        if arguments.emissions not in ["net", "gross"]:
            parser.error("--emissions requires a value of 'net' or 'gross'")
        elif arguments.emissions == "net":
            net_emissions = True
        else:
            net_emissions = False
        match arguments.downscale_SE:
            case "yes":
                downscale_SE = True
            case "no":
                downscale_SE = False
            case _:
                parser.error("--downscale_SE requires a value of 'yes' or 'no'")
        if arguments.global_min is None or arguments.global_max is None:
            downscaling.plot_results(arguments.scenario, arguments.model, arguments.profile, arguments.ssp_baseline, arguments.convergence_year, net_emissions, downscale_SE, None, None)
        else:
            downscaling.plot_results(arguments.scenario, arguments.model, arguments.profile, arguments.ssp_baseline, arguments.convergence_year, net_emissions, downscale_SE, float(arguments.global_min), float(arguments.global_max))
    # TOOLS
    # upload results to Google Earth Engine
    if hasattr(arguments, 'upload') and arguments.upload is True:
        if arguments.model is None or arguments.scenario is None or arguments.profile is None or arguments.ssp_baseline is None or arguments.convergence_year is None or arguments.downscale_SE is None:
            parser.error("--upload requires a --model <model>, --scenario <scenario>, and --profile <profile> to be specified")
        if arguments.emissions not in ["net", "gross"]:
            parser.error("--emissions requires a value of 'net' or 'gross'")
        elif arguments.emissions == "net":
            net_emissions = True
        else:
            net_emissions = False
        downscaling.upload_to_GEE(arguments.scenario, arguments.model, arguments.profile, arguments.ssp_baseline, arguments.convergence_year, net_emissions, arguments.downscale_SE)

    # compare two raster files
    if hasattr(arguments, 'compare_choose') and arguments.compare_choose is True:
        downscaling.compare_two_raster_files()
    if hasattr(arguments, 'compare_tifs') and arguments.compare_tifs is True:
        if arguments.dir1 is None or arguments.dir2 is None or arguments.filename is None:
            parser.error("--compare_tifs requires --dir1 <directory1>, --dir2 <directory2>, --filename <filename>, and --label to be specified")
        downscaling.compare_dirs_tif_raster_files([arguments.dir1, arguments.dir2], arguments.filename, arguments.label)
    if hasattr(arguments, 'compare_ncs') and arguments.compare_ncs is True:
        if arguments.dir1 is None or arguments.dir2 is None or arguments.filename1 is None or arguments.filename2 is None:
            parser.error("--compare_ncs requires --dir1 <directory1>, --dir2 <directory2>, --filename1 <filename1>, and --filename2 <filename2>,  and --label <label> to be specified")
        downscaling.compare_dirs_netcdf_files([arguments.dir1, arguments.dir2],
                                              [arguments.filename1, arguments.filename2],
                                              ["Emissions_CO2_Excl_shipping_aviation_AFOLU", "Emissions_CO2_Excl_shipping_aviation_AFOLU"],
                                              "time", plot_year=2050,
                                              label=arguments.label)


    # if no arguments, print message
    if not any(vars(arguments).values()):
        print("No arguments provided. Use -h or --help for more information.")



