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

def combine_emissions_output(folder: Path) -> pd.DataFrame:

    path = folder / "Emissions_combined.xlsx"
    # remove Excel file it exists
    if path.exists():
        path.unlink()

    frames = []
    for file in folder.glob("Emissions_*.csv"):
        print(file)
        stem = file.stem
        if stem.startswith("Emissions_urban_region_"):
            coverage, rest = "urban", stem.removeprefix("Emissions_urban_region_")
        else:
            coverage, rest = "total", stem.removeprefix("Emissions_region_")
        scenario, round_col, _, harmonised = rest.rsplit("_", 3)
        round_ = f"{round_col}_round"
        df = pd.read_csv(file, sep=";")
        df = df.rename(columns={"Emissions_CO2_Excl_shipping_aviation_AFOLU": "Emissions_CO2_Excl_shipping_aviation_AFOLU_grid_summed",
                       "time": "year"})
        df.insert(0, "Coverage", coverage)
        df.insert(1, "Scenario", scenario)
        df.insert(2, "Round", round_)
        df.insert(3, "Harmonised", harmonised == "harmonised")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_excel(path, sheet_name="Emissions", index=False)

    return combined

# def run_aggregration_to_urban(model: str, scenario: str, SSP_base: str = "SSP2", rounds: dict[str, str] | None = None):
#     '''
#     Run aggregation of emissions to urban level.
#     Rounds: directories as produced by downscale_emissions in downscaling.py
#             names are defined in settings_downscaling.py
#     Example:
#     rounds = {
#             "first_round": "2UP_GHSL_2024_M3_Wang_version_7_EDGAR_2024_net",
#             "second_round": "2UP_GHSL_2024_M3_Murakami_version_2021_1_EDGAR_2024_net",
#             "third_round": "2UP_GHSL_2024_M3_Murakami_version_2021_1_CEDS_CMIP7_2025_04_18_net",
#             "fourth_round": "Zhuang_version_1_Murakami_version_2021_1_CEDS_CMIP7_2025_04_18_net",
#             "fifth_round": "COMPASS_version_2_COMPASS_version_2_CEDS_CMIP7_2025_04_18_net"
#             }
#     '''
#     project_dir = Path(__file__).parent.resolve()
#     print(f"Project directory: {project_dir}")

#     if rounds is None:
#         print(f"{PRINT_COLORS['yellow']}No rounds specified, using default rounds{PRINT_COLORS['end']}")
#     else:
#         # read in urban classification
#         dir_urban = project_dir / Path("data/processed/DLL")
#         print(f"{PRINT_COLORS["green"]}Reading urban classification data from: {dir_urban / 'urban_classification_years.parquet'}{PRINT_COLORS["end"]}")
#         path_urban = dir_urban / "urban_classification_years.parquet"
#         gdf_urban = gpd.read_parquet(path_urban)

#         for i, r in enumerate(rounds.items()):
#             print(f"Processing {r[0]} ({i+1}/{len(rounds)})...")
#             # read in grid emissions
#             varname_EM = "Emissions_CO2_Excl_shipping_aviation_AFOLU"
#             file_EM = f"Emissions_CO2_Excl_shipping_aviation_AFOLU_harmonised_{SSP_base}.nc"
#             scenario_EM = f"{model}_{scenario}"
#             dir_processed = project_dir / Path("data/processed")
#             path_EM = dir_processed / r[1] / scenario_EM / file_EM

#             print(f"Reading emissions data from: {path_EM}")
#             xr_EM = xr.open_dataset(path_EM, engine="netcdf4")
#             print(f"{xr_EM[varname_EM].attrs['unit']}")
#             print(xr_EM)

#             xr_em_urban, df_em_urban, df_em_rural = downscaling.aggregate_urban_emissions(xr_emissions=xr_EM, gdf_urban_classification=gdf_urban,
#                                                                              emissions_varname=varname_EM,
#                                                                              region_varname="region_number",
#                                                                              final_year=2050)


#             dir_urban_classification = project_dir / Path("data/output")
#             df_em_urban.to_csv(dir_urban_classification / f"Emissions_urban_classification_{scenario_EM}_{r[0]}.csv", index=False, sep=";")
#             df_em_rural.to_csv(dir_urban_classification / f"Emissions_rural_classification_{scenario_EM}_{r[0]}.csv", index=False, sep=";")
#             print(f"Saved aggregated emissions for {r[0]} to output directory.")

#             print(xr_em_urban)
#             print("-------------------------------------")
#             print(df_em_urban.head())

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
    parser.add_argument("--driver", type=str, help="driver for the data (Population, GDP_PPP, Emissions)")
    parser.add_argument("--source", type=str, help="data source (e.g. 2UP, Murakami, EDGAR)")
    parser.add_argument("--version", type=str, help="data version (e.g. version_7, version_2021_1, 2025_04_18)")

    parser.add_argument("--create_GADM_raster", action="store_true", help="create GADM raster file for countries")
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

    parser.add_argument("--global_min", type=str, help="minimum for plot range emissions")
    parser.add_argument("--global_max", type=str, help="maximum for plot range emissions")

    parser.add_argument("--plot", action="store_true", help="plot results")

    parser.add_argument("--upload", action="store_true", help="Upload results to Google Earth Engine")

    parser.add_argument("--compare", action="store_true", help="Compare two raster files")

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
        if arguments.emissions not in ["net", "gross"]:
            parser.error("--emissions requires a value of 'net' or 'gross'")
        elif arguments.emissions == "net":
            net_emissions = True
        else:
            net_emissions = False
        downscaling.downscale_emissions(project_dir, arguments.scenario, arguments.model, arguments.profile, arguments.ssp_baseline, net_emissions)

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
        if arguments.global_min is None or arguments.global_max is None:
            downscaling.plot_results(arguments.scenario, arguments.model, arguments.profile, net_emissions, None, None)
        else:
            downscaling.plot_results(arguments.scenario, arguments.model, arguments.profile, net_emissions, float(arguments.global_min), float(arguments.global_max))
    # TOOLS
    # upload results to Google Earth Engine
    if hasattr(arguments, 'upload') and arguments.upload is True:
        if arguments.model is None or arguments.scenario is None or arguments.profile is None:
            parser.error("--upload requires a --model <model>, --scenario <scenario>, and --profile <profile> to be specified")
        downscaling.upload_to_GEE(arguments.scenario, arguments.model, arguments.profile)
    # compare two raster files
    if hasattr(arguments, 'compare') and arguments.compare is True:
        downscaling.compare_two_raster_files()
    # run urban aggregation
    # if hasattr(arguments, 'run_urban_aggregation') and arguments.run_urban_aggregation is True:
    #     run_aggregration_to_urban(model=arguments.model, scenario="ELV-SSP2-CP", SSP_base="SSP2", rounds=rounds)
    #     combine_emissions_output(project_dir / "data" / "output")

    # if no arguments, print message
    if not any(vars(arguments).values()):
        print("No arguments provided. Use -h or --help for more information.")


