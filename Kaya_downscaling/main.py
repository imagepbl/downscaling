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
from tools.general_functions import PRINT_COLORS

"""
Configure GDAL and PROJ data directories for the active Python environment.

The environment root is derived from `sys.executable`, ensuring that both GDAL and PROJ use data files from the same environment (e.g. Pixi/Conda). The paths
to the GDAL and PROJ data folders are then constructed relative to this root(`Library/share/gdal` and `Library/share/proj`).

The environment variables `GDAL_DATA`, `PROJ_LIB`, and `PROJ_DATA` are set so that the underlying native libraries can locate their required resource files.
In addition, `pyproj.datadir.set_data_dir` is used to explicitly direct PROJ tothe correct data directory at runtime.
"""

def _plot_selected_cells(project_dir: Path, xr_em_per_capita_selected: xr.Dataset):
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})

    xr_em_per_capita_selected["emissions"].sel(time=2020).plot(ax=ax, x="lon", y="lat", cmap="viridis",
                                                                transform=ccrs.PlateCarree(),
                                                                cbar_kwargs={"label": "Emissions per capita"})

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_title("Selected cells — per capita emissions near threshold, 2020")

    save_path = project_dir / "figures" / "map_emissions_per_capita_selected.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")

def _plot_selected_pathways(project_dir: Path, region_id: int, xr_em_per_capita: xr.Dataset):
    region_data = xr_em_per_capita["emissions"].where(xr_em_per_capita["region_number"] == region_id)

    df_region = (region_data.to_dataframe(name="emissions")
                .dropna(subset=["emissions"])
                .reset_index())

    summary = (df_region.groupby("time")["emissions"]
            .agg(["min", "max", "median"])
            .reset_index())

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(df_region["time"], df_region["emissions"], alpha=0.2, s=10, color="steelblue", label="Individual cells")
    ax.fill_between(summary["time"], summary["min"], summary["max"], alpha=0.15, color="steelblue", label="Min–max range")
    ax.plot(summary["time"], summary["median"], color="darkblue", marker="o", label="Median")

    ax.set_xlabel("Year")
    ax.set_ylabel("Emissions per capita")
    ax.set_title(f"Per capita emissions over time — region {region_id}")
    ax.legend()

    save_path = project_dir / "figures" / f"emissions_per_capita_selected_region_{region_id}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")

def _calc_avg_emissions_per_capita_selected(xr_em_per_capita_selected: xr.Dataset, xr_population: xr.Dataset, arc_minutes: float) -> pd.DataFrame:

    # Align population data to the emissions dataset using nearest neighbor reindexing
    xr_population_aligned = xr_population.reindex_like(xr_em_per_capita_selected, method="nearest", tolerance=arc_minutes / 60 / 2)
    emissions_absolute = xr_em_per_capita_selected["emissions"] * xr_population_aligned
    emissions_absolute = emissions_absolute.rename("emissions_absolute")

    # Calculate total emissions and population per region, andper capita emissions
    emissions_per_region = (emissions_absolute.groupby(xr_em_per_capita_selected["region_number"])
                            .sum())
    population_per_region = (xr_population_aligned.where(~xr_em_per_capita_selected["emissions"].isnull())
                            .groupby(xr_em_per_capita_selected["region_number"])
                            .sum())
    region_per_capita = (emissions_per_region / population_per_region).rename("region_per_capita")

    # add to dataframe
    df_region_totals = (emissions_per_region.to_dataframe(name="total_emissions")
                        .reset_index()
                        .merge(population_per_region.to_dataframe(name="total_population").reset_index(), on="region_number")
                        .merge(region_per_capita.to_dataframe().reset_index(), on="region_number"))

    return df_region_totals

def select_cells_based_on_per_capita(xr_em_per_capita: xr.Dataset, varname: str,
                                        xr_population: xr.Dataset,
                                        per_capita_threshold: float,
                                        perc_select: float = 0.05, arc_minutes: float = 0.5) -> pd.DataFrame:
    # pre: xre_em_per_capita has at least two variables: <varname> and "region_number"
    tolerance = perc_select * per_capita_threshold
    pop_2020 = xr_em_per_capita[varname].sel(time=2020)
    mask = ((pop_2020 >= per_capita_threshold - tolerance)
             & (pop_2020 <= per_capita_threshold + tolerance)).rename("is_match")

    xr_em_per_capita_selected = xr_em_per_capita.where(mask)

    matches_per_region = (mask.assign_coords(region_number=xr_em_per_capita["region_number"])
                          .groupby("region_number")
                          .sum())
    df = (matches_per_region.to_dataframe(name="match_count")
          .reset_index()
          .sort_values("region_number")
          .reset_index(drop=True))
    print(df.to_string(index=False))

    total_per_region = (xr_em_per_capita["region_number"]
                        .groupby(xr_em_per_capita["region_number"])
                        .count()
                        .to_dataframe(name="total_cells")
                        .reset_index())

    df = df.merge(total_per_region, on="region_number")
    df["pct_matching"] = (100 * df["match_count"] / df["total_cells"]).round(1)
    print(df.to_string(index=False))

    _plot_selected_cells(project_dir, xr_em_per_capita_selected)

    # make range with unique regino_numbers from xr_em_per_capita_selected
    unique_region_numbers = xr_em_per_capita_selected["region_number"].values.ravel()
    unique_region_numbers = unique_region_numbers[~np.isnan(unique_region_numbers)].astype(int)
    for region_id in unique_region_numbers:
        _plot_selected_pathways(project_dir, region_id, xr_em_per_capita)

    df_region_totals = _calc_avg_emissions_per_capita_selected(xr_em_per_capita_selected, xr_population, arc_minutes)
    print(df_region_totals.to_string(index=False))

    return df_region_totals

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

def run_aggregration_to_urban(SSP_base: str = "SSP2", rounds: dict[str, str] | None = None):
    '''
    Run aggregation of emissions to urban level.
    Rounds: directories as produced by downscale_emissions in downscaling.py
            names are defined in settings_downscaling.py
    Example:
    rounds = {
            "first_round": "2UP_GHSL_2024_M3_Wang_version_7_EDGAR_2024_net",
            "second_round": "2UP_GHSL_2024_M3_Murakami_version_2021_1_EDGAR_2024_net",
            "third_round": "2UP_GHSL_2024_M3_Murakami_version_2021_1_CEDS_CMIP7_2025_04_18_net",
            "fourth_round": "Zhuang_version_1_Murakami_version_2021_1_CEDS_CMIP7_2025_04_18_net",
            "fifth_round": "COMPASS_version_2_COMPASS_version_2_CEDS_CMIP7_2025_04_18_net"
            }
    '''
    project_dir = Path(__file__).parent.resolve()
    print(f"Project directory: {project_dir}")

    if rounds is None:
        print(f"{PRINT_COLORS['yellow']}No rounds specified, using default rounds{PRINT_COLORS['end']}")
    else:
        # read in urban classification
        dir_urban = project_dir / Path("data/processed/DLL")
        print(f"{PRINT_COLORS["green"]}Reading urban classification data from: {dir_urban / 'urban_classification_years.parquet'}{PRINT_COLORS["end"]}")
        path_urban = dir_urban / "urban_classification_years.parquet"
        gdf_urban = gpd.read_parquet(path_urban)

        for i, r in enumerate(rounds.items()):
            print(f"Processing {r[0]} ({i+1}/{len(rounds)})...")
            # read in grid emissions
            varname_EM = "Emissions_CO2_Excl_shipping_aviation_AFOLU"
            file_EM = f"Emissions_CO2_Excl_shipping_aviation_AFOLU_harmonised_{SSP_base}.nc"
            scenario_EM = "IMAGE_ELV-SSP2-CP"
            dir_processed = project_dir / Path("data/processed")
            path_EM = dir_processed / r[1] / scenario_EM / file_EM

            print(f"Reading emissions data from: {path_EM}")
            xr_EM = xr.open_dataset(path_EM, engine="netcdf4")
            print(f"{xr_EM[varname_EM].attrs['unit']}")
            print(xr_EM)

            xr_em_urban, df_em_urban, df_em_rural = downscaling.aggregate_urban_emissions(xr_emissions=xr_EM, gdf_urban_classification=gdf_urban,
                                                                             emissions_varname=varname_EM,
                                                                             region_varname="region_number",
                                                                             final_year=2050)


            dir_urban_classification = project_dir / Path("data/output")
            df_em_urban.to_csv(dir_urban_classification / f"Emissions_urban_classification_{scenario_EM}_{r[0]}.csv", index=False, sep=";")
            df_em_rural.to_csv(dir_urban_classification / f"Emissions_rural_classification_{scenario_EM}_{r[0]}.csv", index=False, sep=";")
            print(f"Saved aggregated emissions for {r[0]} to output directory.")

            print(xr_em_urban)
            print("-------------------------------------")
            print(df_em_urban.head())

if __name__ == "__main__":
    '''
    -Process data ('copy' to run folder or 'no_copy)
    python run main.py --process copy --ssp_baseline SSP2
    python run main.py --process no_copy --ssp_baseline SSP2

    -Create GADM raster for countries
    python run main.py --create_GADM_raster --resolution 6.00

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
    parser.add_argument("--process", metavar="copy", choices=["copy", "no_copy"], help="process datasets and 'copy' to run folder or 'no_copy'")
    parser.add_argument("--ssp_baseline", type=str, help="baseline scenario from SSP")

    parser.add_argument("--create_GADM_raster", action="store_true", help="create GADM raster file for countries")
    parser.add_argument("--resolution", type=str, help="Resolution for GADM raster in minutes")

    parser.add_argument("--plot", action="store_true", help="plot results")

    parser.add_argument("--downscale_population", action="store_true", help="downscale population")
    parser.add_argument("--downscale_gdp_ppp", action="store_true", help="downscale GDP (PPP)")
    parser.add_argument("--downscale_emissions", action="store_true", help="downscale emissions")
    parser.add_argument("--scenario", type=str, help="Scenario to downscale for (e.g. ELV-SSP2-CP)")
    parser.add_argument("--model", type=str, help="Model from which scenario input is used (e.g. IMAGE, REMIND")
    parser.add_argument("--profile", type=str, help="Settings for input files")
    parser.add_argument("--emissions", type=str, help="net" or "gross")

    parser.add_argument("--global_min", type=str, help="minimum for plot range emissions")
    parser.add_argument("--global_max", type=str, help="maximum for plot range emissions")

    parser.add_argument("--upload", action="store_true", help="Upload results to Google Earth Engine")

    parser.add_argument("--compare", action="store_true", help="Compare two raster files")

    parser.add_argument("--run_urban_aggregation", action="store_true", help="Run urban aggregation")

    arguments = parser.parse_args()
    print(f"Arguments provided: {arguments}")
    if hasattr(arguments, 'process') and arguments.process is not None:
        if arguments.ssp_baseline is None:
            parser.error("--processing requires a SSP baseline scenario to be specified with --ssp_base")
        # Pre-process population, GDP and emissions datasets
        downscaling.process_datasets(project_dir, arguments.ssp_baseline)
        # Create raster for IMAGE regions based on GADM shapefile and IMAGE region numbers
        IAM_maps.create_GADM_region_raster(project_dir, "IMAGE", float(arguments.resolution), True)
        # process DLL data on urban areas (combine geopandas dataframe with csv dataframe on GDAM_ID)
        process_grid_data.process_urban_classification_data(project_dir)
    if hasattr(arguments, 'downscale_population') and arguments.downscale_population is True:
        if arguments.scenario is None or arguments.profile is None:
            parser.error("--scenario requires a scenario to be specified and/or --profile requires a profile to be specified")
        downscaling.downscale_SE_data(project_dir, "Population", arguments.scenario, arguments.model, False, arguments.profile)
    if hasattr(arguments, 'downscale_gdp_ppp') and arguments.downscale_gdp_ppp is True:
        if arguments.scenario is None or arguments.profile is None:
            parser.error("--scenario requires a scenario to be specified and/or --profile requires a profile to be specified")
        downscaling.downscale_SE_data(project_dir, "GDP|PPP", arguments.scenario, arguments.model, True, arguments.profile)
    if hasattr(arguments, 'downscale_emissions') and arguments.downscale_emissions is True:
        if arguments.scenario is None or arguments.profile is None or arguments.emissions is None:
            parser.error("--scenario requires a scenario to be specified and/or --profile requires a profile to be specified")
        if arguments.emissions not in ["net", "gross"]:
            parser.error("--emissions requires a value of 'net' or 'gross'")
        elif arguments.emissions == "net":
            net_emissions = True
        else:
            net_emissions = False
        downscaling.downscale_emissions(project_dir, arguments.scenario, arguments.model, arguments.profile, net_emissions)
    if hasattr(arguments, 'plot') and arguments.plot is True:
        if arguments.scenario is None or arguments.profile is None:
            parser.error("--scenario requires a scenario to be specified and/or --profile requires a profile to be specified")
        if arguments.emissions not in ["net", "gross"]:
            parser.error("--emissions requires a value of 'net' or 'gross'")
        elif arguments.emissions == "net":
            net_emissions = True
        else:
            net_emissions = False
        if arguments.global_min is None or arguments.global_max is None:
            downscaling.plot_results(arguments.scenario, "IMAGE", arguments.profile, net_emissions, None, None)
        else:
            downscaling.plot_results(arguments.scenario, "IMAGE", arguments.profile, net_emissions, float(arguments.global_min), float(arguments.global_max))
    if hasattr(arguments, 'upload') and arguments.upload is True:
        if arguments.scenario is None or arguments.profile is None:
            parser.error("--upload requires a scenario to be specified and/or --profile requires a profile to be specified")
        downscaling.upload_to_GEE(arguments.scenario, "IMAGE", arguments.profile)
    if hasattr(arguments, 'compare') and arguments.compare is True:
        downscaling.compare_two_raster_files()
    if hasattr(arguments, 'run_urban_aggregation') and arguments.run_urban_aggregation is True:
        run_aggregration_to_urban("SSP2", rounds)
        combine_emissions_output(project_dir / "data" / "output")
    # if no arguments, print message
    if not any(vars(arguments).values()):
        print("No arguments provided. Use -h or --help for more information.")

