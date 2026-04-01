import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import rasterio
from rasterio import DatasetReader
import xarray as xr

from typing import Tuple, Optional

from tools.functions_logging import init_logging
from tools.process_GDP import use_gdpuc
from tools.general_functions import PRINT_COLORS
from downscaling.read_process_grid_data import print_info_rasterio

local_log, dummy_log = init_logging("log", "log/reading_data")

model_unit_conversions = {"IMAGE": {"Emissions|CO2": 1e6,  # Mt to t
                                    "GDP|MER": 1e9,        # billion to 1
                                    "GDP|PPP": 1e9,        # billion to 1
                                    "Population": 1e6      # million to 1
                                   },
                          }

# **************************GENERAL*******************************************
def create_ISO_file_IMAGE():
    # Read in GADM ISO codes raster
    dir_GADM_raster = "K:/PythonWork/Downscaling/data/output"
    file_GADM_raster = "iso_codes_raster_0_50.tif"
    filepath_GADM_raster = f"{dir_GADM_raster}/{file_GADM_raster}"
    with rasterio.open(filepath_GADM_raster) as src:
       print_info_rasterio(src)

def get_IAM_region_info(model="IMAGE"):
    filename_region_grid = ""
    with open("downscaling/settings_data_locations.json", "r") as f:
        data_files = json.load(f)
        filename_region_grid = data_files["IAM"][model]["file_IAM_regions_grid"]
        file_IAM_model_region_numbers = data_files["IAM"][model]["file_IAM_model_region_numbers"]

    return filename_region_grid, file_IAM_model_region_numbers

def read_grid_info_IAM_regions(project_dir:Path, model:str, filename_region_grid:str, file_IAM_model_region_numbers:str, chunk:int, log: logging.Logger=local_log) -> Tuple[xr.Dataset, dict]:
    '''
    Input: netcdf file with region definitions; depending on model it includes region numbers, if not, they should be created
    Output: xarray dataset with region codes (same as IAMC template) and region numbers (type int) as data variables
            The number is zero for OCEAN
    '''

    # init
    xr_IAM_regions = xr.Dataset({"region_number": (["index"], np.array([], dtype="U"))})
    xr_IAM_regions_processed = xr.Dataset({"region_number": (["index"], np.array([], dtype="U"))})
    region_mapping = {}

    if not filename_region_grid == "":
        match model:
            case "IMAGE":
                xr_IAM_regions = xr.open_dataset(filename_region_grid, decode_coords="all")
                # Add region codes to IAMC region codes
                csv_path = project_dir / f"data/input/models/{model}/{file_IAM_model_region_numbers}"
                IMAGE_regions = pd.read_csv(csv_path, sep=",")
                region_mapping = dict(zip(IMAGE_regions["IMAGE number"], IMAGE_regions["IMAGE region"]))
                log.info(region_mapping)

                # process dataset
                xr_IAM_regions_processed = xr_IAM_regions.copy()
                xr_IAM_regions_processed = xr_IAM_regions_processed.rename({"GREG": "region_number"})
                xr_IAM_regions_processed = xr_IAM_regions_processed.isel(time=0, drop=True)
                xr_IAM_regions_processed = xr_IAM_regions_processed.rename({"longitude": "x", "latitude": "y"})
                xr_IAM_regions_processed = xr_IAM_regions_processed.chunk({'y': chunk, 'x': chunk})

                # Change Greenland (region 27) to region 11 (Canada)
                xr_IAM_regions_processed['region_number'] = xr_IAM_regions_processed['region_number'].where(xr_IAM_regions_processed['region_number']!=27, 11)

                # Add OCEAN region
                region_mapping[0] = "OCEAN"
                region_mapping = dict(sorted(region_mapping.items()))
                log.info(xr_IAM_regions.data_vars)
                log.info(f"Original data variables: {xr_IAM_regions.data_vars}")
                xr_IAM_regions_processed["region_number"] = xr_IAM_regions_processed["region_number"].where(~np.isnan(xr_IAM_regions_processed["region_number"]), 0)
                xr_IAM_regions_processed["region_number"] = xr_IAM_regions_processed["region_number"].astype(int)

                # Add region code (used in IAMC template)
                vectorized_map = np.vectorize(region_mapping.get)
                xr_IAM_regions_processed['region_code'] = (xr_IAM_regions_processed['region_number'].dims, vectorized_map(xr_IAM_regions_processed['region_number'].values))

                log.info("Region numbers and codes in IAM regions dataset:")
                log.info(np.unique(xr_IAM_regions_processed["region_number"]))
                log.info(np.unique(xr_IAM_regions_processed["region_code"]))

    return xr_IAM_regions_processed, region_mapping

def read_IAM_regions_data(project_dir: Path, model:str, scenario:str, regions_mapping:dict, region_World:str="World") -> pd.DataFrame:
    # Scenario name should be the name of the IAMC template Excel file


    # Read IAM data
    excel_path = project_dir / f"data/input/models/{model}/SSP/{scenario}.xlsx"
    df_IAM = pd.read_excel(excel_path, sheet_name="data")
    df_IAM = df_IAM[df_IAM["Region"]!="World"]
    df_IAM = df_IAM.melt(id_vars=["Model", "Scenario", "Region", "Variable", "Unit"], var_name="Year", value_name="Value")
    df_IAM["Year"] = df_IAM["Year"].astype(int)

    # Process GDP
    df_IAM = process_GDP(model, df_IAM, "GDP|PPP")

    # Add region numbers
    df_IAM["Region_number"] = df_IAM["Region"].map({v: k for k, v in regions_mapping.items()})
    df_IAM["Region_number"] = df_IAM["Region_number"].astype(int)
    df_IAM.rename(columns={"Region": "Region_code"}, inplace=True)

    return df_IAM

def get_regions(project_dir:Path, model:str, file_IAM_model_region_numbers:str) -> Tuple[pd.DataFrame, dict]:
    csv_path = project_dir / f"data/input/models/{model}/{file_IAM_model_region_numbers}"
    regions = pd.read_csv(csv_path, sep=",")
    regions_mapping = {}
    if model == "IMAGE":
        regions_mapping = dict(zip(regions["IMAGE number"], regions["IMAGE region"]))

    return regions, regions_mapping

def read_process_IAM_data(project_dir:Path, scenario:str, model:str, file_IAM_model_region_numbers:str, vars_downscaling:list, log: logging.Logger=local_log):
    '''
    IAM results must be in IAMC format with region codes
    The years must cover the downscaling years (flexible, but default is [2020, 2025, 2030, 2035, 2040, 2045, 2050, 2060, 2070, 2080, 2090, 2100])
    Abundant years will be deleted, missing years will be (linearly) interpolated
    '''

    regions, regions_mapping = get_regions(project_dir, model, file_IAM_model_region_numbers)

    # Read IAM data
    df_IAM = read_IAM_regions_data(project_dir, model, scenario, regions_mapping, region_World="World")
    df_IAM = df_IAM[df_IAM["Variable"].isin(vars_downscaling)].reset_index(drop=True)
    #df_IAM.to_csv("data/check/IAM_data_before.csv", index=False, sep=";")
    #df_IAM = df_IAM[df_IAM["Variable"].str.contains("Emissions", case=False)].reset_index(drop=True)
    #df_IAM.to_csv("data/check/IAM_data_after.csv", index=False, sep=";")
    df_IAM.columns = [col.lower() for col in df_IAM.columns]

    return df_IAM

def extrapolate_IAM_values_to_convergence_year(dir_procesed: Path, df:pd.DataFrame, conversion_factor:float, convergence_year: int, method: int)->pd.DataFrame:
    # Extends dataframe by extrapolating value with a fixed growth rate per region.
    # Calculate growth rate per region from the last two time steps
    # Method 1: growth rate from last two time steps
    # Method 2: zero growth rate
    # Method 3: growth rate to reach almost zero in convergence year
    # Method 4: absolute growth rate from last two time steps

    df.columns = [col.lower() for col in df.columns]
    varname = df["variable"].unique()[0]

    max_year = df["year"].max()
    second_max_year = df["year"].unique()[-2]

    model = df["model"].unique()[0]
    scenario = df["scenario"].unique()[0]
    variable = df["variable"].unique()[0]
    unit = df["unit"].unique()[0]

    # Get growth rates for each region
    timestep = 10  # your timestep
    rel_growth_rates = {}
    abs_growth_rate = {}
    for region in df["region_number"].unique():
        last_value = df[(df["region_number"] == region) & (df["year"] == max_year)]["value"].iloc[0]
        second_last_value = df[(df["region_number"] == region) & (df["year"] == second_max_year)]["value"].iloc[0]

        # Calculate percentage change
        match method:
            case 1:
                # Method 1
                rel_growth_rates[region] = (last_value / second_last_value)**(1/timestep) - 1
            case 2:
                # Method 2:
                rel_growth_rates[region] = 0
            case 3:
                # Method 3:
                period = convergence_year - max_year
                if (last_value < 0):
                    rel_growth_rates[region] = 0
                else:
                    rel_growth_rates[region] = (0.1/last_value)**(1/timestep)-1 # not solvebale for zero values
            case 4:
                # Method 4:
                abs_growth_rate[region] = (last_value-second_last_value) / timestep
            case _:
                raise ValueError("Invalid method. Choose 1, 2, or 3.")
                df_extended = pd.DataFrame()
                exit()

    # check
    varname_save = varname.replace("&", "and").replace("|", "_").replace(".", " ")
    #Path(project_dir, "data/check").mkdir(parents=True, exist_ok=True)
    csv_path = dir_procesed / f"growth_rates_iam_image_{varname_save}.csv"
    pd.DataFrame(list(rel_growth_rates.items()), columns=["region", "growth_rate"]).to_csv(csv_path, index=False, sep=";")

    # Generate new rows
    new_rows = []

    for year in range(max_year + timestep, convergence_year + 1, timestep):
        for region_number in df["region_number"].unique():
            region_number = df[df["region_number"]==region_number]["region_number"].unique()[0]
            region_code = df[df["region_number"]==region_number]["region_code"].unique()[0]

            last_value = df[df["region_number"] == region_number]["value"].iloc[-1]

            # Apply compound growth
            if method in [1, 2, 3]:
                new_value = last_value * ((1 + rel_growth_rates[region_number]) ** (year-max_year))
            else: # method 4
                new_value = last_value + abs_growth_rate[region_number] * (year - max_year)

            new_rows.append({"model": model, "scenario":scenario, "region_code": region_code, "region_number": region_number, "year": year, "variable": variable, "value": new_value, "unit": unit})

    # Combine original and new data
    df_extended = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df_extended = df_extended.sort_values(by=["region_number", "year"]).reset_index(drop=True)

    df_extended["value"] *= conversion_factor

    return df_extended

# **************************GDP*******************************************

def process_GDP(model:str, df:pd.DataFrame, var:str) -> pd.DataFrame:
    # Read the JSON file
    with open("downscaling/settings_models.json", "r") as f:
        data = json.load(f)
        factor_GDP_PPP = data[model]["factor_GDP_PPP"]
        factor_year_from = data[model]["factor_year_from"]
        factor_year_to = data[model]["factor_year_to"]
    #with open("downscaling/settings_data_locations.json", "r") as f:
    #    data = json.load(f)
    #    R_SCRIPT_PATH = data["general"]["R_SCRIPT_PATH"]
    R_SCRIPT_PATH = r"C:\Program Files\R\R-4.1.1\bin\Rscript.exe"

    # Update GDP|PPP to $2005 dollars
    df_IAM = df.copy()
    mask_update_gdp_ppp = (df["Variable"] == "GDP|PPP") #& (df["Unit"] == "billion USD_2010/yr")
    df_IAM.loc[mask_update_gdp_ppp, "Value"] /= factor_GDP_PPP
    df_IAM.loc[mask_update_gdp_ppp, "Unit"] = df_IAM.loc[mask_update_gdp_ppp, "Unit"].str.replace(f"{factor_year_from}", f"{factor_year_to}")

    # TO DO --> add region/country mapping and calculate conversion with GDPuc
    # df_test = pd.DataFrame({
    #     'country': ['USA', 'CAN', 'MEX', 'FRA', 'DEU'],
    #     'year': [2000, 2000, 2000, 2000, 2000],
    #     'value': [10000, 8000, 5000, 7000, 6000]
    # })
    # result = use_gdpuc(df_test,  R_SCRIPT_PATH, "constant 2010 Int$PPP", "constant 2005 Int$PPP")
    # print(f"GDPuc conversion result:\n{result}")

    return df_IAM

# **************************EM*******************************************

def list_emissions_excl_ship_av_AFOLU():
    list_var_emissions = ["Emissions|CO2|Energy|Supply",
                          "Emissions|CO2|Energy|Demand",
                          "Gross Emissions|CO2|Energy|Supply",
                          "Gross Emissions|CO2|Energy|Demand",
                          "Emissions|CO2|Energy|Demand|Residential and Commercial",
                          "Emissions|CO2|Energy|Demand|Transportation",
                          "Emissions|CO2|Energy|Demand|Industry",
                          "Emissions|CO2|Energy|Demand|AFOFI",
                          "Emissions|CO2|Industrial Processes",
                          "Emissions|CO2|Energy|Demand|Transportation|Domestic Aviation",
                          "Emissions|CO2|Energy|Demand|Transportation|Domestic Shipping",
                          #"Emissions|CO2|Energy|Demand|Bunkers|International Shipping", # in ELV-SSP2-CP and ELV-SSP2-1150F there are only global international shipping emissions
                          "Emissions|CO2|Energy|Demand|Bunkers|International Aviation"]

    return list_var_emissions

def process_EM_regions_data(df:pd.DataFrame, years_downscaling:list, varname_dataset:str, net_emissions:bool=True, model:str="IMAGE", log: logging.Logger=local_log) -> pd.DataFrame:
    # Select CO2 emissions, excluding bunkers and domestic aviation/shipping and AFOLU CO2
    #varname_CO2_excl_ship_av_AFOLU = "Emissions|CO2|Excl. shipping, aviation, AFOLU"
    # vars_CO2_excl_ship_av_AFOLU = ["Emissions|CO2",
    #                                "Emissions|CO2|Energy|Demand|Bunkers|International Aviation", "Emissions|CO2|Energy|Demand|Bunkers|International Shipping",
    #                                "Emissions|CO2|Energy|Demand|Transportation|Domestic Aviation", "Emissions|CO2|Energy|Demand|Transportation|Domestic Shipping",
    #                                "Emissions|CO2|AFOLU"]

    df_IAM_projection_em_downscaling = pd.DataFrame()

    if model == "IMAGE":
        vars_CO2_excl_ship_av_AFOLU = list_emissions_excl_ship_av_AFOLU()
        log.info(f"Variables for CO2 emissions excluding shipping, aviation, and AFOLU: {vars_CO2_excl_ship_av_AFOLU}")

        # check which variables from var_IAM_projectoin_CO2 are not in the variables
        missing_vars = [var for var in vars_CO2_excl_ship_av_AFOLU if var not in df["variable"].unique()]
        if missing_vars:
            print(f"{PRINT_COLORS["red"]}Warning: The following variables are missing in the IAM projection data: {missing_vars}{PRINT_COLORS["end"]}")

        df_CO2_breakdown = df[df["variable"].isin(vars_CO2_excl_ship_av_AFOLU)]
        if net_emissions:
            unit_CO2 = df_CO2_breakdown[df_CO2_breakdown["variable"]=="Emissions|CO2|Energy|Supply"]["unit"].unique()[0]
            print(f"Unit CO2 emissions: {unit_CO2}")
        else:
            unit_CO2 = df_CO2_breakdown[df_CO2_breakdown["variable"]=="Gross Emissions|CO2|Energy|Supply"]["unit"].unique()[0]
            print(f"Unit gross CO2 emissions: {unit_CO2}")

        # process CO2 emissions shipping and aviation
        # mask_CO2_int_shipping = ((df_CO2_breakdown["region_code"] != "World") & (df_CO2_breakdown["variable"] == "Emissions|CO2|Energy|Demand|Bunkers|International Shiping"))
        # df_CO2_breakdown.loc[mask_CO2_int_shipping, "value"] = 0

        # mask_CO2_int_aviation = ((df_CO2_breakdown["region_code"] != "World") & (df_CO2_breakdown["variable"] == "Emissions|CO2|Energy|Demand|Bunkers|International Aviation"))
        # df_CO2_breakdown.loc[mask_CO2_int_aviation, "value"] = 0

        # Collect CO2 indicators for regional IAM  projections
        index_cols = ["model", "scenario", "region_code"] + (["region_number"] if "region_number" in df_CO2_breakdown.columns else []) + ["year"]
        df_CO2_excl_ship_av_AFOLU = df_CO2_breakdown.pivot(index=index_cols ,columns="variable",values="value").fillna(0).reset_index()
        df_CO2_excl_ship_av_AFOLU.to_csv("data/check/vars_CO2_excl_ship_av_AFOLU.csv", index=False)

        if net_emissions:
            df_CO2_excl_ship_av_AFOLU[varname_dataset] = (df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Energy|Supply"] +
                                                          df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Energy|Demand"] +
                                                          df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Industrial Processes"] -
                                                          df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Energy|Demand|Transportation|Domestic Aviation"] -
                                                          df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Energy|Demand|Transportation|Domestic Shipping"]-
                                                          df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Energy|Demand|Bunkers|International Aviation"] #-
                                                          #df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Energy|Demand|Bunkers|International Shipping"] # in ELV-SSP2-CP and ELV-SSP2-1150F there are only global international shipping emissions
                                                          )
            # df_CO2_excl_ship_av_AFOLU[varname_dataset] = (df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Energy|Supply"] +
            #                                               df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Energy|Demand|Industry"] +
            #                                               df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Industrial Processes"] +
            #                                               df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Energy|Demand|Residential and Commercial"] +
            #                                               df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Energy|Demand|Transportation"])
        else: # gross emissions
            df_CO2_excl_ship_av_AFOLU[varname_dataset] = (df_CO2_excl_ship_av_AFOLU["Gross Emissions|CO2|Energy|Supply"] +
                                                          df_CO2_excl_ship_av_AFOLU["Gross Emissions|CO2|Energy|Demand"] +
                                                          df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Industrial Processes"] -
                                                          df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Energy|Demand|Transportation|Domestic Aviation"] -
                                                          df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Energy|Demand|Transportation|Domestic Shipping"] -
                                                          df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Energy|Demand|Bunkers|International Aviation"] #-
                                                          #df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Energy|Demand|Bunkers|International Shipping"] # in ELV-SSP2-CP and ELV-SSP2-1150F there are only global international shipping emissions
                                                          )

            # df_CO2_excl_ship_av_AFOLU[varname_dataset] = (df_CO2_excl_ship_av_AFOLU["Gross Emissions|CO2|Energy|Supply"] +
            #                                               df_CO2_excl_ship_av_AFOLU["Gross Emissions|CO2|Energy|Demand|Industry"] +
            #                                               df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Industrial Processes"] +
            #                                               df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Energy|Demand|Residential and Commercial"] +
            #                                               df_CO2_excl_ship_av_AFOLU["Emissions|CO2|Energy|Demand|Transportation"])


        df_CO2_excl_ship_av_AFOLU = df_CO2_excl_ship_av_AFOLU.melt(id_vars=index_cols, value_vars=[varname_dataset], var_name="variable", value_name="value")
        df_CO2_excl_ship_av_AFOLU["year"] = df_CO2_excl_ship_av_AFOLU["year"].astype(int)
        df_CO2_excl_ship_av_AFOLU["unit"] = unit_CO2

        # check sum of regions in the year 2020
        print("Check IMAGE CO2 emissions 2020")
        value_CO2_2020_projections = df_CO2_excl_ship_av_AFOLU[df_CO2_excl_ship_av_AFOLU["year"]==2020].groupby(["model", "scenario", "year", "variable", "unit"]).sum().reset_index()["value"]
        print(f"Emissions 2020: {value_CO2_2020_projections.iloc[0]:,.0f}")

        df_IAM_projection_em_downscaling = df_CO2_excl_ship_av_AFOLU.copy()
        df_IAM_projection_em_downscaling.columns = [col.lower() for col in df_IAM_projection_em_downscaling.columns]
        df_IAM_projection_em_downscaling = df_IAM_projection_em_downscaling[df_IAM_projection_em_downscaling["year"].isin(years_downscaling)]

    return df_IAM_projection_em_downscaling
