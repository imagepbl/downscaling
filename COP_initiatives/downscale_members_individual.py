from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from openpyxl import load_workbook
import pickle
import sys
import traceback
from typing import Union

DIR = Path(__file__).resolve().parent

# https://stackoverflow.com/questions/78849054/when-using-matplotlib-to-display-graphics-graphics-are-unresponsive-and-the-err
import matplotlib
matplotlib.use('TkAgg')
plt.ion()

# import PBL/own classes/modules
print(f"Importing PBL modules from {DIR.parent}")
from ..IMAGE_tools import settings as IMAGE_settings
from ..IMAGE_tools.IMAGE_regions_settings import IMAGE_regions_nr2ISO, IMAGE_regions_ISO2nr, N_to_N2O
from ..IMAGE_tools import read_process_IMAGE_data as image_data
from ..downscaling import downscale_tool as dt
from ..historical_data import ISO, read_process_historical_data as hist_data
from ..downscaling import log_downscaling_members as log

# global variables
IMAGE_START_YEAR = 1971
#HIST_YEAR = 2020
GWP = []
log_dataframes = []

#output_dir = ""
current_dir = Path().cwd()
error_list = []

def _set_global_variables(Settings, GWP_CH4, GWP_N2O) -> None:
    global IMAGE_PROJECT, IMAGE_START_YEAR, HIST_YEAR_URBAN

    IMAGE_START_YEAR = Settings["IMAGE_START_YEAR"]
    #HIST_YEAR = Settings["HIST_YEAR"]
    GWP.append(GWP_CH4) # CH4
    GWP.append(GWP_N2O) # N2O


def _read_hist_variable(initiative_name: str, var_name_hist: str, hist_year:int, read_original=True) -> pd.DataFrame:
    """
    initiative_name: Methane	Coal	Transport_road	Deforestation	Bunkers_aviation	BT_Power	BT_Road	Steel

    Description: reads in the historical data for the IMAGE variable based on the information in info
    pre: info csv file with TIMER to IEA translation
          and row i
    post: dataframe with historical data and the following columns names:  ["Name", "ISO3", "Country", "Year", "Value", "Unit"]
    """

     #  if file exists, read csv file
    var_name_hist_save = var_name_hist.replace("|", "_")
    hist_file = f"{DIR}/data/processed/df_hist_data_{var_name_hist_save}.csv"
    hist_path = Path(hist_file)
    if hist_path.exists() and not read_original:
        # hist_file is created if not read_original and file does not exists
        print(f"\nReading {str(hist_path)}")
        df_hist_data = pd.read_csv(hist_path, sep=";")
        hist_vars = df_hist_data["Variable"].unique()
        if var_name_hist in hist_vars:
            print(f"Variable {var_name_hist} already in {hist_path}")
            df_hist = df_hist_data[df_hist_data["Variable"]==var_name_hist]
        else:
            print(f"Appending {var_name_hist} to {hist_path}")
            df_hist = hist_data.ReadProcessHistoricalData(var_name_hist, hist_year)
            df_hist["Variable"] = var_name_hist
            df_hist_data = pd.concat([df_hist_data, df_hist], axis=0, ignore_index=True)
            df_hist_data.to_csv(hist_file, sep=";", index=False)
    else:
        # creating new {file hist_path}
        print(f"Creating new {hist_path}")
        df_hist_data = pd.DataFrame()
        df_hist = hist_data.ReadProcessHistoricalData(var_name_hist, hist_year)
        df_hist["Variable"] = var_name_hist
        df_hist_data = pd.concat([df_hist_data, df_hist], axis=0, ignore_index=True)
        df_hist_data.to_csv(hist_file, sep=";", index=False)

    # assumption: all files have the following columns: "ISO3", "Country", "Year", "Value"
    # if there is no World/WLD, this should be calculated
    countries_included = df_hist["Country"].unique()
    if "World" in countries_included:
        pass
    else:
        df_hist_world = df_hist.groupby(["Year"]).sum("Value")
        df_hist_world.reset_index(inplace=True)
        df_hist_world["ISO3"] = "WLD"
        df_hist_world["Country"] = "World"
        print(f"World is not included in {initiative_name}, all countries are aggregated to world")
        df_hist = pd.concat([df_hist, df_hist_world], axis=0, ignore_index=True)

    return df_hist

def _create_plot_hist_data(hist_year: int, df_hist: pd.DataFrame, data_IMAGE: pd.DataFrame, region: int, var_name:str, ShowGraphs: bool, output_dir: str) -> None:
    # create data for plot
    d1 = df_hist[(df_hist["IMAGE_Region_Nr"]==region) & (df_hist["Year"]<=hist_year)]
    d1.insert(len(d1.columns), "Source", "Historical", allow_duplicates=True)
    d1 = d1.loc[:, ["Year", "Value", "Source"]]
    d1 = d1.astype({"Value": float})
    d1 = d1[d1["Year"]>=1990]
    r =  IMAGE_regions_nr2ISO[region]
    print("IMAGE region: ",r)
    data_IMAGE = data_IMAGE[data_IMAGE["Variable"]==var_name]
    mask = (data_IMAGE["Region"]==r) & (data_IMAGE["Year"]>=1990) & (data_IMAGE["Year"]<=hist_year)
    d2 = data_IMAGE[mask]
    d2 = d2.rename(columns={"value":"Value"})
    d2["Source"]="IMAGE"
    d2 = d2.loc[:, ["Year", "Value", "Source", "Unit"]]
    d2 = d2.astype({"Value": float})
    d = d1.merge(d2, how="left", on="Year")
    d.to_csv(f"{DIR}/data/output/downscale_members/{output_dir}/individual/compare_IMAGE_hist.csv", sep=";")
    d1_min = d1["Value"].min()
    d2_min = d2["Value"].min()
    d_min = min(d1_min, d2_min, 0)

    p = d.plot.line(x="Year", y=["Value_x","Value_y"], style='.-', label=["Historical", "IMAGE"])
    v = str.replace(var_name, "|", "_")
    u = d.loc[(d["Year"]==hist_year),"Unit"].iat[0]
    plt.figure
    plt.ylim(d_min, None)
    plt.xlabel("Year")
    plt.ylabel(u)
    plt.title(f"{v}/{region}")
    plt.legend(fontsize=18)
    plt.savefig(f"{DIR}/figures/{output_dir}/fig_hist_IMAGE_global_fullparticipation{v}_{region}.jpg", dpi=300)
    if not ShowGraphs:
        plt.close()

def _downscale_GHG_initiative_to_members(downscaler: dt.DownscaleTool, initiative_name: str, df_hist_IMAGE_Countries: pd.DataFrame, df_IMAGE_baseline: pd.DataFrame, df_IMAGE_initiative: pd.DataFrame, init_members_scenario: str, df_initiative_members: pd.DataFrame, output_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    (df_initiative_country_share, df_IMAGE_initiative_members_allregions, df_IMAGE_initiative_region_share) = downscaler.downscale_to_member_countries(df_IMAGE_baseline, df_IMAGE_initiative, df_hist_IMAGE_Countries, df_initiative_members,
                                                                                                                                              downscaler.hist_year, initiative_name, IMAGE_regions_ISO2nr, output_dir)
    df_IMAGE_initiative_members_allregions["Scenario"] = init_members_scenario
    df_IMAGE_initiative_members_allregions["Model"] = df_IMAGE_initiative.loc[0, "Model"]
    df_units = df_IMAGE_initiative.loc[:, ["Variable", "Unit"]].drop_duplicates()
    df_IMAGE_initiative_members_allregions = pd.merge(df_IMAGE_initiative_members_allregions, df_units, on="Variable", how="left")

    return df_initiative_country_share, df_IMAGE_initiative_members_allregions, df_IMAGE_initiative_region_share

def _plot_GHG_initiative_members(df_IMAGE_initiative_members_allregions: pd.DataFrame, df_IMAGE_initiative_share: pd.DataFrame, initiative_name: str, init_members_scenario: str, region: int,
                                dict_downscale_scenarios: dict, df_IMAGE_initiative: pd.DataFrame, PolicyResults: list[pd.DataFrame], ShowGraphs: bool, output_dir: str) -> None:

    # Add scenarios to one dataframe
    indicator = pd.concat([df_IMAGE_initiative, df_IMAGE_initiative_members_allregions]+PolicyResults, axis=0, ignore_index=True)
    indicator = indicator[indicator["Region"]==IMAGE_regions_nr2ISO[region]]
    indicator["Initiative"] = initiative_name
    indicator.drop(["Initiative", "IMAGE_Region_Nr"], axis=1, inplace=True)
    indicator.to_csv(f"{DIR}/data/output/downscale_members/{output_dir}/individual/indicators_{initiative_name}_{region}.csv", sep=";", index=False)

    # Mt CO2/yr, Mt N2O/yr, Mt CH4/yr)
    unit_list = indicator["Unit"].unique()
    if not set(unit_list).issubset(["Mt CO2/yr", "Mt CH4/yr", "Mt N2O/yr", "kt N2O/yr", "Tg N/yr"]):
        print("\033[92ERROR: One or more units are not correct\033[0m]")
        print(unit_list)
    mask_CO2 = indicator["Unit"]=="Mt CO2/yr"
    mask_CH4 = indicator["Unit"]=="Mt CH4/yr"
    mask_N2O_Mt = indicator["Unit"]=="Mt N2O/yr"
    mask_N2O_kt = indicator["Unit"]=="kt N2O/yr"
    mask_Tg_N = (indicator["Unit"] == "Tg N/yr")
    indicator.loc[mask_CH4, "Value"] *= GWP[0]
    indicator.loc[mask_N2O_Mt, "Value"] *= GWP[1]
    indicator.loc[mask_N2O_kt, "Value"] *= (10**-3*GWP[1])
    indicator.loc[mask_Tg_N, "Value"] *= (N_to_N2O*GWP[1])
    indicator["Unit"]  = "Mt CO2-equiv/yr"
    indicator = indicator.groupby(["Model", "Scenario", "Year", "Unit", "Region", "ScenarioType"]).sum("Value").reset_index() # sum over variables
    indicator.to_csv(f"{DIR}/data/output/downscale_members/{output_dir}/individual/indicators_CO2eq_{initiative_name}_{region}.csv", sep=";", index=False)

    # plot indicators
    # https://www.statology.org/pandas-groupby-plot/
    plt.figure(figsize=(12,10))
    plt.rc('font', size=10)          # controls default text sizes
    indicator_plot = indicator.copy()
    indicator_plot = indicator_plot[indicator_plot["Year"]<=2050]
    indicator_plot.to_csv(f"{DIR}/data/output/downscale_members/{output_dir}/individual/indicator_plot_{initiative_name}_{region}.csv", sep=";", index=True)
    indicator_plot_initiatives = indicator_plot[indicator_plot["Scenario"]==init_members_scenario]

    indicator_plot_initiatives.set_index("Year", inplace=True)
    label_downscaling = indicator_plot_initiatives.loc[2020, "Scenario"]
    indicator_plot_initiatives["Value"].plot(legend=True, linestyle="--", linewidth=3, marker="o", markersize=10, label=label_downscaling)
    mask = indicator_plot["Scenario"]==init_members_scenario
    indicator_plot = indicator_plot[~mask]
    indicator_plot.set_index("Year", inplace=True)
    indicator_plot.groupby("Scenario")["Value"].plot(legend=True, linestyle="-", linewidth=3) # , cmap=my_cmap

    labels={"15degrees_COPInit":"SSP2_2.0W/m2", "2degrees_COPInit":"SSP2_2.6W/m", "COPInitiatives_CPBL_May2025":"COP initiatives (full participation)", "CP":"Current policies"}
    indicator_plot["Scenario"] = indicator_plot["Scenario"].map(labels)

    # add graph aestatics
    r=IMAGE_regions_nr2ISO[region]
    unit = indicator["Unit"].iloc[0]
    mask=(df_IMAGE_initiative_share["IMAGE_Region_Nr"]==region)
    s=100*df_IMAGE_initiative_share.loc[mask, "Country_member_share"].iloc[0]
    plt.figure
    plt.xlabel("Year", fontsize=24)
    plt.ylabel(unit, fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    if indicator_plot["Value"].min() < 0:
        plt.ylim(None, None)
    else:
        plt.ylim(0, None)
    plt.suptitle(f"{initiative_name}/{r}", fontsize=24)
    plt.title(f"Participation in initiative: {s:.1f}%")

    plt.legend(fontsize=18)
    plt.savefig(f"{DIR}/figures/{output_dir}/fig_global_full_vs_members_{initiative_name}_{r}.jpg", dpi=300)
    if not ShowGraphs:
        plt.close()

def _read_process_hist_data(initiative_name: str, var_name_hist: str, hist_year:int, read_original:bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    # read corresponding historical data for countries (all Year, all countries)
    df_hist = _read_hist_variable(initiative_name, var_name_hist, hist_year, read_original)
    print(f"Read historical data {var_name_hist} for {initiative_name}")

    # Calculate country share per region based on historical data
    # group historical data by IMAGE region and calculate aggregated value per region
    df_hist_IMAGE_Countries = ISO.compare_ISO_codes_for_hist_values(df_hist)

    # Aggregate country values to IMAGE regions
    df_hist_IMAGE_Regions = df_hist_IMAGE_Countries.groupby(by=["IMAGE_Region_Nr", "Year"]).sum(["Value"])
    df_hist_IMAGE_Regions.reset_index(drop=False, inplace=True)
    df_hist_IMAGE_Regions = df_hist_IMAGE_Regions.loc[:,["IMAGE_Region_Nr", "Year", "Value"]]

    return (df_hist_IMAGE_Countries, df_hist_IMAGE_Regions)


def downscale_to_individual_initiative_emissions(downscaler: dt.DownscaleTool, input_file: str, dir_RT: str, ShowGraphs: bool, policy_scenarios_IMAGE: list[str], initiatives_scenario: list[str], regions: list[Union[str,int]], GWP_CH4, GWP_N2O, read_original, output_dir) -> None:
    """
    Read from settings file the variables that will be plotted
    1. READ VARIABLE info for IMAGE and historical data from settings file
       Both historical data (emissions/energy) and IMAGE results are added to each plot
    2. Compare IMAGE results with historical data
       2.1. Calculate historical country shares per IMAGE region
            Plot comparison IMAGE and historical data
    3. Downscale global results to participating countries in initiative
       - based on country share of one historical (recent) year
       - apply percentage to regions, add baseline values for non-members and GL values for members
    4. Downscale to urban/subnationals and companies
    """

    # INIT
    if not Path(f"{DIR}/data/output/downscale_members/{output_dir}/individual").exists():
        # create output_dir
        Path(f"{DIR}/data/output/downscale_members/{output_dir}/individual").mkdir(parents=True, exist_ok=True)
    if not Path(f"{DIR}/figures/{output_dir}").exists():
        Path(f"{DIR}/figures/{output_dir}").mkdir(parents=True, exist_ok=True)

    df_ISO_countries_downscaling = pd.DataFrame()

    # Read in paramters from json file
    try:
        parameters_individual_file = f"{DIR}/{input_file}"
        with open(parameters_individual_file) as f:
            parameters = json.loads(f.read())
        settings = parameters["Settings"]
        ICI_signatories_csv = settings["ICI_signatories_file"]
        init_members_scenario = settings["Members_Scenario"]
        ICI_scenario_parameters = parameters["Init_Parameters_ICIs"]
        df_initiative_members = pd.read_csv(f"{DIR}/data/input/{ICI_signatories_csv}", sep=";")
        print(f"\nparameters: {ICI_scenario_parameters}")
        downscale_baseline_scenario = settings["Baseline_Scenario"]
        regions = list(map(int, regions))
        _set_global_variables(settings, GWP_CH4, GWP_N2O)
    except ValueError as Err:
        print(f"ERROR in json file\n")
        print(f"Run 'python -m json.tool {input_file}'")

    # Read variable info for IMAGE and historical data from settings file
    collect_shares_aggregate = pd.DataFrame()
    df_IMAGE_aggregate = pd.DataFrame()
    df_IMAGE_initiative_members_shares = pd.DataFrame()
    for i, p in enumerate(ICI_scenario_parameters):
        if True:
            dict_downscale_scenarios = {"Initiatives": initiatives_scenario[0], "Initiative_members":init_members_scenario}
            initiative_name = p["Initiative"]
            var_name_hist = p["Hist_variable"]
            var_name_ghg = p["GHG_individual_variable"]
            only_global = p["Only_global"]
            only_global = (only_global.lower() == "yes")
            print(f"\n\033[93m{output_dir}-SECTOR: {i+1}. {initiative_name}\033[0m")
            print(f"REGIONS: {regions}")

            # Retrieve and process historical data which is used for downscaling
            print("Retrieve and process historical data")
            df_hist_IMAGE_Countries, df_hist_IMAGE_Regions = _read_process_hist_data(initiative_name, var_name_hist, downscaler.hist_year, read_original)

            # (log) Add countries/ISO to logfile
            df_hist_log = log.LogResults(df_hist_IMAGE_Countries, initiative_name, downscaler.hist_year)
            df_ISO_countries_downscaling = pd.concat([df_ISO_countries_downscaling, df_hist_log], axis=0, ignore_index=True)

            #Read IMAGE data for variable name and aggregate variable name (the scenario results (pandas) are added to list)
            print("Read input IMAGE files")
            PolicyResults = []
            for s in policy_scenarios_IMAGE:
                df_data = image_data.read_IMAGE_input_file(var_name_ghg, s, dir_RT, downscale_baseline_scenario)
                df_data["ScenarioType"] = "Policy"
                PolicyResults.append(df_data)

            df_IMAGE_baseline = image_data.read_IMAGE_input_file(var_name_ghg, downscale_baseline_scenario, dir_RT, output_dir)

            # Read initiative scenario
            df_IMAGE_mitigation = image_data.read_IMAGE_input_file(var_name_ghg, initiatives_scenario[0], dir_RT, output_dir)
            if df_IMAGE_mitigation.empty:
                print(f"\033[92ERROR: {initiatives_scenario[0]} in IMaGE output is empty for variable {var_name_ghg}\033[0m")
                error_list.append(f"ERROR: {initiatives_scenario[0]} is empty for variable {var_name_ghg}")
                continue
            df_IMAGE_mitigation["ScenarioType"] = "Initiatives"

            # Downscale results for ICI to members
            (df_initiative_country_share, df_IMAGE_initiative_members_allregions, df_IMAGE_initiative_region_share) = _downscale_GHG_initiative_to_members(downscaler, initiative_name, df_hist_IMAGE_Countries, df_IMAGE_baseline, df_IMAGE_mitigation,
                                                                                                                                                           init_members_scenario, df_initiative_members, output_dir)
            df_IMAGE_initiative_members_shares = pd.concat([df_IMAGE_initiative_region_share, df_IMAGE_initiative_members_shares], axis=0, ignore_index=True)
            df_IMAGE_initiative_members_allregions["ScenarioType"] = "Initiatives"
            df_IMAGE_initiative_members_allregions.to_csv(f"{DIR}/data/output/downscale_members/{output_dir}/individual/df_IMAGE_initiative_members_individual_{initiative_name}.csv", sep=";", index=False)

            # Plot data for each region in parameters file
            #if ShowGraphs:
            for region in regions:
                if not only_global or region==28:
                    print(region)
                    # Plot comparison IMAGE and historical IEA data
                    if var_name_hist in var_name_ghg:
                        _create_plot_hist_data(downscaler.hist_year, df_hist_IMAGE_Regions, df_IMAGE_mitigation, region, var_name_hist, ShowGraphs, output_dir)
                    print("Downscale to members and create plot")

                    # Plot downscaled results
                    _plot_GHG_initiative_members(df_IMAGE_initiative_members_allregions, df_IMAGE_initiative_region_share,
                                                initiative_name,  init_members_scenario, region, dict_downscale_scenarios, df_IMAGE_mitigation,
                                                PolicyResults, ShowGraphs, output_dir)
            # Collect initiatives' shares and baseline/mitigation emissions
            if not (initiative_name=="Flaring"):
                # shares
                collect_share = df_initiative_country_share.copy()
                collect_share.drop(["Value"], axis=1, inplace=True)
                collect_share["ICI"] = initiative_name
                collect_shares_aggregate = pd.concat([collect_shares_aggregate, collect_share], axis=0, ignore_index=True)

    # save members shares per region
    df_IMAGE_initiative_members_shares.to_csv(f"{DIR}/data/output/downscale_members/{output_dir}/individual/df_IMAGE_initiative_members_shares.csv", sep=";", index=False)

    # merge country shares ICI (see IMAGE_sectors.xlsx) and globally
    if not collect_shares_aggregate.empty:
        collect_shares_aggregate["Region"] = collect_shares_aggregate["IMAGE_Region_Nr"].map(IMAGE_regions_nr2ISO)
        collect_shares_aggregate = collect_shares_aggregate[~(collect_shares_aggregate["Country_share"].isnull())]
        collect_shares_aggregate.sort_values(by=["ICI", "ISO3"], inplace=True)
        collect_shares_aggregate.to_csv(f"{DIR}/data/output/downscale_members/{output_dir}/individual/initiatives_country_shares_individual.csv", sep=";", index=False)
    else:
        collect_shares_aggregate.to_csv(f"{DIR}/data/output/downscale_members/{output_dir}/individual/initiatives_country_shares_individual.csv", sep=";", index=False)

    # (log) Save results to logfile
    log.SaveResults(df_ISO_countries_downscaling, f"{DIR}/runlog/downscale_members/{output_dir}", downscaler.hist_year)

    if error_list:
        print("\n\033[92mOverview ERRORS in downscaling individual initiatives:\033[0m")
        for e in error_list:
            print(e)

def main():
    pass

if __name__ == "__main__":
    main()



