from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
import seaborn as sns

# import PBL/own classes/modules
# from ..IMAGE_tools.settings import dir_RT
from ..IMAGE_tools.IMAGE_regions_settings import IMAGE_regions_nr2UN, IMAGE_IAMC_regions_nr2ISO
from ..downscaling import downscale_tool as dt
from ..IMAGE_tools import read_process_IMAGE_data as image_data

pd.options.display.float_format = "{:20,.2f}".format

# TEMP --> change to get this from json file
dir_RT = "X:/user/roelfsemam/timer/COP_Initiatives_3_4/9_Python/COP_Initiatives/data/input/paper"

# global variables
DIR_RT = ""
HIST_YEAR_URBAN = 2020

output_dir = "methods_paper"

dir_current = Path(__file__).parent

# TO DO: make policy scenarios flexible, now they are fixed to NDC, OneHalfD and initiative

def set_global_variables(Settings):
    global DIR_RT, HIST_YEAR_URBAN
    DIR_RT = dir_RT
    HIST_YEAR_URBAN = Settings["HIST_YEAR_URBAN"]

def plot_downscaling_results(lst_df_emissions_var: list[pd.DataFrame], scenario_name_to_label, region, title:str) -> None:

    # prepare data for plotting
    df_emissions_var_plot = pd.concat(lst_df_emissions_var, axis=0, ignore_index=True)
    mask = (df_emissions_var_plot["Region"]==region) & (df_emissions_var_plot["Year"]>=2015) & (df_emissions_var_plot["Year"]<=2050)
    df_emissions_var_plot = df_emissions_var_plot[mask]
    df_emissions_var_plot = df_emissions_var_plot.copy().loc[:, ["Scenario", "Year", "Value"]]
    df_emissions_var_plot["Scenario"] = df_emissions_var_plot["Scenario"].map(scenario_name_to_label)
    # plot
    plt.figure()
    df_emissions_var_plot.set_index("Year", inplace=True)
    df_emissions_var_plot.groupby("Scenario")["Value"].plot(legend=True)
    plt.ylim(bottom=0)
    plt.xlabel("Year")
    plt.ylabel("MtCO2eq")
    plt.title(title)
    plt.legend()
    plt.savefig(f"{dir_current}/figures/ActorsDownscaling/fig_global_downcaled_emissions_{title}.jpg", dpi=300)

def combine_downscaled_results(lst_df_emissions_total: list[pd.DataFrame],
                               lst_df_emissions_urban: list[pd.DataFrame],
                               lst_df_emissions_comnpanies: list[pd.DataFrame]) -> pd.DataFrame:
    
    
    df_emissions_total= pd.concat(lst_df_emissions_total, axis=0, ignore_index=True)
    df_emissions_urban= pd.concat(lst_df_emissions_urban, axis=0, ignore_index=True)
    df_emissions_comnpanies= pd.concat(lst_df_emissions_comnpanies, axis=0, ignore_index=True)

    df_emissions_total["Actor_Group"] = "All"
    df_emissions_urban["Actor_Group"] = "Urban"
    df_emissions_comnpanies["Actor_Group"] = "Companies"

    df_IMAGE_CO_Combine = pd.concat([df_emissions_total, df_emissions_urban, df_emissions_comnpanies])
    df_IMAGE_CO_Combine.to_csv(f"{dir_current}/data/output/figures/{output_dir}/actor_group_emissions.csv", sep=";")

    return df_IMAGE_CO_Combine

def plot_combine_downscaled_actor_emissions(df: pd.DataFrame, scenario_name_to_label, region, title) -> None:
    
    # collect data for plotting
    df_plot = df.copy()
    mask = (df_plot["Region"]==region) & (df_plot["Year"]>=2015) & (df_plot["Year"]<=2050)
    df_plot = df_plot[mask]
    df_plot["Scenario"] = df_plot["Scenario"].map(scenario_name_to_label)
    df_plot_all=df_plot[df_plot["Actor_Group"]=="All"]
    df_plot_urban=df_plot[df_plot["Actor_Group"]=="Urban"]
    df_plot_companies=df_plot[df_plot["Actor_Group"]=="Companies"]

    min_value = min(0, 0.9*df_plot["Value"].min())
    max_value = 1.1*df_plot["Value"].max()    
    print(f"plot range: {min_value}-{max_value}")

    # plot
    sns.set_context('paper', font_scale=2)
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20,10))
    sns.lineplot(data=df_plot_all, x="Year", y="Value", hue="Scenario", linewidth=5, ax=axs[0], legend=True).set_title("Global GHG emissions")
    sns.lineplot(data=df_plot_urban, x="Year", y="Value", hue="Scenario", linewidth=5, ax=axs[1], legend=False).set_title("Global GHG emissions on urbal level")
    sns.lineplot(data=df_plot_companies, x="Year", y="Value", hue="Scenario", linewidth=5, ax=axs[2], legend=False).set_title("Global GHG emissions for companies")

    axs[0].set(ylim=(min_value, max_value))
    axs[1].set(ylim=(min_value, max_value))
    axs[2].set(ylim=(min_value, max_value))

    #fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(f"{dir_current}/figures/ActorsDownscaling/fig_subplots_global_downcaled_emissions.jpg", dpi=300)

def smooth_IMAGE_results(df_IMAGE: pd.DataFrame, year1:int, year2:int) -> pd.DataFrame:
    df_IMAGE["tmp"] = df_IMAGE["Value"].copy().rolling(window=3).mean()
    mask1 = df_IMAGE["Year"]==year1
    mask2 = df_IMAGE["Year"]==year2
    interpolated_value_y1 = df_IMAGE.loc[mask1, "tmp"].values[0]
    interpolated_value_y2 = df_IMAGE.loc[mask2, "tmp"].values[0]
    df_IMAGE.loc[df_IMAGE["Year"]==year1, "Value"] = interpolated_value_y1
    df_IMAGE.loc[df_IMAGE["Year"]==year2, "Value"] = interpolated_value_y2
    df_IMAGE.drop(["tmp"], axis=1, inplace=True)
    
    return df_IMAGE

def calculate_downscaled_actor_CO2_emissions(downscaler: dt, initiative_scenario: str, ShowGraphs: bool) -> None:
    
    # retrieve settings in json file
    try:
        with open("Parameters_actor_downscaling.json") as f:
            parameters = json.loads(f.read())
        settings = parameters["Settings"]
    except ValueError as Err:
        print(f"Error in json file\n")
        print(f"Run 'python -m json.tool Parameters.json'")
    set_global_variables(settings)
    policy_scenario_labels = settings["PolicyScenariosLabels"]
    policy_scenarios_IMAGE = settings["PolicyScenariosIMAGENames"]
    var = settings["Downscale_variable_emissions"]

    scenario_name_to_label = dict(map(lambda i,j : (i,j) , policy_scenarios_IMAGE,policy_scenario_labels))

    # Downscale emissions to urban/subnationals and companies to GHG sources
    if not var.startswith("Emissions"):
        print("Downscale variable should be an emissions variable that starts with 'Emissions'")
        exit()

    # Read IMAGE data for variable name (the scenario results (pandas) are added to list)
    print("Read input files")
    policy_scenario_data_emissions = []
    policy_scenario_data_emissions_var = []
    downscaled_urban_scenario_data_emissions_var = []
    downscaled_companies_scenario_data_emissions_var = []
    for s in policy_scenarios_IMAGE:
        # read IMAGE data
        df_emissions = image_data.ImageRT_start(s, "Emissions",f"{DIR_RT}/{output_dir}")
        policy_scenario_data_emissions.append(df_emissions)
        df_emissions_var = df_emissions[df_emissions["Variable"]==var]
        policy_scenario_data_emissions_var.append(df_emissions_var)
        # downscale to urban and companies
        df_IMAGE_CO2_Urban_NP = downscaler.downscale_initiative_to_urban(df_emissions_var, dir_current, HIST_YEAR_URBAN, IMAGE_regions_nr2UN, IMAGE_IAMC_regions_nr2ISO)
        downscaled_urban_scenario_data_emissions_var.append(df_IMAGE_CO2_Urban_NP)
        df_IMAGE_CO2_Companies_NP = downscaler.downscale_initiative_to_companies(df_emissions)
        downscaled_companies_scenario_data_emissions_var.append(df_IMAGE_CO2_Companies_NP)

    # Create plots of 1) total GHG emissions 2) downscaled urban emissions, 3) downscaled company emissions
    plot_downscaling_results(policy_scenario_data_emissions_var, scenario_name_to_label, region="World", title="Total GHG emissions")
    plot_downscaling_results(downscaled_urban_scenario_data_emissions_var, scenario_name_to_label, region="World", title="Urban GHG emissions")
    plot_downscaling_results(downscaled_companies_scenario_data_emissions_var, scenario_name_to_label, region="World", title="Companies' GHG emissions")
    df_IMAGE_CO_Combine = combine_downscaled_results(policy_scenario_data_emissions_var, downscaled_urban_scenario_data_emissions_var, downscaled_companies_scenario_data_emissions_var)
    
    plot_combine_downscaled_actor_emissions(df_IMAGE_CO_Combine, scenario_name_to_label, region="World", title = "Total GHG emissions, and downscaled to urban and companies")
    if ShowGraphs:
        plt.show()

def downscale_actors(scenario: str, show_graphs: bool) -> None:
    downscaler = dt.DownscaleTool()
    calculate_downscaled_actor_CO2_emissions(downscaler, scenario, show_graphs)
    if show_graphs:
        plt.show()
    
def main() -> None:
    show_graphs = True
    scenario = "GlasgowInitiatives_CPBL"
    downscale_actors(scenario, show_graphs)

if __name__ == "__main__":
    main()