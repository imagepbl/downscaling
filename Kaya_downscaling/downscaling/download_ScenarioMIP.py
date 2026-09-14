# %%
from pathlib import Path
from tabulate import tabulate
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyam
import ixmp4

def _label(value):
    """Handle model/scenario being either a plain string or a facade object."""
    return getattr(value, "name", value)


def discover_emission_variables(platform_name):
    """Return all "Emissions|..." variable names, read from one sample run."""
    platform = ixmp4.Platform(platform_name)
    sample = next(iter(platform.runs.list())).iamc.tabulate()
    return sorted(v for v in sample["variable"].unique() if v.startswith("Emissions|"))


def load_emissions(platform_name, regions, variables, models):
    """Retrieve emissions by iterating runs and tagging each with its own model/scenario."""

    if variables is None:
        variables = discover_emission_variables(platform_name)
    platform = ixmp4.Platform(platform_name)

    frames = []
    for run in platform.runs.list():
        model_name = _label(run.model)
        if models is not None and not any(model_name.startswith(m) for m in models):
            continue
        print(f"pulling {model_name} / {_label(run.scenario)}", flush=True)
        data = run.iamc.tabulate(region=REGIONS, variable=VARIABLES)
        if "region" in data.columns:
            data = data[data["region"].isin(regions)]
        if "variable" in data.columns:
            data = data[data["variable"].isin(variables)]
        if data.empty:
            continue
        data["model"] = model_name
        data["scenario"] = _label(run.scenario)
        frames.append(data)

    if not frames:
        raise ValueError("No matching rows for the given models, regions, and variables.")
    df = pyam.IamDataFrame(pd.concat(frames, ignore_index=True))
    
    return df

def download_ScenarioMIP_emissions(data_dir,ScenarioMIP_scenarios, PLATFORM, REGIONS, VARIABLES, MODELS):
    """Download ScenarioMIP emissions data from the ixmp4 platform and save to CSV."""

    file = "scenariomip_emissions.csv"
    file_path = data_dir / file
    data_dir_scenarios = data_dir / "outxlsx"
    data_dir_scenarios.mkdir(parents=True, exist_ok=True)

    df_pyam = load_emissions(PLATFORM, REGIONS, VARIABLES, MODELS)
    print(df_pyam.head(5))

    df = df_pyam.data
    df["scenario"] = df["scenario"].str.replace(" (Marker)", "")  # remove SSP number from scenario name
    # df["scenario"] = df["scenario"].str.replace("-", "_")  # remove SSP number from scenario name
    # df["scenario"] = df["scenario"].str.replace(" ", "")  # remove SSP number from scenario name
    df.to_csv(file_path, index=False, sep=";")           # long format, no index column
    print(f"ScenarioMIP emissions data saved to {file_path}")

    # create empty dataframe with columns to store model and scenario combinations
    df_model_scenario_combinations = pd.DataFrame(columns=['model', 'scenario', 'baseline'])

    scenarios = df['scenario'].unique()
    scenarios_present = pd.DataFrame(columns=["scenario"])
    for model in MODELS:
        for scenario in scenarios:
            # separate the scenario name and SSP number using regex
            match = re.search(r"^(.*?)\s*-\s*(SSP\d+)$", scenario)
            if match:
                scenario_base, ssp = match.group(1), match.group(2) 
            else:
                scenario_base, ssp = scenario, None
            if scenario_base not in scenarios_present['scenario'].values:
                scenarios_present = pd.concat([scenarios_present, pd.DataFrame([[scenario_base]], columns=["scenario"])])
            df_model_scenario_combinations.loc[len(df_model_scenario_combinations)] = [model, scenario_base, ssp]
            #print(f"\nSaving data for model: {model}, scenario: {scenario}")

            mask_model_scenario = (df['model'] == model) & (df['scenario'] == scenario)
            df_model_scenario = df[mask_model_scenario]
            if len(df_model_scenario) == 0:
                print(f"No data found for model: {model}, scenario: {scenario}")
            else:
                # check which variables from VARIABLES are not present in the data for this model and scenario
                missing_variables = [var for var in VARIABLES if var not in df_model_scenario['variable'].unique()]
                if missing_variables:
                    print(f"Missing variables for model: {model}, scenario: {scenario}: {missing_variables}")
                # save to excel file in data sheet
                df_model_scenario.to_excel(data_dir_scenarios / f"{model}_{scenario}.xlsx", index=False)

    print(tabulate(df_model_scenario_combinations, headers='keys', tablefmt='psql', showindex=False))

    print(f"\nScenarios present in the data: {list(scenarios_present['scenario'])}")
    print(f"\nScenarios in the ScenarioMIP list: {ScenarioMIP_scenarios}")
    set_scenarios_present = list(scenarios_present["scenario"])
    missing_scenarios = [s for s in ScenarioMIP_scenarios if s not in set_scenarios_present]
    if missing_scenarios:
        print(f"\nMissing scenarios: {missing_scenarios}")
    else:
        print("\nAll ScenarioMIP scenarios are present in the data")


if __name__ == "__main__":
    PLATFORM = "scenariomip-cmip7"
    REGIONS = ["World"]
    VARIABLES = ["Emissions|CO2", 
                "Emissions|CO2|Energy|Supply", "Emissions|CO2|Energy|Demand", 
                "Emissions|CO2|Energy|Demand|Industry", "Emissions|CO2|Energy|Demand|Transportation", "Emissions|CO2|Energy|Demand|Residential and Commercial", "Emissions|CO2|Energy|Demand|Other Sector", 
                "Emissions|CO2|Energy|Demand|Bunkers|International Aviation", "Emissions|CO2|Energy|Demand|Bunkers|International Shipping", "Emissions|CO2|Energy|Demand|Transportation|Domestic Aviation",
                "Emissions|CO2|Industrial Processes", 
                "Emissions|CO2|AFOLU",
                "Gross Emissions|CO2|Energy|Supply", "Gross Emissions|CO2|Energy|Demand", "Gross Emissions|CO2|Energy|Demand|Industry",
                "Population", "GDP|PPP"]
    MODELS = ["IMAGE 3.4", "REMIND-MAgPIE 3.5-4.11"]

    ScenarioMIP_scenarios = ["High", "High-to-Low", "Medium", "Medium-to-Low", "Low", "Very Low", "Low-to-Negative"]

    project_dir = Path.cwd()
    data_dir = project_dir / "data" / "processed" / "models"
    data_dir.mkdir(parents=True, exist_ok=True)

    download_ScenarioMIP_emissions(data_dir, ScenarioMIP_scenarios, PLATFORM, REGIONS, VARIABLES, MODELS)

    
