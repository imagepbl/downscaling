import pandas as pd
import numpy as np
from pathlib import Path

DIR = Path(__file__).resolve().parent

class DownscaleTool():


    def __init__(self, hist_year:int=2021):
        # init
    	self.hist_year = hist_year

    def test_tool(self):
        print("downscaling tool test")

    def downscale_initiative_to_companies(self, df_IMAGE_emissions):
        # Downscale emissions companies
        # Read IMAGE data for
        # Industry (energy/process), Power/heat generation, Domestic Freight Transport, Services Buildings, international aviation, international shipping

        scen = df_IMAGE_emissions["Scenario"].iloc[0]

        #df_IMAGE_industry_energy = df_IMAGE_emissions[df_IMAGE_emissions["Variable"]=="Emissions|CO2|Industrial Processes"]
        #df_IMAGE_industry_process = df_IMAGE_emissions[df_IMAGE_emissions["Variable"]=="Emissions|CO2|Energy|Demand|Industry"]
        df_IMAGE_industry_energy = df_IMAGE_emissions[df_IMAGE_emissions["Variable"]=="Gross Emissions|CO2|Industrial Processes"]
        df_IMAGE_industry_process = df_IMAGE_emissions[df_IMAGE_emissions["Variable"]==""]
        df_IMAGE_power = df_IMAGE_emissions[df_IMAGE_emissions["Variable"]=="Emissions|CO2|Energy|Supply|Electricity"]
        df_IMAGE_heat = df_IMAGE_emissions[df_IMAGE_emissions["Variable"]=="Emissions|CO2|Energy|Supply|Heat"]
        df_IMAGE_domestic_freight = df_IMAGE_emissions[df_IMAGE_emissions["Variable"]=="Emissions|CO2|Transportation|Freight|Domestic"]
        df_IMAGE_commercial_buildings = df_IMAGE_emissions[df_IMAGE_emissions["Variable"]=="Emissions|CO2|Energy|Demand|Commercial"]
        df_IMAGE_international_aviation = df_IMAGE_emissions[df_IMAGE_emissions["Variable"]=="Emissions|CO2|Energy|Demand|Transportation|Aviation|International"]
        df_IMAGE_international_shipping = df_IMAGE_emissions[df_IMAGE_emissions["Variable"]=="Emissions|CO2|Energy|Demand|Transportation|Shipping|International"]

        company_categories = [df_IMAGE_industry_energy, df_IMAGE_industry_process, df_IMAGE_power, df_IMAGE_heat,
                              df_IMAGE_domestic_freight, df_IMAGE_commercial_buildings, df_IMAGE_international_aviation, df_IMAGE_international_shipping]

        df_IMAGE_CO2_companies = pd.concat(company_categories).groupby(["Year", "Region"]).sum(numeric_only=True).reset_index()
        df_IMAGE_CO2_companies["Scenario"] = scen

        return df_IMAGE_CO2_companies

    def downscale_initiative_to_urban(self, df_IMAGE_emissions,  hist_year_urban, IMAGE_regions_UN_regions, IMAGE_IAMC_regions_nr2ISO):

        # 1. Downscale emissions to urban/subnationals and companies
        # - urban/subnationals
        # based on Marcotullio et al (2013), https://link.springer.com/article/10.1007/s10584-013-0977-z
        # regions can be found at https://unstats.un.org/unsd/methodology/m49/

        scen = df_IMAGE_emissions["Scenario"].iloc[0]

        # 1.1 calculate % urban emisssion per UN region per Source
        # Read urban_ratio
        df_UN_WPP_urban_ratio = pd.read_csv(f"{DIR}/data/input/urban_population_per_region.csv", sep=";")
        # Select data after hist_year_urban
        df_urban_ratio_continents = (df_UN_WPP_urban_ratio
                                        .loc[(df_UN_WPP_urban_ratio["year"]>=hist_year_urban) & (df_UN_WPP_urban_ratio["year"] <= 2050)]
                                        .rename(columns={"year":"Year", "TIMER":"Region"})
                                        )
        # aggregate urban population and ratio to UN regions
        # https://www.youtube.com/watch?v=7StAlfH78C0
        # tmp = df_urban_ratio_continents
        df_urban_ratio_continents["Continent"] = df_urban_ratio_continents["Region"].map(IMAGE_regions_UN_regions)
        df_urban_ratio_continents["perc_urban"] = df_urban_ratio_continents["urban_population"]/df_urban_ratio_continents.groupby(["Year", "Continent"])["urban_population"].transform("sum")
        df_urban_ratio_continents["weight_urban"] = df_urban_ratio_continents["urban_ratio"]*df_urban_ratio_continents["perc_urban"]
        df_urban_ratio_continents.drop(["Region", "urban_ratio", "perc_urban"], axis=1, inplace=True)
        df_urban_ratio_continents.rename(columns={"weight_urban":"urban_ratio"}, inplace=True)
        df_urban_ratio_continents = df_urban_ratio_continents.groupby(["Year", "Continent"]).sum(numeric_only=True).reset_index()
        # Calculate urban emissions per sector (Agriculture, Energy, Industry, Buildings, Transportation, Waste, Urban)
        # - first calculate change in urban ratio per year per UN region
        df_urban_ratio_continents["urban_change"] = df_urban_ratio_continents.groupby(["Continent"])["urban_ratio"].pct_change(periods=1)
        df_urban_ratio_continents["urban_change"] = df_urban_ratio_continents["urban_change"].replace(np.nan, 0)
        # - second, apply this change in urban_ratio to urban emissions percentage after year 2000 from Marcotullio et al (2013) (on continent and GHG source)
        filename_urban_emissions = f"{DIR}/data/input/Urban_emissions_per_UN_continent_2000.xlsx"
        df_urban_emissions_per_continent_hist_year = pd.read_excel(filename_urban_emissions,sheet_name="urban_emissions")
        df_urban_emissions_per_continent_hist_year = pd.melt(df_urban_emissions_per_continent_hist_year,id_vars=["Source"], var_name="Continent", value_name="urban_ratio_emissions_2000")
        df_urban_emissions_per_continent = pd.DataFrame(np.repeat(df_urban_emissions_per_continent_hist_year.values, (2050-hist_year_urban+1), axis=0)) # emtpy dataframe to calculate annual urban emissions ratio
        df_urban_emissions_per_continent.columns = df_urban_emissions_per_continent_hist_year.columns
        df_urban_emissions_per_continent["Year"] = df_urban_emissions_per_continent.groupby(["Continent", "Source"]).cumcount() + hist_year_urban
        df_urban_emissions_per_continent = df_urban_emissions_per_continent.merge(df_urban_ratio_continents, on=["Continent", "Year"])
        df_urban_emissions_per_continent["urban_change_gross"] = 1 + df_urban_emissions_per_continent["urban_change"]
        df_urban_emissions_per_continent["cum_urban_change"] = df_urban_emissions_per_continent.groupby(["Continent", "Source"])["urban_change_gross"].cumprod()-1
        df_urban_emissions_per_continent["urban_ratio_emissions"] = df_urban_emissions_per_continent["urban_ratio_emissions_2000"]*(1+df_urban_emissions_per_continent["cum_urban_change"])
        df_urban_emissions_per_continent.drop(["urban_ratio_emissions_2000", "urban_change", "urban_change_gross", "cum_urban_change", "urban_ratio"], axis=1, inplace=True)

        # aggregate IMAGE values to 5 regions and aggregate to world (for df_IMAGE_NP and df_IMAGE_GL) --> only for Emissions|Kyoto Gases
        # TO DO: extend to Sources from UN definitions
        # TO DO: interpolate emissions#  https://stackoverflow.com/questions/37057187/pandas-interpolate-within-a-groupby
        df_IMAGE_emissions_urban = df_IMAGE_emissions.rename(columns={"Year":"Year"})
        df_IMAGE_emissions_urban["Continent"] = df_IMAGE_emissions_urban["Region"].map(IMAGE_IAMC_regions_nr2ISO)
        df_IMAGE_emissions_urban.drop(["Region"], axis=1, inplace=True)
        df_IMAGE_emissions_urban["Year"] = df_IMAGE_emissions_urban["Year"].astype(str)
        df_IMAGE_emissions_urban = df_IMAGE_emissions_urban.groupby(["Year", "Continent"]).sum(numeric_only=True).reset_index()
        df_IMAGE_emissions_urban["Year"] = df_IMAGE_emissions_urban["Year"].astype(int)
        df_urban_emissions_total = df_urban_emissions_per_continent.loc[df_urban_emissions_per_continent["Source"]=="All urban (low)"]
        df_IMAGE_emissions_urban = pd.merge(df_IMAGE_emissions_urban, df_urban_emissions_total, on=["Continent", "Year"])
        df_IMAGE_emissions_urban["urban_emissions"] = df_IMAGE_emissions_urban["Value"] * df_IMAGE_emissions_urban["urban_ratio_emissions"] / 100
        df_IMAGE_emissions_urban.drop(["urban_population", "urban_ratio_emissions", "Value"], axis=1, inplace=True)
        df_IMAGE_emissions_urban.rename(columns={"urban_emissions":"Value"}, inplace=True)
        df_IMAGE_emissions_urban["Value"] = df_IMAGE_emissions_urban["Value"].astype(float)
        # TO DO: compare to world=emissions_urban_ratio*GHG_Kyoto_World
        df_IMAGE_emissions_urban_world = df_IMAGE_emissions_urban.groupby(["Year", "Source"]).sum(numeric_only=True)
        df_IMAGE_emissions_urban_world.reset_index(drop=False, inplace=True)
        df_IMAGE_emissions_urban_world["Continent"] = "World"
        df_IMAGE_emissions_urban_world = df_IMAGE_emissions_urban_world.rename(columns={"value":"Value"})
        df_IMAGE_emissions_urban = pd.concat([df_IMAGE_emissions_urban, df_IMAGE_emissions_urban_world], axis=0, ignore_index=True)
        df_IMAGE_emissions_urban.rename(columns={"Continent":"Region"}, inplace=True)
        df_IMAGE_emissions_urban["Scenario"] = scen

        return df_IMAGE_emissions_urban

    def calculate_country_share_in_region(self, df_hist_IMAGE_Countries: pd.DataFrame, output_dir: str) -> pd.DataFrame:
        df_hist_IMAGE_Countries = df_hist_IMAGE_Countries.copy()[df_hist_IMAGE_Countries["Value"].notna()]
        var = df_hist_IMAGE_Countries["Variable"].iloc[0].replace("|", "_")

        # calculate % for country of region total from historical data
        mask= (df_hist_IMAGE_Countries["ISO_IMAGE"] == True) & (df_hist_IMAGE_Countries["ISO_hist"]==True) & (df_hist_IMAGE_Countries["Year"]==self.hist_year) & (~(df_hist_IMAGE_Countries["IMAGE_Region_Nr"]==28))
        df_var_country_share_in_region = df_hist_IMAGE_Countries.loc[mask].copy()
        df_var_country_share_in_region.to_csv(f"{DIR}/data/output/downscale_tool/{output_dir}/df_var_country_share_in_region.csv", sep=";")
        df_var_country_share_in_region["perc_region"] = 100 * df_var_country_share_in_region["Value"] / df_var_country_share_in_region.groupby("IMAGE_Region_Nr")["Value"].transform("sum") # just for checking
        df_var_country_share_in_region= df_var_country_share_in_region.loc[:, ["ISO3", "Country_name", "IMAGE_Region_Nr", "Value"]]
        df_var_country_share_in_region["Country_share"] = df_var_country_share_in_region["Value"]/df_var_country_share_in_region.groupby("IMAGE_Region_Nr")["Value"].transform("sum")
        df_var_country_share_in_region.to_csv(f"{DIR}/data/output/downscale_tool/{output_dir}/df_var_country_share_in_region_{var}.csv", sep=";")

        return df_var_country_share_in_region

    def calculate_member_perc_in_region(self, df_var_country_share_in_region, df_initiative_members, ICI, output_dir) -> pd.DataFrame:
        # read in ICI country members
        lst_initiatives = df_initiative_members.columns.tolist()[4:]
        print(f"Initiatives included: {lst_initiatives}")
        if ICI == "Flaring":
            df_initiative_members_incl_share = pd.read_csv(f"{DIR}/data/input/ICI_members_flaring.csv", sep=";")
            df_initiative_members_incl_share["Value"] = df_initiative_members_incl_share["Value"].astype(float)
            df_initiative_not_members = df_initiative_members_incl_share.copy()
            df_initiative_not_members["Value"] = 1- df_initiative_not_members["Value"]
            # determine baseline share (World only)
            df_initiative_members_share_per_region_value = df_initiative_members_incl_share.loc[:, "Value"].mean()
            df_initiative_members_share_per_region = pd.DataFrame({"IMAGE_Region_Nr":[28], "Country_member_share":[df_initiative_members_share_per_region_value]})
            df_var_country_share_in_region = pd.DataFrame()
        else:
            #df_initiative_members_incl_share = pd.read_csv(f"{DIR}/data/input/ICI_members.csv", sep=";")
            df_initiative_members_incl_share = df_initiative_members.copy()
            df_initiative_members_incl_share = df_initiative_members_incl_share.loc[:, ~df_initiative_members_incl_share.columns.str.startswith(("BT_Power", "BT_Road"))] # exclude BT_Power and BT_Road that are included in Coal and Transport_cars_buses
            df_initiative_members_incl_share = pd.melt(df_initiative_members_incl_share, id_vars=["ISO", "IMAGE_region_nr", "Country_name", "Included by NCI?"],
                                                    #value_vars=["Methane",  "Coal", "Transport_cars_buses", "Transport_trucks", "Deforestation", "Bunkers_aviation", "Steel", "Renewable", "Efficiency", "Cooling", "Cement", "Buildings"],
                                                    value_vars=lst_initiatives,
                                                    var_name="ICI", value_name="Member")
            df_initiative_members_incl_share["IMAGE_region_nr"].astype(int)
            df_initiative_members_incl_share["ICI"].astype("category")
            df_initiative_members_incl_share.to_csv(f"{DIR}/data/output/downscale_tool/{output_dir}/df_initiative_members_incl_share.csv", sep=";")

            # determine member share per region
            df_initiative_members_incl_share_sector = df_initiative_members_incl_share[df_initiative_members_incl_share["ICI"]==ICI]
            df_initiative_members_share_per_region = pd.merge(df_var_country_share_in_region, df_initiative_members_incl_share_sector, left_on="ISO3", right_on="ISO", how="outer")
            df_initiative_members_share_per_region = df_initiative_members_share_per_region[df_initiative_members_share_per_region["ISO3"].notna()]
            df_initiative_members_share_per_region_World = df_initiative_members_share_per_region.copy() # use later
            df_initiative_members_share_per_region['Country_member_share'] = df_initiative_members_share_per_region['Country_share'] * df_initiative_members_share_per_region['Member']
            df_initiative_members_share_per_region['Country_member_share'] = df_initiative_members_share_per_region['Country_member_share'].fillna(0)
            df_initiative_members_share_per_region = df_initiative_members_share_per_region.loc[:,["IMAGE_Region_Nr", "Country_member_share"]].groupby("IMAGE_Region_Nr").sum(numeric_only=True)
            df_initiative_members_share_per_region.reset_index(inplace=True)

            # calculate member share for World
            df_initiative_members_share_per_region_World.drop(["Country_name_x", "Country_name_y","IMAGE_Region_Nr", "ISO", "Included by NCI?"], axis=1, inplace=True)
            df_initiative_members_share_per_region_World["w_value"] = df_initiative_members_share_per_region_World["Value"] * df_initiative_members_share_per_region_World["Country_share"] * df_initiative_members_share_per_region_World["Member"]
            share_World = df_initiative_members_share_per_region_World["w_value"].sum()/df_initiative_members_share_per_region_World["Value"].sum()
            df_tmp = pd.DataFrame({"IMAGE_Region_Nr":[28], "Country_member_share":[share_World]})
            #df_tmp.set_index("IMAGE_Region_Nr", inplace=True)
            df_initiative_members_share_per_region = pd.concat([df_initiative_members_share_per_region, df_tmp], axis=0, ignore_index=True)
            print(f"\033[32mGlobal signatories share for {ICI} in {self.hist_year}: {share_World}\033[0m")

        df_initiative_members_share_per_region["ICI"] = ICI
        df_initiative_members_share_per_region.to_csv(f"{DIR}/data/output/downscale_tool/{output_dir}/df_initiative_members_share_per_region_{ICI}.csv", sep=";")
        df_initiative_members_share_per_region.to_csv(f"{DIR}/runlog/downscale_tool/{output_dir}/df_initiative_members_share_per_region_{ICI}.csv", sep=";")

        return df_initiative_members_share_per_region

    def _construct_new_pathway_initiatives(self, df_IMAGE_baseline, IMAGE_regions_ISO2Nr, df_initiative_members_share_per_region,  ICI, df_IMAGE_mitigation_scenario, output_dir):
    # construct new pathway were members follow GL_scenario and other NP_Scenario based on member shares

        # - member scneario for regions excluding world and flaring
        if not ICI == "Flaring":
            df_IMAGE_baseline_tmp = df_IMAGE_baseline.copy()
            df_IMAGE_baseline_tmp["Region_nr"] = df_IMAGE_baseline_tmp["Region"].map(IMAGE_regions_ISO2Nr)
            mask = ~(df_IMAGE_baseline_tmp["IMAGE_Region_Nr"]==28)
            df_IMAGE_baseline_tmp = df_IMAGE_baseline_tmp[mask]
            df_IMAGE_baseline_share = pd.merge(df_IMAGE_baseline_tmp, df_initiative_members_share_per_region, left_on="Region_nr", right_on="IMAGE_Region_Nr", how="left")
            df_IMAGE_baseline_share.to_csv(f"{DIR}/runlog/downscale_tool/{output_dir}/df_IMAGE_baseline_share1_{ICI}.csv", sep=";")
            df_IMAGE_baseline_share["Value_share"] = (1-df_IMAGE_baseline_share["Country_member_share"])*df_IMAGE_baseline_share["Value"]
            df_IMAGE_baseline_share.rename(columns={"IMAGE_Region_Nr_x":"IMAGE_Region_Nr"}, inplace=True)
            df_IMAGE_baseline_share = df_IMAGE_baseline_share.loc[:, ["Region", "Year", "Variable", "Value_share", "IMAGE_Region_Nr", "Country_member_share"]]
            #df_IMAGE_baseline_share = df_IMAGE_baseline_share[df_IMAGE_baseline_share["Region"]!="World"]
            df_IMAGE_baseline_share.to_csv(f"{DIR}/runlog/downscale_tool/{output_dir}/df_IMAGE_baseline_share2_{ICI}.csv", sep=";")

            # - Initiatives
            df_IMAGE_mitigation_scenario_tmp = df_IMAGE_mitigation_scenario.copy()
            df_IMAGE_mitigation_scenario_tmp["IMAGE_Region_Nr"] = df_IMAGE_mitigation_scenario_tmp["Region"].map(IMAGE_regions_ISO2Nr)
            mask = ~(df_IMAGE_mitigation_scenario_tmp["IMAGE_Region_Nr"]==28)
            df_IMAGE_mitigation_scenario_tmp = df_IMAGE_mitigation_scenario_tmp[mask]
            df_IMAGE_mitigation_scenario_share = pd.merge(df_IMAGE_mitigation_scenario_tmp, df_initiative_members_share_per_region, on="IMAGE_Region_Nr", how="left")
            df_IMAGE_mitigation_scenario_share.to_csv(f"{DIR}/runlog/downscale_tool/{output_dir}/df_IMAGE_mitigation_share1_{ICI}.csv", sep=";")
            df_IMAGE_mitigation_scenario_share["Value_share"] = df_IMAGE_mitigation_scenario_share["Country_member_share"]*df_IMAGE_mitigation_scenario_share["Value"]
            df_IMAGE_mitigation_scenario_share.rename(columns={"IMAGE_Region_Nr_x":"IMAGE_Region_Nr"}, inplace=True)
            df_IMAGE_mitigation_scenario_share.to_csv(f"{DIR}/runlog/downscale_tool/{output_dir}/df_IMAGE_mitigation_scenario_share_{ICI}.csv", sep=";")
            df_IMAGE_mitigation_scenario_share = df_IMAGE_mitigation_scenario_share.loc[:, ["Region", "Variable", "Year", "Value_share", "IMAGE_Region_Nr", "Country_member_share"]]
            df_IMAGE_mitigation_scenario_share.to_csv(f"{DIR}/runlog/downscale_tool/{output_dir}/df_IMAGE_mitigation_share2_{ICI}.csv", sep=";")

            # Merge baseline and Climate initiative projections per region
            df_IMAGE_mitigation_scenario_members = pd.merge(df_IMAGE_baseline_share, df_IMAGE_mitigation_scenario_share, on=["Region", "Variable", "Year"], how="left")
            df_IMAGE_mitigation_scenario_members["Value"] = df_IMAGE_mitigation_scenario_members["Value_share_x"] + df_IMAGE_mitigation_scenario_members["Value_share_y"]
            df_IMAGE_mitigation_scenario_members.to_csv(f"{DIR}/runlog/downscale_tool/{output_dir}/df_IMAGE_mitigation_scenario_members_{ICI}_extended.csv", sep=";")
            df_IMAGE_mitigation_scenario_members = df_IMAGE_mitigation_scenario_members.loc[:, ["Region", "Variable", "Year", "Value"]]

        # add World
        if not ICI == "Flaring":
            df_IMAGE_mitigation_scenario_members_World = df_IMAGE_mitigation_scenario_members.groupby(["Year", "Variable"]).sum(numeric_only=True).reset_index()
            df_IMAGE_mitigation_scenario_members_World["Region"] = "World"
            df_IMAGE_mitigation_scenario_members = pd.concat([df_IMAGE_mitigation_scenario_members, df_IMAGE_mitigation_scenario_members_World], axis=0, ignore_index=True)
            df_IMAGE_mitigation_scenario_members.to_csv(f"{DIR}/runlog/downscale_tool/{output_dir}/df_IMAGE_mitigation_scenario_members_{ICI}.csv", sep=";")
        else: # flaring
            df_IMAGE_baseline_share = df_IMAGE_baseline.copy()
            df_IMAGE_baseline_share.loc[df_IMAGE_baseline_share["Region"] != "World", ["Country_member_share"]] = 1
            df_IMAGE_baseline_share.loc[df_IMAGE_baseline_share["Region"] == "World", ["Country_member_share"]] = 1 - df_initiative_members_share_per_region.loc[df_initiative_members_share_per_region["IMAGE_Region_Nr"] == 28, "Country_member_share"].values[0]
            df_IMAGE_baseline_share.drop(["Model", "Scenario", "Unit"], axis=1, inplace=True)
            mask = (df_IMAGE_baseline_share["Region"] == "World") #& (df_IMAGE_baseline_share["Year"]!=2020)
            df_IMAGE_baseline_share = df_IMAGE_baseline_share[mask]
            df_IMAGE_baseline_share["Value_share"] = df_IMAGE_baseline_share["Value"]*df_IMAGE_baseline_share["Country_member_share"]
            df_IMAGE_baseline_share.drop(["Value"], axis=1, inplace=True)
            # determine mitigation scenario share (World only)
            df_IMAGE_mitigation_scenario_share = df_IMAGE_mitigation_scenario.copy()
            df_IMAGE_mitigation_scenario_share.loc[df_IMAGE_mitigation_scenario_share["Region"] != "World", ["Country_member_share"]] = 0
            df_IMAGE_mitigation_scenario_share.loc[df_IMAGE_mitigation_scenario_share["Region"] == "World", ["Country_member_share"]] = df_initiative_members_share_per_region.loc[df_initiative_members_share_per_region["IMAGE_Region_Nr"]==28, "Country_member_share"].values[0]
            df_IMAGE_mitigation_scenario_share.drop(["Model", "Scenario", "Unit"], axis=1, inplace=True)
            mask = (df_IMAGE_mitigation_scenario_share["Region"] == "World") #& (df_IMAGE_mitigation_scenario_share["Year"]!=2020)
            df_IMAGE_mitigation_scenario_share = df_IMAGE_mitigation_scenario_share.loc[mask]
            df_IMAGE_mitigation_scenario_share["Value_share"] = df_IMAGE_mitigation_scenario_share["Value"]*df_IMAGE_mitigation_scenario_share["Country_member_share"]
            df_IMAGE_mitigation_scenario_share.drop(["Value"], axis=1, inplace=True)

            # Merge flaring baseline and initiative projections
            df_IMAGE_mitigation_scenario_members = pd.merge(df_IMAGE_baseline_share, df_IMAGE_mitigation_scenario_share, on=["Region", "Variable", "Year"], how="left")
            df_IMAGE_mitigation_scenario_members["Value"] = df_IMAGE_mitigation_scenario_members["Value_share_x"] + df_IMAGE_mitigation_scenario_members["Value_share_y"]
            df_IMAGE_mitigation_scenario_members.to_csv(f"{DIR}/runlog/downscale_tool/{output_dir}/df_IMAGE_mitigation_scenario_members_{ICI}_extended.csv", sep=";")
            df_IMAGE_mitigation_scenario_members = df_IMAGE_mitigation_scenario_members.loc[:, ["Region", "Variable", "Year", "Value"]]

        return df_IMAGE_mitigation_scenario_members

    def downscale_to_member_countries(self, df_IMAGE_baseline, df_IMAGE_mitigation_scenario, df_hist_IMAGE_Countries, df_initiative_members,
                                      hist_year, ICI, IMAGE_regions_ISO2Nr, output_dir):

        # 3. DOWNSCALE GLOBAL RESULTS TO PARTICIPATING COUNTRIES IN INITIATIVE
        # based on historical year
        # calcualte % of participating countries in each region
        # this % is kept constant over the projection period
        # - members follow Glasgow Initiative scenario
        # - other follow baseline (input)
        print(f"Output type: {output_dir}")
        #var = df_IMAGE_mitigation_scenario["Variable"].iloc[0]
        # Flaring has no country members, only companies

        # init
        self.hist_year = hist_year

        # Calculate country share in region for emissions variable ICI in historical year
        df_var_country_share_in_region = self.calculate_country_share_in_region(df_hist_IMAGE_Countries, output_dir)

        # Calculate share per region covered by members
        df_initiative_members_share_per_region = self.calculate_member_perc_in_region(df_var_country_share_in_region, df_initiative_members, ICI, output_dir)

        # calculate new pathway
        df_IMAGE_mitigation_scenario_members = self._construct_new_pathway_initiatives(df_IMAGE_baseline, IMAGE_regions_ISO2Nr, df_initiative_members_share_per_region,  ICI,
                                                                                       df_IMAGE_mitigation_scenario, output_dir)

        return (df_var_country_share_in_region, df_IMAGE_mitigation_scenario_members, df_initiative_members_share_per_region)
