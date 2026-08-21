# ---------------------------------------------------------------------------
# Source profiles
# ---------------------------------------------------------------------------
SOURCE_PROFILES = {
    "default": {
        "source_POP": "2UP",
        "version_POP": "GHSL_2024_M3",   # options: "M3", "GHSL_2024_M1", "M1", "version_2", "version_3"
        "source_GDP": "Murakami",
        "version_GDP": "version_2021_1",  # options: "version_7", "version_3"
        "source_EM":  "CEDS_CMIP7",
        "version_EM": "2025_04_18",       # options: "2024", "2025_04_18"
    },
    "fifth_round": {
        "source_POP": "COMPASS",
        "version_POP": "version_2",
        "source_GDP": "COMPASS",
        "version_GDP": "version_2",
        "source_EM":  "CEDS_CMIP7",
        "version_EM": "2025_04_18",
    },
    "fourth_round": {
        "source_POP": "Zhuang",
        "version_POP": "version_1",
        "source_GDP": "Murakami",
        "version_GDP": "version_2021_1",
        "source_EM":  "CEDS_CMIP7",
        "version_EM": "2025_04_18",
    },
    "third_round": {
        "source_POP": "2UP",
        "version_POP": "GHSL_2024_M3",
        "source_GDP": "Murakami",
        "version_GDP": "version_2021_1",
        "source_EM":  "CEDS_CMIP7",
        "version_EM": "2025_04_18",
    },
    "second_round_v2": {
        "source_POP": "2UP",
        "version_POP": "GHSL_2024_M3",
        "source_GDP": "Murakami",
        "version_GDP": "version_2021_1",
        "source_EM":  "EDGAR",
        "version_EM": "2024",
    },
    "second_round": {
        "source_POP": "2UP",
        "version_POP": "GHSL_2024_M3",
        "source_GDP": "Murakami",
        "version_GDP": "version_2021_1",
        "source_EM":  "EDGAR",
        "version_EM": "2024",
    },
    "first_round": {
        "source_POP": "2UP",
        "version_POP": "GHSL_2024_M3",
        "source_GDP": "Wang",
        "version_GDP": "version_7",
        "source_EM":  "EDGAR",
        "version_EM": "2024",
    }
}


# ---------------------------------------------------------------------------
# Process flags
# ---------------------------------------------------------------------------
process_flags = {
    "read_process_IAM": False,
    "read_process_POP": False,
    "read_process_GDP_PPP": False,
    "read_process_EM": False,
    "process_GDP_POP_grid": False,
    "process_GDP_per_POP": False,
    "process_df_EM_per_GDP": False,
    "process_grid_EM_per_GDP": False,
    "save_tiffs_intermediate": True,
    "save_tiffs_results": True,
    "process_urban_classification": True,
    "process_SE": True
}

check_flags = {
    "check_POP_data": False,
    "check_GDP_data": False,
    "check_EM_data": False,
    "check_GDP_POP": False,
    "check_IAM_data": False,
    "check_IAM_grid_data": False,
    "check_grid_GDP_per_pop": False,
    "check_IAM_GDP_per_pop": False,
    "check_SE_correction_factors": False,
    "check_SE_harmonised": False,
    "check_emissions": False,
}

# ---------------------------------------------------------------------------
# Variable names
# ---------------------------------------------------------------------------
varname_POP = "Population"
varname_GDP = "GDP|PPP"
varname_EM  = "Emissions_CO2_Excl_shipping_aviation_AFOLU"
varname_gdp_per_pop = "GDP (PPP) per capita"
varname_em_per_gdp_ppp = f"{varname_EM}_per_{varname_gdp_per_pop}"  # derived

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
unit_POP     = "people"
unit_GDP_PPP = "USD_2005/yr"
unit_EM      = "tonnes CO2/year"

# ---------------------------------------------------------------------------
# Downscaling
# ---------------------------------------------------------------------------
SSP_base          = "SSP2"
base_year         = 2020
convergence_year  = 2150
method_extension  = 2  # 1: growth rate from last two steps, 2: zero growth rate
                       # 3: growth rate to near-zero at convergence year, 4: absolute growth rate

years_downscaling = [2020, 2025, 2030, 2035, 2040, 2045, 2050, 2060, 2070, 2080, 2090, 2100]

# TO DO --> move this to settings_models.py
vars_downscaling = [
    "Population",
    "GDP|PPP",
    "Emissions|CO2",
    "Emissions|CO2|Energy|Supply",
    "Emissions|CO2|Energy|Demand",
    "Emissions|CO2|Energy|Demand|Industry",
    "Emissions|CO2|Energy|Demand|Transportation",
    "Emissions|CO2|Energy|Demand|Residential and Commercial",
    "Emissions|CO2|Energy|Demand|Other Sector",
    "Emissions|CO2|Energy|Demand|AFOFI",
    "Emissions|CO2|Industrial Processes",
    "Emissions|CO2|Energy|Demand|Bunkers|International Aviation",
    #"Emissions|CO2|Energy|Demand|Bunkers|International Shipping", # shipping is only available at World region, which is excluded
    "Emissions|CO2|Energy|Demand|Transportation|Domestic Aviation",
    "Emissions|CO2|Energy|Demand|Transportation|Domestic Shipping",
    "Emissions|CO2|AFOLU",
    "Gross Emissions|CO2|Energy|Supply",
    "Gross Emissions|CO2|Energy|Demand",
    "Gross Emissions|CO2|Energy|Demand|Industry"
    ]

