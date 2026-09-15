models = {
    "IMAGE": {"factor_GDP_PPP": 1.10774,
              "factor_year_from": 2010,
              "factor_year_to": 2005,
              "model_unit_conversions": {"Emissions|CO2": 1e6,
                                         "GDP|MER": 1e9,
                                         "GDP|PPP": 1e9,
                                         "Population": 1e6
                                        },

              "file_IAM_model_country_region": "data/input/models/IMAGE/IMAGE_country_to_regions.csv",
              "file_IAM_model_region_numbers": "data/input/models/IMAGE/IMAGE_region_numbers.csv",
              "file_model_grid_regions": "IMAGE_GADM_regions_raster.nc",
              "file_IAM_regions_grid": "data/input/models/IMAGE/IMAGE_GADM_regions_raster.nc",

              "vars_downscaling": [
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
              },

    "IMAGE_ScenarioMIP": {"factor_GDP_PPP": 1.10774,
              "factor_year_from": 2010,
              "factor_year_to": 2005,
              "model_unit_conversions": {"Emissions|CO2": 1e6,
                                         "GDP|MER": 1e9,
                                         "GDP|PPP": 1e9,
                                         "Population": 1e6
                                        },
              "file_IAM_model_country_region": "data/input/models/IMAGE_ScenarioMIP/IMAGE_ScenarioMIP_country_to_regions.csv",
              "file_IAM_model_region_numbers": "data/input/models/IMAGE_ScenarioMIP/IMAGE_ScenarioMIP_region_numbers.csv",
              "file_model_grid_regions": "IMAGE_ScenarioMIP_GADM_regions_raster.nc",
              "file_IAM_regions_grid": f"data/input/models/IMAGE_ScenarioMIP/IMAGE_ScenarioMIP_GADM_regions_raster.nc",

              "vars_downscaling": [
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
                                            #"Emissions|CO2|Energy|Demand|Transportation|Domestic Shipping",
                                            "Emissions|CO2|AFOLU",
                                            "Gross Emissions|CO2|Energy|Supply",
                                            "Gross Emissions|CO2|Energy|Demand",
                                            "Gross Emissions|CO2|Energy|Demand|Industry"
                                            ]
              },

      "REMIND_ScenarioMIP": {"factor_GDP_PPP": 1.10774,
              "factor_year_from": 2010,
              "factor_year_to": 2005,
              "model_unit_conversions": {"Emissions|CO2": 1e6,
                                         "GDP|MER": 1e9,
                                         "GDP|PPP": 1e9,
                                         "Population": 1e6
                                        },

              "file_IAM_model_country_region": "data/input/models/REMIND_ScenarioMIP/REMIND_ScenarioMIP_country_to_regions.csv",
              "file_IAM_model_region_numbers": "data/input/models/REMIND_ScenarioMIP/REMIND_ScenarioMIP_region_numbers.csv",
              "file_model_grid_regions": "REMIND_ScenarioMIP_GADM_regions_raster.nc",
              "file_IAM_regions_grid": "data/input/models/REMIND_ScenarioMIP/REMIND_ScenarioMIP_GADM_regions_raster.nc",

              "vars_downscaling": [
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
                                  #"Emissions|CO2|Energy|Demand|Transportation|Domestic Shipping",
                                  "Emissions|CO2|AFOLU",
                                  "Gross Emissions|CO2|Energy|Supply",
                                  "Gross Emissions|CO2|Energy|Demand",
                                  "Gross Emissions|CO2|Energy|Demand|Industry"
                                  ]
              }
            }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
#model                         = "IMAGE"

