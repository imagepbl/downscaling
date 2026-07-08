models = {
    "IMAGE": {"factor_GDP_PPP": 1.10774,
              "factor_year_from": 2010,
              "factor_year_to": 2005,
              "model_unit_conversions": {"Emissions|CO2": 1e6,
                                         "GDP|MER": 1e9,
                                         "GDP|PPP": 1e9,
                                         "Population": 1e6
                                        },

              "file_model_grid_regions": "IMAGE_GADM_regions_raster.nc",
            #   "file_model_grid_regions_0_50": "IMAGE_GADM_regions_raster_0_50_arcmin.nc",
            #   "file_model_grid_regions_6_00": "IMAGE_GADM_regions_raster_6_00_arcmin.nc",
              "file_IAM_model_region_numbers": "image_region_numbers.csv",

              "file_IAM_model_country_region": "data/input/models/IMAGE/country_to_regions.csv",
              "file_IAM_model_region_numbers": "data/input/models/IMAGE/image_region_numbers.csv",
              "file_IAM_regions_grid": "data/input/models/IMAGE/IMAGE_GADM_regions_raster.nc"
              }
            }

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
#model                         = "IMAGE"

