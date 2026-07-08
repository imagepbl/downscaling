import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd

import downscaling.downscaling as downscaling

# read in grid emissions
varname_EM = "Emissions_CO2_Excl_shipping_aviation_AFOLU"
dir_EM = "Z:/cold_data_storage/users/roelfsemam/surdrive_UU/NSA/Downscaling_share/data_downscaling/emissions/processed/EDGAR/2024"
path_EM = f"{dir_EM}/Emissions_CO2_Excl_shipping_aviation_AFOLU.nc"
xr_EM = xr.open_dataset(path_EM)
print(f"{xr_EM[varname_EM].attrs["unit"]}")

# read in urban classification
dir_urban = "data/input/DLL"
path_urban = f"{dir_urban}/urban_classification_years.parquet"
gdf_urban = gpd.read_parquet(path_urban)

xr_em_urban, df_em_urban = downscaling.aggregate_emissions_urban(xr_emissions=xr_EM, gdf_urban_classification=gdf_urban,
                                                                 emissions_varname="Emissions_CO2_Excl_shipping_aviation_AFOLU",
                                                                 region_varname="region_number",
                                                                 suffix="_first_round")
