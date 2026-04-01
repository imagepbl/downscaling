from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
from affine import Affine
import xarray as xr

from tools.general_functions import replace_punctuation_in_filenames
from downscaling.read_process_grid_data import calculate_resolution

DIR = Path(__file__).parent

# read in processsed data
# population_2UP_GHSL_2024_M3_SSP2.nc_12.nc
# gdp_ppp_Wang_SSP2_version_7_cf_12.nc
# emissions_CO2_excl_shipping_aviation_EDGAR_cf_1.nc

# Three sections
# 1. General
# 2. GDP per capita
# 3. EM per GDP (PPP)

#**************************GENERAL*******************************************
def sum_grid_to_IAM_regions(xr_main:xr.DataArray, xr_weight:xr.DataArray, xr_regions):
    """
    Calculate weighted sum for each region and each time step using full vectorization.
    Returns a DataFrame with region_number as index and years as columns.
    """
    varname_main = xr_main.name
    varname_weight = xr_weight.name

    xr_main = xr_main.chunk({"time": 1, "y": 1024, "x": 1024})

    # Reindex weight data to match main data grid
    #xr_weight_reindex = xr_weight.reindex_like(xr_main, method="nearest")
    xr_weight_reindex = xr_weight.copy()
    region_numbers = xr_regions["region_number"].compute()

    # Add region information as coordinates
    xr_main = xr_main.assign_coords(region_number=(["y", "x"], region_numbers.values))
    xr_weight_reindex = xr_weight_reindex.assign_coords(region_number=(["y", "x"], region_numbers.values))

    print("Calculating weighted average by region and time (vectorized)...")

    # Add region information to both datasets
    #xr_main["region_number"] = xr_regions["region_number"].compute()
    #xr_weight_reindex["region_number"] = xr_regions["region_number"].compute()
    xr_main = xr_main.assign_coords(region_number=(["y", "x"], region_numbers.values))
    xr_weight_reindex = xr_weight_reindex.assign_coords(region_number=(["y", "x"], region_numbers.values))

    print("Calculating weighted average by region and time (vectorized)...")

    # # Create valid mask
    valid_mask = (
        ~np.isnan(xr_main) &
        ~np.isnan(xr_weight_reindex) &
    #     ~np.isinf(xr_main) &
    #     ~np.isinf(xr_weight_reindex) &
         (xr_weight_reindex > 0) &
         (region_numbers > 0) &
         ~np.isnan(region_numbers)
    )

    # Apply mask
    main_masked = xr_main.where(valid_mask)
    weight_masked = xr_weight_reindex.where(valid_mask)
    #main_masked = xr_main.where(xr_main["region_number"] > 0)
    #weight_masked = xr_weight_reindex.where(xr_weight_reindex["region_number"] > 0)

    # Stack spatial dimensions
    main_stacked = main_masked.stack(space=["y", "x"])
    weight_stacked = weight_masked.stack(space=["y", "x"])

    combined = xr.Dataset({varname_main: main_stacked,varname_weight: weight_stacked})
    combined = combined.dropna(dim="space", how="any")

    main_stacked = combined[varname_main]
    weight_stacked = combined[varname_weight]

    print(np.unique(main_stacked["region_number"]))
    print(np.unique(weight_stacked["region_number"]))

    # Group by region and calculate weighted average for all regions at once
    numerator = (main_stacked * weight_stacked).groupby("region_number").sum(dim="space")
    denominator = weight_stacked.groupby("region_number").sum(dim="space")

    weighted_avg = numerator / denominator

    # Convert to DataFrame
    df_results = weighted_avg.to_dataframe(name="value").reset_index()
    df_results.drop(["spatial_ref"], axis=1, inplace=True, errors='ignore')
    df_results.rename(columns={"time": "year"}, inplace=True, errors='ignore')

    return df_results

def compare_IAM_grid_regions_GDP_per_capita(ds:xr.Dataset, ds_varname:str,
                             ds_weight:xr.Dataset, ds_weight_varname,
                             df_IAM:pd.DataFrame,
                             xr_IAM_regions_grid:xr.Dataset,
                             selection_years=[2020, 2030]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Compare the regional difference between IAM and grid data for GDP per capita
    '''
    df_grid_regional = sum_grid_to_IAM_regions(ds[ds_varname], ds_weight[ds_weight_varname], xr_IAM_regions_grid)
    #df_grid_gdp_ppp_per_population_regional["region_code"] = df_grid_gdp_ppp_per_population_regional["region_number"].map(convert_IMAGE_regions_to_names)
    df_grid_regional = df_grid_regional[df_grid_regional["region_number"]>0]
    df_grid_regional = df_grid_regional.sort_values(by=["year", "region_number"])
    # select
    df_grid_regional_selection = df_grid_regional[df_grid_regional["year"].isin(selection_years)]
    df_grid_regional_selection = df_grid_regional_selection.pivot(index=["region_number"], columns="year", values="value").reset_index()
    df_grid_regional_selection = df_grid_regional_selection.sort_values(by=["region_number"])
    # display
    df_display_grid = df_grid_regional_selection.copy()
    # Format region_number as integer
    df_display_grid['region_number'] = df_display_grid['region_number'].astype(int)
    # Format all other numeric columns with thousands separator
    numeric_cols = df_display_grid.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col != 'region_number':  # Skip region_number
            df_display_grid[col] = df_display_grid[col].apply(lambda x: f'{x:,.0f}')

    # compare projections and summed grid data per region
    df_IAM_projection_gdp_ppp_per_population_check = df_IAM.copy()
    df_IAM_projection_gdp_ppp_per_population_check.drop(["model", "scenario"], axis=1, inplace=True)
    df_IAM_projection_gdp_ppp_per_population_check.columns = df_IAM_projection_gdp_ppp_per_population_check.columns.str.lower()

    df_grid_gdp_ppp_per_population_regional_check = df_grid_regional.copy()
    df_grid_gdp_ppp_per_population_regional_check["value"] /= 1e3  # convert to million unit

    df_gdp_ppp_per_population_regional_check = pd.merge(df_IAM_projection_gdp_ppp_per_population_check, df_grid_gdp_ppp_per_population_regional_check, on=["region_number", "year"], suffixes=("_IAM", "_grid_sum"))
    df_gdp_ppp_per_population_regional_check["diff_perc"] = 100*df_gdp_ppp_per_population_regional_check["value_grid_sum"]/df_gdp_ppp_per_population_regional_check["value_IAM"]
    df_gdp_ppp_per_population_regional_check = df_gdp_ppp_per_population_regional_check[df_gdp_ppp_per_population_regional_check["year"].isin(selection_years)]
    df_gdp_ppp_per_population_regional_check = df_gdp_ppp_per_population_regional_check.pivot(index=["region_number"], columns="year", values=["value_IAM", "value_grid_sum", "diff_perc"])
    # Create a formatted copy
    df_display_compare = df_gdp_ppp_per_population_regional_check.copy()
    for col in df_display_compare.columns:
        df_display_compare[col] = df_display_compare[col].apply(lambda x: f'{x:.1f}')

    return df_display_grid, df_display_compare

#**************************GDP PER CAPITA*******************************************
def check_location_for_GDP_per_pop_calculation(ds:xr.Dataset, varname):

    # Define city center coordinates (lat, lon)
    cities = {
        "Amsterdam": {"lat": 52.37, "lon": 4.89},
        "Berlin": {"lat": 52.52, "lon": 13.40},
        "New York": {"lat": 40.71, "lon": -74.01},
        "Beijing": {"lat": 39.90, "lon": 116.40}
    }

    # Extract values for each city
    # Note: adjust dimension names based on your dataset (lat/lon, y/x, or latitude/longitude)
    for city_name, coords in cities.items():
        point = ds.sel(y=coords["lat"], x=coords["lon"], method="nearest")

        # Get the GDP per capita value (adjust variable name as needed)
        gdp_value = point[varname].values

        # Show actual coordinates of selected pixel (for verification)
        #actual_lat = point["lat"].values
        #actual_lon = point["lon"].values
        actual_y = point["y"].values
        actual_x = point["x"].values

        print(f"{city_name}: GDP per capita = {gdp_value}")
        print(f"  Actual coordinates: lat={actual_y}, lon={actual_x}")

def check_POP_GDP_alignment(dir_processed:Path, xr_population_processed, xr_gdp_ppp_processed, varname_POP, varname_GDP):

    save_dir = dir_processed / "check"
    save_dir.mkdir(parents=True, exist_ok=True)

    gdp = xr_gdp_ppp_processed[varname_GDP]
    population = xr_population_processed[varname_POP]

    # determine values where GDP is a number (not NaN or zero) and population is NaN, zero, or a number
    valid_mask = (gdp != 0) & gdp.notnull()
    valid_mask_computed = valid_mask.compute()

    pop_vals = population.values[valid_mask_computed.values]
    gdp_vals = gdp.values[valid_mask_computed.values]

    df = pd.DataFrame({"population": population.values[valid_mask.values],
                       "gdp": gdp.values[valid_mask.values]})
    unique_combinations = df.drop_duplicates()
    csv_file_path_check = save_dir / "unique_combinations.csv"
    unique_combinations.to_csv(csv_file_path_check, sep=";", index=False)

    # determine values where GDP is a number (not NaN or zero) and population is +1
    gdp_where_pop_1 = unique_combinations[unique_combinations["population"] == 1]["gdp"]
    csv_file_path_one = save_dir / "pop_1_gdp_value.csv"
    gdp_where_pop_1.to_csv(csv_file_path_one, sep=";", index=False)

    # determine values where GDP is a number (not NaN or zero) and population is zero
    gdp_where_pop_0 = unique_combinations[unique_combinations["population"] == 0]["gdp"]
    csv_file_path_zero = save_dir / "pop_0_gdp_value.csv"
    gdp_where_pop_0.to_csv(csv_file_path_zero, sep=";", index=False)

    # determine values where GDP is a number (not NaN or zero) and population is NaN
    gdp_where_pop_nan = unique_combinations[unique_combinations["population"].isna()]["gdp"]
    csv_file_path_nan = save_dir / "pop_nan_gdp_value.csv"
    gdp_where_pop_nan.to_csv(csv_file_path_nan, sep=";", index=False)

    # plot
    years = [2020, 2030, 2050]
    counts = {"NaN": [], "Zero": [], "Number": []}
    for year in years:
        gdp_year = gdp.sel(time=year)
        pop_year = population.sel(time=year)

        valid_gdp_mask = (gdp_year != 0) & gdp_year.notnull()
        pop_valid_gdp = pop_year.values[valid_gdp_mask.values]

        counts["NaN"].append(np.sum(np.isnan(pop_valid_gdp)))
        counts["Zero"].append(np.sum(pop_valid_gdp == 0))
        counts["Number"].append(np.sum((pop_valid_gdp != 0) & ~np.isnan(pop_valid_gdp)))

    x = np.arange(len(years))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width, counts["NaN"], width, label="NaN")
    ax.bar(x, counts["Zero"], width, label="Zero")
    ax.bar(x + width, counts["Number"], width, label="Number")

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of cells")
    ax.set_title("Population category per year (for cells where GDP is a number)")
    ax.legend()

    plt.tight_layout()
    save_path = save_dir / "population_gdp_alignment.png"
    plt.savefig(save_path)


def process_factors_GDP_POP(ds_population:xr.Dataset, ds_gdp_ppp:xr.Dataset, rxr_emissions:xr.Dataset,
                            varname_population:str, varname_gdp_ppp:str,
                            unit_population:str, unit_gdp_ppp:str,
                            years_downscaling,
                            check=False) -> Tuple[xr.Dataset, xr.Dataset]:
    '''
    1. align coordinates datasets
    2. align downscaling years
    2. set 1 where population is nan and gdp is not nan

    '''
    # 2. align xr_population with downscaling years (downscaling)
    #ds_population_aligned = ds_population.copy()
    #ds_gdp_aligned = ds_gdp_ppp.copy()

    ds_population_downscaling = ds_population.interp(time=years_downscaling, method="linear")
    print(f"(process_factors_GDP_POP) Years in population grid aligned with downscaling years: {ds_population_downscaling.time.values}")
    ds_gdp_ppp_downscaling = ds_gdp_ppp.interp(time=years_downscaling, method="linear")
    print(f"(process_factors_GDP_POP) Years in GDP (PPP) grid aligned with downscaling years: {ds_gdp_ppp_downscaling.time.values}")
    # check
    total_pop_2020 = ds_population_downscaling[varname_population].sel(time=2020).sum().compute().item()
    total_gdp_2020 = ds_gdp_ppp_downscaling[varname_gdp_ppp].sel(time=2020).sum().compute().item()
    print(f"(process_factors_GDP_POP) 1.Total population: {total_pop_2020:,.0f}")
    print(f"(process_factors_GDP_POP) 1.Total GDP (PPP): {total_gdp_2020:,.0f}")

    # # # 1. align coordinates with emissions grid (aligned)
    # ds_population_aligned = ds_population_downscaling.reindex_like(rxr_emissions.sel(time=2020), method="nearest", tolerance=0.01)
    ds_population_aligned = ds_population_downscaling.copy()
    ds_gdp_ppp_aligned = ds_gdp_ppp_downscaling.reindex_like(ds_population_downscaling.sel(time=2020), method="nearest", tolerance=0.01)

    # 3.set population to 1 where population is nan and gdp is not nan (adjusted)
    ds_population_adjusted = ds_population_aligned.copy()
    ds_gdp_adjusted = ds_gdp_ppp_aligned.copy()
    mask_pop_nan = (ds_population_adjusted[varname_population].isnull()) & (ds_gdp_adjusted["GDP|PPP"]>0)
    total_pop_2020 = None
    check_pop = None
    num_cells_pop_nan_gdp_not_nan_2020 = None
    if check:
        print("(process_factors_GDP_POP) Checking population and GDP (PPP) datasets after coarsening ...")
        total_pop_2020 = ds_population_adjusted[varname_population].sel(time=2020).sum().compute().item()
        total_gdp_2020 = ds_gdp_adjusted[varname_gdp_ppp].sel(time=2020).sum().compute().item()
        print(f"(process_factors_GDP_POP) Total population: {total_pop_2020:,.0f}")
        print(f"(process_factors_GDP_POP) Total GDP (PPP): {total_gdp_2020:,.0f}")
        mask_pop_nan_2020 = mask_pop_nan.sel(time=2020)
        # check 1: calculate number of cells where population is nan and gdp is not nan
        num_cells_pop_nan_gdp_not_nan_2020 = mask_pop_nan_2020.sum().compute().item()
        print(f"(process_factors_GDP_POP) Number of cells where population is NaN/zero and GDP (PPP) is not NaN: {num_cells_pop_nan_gdp_not_nan_2020:,.0f}")
        check_pop = total_pop_2020 + num_cells_pop_nan_gdp_not_nan_2020
        # check 2: calculate sum of gdp where population is nan, and also the percentage of total gdp
        sum_gdp_pop_nan_2020 = ds_gdp_adjusted["GDP|PPP"].sel(time=2020).where(mask_pop_nan_2020).compute().sum().item()
        print(f"(process_factors_GDP_POP) Sum of GDP (PPP) where population is NaN: {sum_gdp_pop_nan_2020:,.0f}")
        percentage_gdp_pop_nan_2020 = (sum_gdp_pop_nan_2020 / total_gdp_2020) * 100 if total_gdp_2020 != 0 else 0
        print(f"(process_factors_GDP_POP) Percentage of total GDP (PPP) where population is NaN: {percentage_gdp_pop_nan_2020:.2f}%")

    # change values to 1
    ds_population_adjusted[varname_population] = ds_population_adjusted[varname_population].where(~mask_pop_nan, 1)
    # check population after processing
    total_pop_aligned_2020 = ds_population_adjusted[varname_population].sel(time=2020).sum().compute().item()
    print("--------------------------------")
    if check_pop:
        print(f"(process_factors_GDP_POP) Check: Total population + number of cells with population NaN/zero and GDP not NaN: {check_pop:,.0f}")
    print(f"Total population + number of cells with population NaN/zero and GDP not NaN: {total_pop_aligned_2020:,.0f}")

    ds_population_processed = ds_population_adjusted
    ds_gdp_ppp_processed = ds_gdp_adjusted

    ds_population_processed.attrs["unit"] = unit_population
    ds_gdp_ppp_processed.attrs["unit"] = unit_gdp_ppp

    return ds_population_processed, ds_gdp_ppp_processed

def _transforms_are_close(t1: Affine, t2: Affine, rtol: float = 1e-5) -> bool:
    """Compare two Affine transforms element-wise with a tolerance."""
    return np.allclose(t1[:6], t2[:6], rtol=rtol)

def calculate_gdp_per_pop(ds_population, ds_gdp,
                          varname_POP, varname_GDP, varname_gpd_per_pop:str,
                          unit_pop:str, unit_gdp_ppp:str) -> xr.Dataset:
    '''
    Calculate GDP per capita
    GDP|PPP / Population
    '''

     # --- Check CRS and transform consistency between inputs ---
    crs_pop = ds_population.rio.crs
    crs_gdp = ds_gdp.rio.crs
    transform_pop = ds_population.rio.transform()
    transform_gdp = ds_gdp.rio.transform()
    print(f"CRS and transform for population dataset: CRS={crs_pop}, \nTransform={transform_pop}")
    print(f"CRS and transform for GDP dataset: CRS={crs_gdp}, \nTransform={transform_gdp}")

    if crs_pop is None or crs_gdp is None:
        raise ValueError(f"CRS missing: ds_population CRS = {crs_pop}, ds_gdp CRS = {crs_gdp}")
    if crs_pop != crs_gdp:
        raise ValueError(f"CRS mismatch: ds_population has {crs_pop}, ds_gdp has {crs_gdp}")
    if not _transforms_are_close(transform_pop, transform_gdp):
        raise ValueError(f"Transform mismatch:\n  ds_population: {transform_pop}\n  ds_gdp: {transform_gdp}")

    #unit_pop = ds_population[varname_POP].attrs["unit"]
    #unit_gdp_ppp = ds_gdp[varname_GDP].attrs["unit"]
    ds_gdp_per_pop = xr.where((ds_population[varname_POP] > 0), ds_gdp[varname_GDP] / ds_population[varname_POP], float("nan"))
    ds_gdp_per_pop.name = varname_gpd_per_pop
    ds_gdp_per_pop.attrs["unit"] = f"{unit_gdp_ppp} / {unit_pop}"

    ds_gdp_per_pop = ds_gdp_per_pop.to_dataset(name=varname_gpd_per_pop)

    # Write CRS and transform onto the result ---
    ds_gdp_per_pop = ds_gdp_per_pop.rio.write_crs(crs_pop)
    ds_gdp_per_pop = ds_gdp_per_pop.rio.write_transform(transform_pop)

    return ds_gdp_per_pop


#**************************EM per GDP (PPP) *******************************************

def calc_scaling_factors_EM_per_GDP(xr_IAM_regions_grid_downscaling:xr.Dataset,
                                    base_year:int,
                                    xr_gdp_ppp_by_downscaling:xr.DataArray,
                                    xr_em_per_gdp_ppp_by_downscaling:xr.DataArray,
                                    df_IAM_projection_em_per_GDP_PPP_downscaling_extended:pd.DataFrame) -> Tuple[xr.DataArray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate scaling factors for each grid cell based on the ratio of
    emissions per GDP in the base year (2020) between the grid cell and the

    Parameters:
    -----------
    em_per_gdp_by : xarray.DataArray
        Grid of emissions per GDP for 2020 (dimensions: x, y, time)
    region_grid : xarray.DataArray
        Grid indicating region ID for each cell (dimensions: x, y)
    df_regional : pandas.DataFrame
        Regional emissions per GDP with columns for each year
        (index: region IDs, columns: years)
    years_downscaling : list
        Years available in df_regional
    target_year : int
        Final year to interpolate to (default: 2150)

    Returns:
    --------
    xarray.Dataset with emissions per GDP for all years
    """

    # Get unique regions
    vals = xr_IAM_regions_grid_downscaling["region_number"].values
    regions = np.unique(vals[(vals > 0) & np.isfinite(vals)])
    print(f"Regions: {regions}")

    # Get grid dimensions
    x_coords = xr_gdp_ppp_by_downscaling.x
    y_coords = xr_gdp_ppp_by_downscaling.y
    print(f"Grid dimensions: x={len(x_coords)}, y={len(y_coords)}")

    # Get baseline (2020) grid values
    # baseline_grid = em_per_gdp_by.sel(time=2020).values

    # Calculate initial scaling factor for each grid cell (2020)
    # scaling_factor_by = CO2perGDP_grid(by) / CO2perGDP_region_IAM(by)
    np_scaling_factor_by = np.full((len(y_coords), len(x_coords)), np.nan)
    #convert = one_unit_IMAGE_GDP_PPP / one_unit_IMAGE_em
    for region_id in regions:
        region_mask_xr = (xr_IAM_regions_grid_downscaling["region_number"].values == region_id)

        region_mask_df = (df_IAM_projection_em_per_GDP_PPP_downscaling_extended["region_number"] == region_id) & (df_IAM_projection_em_per_GDP_PPP_downscaling_extended["year"] == base_year)
        baseline_regional_IAM = df_IAM_projection_em_per_GDP_PPP_downscaling_extended.loc[region_mask_df, "value"].iloc[0]

        if baseline_regional_IAM != 0:
            # Both in grid units now - no conversion needed
            np_scaling_factor_by[region_mask_xr] = xr_em_per_gdp_ppp_by_downscaling.values[region_mask_xr] / baseline_regional_IAM
        else:
            np_scaling_factor_by[region_mask_xr] = 1.0

    # Scaling factor in 2150 is 1 for all grid cells
    scaling_factor_2150 = 1.0

    # Convert to xarray DataArray
    xr_scaling_factor_by = xr.DataArray(
        np_scaling_factor_by,
        coords={"y": y_coords, "x": x_coords},
        dims=["y", "x"],
        name="scaling_factor_by"
    )

    xr_scaling_factor_by = xr_scaling_factor_by.rio.write_crs(xr_em_per_gdp_ppp_by_downscaling.rio.crs)
    xr_scaling_factor_by = xr_scaling_factor_by.rio.write_transform(xr_em_per_gdp_ppp_by_downscaling.rio.transform())

    return xr_scaling_factor_by, regions, x_coords, y_coords

def downscale_em_per_gdp(xr_scaling_factor_by:xr.DataArray, varname_em_per_gdp_ppp:str,
                         xr_IAM_regions_grid_downscaling:xr.Dataset,
                         df_IAM_projection_em_per_GDP_PPP_downscaling:pd.DataFrame,
                         years_downscaling_extended, base_year, convergence_year,
                         regions, x_coords, y_coords) -> xr.Dataset:
    """
    Downscale emissions per GDP (PPP) to a gridded spatial resolution.

    This function downscales regional emissions per GDP values to a regular grid by applying
    spatially-varying scaling factors that evolve over time. The scaling factors transition
    linearly from their base-year values (CO2/GDP in 2020) to unity (1.0) over a specified convergence period,
    enabling the representation of convergence in regional emission intensities.

    Steps:
    1. For each year in the downscaling period:
       - Determine the appropriate scaling factor based on temporal interpolation:
         * Before base year: use base-year scaling factors
         * After convergence year: use scaling factor of 1.0 (full convergence)
         * Between years: linearly interpolate scaling factors between base and convergence
       - For each IAM region:
         * Retrieve the regional emissions per GDP value for the given year
         * Identify grid cells belonging to the region
         * Apply the scaling factor: downscaled_value = scaling_factor × regional_value
       - Store the resulting gridded data for the year
    2. Combine all yearly grids into a single xarray DataArray with time dimension
    3. Mask invalid (non-finite) values
    4. Convert to xarray Dataset and attach geospatial metadata (CRS, transform)
    5. Assign unit metadata from the input DataFrame

    Args:
        xr_scaling_factor_by: DataArray containing base-year scaling factors with spatial dimensions
        varname_em_per_gdp_ppp: Name for the output variable in the resulting Dataset
        xr_IAM_regions_grid_downscaling: Dataset containing spatial region identifiers (region_number)
        df_IAM_projection_em_per_GDP_PPP_downscaling: DataFrame with projected regional emissions per GDP
                                                      (columns: region_number, year, value, unit)
        years_downscaling_extended: Sequence of years to downscale
        base_year: Reference year for scaling factors (typically 2020)
        convergence_year: Year by which regional intensities converge to 1.0 (typically 2150)
        regions: Sequence of unique region identifiers
        x_coords: X-axis coordinates for the output grid
        y_coords: Y-axis coordinates for the output grid

    Returns:
        xr.Dataset: Gridded dataset with downscaled emissions per GDP containing:
                    - Variable with name varname_em_per_gdp_ppp
                    - Dimensions: (time, y, x)
                    - Coordinates: time, y, x
                    - Attributes: unit, CRS, and geospatial transform
    """

    result_list = []

    for year in years_downscaling_extended:
        print(f"Processing year {year} ...")

        # Linearly interpolate scaling factor between 2020 and 2150
        if year <= base_year:
            scaling_factor_year = xr_scaling_factor_by.values  # at by year, scalig factor is the same as in the base year
        elif year >= convergence_year:
            scaling_factor_year = np.ones_like(xr_scaling_factor_by.values)  # after convergence year, scaling factor is 1
        else: # Linear interpolation between by scalign and 1
            #weight of scaling factor
            weight = (year - base_year) / (convergence_year - base_year)
            scaling_factor_year = np.full_like(xr_scaling_factor_by.values, np.nan) # create empty array with np.nan with same shape as xr_scaling_factor_by
            finite_mask = np.isfinite(xr_scaling_factor_by.values)
            # check for infs
            scaling_factor_year[finite_mask] = (xr_scaling_factor_by.values[finite_mask] + (1 - xr_scaling_factor_by.values[finite_mask]) * weight)

        # Create empty grid for this year
        year_grid = np.full((len(y_coords), len(x_coords)), np.nan)

        # For each region, multiply scaling factor with regional CO2perGDP
        print(f"Processing region_id: ", end="")
        for region_id in regions:
            print(f"{region_id}...", end="")
            region_mask = (xr_IAM_regions_grid_downscaling["region_number"].values == region_id)

            # Get regional value for this year (should exist in df_regional)
            regional_mask = (df_IAM_projection_em_per_GDP_PPP_downscaling["region_number"] == region_id) & (df_IAM_projection_em_per_GDP_PPP_downscaling["year"] == year)
            regional_value = df_IAM_projection_em_per_GDP_PPP_downscaling.loc[regional_mask, "value"].iloc[0]  # Add .iloc[0] to get scalar

            # Apply: CO2perGDP_grid(year) = scaling_factor(year) * CO2perGDP_region(year)
            year_grid[region_mask] = scaling_factor_year[region_mask] * regional_value
        print()
        # Create DataArray for this year
        year_da = xr.DataArray(
            year_grid,
            coords={"y": y_coords, "x": x_coords, "time": year},
            dims=["y", "x"]
        )
        result_list.append(year_da)

    # Combine all years into a single DataArray
    xr_em_per_gdp_ppp = xr.concat(result_list, dim="time")
    xr_em_per_gdp_ppp = xr_em_per_gdp_ppp.where(np.isfinite(xr_em_per_gdp_ppp))

    # to dataset
    xr_em_per_gdp_ppp = xr_em_per_gdp_ppp.to_dataset(name=varname_em_per_gdp_ppp)
    xr_em_per_gdp_ppp = xr_em_per_gdp_ppp.rio.write_crs(xr_scaling_factor_by.rio.crs)
    xr_em_per_gdp_ppp = xr_em_per_gdp_ppp.rio.write_transform(xr_scaling_factor_by.rio.transform())
    xr_em_per_gdp_ppp[varname_em_per_gdp_ppp].attrs["unit"] = df_IAM_projection_em_per_GDP_PPP_downscaling['unit'].iloc[0]

    return xr_em_per_gdp_ppp

#************************** EM  *******************************************

def calc_urban_regional_emissions(xr_grid:xr.Dataset, varname:str,
                                  #xr_IAM_regions_grid_downscaling:xr.Dataset,
                                  xr_urban_classification:xr.Dataset,
                                  years_downscaling:list) -> pd.DataFrame:
    xr_urban_classification = xr_urban_classification.rename({"band_data": "urban_classification"})
    xr_grid["urban_classification"] = xr_urban_classification["urban_classification"].reindex_like(xr_grid, method="nearest", tolerance=1e-5)

    df_emissions_urban_regional_sums = (xr_grid
                                        .sel(time=years_downscaling)
                                        .groupby(["region_number", "urban_classification"])
                                        .sum()
                                        .to_dataframe()
                                        .rename(columns={varname: varname})
                                        .reset_index())
    df_emissions_urban_regional_sums.drop(["correction_factor", "spatial_ref", "band"], axis=1, inplace=True, errors='ignore')

    # add percentage emissions in urban areas per region
    group_sum = (df_emissions_urban_regional_sums
                 .groupby(["region_number", "time"])["Emissions_CO2_Excl_shipping_aviation_AFOLU"]
                 .transform("sum")
                .replace(0, np.nan))

    df_emissions_urban_regional_sums["percentage_class"] = (df_emissions_urban_regional_sums["Emissions_CO2_Excl_shipping_aviation_AFOLU"]/ group_sum* 100)

    return df_emissions_urban_regional_sums

def calc_regional_values(xr_grid:xr.Dataset, varname:str,
                            xr_IAM_regions_grid_downscaling:xr.Dataset,
                            df_IAM:pd.DataFrame,
                            years_downscaling:list) -> Tuple[pd.DataFrame, xr.Dataset]:

    # varname_save = varname.replace("|", "_").replace(" ", "_")

    # land_mask = (xr_IAM_regions_grid_downscaling["region_number"] > 0)
    # ocean_mask = (xr_IAM_regions_grid_downscaling["region_number"] == 0)

    # # Create grid with regional sums for se_indicator (for emissions, only base year)
    # xr_regional_sums = None
    # df_output = None

    # # Calculate regional sums for all downscaling years: determine region numbers for each grid cell
    # print("Determine region numbers for each grid cell")
    # region_numbers = xr_IAM_regions_grid_downscaling.region_number.compute()

    # # determine regional sums for grid data
    # print("Determine regional sums for grid data")
    # if "region_number" not in xr_grid.coords:
    #     region_numbers_clean = (region_numbers
    #                             .drop_vars([c for c in region_numbers.coords
    #                                         if c not in ("y", "x")])
    #                             .assign_coords(y=xr_grid.y, x=xr_grid.x))
    #     xr_grid = xr_grid.assign_coords(region_number=region_numbers_clean)
    # with ProgressBar():
    #     xr_regional_sums = (xr_grid
    #                         .sel(time=years_downscaling)
    #                         .where(land_mask)
    #                         .groupby("region_number")
    #                         .sum())
    # df_regional_sums_compare = None

    # prepare region ids once, outside the loop
    region_ids = xr_IAM_regions_grid_downscaling["region_number"].values.ravel().astype(int)
    n_regions = int(region_ids.max()) + 1

    # calculate regional sums one time step at a time using np.bincount
    print("Determine regional sums for grid data")
    regional_sums = []
    for year in years_downscaling:
        print(f"  {year}")
        values = xr_grid[varname].sel(time=year).values  # triggers compute for this slice only
        sums = np.bincount(
            region_ids,
            weights=np.nan_to_num(values.ravel(), nan=0.0),
            minlength=n_regions)
        regional_sums.append(sums)

    regional_sums_array = np.stack(regional_sums, axis=0)  # shape (time, n_regions)

    xr_regional_sums = xr.Dataset({
        varname: xr.DataArray(
            regional_sums_array,
            dims=["time", "region_number"],
            coords={"time": years_downscaling,
                    "region_number": np.arange(n_regions)})})

    print(varname)
    xr_regional_sums_check = xr_regional_sums.rename({varname: f"{varname}_grid"})
    print(f"Unique region numbers in regional summations: {np.unique(xr_regional_sums.region_number.values)}")
    print(f"Years in regional summations: {xr_regional_sums_check.time.values}")

    #---------------------
    # Compare grid and IAM regional sums for se_indicator (for emissions, only 2020)
    # for comparison, different variable names are created
    varname_grid = f"{varname}_grid"
    varname_grid_IAM = f"{varname}_IAM"
    varname_grid_summed = f"{varname}_grid_summed"

    # compare se_indicator from xr_se_regional_sums_check with xr_IAM_regions_grid_downscaling and add to csv file
    df_regional_sums_compare = pd.merge(
            xr_regional_sums_check[varname_grid]
            .to_dataframe()
            .reset_index()
            .rename(columns={"region_number": "region_number", "time": "year", varname_grid: varname_grid_summed}), df_IAM
            .rename(columns={"value": varname_grid_IAM})[["region_number", "year", varname_grid_IAM]], on=["region_number", "year"], how="left")
    df_regional_sums_compare = df_regional_sums_compare[df_regional_sums_compare["year"].isin(years_downscaling)].copy()

    df_regional_sums_compare["difference"] = df_regional_sums_compare[varname_grid_summed] - df_regional_sums_compare[varname_grid_IAM]
    df_regional_sums_compare["relative_difference_%"] = df_regional_sums_compare["difference"] / df_regional_sums_compare[varname_grid_IAM] * 100
    df_regional_sums_compare = df_regional_sums_compare.sort_values(["year", "region_number"]).reset_index(drop=True)
    df_regional_sums_compare["indicator_grid_xr_million"] = df_regional_sums_compare[varname_grid_summed] * 10**-6
    df_regional_sums_compare["indicator_df_million"] = df_regional_sums_compare[varname_grid_IAM] * 10**-6

    return df_regional_sums_compare, xr_regional_sums

def calculate_harmonisation_factors_emissions(xr_em: xr.Dataset, varname_EM:str,
                                              xr_regional_sums:xr.Dataset,
                                              xr_IAM_regions_grid_downscaling:xr.Dataset,
                                              df_IAM_EM:pd.DataFrame,
                                              years_downscaling:list) -> xr.DataArray:
    # Calculate correction factors and redistribute se_indicator (for emissions, only base year)
    nr_regions = df_IAM_EM["region_number"].nunique()

    print("Calculate correction factors and redistribute se_indicator...")
    # Prepare harmonised (target) se_indicator values from IAM projections
    xr_harmonised = (df_IAM_EM
                    .set_index(['year', 'region_number'])['value']
                    .to_xarray()
                    .sel(year=years_downscaling, region_number=xr_regional_sums.region_number))

    print(f"Unique region numbers in harmonised se_indicator data: {np.unique(xr_harmonised.region_number.values)}")
    xr_harmonised = xr_harmonised.rename({'year': 'time'})

    print(f"Years in harmonised se_indicator data: {xr_harmonised.time.values}")
    # Calculate regional correction factors
    #xr_correction_factors_regional = (xr_harmonised / xr_regional_sums[varname_EM]).fillna(0)
    regional_sums_not_harmonised = xr_regional_sums[varname_EM]
    # regional_sums_not_harmonised == 0 → 1
    # regional_sums_not_harmonised == inf → 0 (from the division itself)
    # regional_sums_not_harmonised is a normal non-zero finite value → normal division result passes through unchanged, neither .where() nor .fillna() touch it
    xr_correction_factors_regional = ((xr_harmonised / regional_sums_not_harmonised
                                        .where(regional_sums_not_harmonised != 0))
                                        .where(regional_sums_not_harmonised != 0, other=1)
                                        .fillna(0))

    # Map to spatial grid using numpy
    region_ids = xr_IAM_regions_grid_downscaling.region_number.values
    print(f"Unique region numbers in IAM regions grid: {np.unique(region_ids)}")

    # Process x time steps at a time instead of all or one
    print("Processing correction factors in blocks...")
    block_size = 4
    correction_factors_chunks = []

    for i in range(0, len(xr_correction_factors_regional.time), block_size):
        # Calculate actual block size (last block might be smaller)
        actual_block_size = min(block_size, len(xr_correction_factors_regional.time) - i)
        print(f"Processing block {i // block_size + 1} of {(len(xr_correction_factors_regional.time) + block_size - 1) // block_size}...")

        time_slice = slice(i, i + actual_block_size)

        # Use actual_block_size instead of block_size
        lookup = np.zeros((actual_block_size, nr_regions + 1), dtype='float32')
        lookup[:, xr_correction_factors_regional.region_number.values.astype(int)] = \
            xr_correction_factors_regional.isel(time=time_slice).values

        correction_factors_chunks.append(lookup[:, region_ids])

    # Concatenate all chunks AFTER the loop
    print("Concatenating correction factor chunks...")
    correction_factors_array = np.concatenate(correction_factors_chunks, axis=0)

    # Create the final DataArray with the full concatenated array
    print("Creating final DataArray for correction factors...")
    xr_em_correction_factors = xr.DataArray(
        correction_factors_array,
        coords={'time': xr_correction_factors_regional.time,
                'y': xr_em.y,
                'x': xr_em.x},
        dims=['time', 'y', 'x']
    )

    return xr_em_correction_factors

def apply_harmonisation_factors_emissions(xr_correction_factors:xr.DataArray,
                                          xr_em:xr.Dataset, varname:str,
                                          xr_IAM_regions_grid_downscaling:xr.Dataset,
                                          model: str, scenario: str) -> xr.Dataset:

    # Apply correction factor
    nr_regions = len(np.unique(xr_IAM_regions_grid_downscaling["region_number"].values)) - 1  # Exclude ocean (0)

    # Merge grid se_indicator and correction factors into one Dataset
    # 1) Make sure time coords match (important!)
    if not np.array_equal(xr_em.time.values, xr_correction_factors.time.values):
        print("Aligning time coordinates between se_indicator grid and correction factors...")
        xr_correction_factors = xr_correction_factors.interp(time=xr_em.time)
    # 2) (optional) mask ocean so zeros don’t skew colors
    xr_correction_factors_masked = xr_correction_factors.where(xr_IAM_regions_grid_downscaling.region_number > 0)
    # 3) Put the factors into the se_indicator Dataset
    if isinstance(xr_em, xr.Dataset):
        xr_grid_correction = xr_em.assign(correction_factor=xr_correction_factors_masked)
    else:
        # if xr_SE_grid is a DataArray named 'se_indicator'
        xr_grid_correction = xr.Dataset(
            data_vars=dict(
                se_indicator=xr_SE_grid,
                correction_factor=xr_correction_factors_masked
            )
        )
    # add region number to Dataset
    # 2-D region numbers (y, x), int and aligned to SE grid
    region2d = xr_IAM_regions_grid_downscaling.region_number.astype("int8")

    # attach as a coordinate; stays time-invariant
    xr_grid_correction = xr_grid_correction.assign_coords(region_number=(("y", "x"), region2d.values))

    # optional: helpful attrs
    xr_grid_correction.coords["region_number"].attrs.update(
        long_name=f"{model} {scenario} region number (0=ocean, 1–{nr_regions}=land regions)"
    )

    # Apply correction factor to se_indicator variable
    #varname_corrected = f"{varname}_corrected"
    xr_grid_correction[varname] = xr_grid_correction[varname] * xr_grid_correction["correction_factor"]

    return xr_grid_correction


