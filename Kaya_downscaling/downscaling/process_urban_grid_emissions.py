from pathlib import Path
import time
import logging
from tabulate import tabulate
from pprint import pformat

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

import xarray as xr
import geopandas as gpd
import rasterio.features
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from tools.functions_logging import init_logging
from tools.general_functions import PRINT_COLORS

local_log, dummy_log = init_logging("log", "log/reading_processing_data/local")

def process_urban_classification_data(project_dir: Path, save_gdf:bool=False, plot:bool=False, log: logging.Logger=local_log) -> None:
    '''
    Process urban classification data by merging geopandas dataframe with csv dataframe on GDAM_ID.
    The results is a geopandas dataframe with urban classification, including polygons for each GDAM_ID for each year, which is saved as a parquet file.
    '''

    # 1. Process urban classification data by merging geopandas dataframe with csv dataframe on GDAM_ID.

    # Combine geopandas dataframe with csv dataframe on GDAM_ID
    gpkg_path = project_dir / "data" / "input" / "DLL" / "data_final.gpkg"
    csv_path = project_dir / "data" / "input" / "DLL" / "data_timeseries_70.csv"
    merged_output_path = project_dir / "data" / "processed" / "DLL" / "urban_classification_years.parquet"
    merged_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Only load the ID column, other columns and geometry from the gpkg first, to check the join before pulling everything in
    start_time = time.time()
    log.info(f"\n{PRINT_COLORS["green"]}Reading geopackage from {gpkg_path}{PRINT_COLORS["end"]}")
    gdf = gpd.read_file(gpkg_path, layer="data_final")
    print(f"{gdf.head(5)}")
    gdf.drop(columns=['geometry']).to_csv(project_dir / Path("data/check") / "gdf.csv", index=False, sep=";")
    log.info(f"Geopackage read in {time.time() - start_time:,.2f} seconds")
    log.info(f"\n{tabulate(gdf.drop(columns=['geometry']).head(5), headers='keys', tablefmt='grid', intfmt=',', showindex=False)}")
    log.info(f"dtypes:\n{pformat(gdf.dtypes.to_dict())}")
    log.info(f"Geopackage geometry type: {gdf.geometry.type.iloc[0]}")
    log.info(f"Reading csv from {csv_path}")

    log.info(f"\n{PRINT_COLORS["green"]}Reading csv from {csv_path}{PRINT_COLORS["end"]}")
    df_ts = pd.read_csv(csv_path)
    df_ts = df_ts[df_ts["GDAM_id"]!="?"]
    df_ts["GADM_level"] = df_ts["GDAM_id"].astype(str).str.count(r"\.").astype(int)  # count number of dots in GDAM_id to determine GADM_level
    df_ts.to_csv(project_dir / Path("data/check") / "df_ts.csv", index=False, sep=";")
    # count nr occurences for each GADM_level
    gdam_level_counts = df_ts["GADM_level"].value_counts().sort_index()
    log.info(f"GADM level counts:\n{gdam_level_counts}")
    log.info(f"dtypes:\n{pformat(df_ts.dtypes.to_dict())}")
    log.info(f"\n{tabulate(df_ts.head(5), headers='keys', tablefmt='grid', intfmt=',', showindex=False)}")
    if plot:
        # plot
        print(f"\n{PRINT_COLORS["green"]}Plotting urban classification map and saving to {project_dir / Path('figures') / 'DLL_urban_classification_map.png'}{PRINT_COLORS["end"]}")
        fig, axs = plt.subplots(figsize=(12, 8), ncols=2)
        gdf.plot(ax=axs[0], column="NAME_0", cmap="cividis", legend=False)
        axs[0].set_title("Urban Classification by Country")
        gdf.plot(ax=axs[1], column="NAME_1", cmap="cividis", legend=False)
        axs[1].set_title("Urban Classification by Province")
        for ax in axs:
            ax.set_axis_off()
        #fig.savefig(project_dir / Path("figures") / "DLL_urban_classification_map.png", dpi=300, bbox_inches="tight")

    # Check join coverage before committing to a full merge
    gpkg_ids = set(gdf["GDAM_ID"])
    csv_ids = set(df_ts["GDAM_id"])
    log.info(f"IDs in gpkg but not in csv: {len(gpkg_ids - csv_ids)}")
    log.info(f"IDs in csv but not in gpkg: {len(csv_ids - gpkg_ids)}")

    print(f"\n{PRINT_COLORS["green"]}Merging geopackage and csv data...{PRINT_COLORS["end"]}")
    GADM_levels = np.sort(df_ts["GADM_level"].unique())
    merge_key = gdf["GID_0"]
    for level in range(1, 6):
        merge_key = merge_key.where(merge_key.isin(df_ts["GDAM_id"]), gdf[f"GID_{level}"])
    gdf["merge_key"] = merge_key
    merged_gdf = gdf.merge(df_ts.drop(columns="GID_1"), how="left", left_on="merge_key", right_on="GDAM_id")
    merged_gdf.to_csv(project_dir / Path("data/check") / f"merged_gdf.csv", index=False, sep=";")
    base_cols = ["UID", "GDAM_id", "GADM_level", "NAME_0", "GID_0", "NAME_1", "GID_1", "NAME_2", "GID_2", "NAME_3", "GID_3", "NAME_4", "GID_4", "NAME_5", "GID_5", "geometry"]
    cluster_cols = [col for col in merged_gdf.columns if col.startswith("cluster_")]
    merged_gdf = merged_gdf[base_cols + cluster_cols]
    merged_gdf.rename(columns={"NAME_0": "Country", "NAME_1": "State/province", "NAME_2": "County/district", "NAME_3": "Commune/municipality", "NAME_4": "admin_level_4", "NAME_5": "admin_level_5"}, inplace=True)
    print(f"{merged_gdf.head(5)}")
    # find duplicates in
    # print sample
    print(f"\n{PRINT_COLORS["green"]}Merged geopandas dataframe sample:{PRINT_COLORS["end"]}")
    log.info(f"\n{tabulate(merged_gdf.drop(columns=['geometry']).head(5), headers='keys', tablefmt='grid', intfmt=',', showindex=False)}")
    log.info(f"dtypes:\n{pformat(merged_gdf.dtypes.to_dict())}")
    log.info(f"Geopackage geometry type: {merged_gdf.geometry.type.iloc[0]}")
    log.info("Sample of merged data:")
    log.info(merged_gdf.sample(10))

    if plot:
        # plot urban vs rural polygons for 2020 and 2050
        print(f"\n{PRINT_COLORS["green"]}Plotting urban/rural classification map and saving to {project_dir / Path('figures') / 'DLL_urban_rural_map.png'}{PRINT_COLORS["end"]}")
        country_borders = merged_gdf.dissolve(by="GID_0")
        years = ["2020", "2050"]
        colors = {1: "firebrick", 0: "forestgreen"}
        labels = {1: "Urban", 0: "Rural"}
        fig, axs = plt.subplots(figsize=(16, 8), ncols=2)
        for ax, year in zip(axs, years):
            column = f"cluster_{year}"
            merged_gdf.plot(ax=ax, color="lightgrey", edgecolor="none")  # background for missing/no-data
            for value, color in colors.items():
                merged_gdf[merged_gdf[column] == value].plot(ax=ax, color=color)
            country_borders.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5)
            ax.set_title(f"Urban/Rural Classification {year}")
            ax.set_axis_off()
        legend_handles = [Patch(color=color, label=labels[value]) for value, color in colors.items()]
        legend_handles.append(Patch(color="lightgrey", label="No data"))
        fig.legend(handles=legend_handles, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.02))
        # save figure
        fig.savefig(project_dir / Path("figures") / "DLL_urban_rural_map.png", dpi=300, bbox_inches="tight")

    print(f"\n{PRINT_COLORS["green"]}Saving results to geopandas and parquet files.{PRINT_COLORS["end"]}")
    # save to geopandas
    if save_gdf:
        start_time = time.time()
        merged_gdf.to_file(merged_output_path.with_suffix(".gpkg"), driver="GPKG")
        elapsed_time_geo = time.time() - start_time
        log.info(f"Saved merged geopandas dataframe to {merged_output_path.with_suffix('.gpkg')} in {elapsed_time_geo:,.2f} seconds")

    # save to parquet files
    merged_gdf.to_parquet(merged_output_path)
    elapsed_time_parquet = time.time() - start_time
    log.info(f"Saved merged parquet file to {merged_output_path} in {elapsed_time_parquet:,.2f} seconds")

def precompute_urban_masks(gdf_urban: gpd.GeoDataFrame, ds_ref: xr.Dataset, out_path: Path,
                           cluster_years: list[int] | None = None) -> xr.DataArray:
    '''
    Rasterize the GADM urban classification for every available year onto the grid of ds_ref and save it as
    one netCDF (dims: year, y, x; uint8, 1 = urban, 0 = not urban / outside any polygon). Run once; the
    resolution and extent are taken from ds_ref, so the result aligns cell-for-cell with the emissions data.
    '''
    if gdf_urban.crs != ds_ref.rio.crs:
        gdf_urban = gdf_urban.to_crs(ds_ref.rio.crs)
    if cluster_years is None:
        cluster_years = sorted(int(col.replace("cluster_", "")) for col in gdf_urban.columns if col.startswith("cluster_"))

    y_dim, x_dim = ds_ref.rio.y_dim, ds_ref.rio.x_dim
    out_shape, transform = (ds_ref.rio.height, ds_ref.rio.width), ds_ref.rio.transform()

    slices = []
    for year in cluster_years:
        geoms = gdf_urban.geometry[gdf_urban[f"cluster_{year}"] == 1.0]
        burned = rasterio.features.rasterize(((geom, 1) for geom in geoms), out_shape=out_shape,
                                             transform=transform, fill=0, all_touched=False, dtype="uint8")
        slices.append(xr.DataArray(burned, coords={y_dim: ds_ref[y_dim], x_dim: ds_ref[x_dim]},
                                   dims=(y_dim, x_dim)).expand_dims(year=[year]))

    da_urban = xr.concat(slices, dim="year").rename("is_urban").rio.write_crs(ds_ref.rio.crs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da_urban.to_netcdf(out_path)

    return da_urban

def load_urban_masks(path: Path) -> xr.DataArray:
    '''Load the precomputed urban masks, keeping the CRS/spatial_ref coordinate.'''
    return xr.open_dataarray(Path(path), decode_coords="all")

def _assert_grids_match(da_ref: xr.DataArray, da_other: xr.DataArray, tol_fraction: float = 0.01) -> None:
    '''
    Raise if da_other is not on the same (y, x) grid as da_ref. Checks dim names, sizes, coordinate values
    (within tol_fraction of one cell), and CRS. Only spatial dims are compared, so extra dims like "year"
    on da_other are ignored.
    '''
    y_dim, x_dim = da_ref.rio.y_dim, da_ref.rio.x_dim

    for dim in (y_dim, x_dim):
        if dim not in da_other.dims:
            raise ValueError(f"Grid mismatch: reference dim '{dim}' missing from other array (dims: {da_other.dims}).")
        if da_ref.sizes[dim] != da_other.sizes[dim]:
            raise ValueError(f"Grid mismatch on '{dim}': {da_ref.sizes[dim]} vs {da_other.sizes[dim]} cells.")

    for dim in (y_dim, x_dim):
        a, b = da_ref[dim].values, da_other[dim].values
        cell = abs(float(a[1] - a[0])) if a.size > 1 else 1.0
        max_diff = float(np.abs(a - b).max())
        if max_diff > tol_fraction * cell:
            raise ValueError(f"Grid mismatch on '{dim}' coords: max diff {max_diff:g} > {tol_fraction:g} of "
                             f"cell size {cell:g}. The mask grid does not line up with the emissions grid.")

    crs_ref, crs_other = da_ref.rio.crs, da_other.rio.crs
    if crs_ref is not None and crs_other is not None and crs_ref != crs_other:
        raise ValueError(f"CRS mismatch: reference {crs_ref} vs other {crs_other}.")

def create_urban_id_raster(gdf_urban_classification: gpd.GeoDataFrame, ds_ref: xr.Dataset, cache_path: Path | None = None,
                           use_saved: bool = True, log: logging.Logger=dummy_log) -> np.ndarray:
    '''
    Burn a 1-based polygon index onto the grid of ds_ref once (0 = outside every polygon) and return it as a raw
    NumPy array. If use_saved is True and cache_path exists (and its shape matches the grid), load it from disk;
    otherwise rasterize, and when cache_path is given save it as a .npy for reuse. Because GADM districts do not
    overlap, this id grid reproduces a direct per-value burn exactly.

    Each cell in the saved .npy array holds a polygon id: a small unsigned integer that says which polygon of
    the gdf (gdf_urban_classification) that grid cell's centre falls inside. It's a label, not a value.
    0 is special: it's the fill -- it means "this cell's centre landed inside no polygon".
    Any other number is the 1-based row index of the polygon in gdf_urban_classification whose interior contains
    the cell centre (1 = first row, 2 = second row, etc.), independent of whether that polygon is urban in any
    given year -- urban/non-urban is applied later via each year's cluster_<year> column.
    '''
    out_shape = (ds_ref.rio.height, ds_ref.rio.width)

    # If use_saved is True and cache_path exists, load it and check the shape
    if use_saved and cache_path is not None and Path(cache_path).exists():
        ids = np.load(Path(cache_path))
        if ids.shape != out_shape:
            raise ValueError(f"Cached id raster at {cache_path} has shape {ids.shape}, but the current grid is "
                             f"{out_shape}. Delete it or pass use_saved=False to rebuild.")
        return ids
    elif not Path(cache_path).exists():
        log.info(f"{PRINT_COLORS['yellow']}Cached id raster at {cache_path} does not exist; creating it now.{PRINT_COLORS['end']}")

    # Rasterize the polygon geometries onto the grid of ds_ref, using a 1-based index for each polygon. Cells outside every polygon get 0.
    if gdf_urban_classification.crs != ds_ref.rio.crs:
        gdf_urban_classification = gdf_urban_classification.to_crs(ds_ref.rio.crs)
    id_dtype = "uint16" if len(gdf_urban_classification) <= np.iinfo("uint16").max else "int32"
    ids = rasterio.features.rasterize(((geom, i) for i, geom in enumerate(gdf_urban_classification.geometry, start=1)), out_shape=out_shape,
                                      transform=ds_ref.rio.transform(), fill=0, all_touched=False, dtype=id_dtype)

    # If cache_path is given, save the id raster to disk for reuse
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # print metadata about the rasterized ids
        log.info(f"\nRasterized {len(gdf_urban_classification)} polygons onto grid {out_shape} (dtype={id_dtype}); saving to {cache_path}")
        log.info(f"Unique ids: {np.unique(ids)}")
        np.save(cache_path, ids)

    log.info(f"Created urban id raster with shape {ids.shape}, dtype {ids.dtype}, and {len(np.unique(ids))} unique ids.")

    return ids

def plot_urba_nan(plot_dir: Path, xr_urban:xr.Dataset, varname:str, add_txt:str, log: logging.Logger=dummy_log) -> None:
    ds_check = xr_urban.sel(time=2020)
    xr_check = ds_check[varname]
    urban = ds_check["urban"]
    print(f"{PRINT_COLORS['yellow']}Check: unique 2020 urban values: {np.unique(urban.values)}{PRINT_COLORS['end']}")
    check_urban_null = float(xr_check.where(urban.isnull()).sum())
    check_total_urban = float(xr_check.where(urban==1).sum())
    check_perc_urban_null = check_urban_null / (check_urban_null + check_total_urban) * 100
    print(f"{PRINT_COLORS['yellow']}Check: 2020 urban emissions unharmonised: urban_null={check_urban_null:,.0f}, total_urban={check_total_urban:,.0f}, percentage_null={check_perc_urban_null:.2f}%{PRINT_COLORS['end']}")

    ds_2020 = xr_urban.sel(time=2020)
    em_na = ds_2020[varname].where(ds_2020["urban"].isnull())
    pts = em_na.stack(cell=("y", "x")).dropna("cell")
    xs, ys = pts["x"].values, pts["y"].values
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.9")
    ax.coastlines(linewidth=0.4, color="0.7")
    ax.scatter(xs, ys, color="red", s=14, edgecolors="none", transform=ccrs.PlateCarree())
    ax.set_global()
    add_txt_plot = add_txt[1:] if add_txt.startswith("_") else add_txt
    ax.set_title(f"2020 values where urban is NA: {add_txt_plot.lstrip('_')}")
    plt.tight_layout()
    fig.savefig(plot_dir / f"emissions_urban_na_2020{add_txt}.png", dpi=150, bbox_inches="tight")

def _rasterize_urban_value(gdf: gpd.GeoDataFrame, cluster_col: str, ds_ref: xr.Dataset,
                           ids: np.ndarray | None = None) -> xr.DataArray:
    '''
    Map each polygon's cluster_col value (1.0 urban, 0.0 non-urban) onto the grid of ds_ref through a 1-based
    polygon id grid; cells outside every polygon get NaN. If ids is supplied (from create_urban_id_raster) it is
    reused as a cheap lookup, otherwise the id grid is burned here. Output matches the old direct value burn.
    '''
    if ids is None:
        ids = create_urban_id_raster(gdf, ds_ref, cache_path=None, use_saved=False)
    y_dim, x_dim = ds_ref.rio.y_dim, ds_ref.rio.x_dim
    lut = np.concatenate(([np.nan], gdf[cluster_col].to_numpy().astype("float32")))
    return xr.DataArray(lut[ids], coords={y_dim: ds_ref[y_dim], x_dim: ds_ref[x_dim]}, dims=(y_dim, x_dim), name="urban")

def aggregate_urban_values(project_dir: Path,
                           profile: str, add_txt: str,
                           xr_dataset: xr.Dataset,
                           gdf_urban_classification: gpd.GeoDataFrame,
                           base_year: int = 2020,
                           varname: str = "Emissions_CO2_Excl_shipping_aviation_AFOLU",
                           region_varname: str = "region_number", final_year: int = 2100,
                           use_saved: bool = True, log: logging.Logger = dummy_log) -> xr.Dataset:
    '''
    Aggregate emissions per region and year based on urban classification, using rasterio.features.rasterize
    (centre-based, no geocube, no fraction) so it can be compared against the geocube version.
    '''
    if not xr_dataset.chunks:
        xr_dataset = xr_dataset.chunk({"time": 1, "y": 2048, "x": 2048})

    print(f"Calculating common years between gdf_urban_classification and xr_dataset...")
    cluster_years = sorted(int(col.replace("cluster_", "")) for col in gdf_urban_classification.columns
                           if col.startswith("cluster_"))
    emissions_years = set(int(y) for y in xr_dataset["time"].values)
    common_years = sorted(y for y in (set(cluster_years) & emissions_years) if y <= final_year)
    if not common_years:
        raise ValueError("No overlapping years found between gdf_urban_classification and xr_dataset.")
    log.info(f"Using {len(common_years)} common years: {common_years}")

    # burn the polygon geometry once (or load it); every year is then a cheap lookup on this id grid
    height, width = xr_dataset.rio.height, xr_dataset.rio.width
    id_cache_path = project_dir / "data" / "processed" / f"{profile}" / f"urban_id_raster_{profile}_{height}x{width}.npy"
    print(f"Creating or loading polygon id raster (use_saved={use_saved}) at {id_cache_path}...")
    ids = create_urban_id_raster(gdf_urban_classification, xr_dataset, cache_path=id_cache_path, use_saved=use_saved)

    # one (y, x) value grid per year; coords taken from xr_dataset so they align exactly
    print(f"Building urban classification grids for {len(common_years)} common years...")
    start_time = time.time()
    urban_slices = [_rasterize_urban_value(gdf_urban_classification, f"cluster_{year}", xr_dataset, ids=ids).expand_dims(time=[year])
                    for year in common_years]
    t_rasterize = time.time()
    print(f"Time elapsed for mapping urban values for common years: {(t_rasterize - start_time)/60:,.2f} minutes")
    urban_by_year = xr.concat(urban_slices, dim="time")
    t_concat = time.time()
    print(f"Time elapsed for concatenating urban grids: {(t_concat - t_rasterize)/60:,.2f} minutes")
    xr_combined = xr.merge([xr_dataset.sel(time=common_years), urban_by_year.to_dataset()], join="exact", compat="no_conflicts")
    t_merge = time.time()
    print(f"Time elapsed for merging emissions and urban data: {(t_merge - t_concat)/60:,.2f} minutes")

    check = xr_combined[varname].sel(time=base_year).where(xr_combined["region_number"]==1).sum().compute().values
    print(f"{PRINT_COLORS["yellow"]}Check xr_combined om aggregate_urban_values for region 1 in 2020: {check:,.2f}{PRINT_COLORS["end"]}")

    print(f"Aggregating values per {region_varname} for {len(common_years)} years...")
    log.info(f"Unique urban values (first year): {np.unique(xr_combined["urban"].isel(time=0).values)}")
    log.info(f"Unique region values (first year): {np.unique(xr_combined[region_varname].values)}")
    rows = []
    nr_regions = np.unique(xr_combined[region_varname].values).size
    print(f"Counting cells per region (1..{nr_regions})...")
    for r in range(1, nr_regions):
        total = int((xr_combined[region_varname] == r).sum().compute())
        urban = int(((xr_combined[region_varname] == r) & (xr_combined["urban"].isel(time=0) == 1)).sum().compute())
        rural = int(((xr_combined[region_varname] == r) & (xr_combined["urban"].isel(time=0) == 0)).sum().compute())
        rows.append({"region": r, "total": total, "urban": urban, "rural": rural})
    df_cell_counts = pd.DataFrame(rows)
    log.info(f"Cell counts per region:\n{tabulate(df_cell_counts, headers="keys", tablefmt="grid", intfmt=",", showindex=False)}")
    df_cell_counts.to_csv(project_dir / "data/check/urban_comparison" / f"cell_counts_per_region{add_txt}.csv", index=False)

    plot_dir = project_dir / "figures" / "check"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_urba_nan(plot_dir, xr_combined, varname, add_txt, log=log)

    return xr_combined

def calculate_urban_rural_totals(xr_dataset: xr.Dataset, varname: str, region_varname: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # pre: xr_dataset must have a "urban" variable (1 = urban, 0 = rural) and a region variable (e.g., "region_number")

    urban_totals = (xr_dataset[varname]
                    .where(xr_dataset["urban"]==1)
                    .groupby(xr_dataset[region_varname].compute())
                    .sum()
                    .compute())
    df_urban_values = (urban_totals.to_dataframe(name=varname).reset_index()
                          .rename(columns={"time": "year"}))

    rural_totals = (xr_dataset[varname]
                    .where(xr_dataset["urban"] == 0)
                    .groupby(xr_dataset[region_varname].compute())
                    .sum()
                    .compute())
    df_rural_values = (rural_totals.to_dataframe(name=varname).reset_index()
                       .rename(columns={"time": "year"}))

    return df_urban_values, df_rural_values
