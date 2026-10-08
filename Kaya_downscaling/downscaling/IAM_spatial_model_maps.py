from pathlib import Path
import json
from typing import Tuple
import logging

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import rasterio.enums
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import geopandas as gpd
import xarray as xr
import rioxarray as rxr

from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from tools.general_functions import PRINT_COLORS, apply_root_json
from tools.functions_logging import init_logging

colour_red = "\033[91m"
colour_green = "\033[92m"
colour_yellow = "\033[93m"
color_end = "\033[0m"

local_log, dummy_log = init_logging("log", "log/reading_processing_data/local")

def read_GADM_vector(input_dir:Path, output_dir:Path) -> [gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    # Read in
    file_GADM_countries = input_dir / "gadm_410.gpkg"
    print(f"Reading GADM countries from: {file_GADM_countries}")
    countries = gpd.read_file(file_GADM_countries)
    # remove countries not needed
    mask_countries_excluded = ~countries["GID_0"].str.match(r"^(X(?!KO)|Z\d{2})") # Do include 'XKO' (Kosovo)
    excluded_countries_df = (countries.loc[~mask_countries_excluded, ["GID_0", "NAME_0"]]
                                .drop_duplicates()
                                .sort_values("GID_0"))
    excluded_countries_df.to_csv(output_dir / "excluded_countries.csv", sep=";", index=False)
    print(f"Excluded countries from GADM dataset (disputed regions ('Z-') and special territories ('X-')):\n{excluded_countries_df}")
    countries = countries[mask_countries_excluded]

    # Save check file
    check_countries = pd.DataFrame(countries.drop(columns="geometry"))
    check_countries.to_csv(f"{output_dir}/GADM_countries.csv", sep=";", index=False)

    # Plot
    import matplotlib.pyplot as plt
    print("Plotting GADM countries...")
    fig, ax = plt.subplots(figsize=(12, 8))
    countries.plot(ax=ax, edgecolor="black", facecolor="lightblue", linewidth=0.5)
    ax.set_title("GADM Countries")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.savefig(output_dir / "figures/GADM_countries.png", dpi=300)
    plt.close()

    # Convert
    print("Converting GADM vector data to raster format...")
    iso_to_name = (countries
                    .drop_duplicates("GID_0")
                    .set_index("GID_0")["NAME_0"]
                    .to_dict())
    # Map ISO codes to unique integer IDs (raster pixels need numeric values)
    iso_codes = countries["GID_0"].dropna().unique()
    iso_to_id = {iso: i + 1 for i, iso in enumerate(iso_codes)}  # Start from 1, reserve 0 for NoData
    id_to_iso = {i: iso for iso, i in iso_to_id.items()}
    #id_to_name = {i: iso_to_name[iso] for iso, i in iso_to_id.items()}

    # Save ISO --> id --> NAME
    print(f"\nMapping {len(iso_codes)} unique ISO codes to integer IDs")
    df_iso_to_id = pd.DataFrame({"ISO": list(iso_to_id.keys()), "id": list(iso_to_id.values())})
    df_iso_to_id["NAME"] = df_iso_to_id["ISO"].map(iso_to_name)
    df_iso_to_id.to_csv(f"{output_dir}/iso_to_id_mapping.csv", sep=";", index=False)
    # Save id → ISO → NAME
    df_id_to_iso = pd.DataFrame({"id": list(id_to_iso.keys()), "ISO": list(id_to_iso.values())})
    df_id_to_iso["NAME"] = df_id_to_iso["ISO"].map(iso_to_name)
    df_id_to_iso.to_csv(f"{output_dir}/id_to_iso_mapping.csv", sep=";", index=False)
    countries["iso_id"] = countries["GID_0"].map(iso_to_id)

    return countries, df_iso_to_id, df_id_to_iso

def GADM_vector_to_raster(input_dir:Path, output_dir:Path, resolution_degrees: float = 5) -> Tuple[Path, pd.DataFrame, pd.DataFrame]:
    """
    Read in GADM countries file and convert vector data to raster format using rasterio.

    Parameters
    project_dir : str - Project directory path
    resolution_degrees : float - Resolution in degrees. Default 1/120 corresponds to 0.5 arc-minutes
    plot : bool - Whether to create a plot of the countries
    """
    print(f"\nConverting GADM vector data to raster format with resolution {resolution_degrees:.6f} degrees ({60 * resolution_degrees:.2f} arc-minutes)...")

    # Read in GADM countries vector data
    countries, df_iso_to_id, df_id_to_iso = read_GADM_vector(input_dir, output_dir)

    # Define output filename
    resolution_minutes_str = f"{60 * resolution_degrees:.2f}".replace(".", "_")
    print(f"\nCharacteristics of the raster to be created:")
    print(f"Resolution: {60 * resolution_degrees:.2f} arc-minutes ({resolution_degrees:.6f} degrees)")
    raster_file = Path(f"{output_dir}/iso_codes_raster_{resolution_minutes_str}.tif")

    # Define raster extent (global, aligned to resolution)
    minx, miny, maxx, maxy = -180.0, -90.0, 180.0, 90.0

    # Calculate dimensions
    width = int(round((maxx - minx) / resolution_degrees))
    height = int(round((maxy - miny) / resolution_degrees))
    print(f"Raster dimensions: {width} x {height} pixels")
    print(f"Raster extent: x=[{minx}, {maxx}], y=[{miny}, {maxy}]")

    # Create transform (affine transformation from pixel to geographic coordinates)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Prepare shapes for rasterization: list of (geometry, value) tuples
    shapes = [(geom, value) for geom, value in zip(countries.geometry, countries["iso_id"])]

    # Rasterize
    print("Rasterizing (this may take a while for high resolution)...")
    nodata_value = 0

    rasterized = rasterize(shapes=shapes, out_shape=(height, width), transform=transform, fill=nodata_value, dtype=np.int16,
                           all_touched=True)  # Set True if you want all pixels touched by polygons

    # Write to GeoTIFF
    print(f"Writing raster to: {raster_file}")
    with rasterio.open(
        raster_file,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.int16,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata_value,
        compress="LZW",  # Compression reduces file size significantly
        tiled=True,      # Tiled storage improves read performance for large files
        blockxsize=512,
        blockysize=512,
    ) as dst:
        dst.write(rasterized, 1)

    print(f"Rasterization complete: {raster_file}")
    print(f"File size: {Path(raster_file).stat().st_size / (1024**2):.1f} MB")

    return raster_file, df_iso_to_id, df_id_to_iso

def _map_country_ids_to_region_numbers(country_id_da: xr.DataArray, country_to_region: dict) -> xr.DataArray:
    """
    Maps country IDs to region numbers using a vectorised NumPy index lookup.
    country_id_da: xr.DataArray with integer country IDs (may contain NaN for nodata)
    country_to_region: dict mapping country_id (int) to region_number (int)
    returns: xr.DataArray with region numbers as int16, nodata pixels set to 0
    """
    country_ids_vals = country_id_da.values
    nan_mask = np.isnan(country_ids_vals)
    country_ids_int = np.where(nan_mask, 0, country_ids_vals).astype(np.int16)

    max_id = int(country_ids_int.max()) + 1
    lookup_arr = np.zeros(max_id, dtype=np.int16)
    for cid, rnum in country_to_region.items():
        cid_int = int(cid)
        if 0 <= cid_int < max_id:
            lookup_arr[cid_int] = rnum

    region_values = lookup_arr[country_ids_int]
    region_values[nan_mask] = 0

    return xr.DataArray(
        region_values,
        dims=country_id_da.dims,
        coords=country_id_da.coords,
    ).astype(np.int16)

def plot_countries_regions(tiff_file:Path, fig_dir:Path) -> None:
    print("\n********************************")
    print(f"Plotting TIFF file: {tiff_file}")

    with rasterio.open(tiff_file) as src:
        # calc resolution to decide on scaling
        transform = src.transform
        res_deg_x = abs(transform.a)
        res_deg_y = abs(transform.e)
        res_arcmin_x = res_deg_x * 60
        res_arcmin_y = res_deg_y * 60
        scale = None
        if res_arcmin_x < 5:
            print(f"Resolution is {res_arcmin_x:.1f} x {res_arcmin_y:.1f} arc-minutes, applying scaling for plotting.")
            max_pixels = 3000
            scale_res = int(np.ceil(5 / res_arcmin_x))
            scale_auto = int(np.ceil(src.width / max_pixels))
            scale = int(max(1, scale_res, scale_auto))
            print(f"Using scale factor of {scale} for plotting.")
            out_height = int(src.height // scale)
            out_width  = int(src.width  // scale)
            countries = src.read(1, out_shape=(out_height, out_width), resampling=rasterio.enums.Resampling.nearest)
            regions = src.read(2, out_shape=(out_height, out_width), resampling=rasterio.enums.Resampling.nearest)
        else:
            countries = src.read(1)#.astype(float)
            regions   = src.read(2)#.astype(float)

    fig, axes = plt.subplots(3, 2, figsize=(18, 12), subplot_kw={"projection": ccrs.PlateCarree()})

    # Countries
    countries_plot = np.where(countries == 0, np.nan, countries)
    cmap_c = plt.get_cmap("hsv", 263)
    cmap_c.set_bad("white")
    axes[0,0].imshow(countries_plot, origin="upper", extent=[-180, 180, -90, 90],
                transform=ccrs.PlateCarree(),
                cmap=cmap_c,
                vmin=1, vmax=263, interpolation="none")
    axes[0,0].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
    axes[0,0].add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="black")
    axes[0,0].set_title("Band 1 — Country IDs")
    axes[0,0].set_global()

    # Oceans in countries
    ocean_cmap = ListedColormap(["white", "lightblue"])
    #countries_oceans_plot = np.where(countries!=0, np.nan, countries)
    countries_oceans_plot = (countries == 0)
    #cmap_c_ocean = plt.get_cmap("Blues", 263)
    #cmap_c_ocean.set_bad("white")
    axes[0,1].imshow(countries_oceans_plot, origin="upper", extent=[-180, 180, -90, 90],
                transform=ccrs.PlateCarree(), cmap=ocean_cmap,interpolation="none")
    axes[0,1].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
    axes[0,1].add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="black")
    axes[0,1].set_title("Band 1 — Oceans (light blue) vs land (white)")
    axes[0,1].set_global()

    # Regions
    n_regions = 26
    cmap_r = plt.get_cmap("tab20", n_regions)
    cmap_r.set_bad(color="white")
    #Map discrete integer values (1, 2, 3, …) to discrete colors, not a gradient.
    norm_r = BoundaryNorm(boundaries=np.arange(0.5, n_regions + 1.5), ncolors=n_regions)
    #regions_plot = np.where(regions == 0, np.nan, regions)  # Set 0 values to NaN for better visualization
    regions_plot = np.where(countries == 0, np.nan, regions)
    axes[1,0].imshow(regions_plot, origin="upper", extent=[-180, 180, -90, 90],
                   transform=ccrs.PlateCarree(), cmap=cmap_r, norm=norm_r, interpolation="none")
    axes[1,0].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
    axes[1,0].add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="black")
    axes[1,0].set_title("Band 2 — IAM model Region Numbers")
    axes[1,0].set_global()

    # oceans in regions
    #cmap_r_ocean = plt.get_cmap("Blues", n_regions)
    #cmap_r_ocean.set_bad(color="white")
    #Map discrete integer values (1, 2, 3, …) to discrete colors, not a gradient.
    #regions_oceans_plot = np.where(regions!=0, np.nan, regions)
    regions_oceans_plot = (regions == 0)
    axes[1,1].imshow(regions_oceans_plot, origin="upper", extent=[-180, 180, -90, 90],
                   transform=ccrs.PlateCarree(), cmap=ocean_cmap, interpolation="none")
    axes[1,1].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
    axes[1,1].add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="black")
    axes[1,1].set_title("Band 2 — Oceans (light blue), l vs land (white)")
    axes[1,1].set_global()

    # check
    missing_region_mask = (regions == 0) & (countries != 0)
    missing_cmap = ListedColormap(["white", "red"])
    axes[2,0].imshow(missing_region_mask, origin="upper", extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree(), cmap=missing_cmap, interpolation="none")
    axes[2,0].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
    axes[2,0].add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="black")
    axes[2,0].set_title("Missing regions (region == 0 on land)")

    plt.tight_layout()
    scale_text = ""
    if scale is not None:
        scale_text = f"_scaled_{scale}x" if scale > 1 else ""
    fig_path = fig_dir / f"gadm_check_coastlines_{res_arcmin_x}_arcmin{scale_text}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")

def _fill_nearest_neighbour(da: xr.DataArray, max_fill_pixels: int = 1, land_mask: xr.DataArray|None=None) -> xr.DataArray:

    values = da.values.copy().astype(float)
    # valid = land pixels only
    mask_valid = np.isfinite(values) & (values != 0)
    distances = np.empty(values.shape, dtype=np.float64)
    indices   = np.empty((values.ndim,) + values.shape, dtype=np.int32)
    distance_transform_edt(~mask_valid, return_distances=True, return_indices=True, distances=distances, indices=indices)
    filled = values[tuple(indices)]
    fill_mask = (distances <= max_fill_pixels) & (~mask_valid)
    if land_mask is not None:
        fill_mask = fill_mask & land_mask
    result = values.copy()
    result[fill_mask] = filled[fill_mask]

    return da.copy(data=result.astype(da.dtype))

def gadm_levels_to_csv(gpkg_file_path: Path, output_dir: Path) -> Path:
    """
    Read all six GADM 4.1 admin levels from a local GeoPackage and write
    the attribute data (excluding geometry) to a single CSV file.

    Each row in the output CSV has a 'level' column indicating which admin
    level it came from (0-5). Columns not present in a given level are
    filled with NaN.

    Parameters:
    GADM_gpkg_path : Path - Path to the local gadm_410-levels.gpkg file.
    output_dir : Path - Directory where the output CSV will be saved.

    Returns:
    Path - Path to the saved CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_levels = []

    level_names = {0: "Country" , 1: "State_Province", 2: "County_District", 3: "Commune_Municipality", 4: "Sub-municipal_1", 5: "Sub-municipal_2"}

    if not gpkg_file_path.exists():
        raise FileNotFoundError(f"GADM GeoPackage file not found: {gpkg_file_path}")
    for level in range(6):
        layer = f"ADM_{level}"
        print(f"Reading {layer}...")
        try:
            gdf = gpd.read_file(gpkg_file_path, layer=layer)
        except Exception as e:
            print(f"Could not read layer {layer}: {e}")
            continue

        df = gdf.drop(columns="geometry")
        df["level"] = level
        all_levels.append(df)
        print(f"  {layer}: {len(df)} features, columns: {list(df.columns)}")

        # Write level to CSV immediately and discard to reduce memory pressure
        level_csv = output_dir / f"ADM_{level}_{level_names[level]}.csv"
        df.to_csv(level_csv, sep=";", index=False)
        print(f"  Written to: {level_csv}")

    print("Concatenating all levels...")
    combined = pd.concat(all_levels, ignore_index=True)
    del all_levels

    out_path = output_dir / "gadm_all_levels.csv"
    combined.to_csv(out_path, sep=";", index=False)
    print(f"Combined CSV saved to: {out_path} ({out_path.stat().st_size / (1024**2):.1f} MB)")

    return out_path

def create_GADM_region_raster(project_dir:Path, model:str="IMAGE", resolution_minutes:float=0.5, label:str="_", plot=False) -> str:
    """
    Create a GADM-based country/region raster for a target IAM model (default: IMAGE),
    and export both NetCDF and GeoTIFF outputs with validation diagnostics.

    This routine builds (or reuses) a country-ID raster derived from GADM vector boundaries,
    maps each country to an IAM region number, writes combined country/region grids, and
    prints quality checks to help verify spatial integrity.

    Parameters:
        project_dir : pathlib.Path
            Root directory of the project. Used to locate:
            - input settings (`downscaling/settings_data_locations.json`)
            - model mapping files (`data/input/models/IMAGE/...`)
            - output folder (`data/processed/GADM`)
        model : str, default "IMAGE"
            IAM model identifier. Current logic is implemented for `"IMAGE"` mappings.
        resolution_minutes : float, default 0.5
            Target grid resolution in arc-minutes for GADM rasterization.
        plot : bool, default False
            Reserved plotting flag (currently not actively used in this function body).

    Workflow:
        1. Prepare output directory (`data/processed/GADM`) and load data-location settings.
        2. Build expected raster filename suffix from requested resolution.
        3. If country raster does not exist:
        - Rasterize GADM vector data via `GADM_vector_to_raster(...)`.
        - Retrieve ISO↔ID mapping returned by rasterization.
        Else:
        - Reuse existing raster and read stored ID↔ISO mapping CSV.
        4. Open the country raster into xarray/rioxarray and read CRS/transform with rasterio.
        5. For model `"IMAGE"`:
        - Load country-to-region mapping (ISO3 → IMAGE region code).
        - Apply manual correction for Greenland (`GRL -> WEU`).
        - Append HKG and MAC rows to align GADM/model coverage.
        - Outer-merge model mapping with GADM ID mapping.
        - Print warnings for ISO codes missing in either source.
        - Join IMAGE region-code → numeric region table.
        - Fill missing region numbers with 0, cast dtypes, add ocean row (ID 0 → region 0).
        - Save country-to-region mapping table to CSV.
        6. Create `region_number` raster by mapping each `country_id_GADM` pixel to model region.
        7. Export:
        - NetCDF: `IMAGE_GADM_regions_raster_<res>_arcmin.nc`
        - 2-band GeoTIFF: band 1 = country IDs, band 2 = region numbers
            with original CRS/transform and LZW tiled compression.
        8. Compute and print effective spatial resolution from coordinates.
        9. Generate map plots (`plot_countries_regions`) and run raster sanity checks:
        - sample pixel values at reference longitudes
        - min/unique values by band
        - zero/non-zero statistics
        - count of land pixels with missing region assignment (`country>0 & region==0`)

    Returns:
        None (results are written to disk; the function primarily performs I/O and diagnostics).

    Outputs:
        - `iso_codes_raster_<res>.tif` (if newly rasterized)
        - `IMAGE_GADM_country_to_region_codes.csv`
        - `IMAGE_GADM_regions_raster_<res>_arcmin.nc`
        - `IMAGE_GADM_regions_raster_<res>_arcmin.tif` (2-band)
        - figures in `data/processed/GADM/figures`

    Notes:
        - The function currently contains model-specific branching only for `"IMAGE"`.
        - Missing or unmatched ISO entries are handled by assigning region number `0`
        (treated as ocean/unassigned in diagnostics).
        - CRS and affine transform are intentionally taken from rasterio to avoid metadata
        inconsistencies that can occur after intermediate xarray/rioxarray operations.
    """
    '''

    '''
    dir_GADM = project_dir / "data" / "processed" / "GADM"
    print(f"PROJECT_DIR: {project_dir}")
    print(f"dir_GADM: {dir_GADM}")
    dir_GADM.mkdir(parents=True, exist_ok=True)

    settings_file_data_locations = project_dir / "downscaling" / "settings_data_locations.json"
    with open(settings_file_data_locations, "r") as f:
        data_files_data_locations = json.load(f)
    data_files_data_locations = apply_root_json(data_files_data_locations, data_files_data_locations["data_root"])
    data_dir_GADM = Path(data_files_data_locations["GADM"]["dir_GADM_single"])

    print("Creating GADM raster file for regions...")

    # 1. check if file with GADM raster countries exists
    file_iso_to_id = Path(f"{dir_GADM}/iso_to_id_mapping.csv")
    file_id_to_iso = Path(f"{dir_GADM}/id_to_iso_mapping.csv")
    res_min_file_end = f"{resolution_minutes:.2f}".replace(".", "_")
    iso_GADM_raster_file = f"{dir_GADM}/iso_codes_raster_{res_min_file_end}.tif"
    print(f"Checking if GADM raster file exists at: {iso_GADM_raster_file}")
    if not Path(iso_GADM_raster_file).exists():
        print(f"Reading in GADM raster with resolution {resolution_minutes} arc minutes file for countries: {iso_GADM_raster_file}")
        raster_file, df_iso_to_id, df_id_to_iso = GADM_vector_to_raster(data_dir_GADM, dir_GADM, resolution_degrees = resolution_minutes/60)
        df_iso_to_id.to_csv(file_iso_to_id, sep=";", index=False)
        df_id_to_iso.to_csv(file_id_to_iso, sep=";", index=False)
    else:
        print(f"GADM raster with resolution {resolution_minutes} arc minutes file already exists at: {iso_GADM_raster_file}, skipping creation.")
        if not file_iso_to_id.exists() or not file_id_to_iso.exists():
            raster_file, df_iso_to_id, df_id_to_iso = read_GADM_vector(data_dir_GADM, dir_GADM)
            df_iso_to_id.to_csv(file_iso_to_id, sep=";", index=False)
            df_id_to_iso.to_csv(file_id_to_iso, sep=";", index=False)
        else:
            print(f"Reading existing ISO↔ID mapping files from: {file_iso_to_id} and {file_id_to_iso}")
            df_iso_to_id = pd.read_csv(f"{dir_GADM}/id_to_iso_mapping.csv", sep=";")
            df_id_to_iso = pd.read_csv(f"{dir_GADM}/iso_to_id_mapping.csv", sep=";")

    # 2. convert to rioxarray and rasterio dataset and add model region numbers
    # Open GADM raster file
    ds_GADM_raster = rxr.open_rasterio(iso_GADM_raster_file)
    ds_GADM_raster = ds_GADM_raster.squeeze("band", drop=True)
    ds_GADM_raster = ds_GADM_raster.to_dataset(name="country_id_GADM")
    print(f"\nGADM raster dataset: {ds_GADM_raster}")

    # retrieve transform and crs using rasterio which is more reliable than using rioxarray attributes, which can be incorrect after processing steps
    with rasterio.open(iso_GADM_raster_file) as src:
        transform = src.transform
        crs = src.crs

    # add country_ID_GADM code from GADM
    df_model_GADM_region_code_number = pd.DataFrame()

    country_to_region_file = project_dir / "data" / "input" / "models" / f"{model}" / f"{model}_country_to_regions.csv"
    df_model_coutry_to_region = pd.read_csv(country_to_region_file, sep=";") # ISO3
    df_model_coutry_to_region.rename(columns={"Country name": "country_name_model"}, inplace=True)
    df_model_coutry_to_region.loc[df_model_coutry_to_region["ISO3"]=="GRL", "Region code"] = "WEU" # change GRL region code to WEU
    # correct "HKG" and "MAC" ISO3 codes in GADM
    df_iso_to_id_IAM_model = df_iso_to_id.copy()
    new_rows = pd.DataFrame({"id": [None, None],
                                "ISO": ["HKG", "MAC"],
                                "NAME": ["Hong Kong", "Macau"]})
    df_iso_to_id_IAM_model = pd.concat([df_iso_to_id_IAM_model, new_rows], ignore_index=True)
    df_model_GADM_region_code = pd.merge(df_model_coutry_to_region, df_iso_to_id_IAM_model, left_on="ISO3", right_on="ISO", how="outer")
    df_model_GADM_region_code.rename(columns={"ISO3": "ISO3_model", "ISO": "ISO3_GADM", "NAME": "country_name_GADM"}, inplace=True) # TO DO --> check missing ISO3 codes between model/GADM

    # print missing countries
    # print ISO3_model codes that are missing in model
    missing_ISO3_GADM = df_model_GADM_region_code[df_model_GADM_region_code["ISO3_GADM"].isna()][["ISO3_model", "country_name_model", "country_name_GADM"]].drop_duplicates()
    if len(missing_ISO3_GADM) > 0:
        print(f"\n{colour_red}Warning: The following ISO3 codes from the model are missing in GADM and will be assigned a region number of 0:{color_end}")
        print(missing_ISO3_GADM)
    # print ISO3_GADM codes that are missing in model
    missing_ISO3_model = df_model_GADM_region_code[df_model_GADM_region_code["ISO3_model"].isna()][["ISO3_GADM", "country_name_model", "country_name_GADM"]].drop_duplicates()
    if len(missing_ISO3_model) > 0:
        print(f"\n{colour_yellow}Warning: The following ISO3 codes from GADM are missing in the model and will be ignored:{color_end}")
        print(missing_ISO3_model)

    # map to region numbers
    region_numbers_file = project_dir / "data" / "input" / "models" / f"{model}" / f"{model}_region_numbers.csv"
    df_region_numbers = pd.read_csv(region_numbers_file, sep=";") # assumption: columns "region" and "number"
    df_model_GADM_region_code_number = pd.merge(df_model_GADM_region_code, df_region_numbers, left_on="Region code", right_on="region", how="left")
    #df_model_GADM_region_code_number.drop(columns=["Region code", "IMAGE region", "country_name_GADM", "ISO3_GADM", "ISO3_model"], inplace=True)
    df_model_GADM_region_code_number.drop(columns=["Region code", "region", "ISO3_model"], inplace=True)
    df_model_GADM_region_code_number.rename(columns={"id": "country_id_GADM",  "number": "model_region_number"}, inplace=True)
    df_model_GADM_region_code_number["model_region_number"] = df_model_GADM_region_code_number["model_region_number"].fillna(0).astype(np.int8)
    # add ocean with region number 0 to mapping
    ocean_row = pd.DataFrame({"country_id_GADM": [0], "country_name_model": ["Ocean"], "model_region_number": [0]})
    df_model_GADM_region_code_number = pd.concat([df_model_GADM_region_code_number, ocean_row],ignore_index=True)
    df_model_GADM_region_code_number.rename(columns={"model_region_number": "region_number"}, inplace=True)
    df_model_GADM_region_code_number["country_id_GADM"] = (pd.to_numeric(df_model_GADM_region_code_number["country_id_GADM"], errors="coerce")
                                                                            .fillna(0)
                                                                            .astype(np.int16))
    df_model_GADM_region_code_number.rename(columns={"number": "region_number"}, inplace=True)
    df_model_GADM_region_code_number.to_csv(f"{dir_GADM}/{model}_GADM_country_to_region_codes.csv", sep=";", index=False)

    print("\nMerging GADM raster with model region numbers...")
    # add region numbers to GADM raster (to save memory, inly the GADM country ID and the region number are kept in the raster)
    country_to_region = df_model_GADM_region_code_number.set_index("country_id_GADM")["region_number"].to_dict()
    ds_GADM_raster["region_number"] = xr.full_like(ds_GADM_raster["country_id_GADM"], fill_value=0, dtype=np.int16) # Return a new object with the same shape and type as a given object.
    ds_GADM_raster["region_number"] = xr.apply_ufunc(np.vectorize(lambda x: 0 if np.isnan(x) else country_to_region.get(x, 0)),
                                                     ds_GADM_raster["country_id_GADM"],
                                                     output_dtypes=[np.int16]) # np.vectorize allows us to apply a function to each element of an array, handling NaN values and missing keys gracefully.

    # save to netcdf and tiff
    print(f"CRS: {ds_GADM_raster.rio.crs}")
    print(f"\nSaving GADM raster to {colour_green}netcdf {color_end}file in {dir_GADM} with region numbers...")
    ds_GADM_raster.to_netcdf(f"{dir_GADM}/{model}_GADM_regions_raster_6_00_arcmin.nc", mode="w", engine="netcdf4")
    print(f"\nSaving GADM raster to {colour_yellow}tiff {color_end}file in {dir_GADM} with region numbers...")
    data = np.stack([ds_GADM_raster["country_id_GADM"].values, ds_GADM_raster["region_number"].values])
    tiff_file = f"{dir_GADM}/{model}_GADM_regions_raster_{res_min_file_end}{label}_arcmin.tif"
    with rasterio.open(
        tiff_file,
        "w",
        driver="GTiff",
        height=data.shape[1], width=data.shape[2],
        count=2,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        compress="LZW",
        predictor=2,  # Improves compression for integer data
        tiled=True,   # Better for large files
        all_touched=True,
        blockxsize=256, blockysize=256) as dst: dst.write(data)

    # calculate resolutions from coordinates
    lat = ds_GADM_raster["region_number"]["y"].values  # or "y"
    lon = ds_GADM_raster["region_number"]["x"].values  # or "x"
    arc_degrees = abs(float(np.diff(lon).mean()))
    arc_minutes = arc_degrees * 60
    arc_seconds = arc_degrees * 3600
    print(f"resolution EM grid: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.1f} arc degrees")

    # 3. plot and checks
    dir_input = Path(f"{dir_GADM}/{model}_GADM_regions_raster_{res_min_file_end}{label}_arcmin.tif")
    dir_fig = Path(f"{dir_GADM}/figures")
    dir_fig.mkdir(parents=True, exist_ok=True)
    plot_countries_regions(dir_input, dir_fig)

    with rasterio.open(tiff_file) as src:
         # checks
        countries = src.read(1).astype("int16") #.astype(float)
        regions   = src.read(2).astype("int8") #.astype(float)

        # Check: what is the value at a known location?
        # e.g. pixel at lon=0 should be somewhere in Africa/Europe, not ocean
        print("Value at lon=0 center:", countries[900, 1800])   # row 900 = equator, col 1800 = lon 0 in -180:180
        print("Value at lon=180 center:", countries[900, 3599]) # col 3599 = lon 180
        print("Min country value:", np.nanmin(countries))
        print("Unique countries:", np.unique(countries))
        print("Min region value:", np.nanmin(regions))
        print("Unique regions:", np.unique(regions))

        n_total_countries = countries.size
        n_countries_zero     = np.count_nonzero(countries == 0)
        n_countries_nonzero  = np.count_nonzero(countries != 0)
        n_total_regions = regions.size
        n_regions_zero       = np.count_nonzero(regions == 0)
        n_regions_nonzero    = np.count_nonzero(regions != 0)
        mask_country_land_region_zero = (countries > 0) & (regions == 0)
        n_country_land_region_zero = np.count_nonzero(mask_country_land_region_zero)
        pct_country_land_region_zero = n_country_land_region_zero / n_total_countries

        # print checks for countries and regions as a table
        check_table = pd.DataFrame([
            {"Layer": "Countries",
             "Total pixels": n_total_countries,
             "Zero pixels": n_countries_zero,
             "Zero %": n_countries_zero / n_total_countries,
             "Non-zero pixels": n_countries_nonzero,
             "Non-zero %": n_countries_nonzero / n_total_countries,
            },
            {"Layer": "Regions",
             "Total pixels": n_total_regions,
             "Zero pixels": n_regions_zero,
             "Zero %": n_regions_zero / n_total_regions,
             "Non-zero pixels": n_regions_nonzero,
             "Non-zero %": n_regions_nonzero / n_total_regions,
            },
            {"Layer": "Land but region=0",
             "Total pixels": n_total_countries,
             "Zero pixels": n_country_land_region_zero,
             "Zero %": pct_country_land_region_zero,
             "Non-zero pixels": n_total_countries - n_country_land_region_zero,
             "Non-zero %": 1 - pct_country_land_region_zero,
            },
        ])

        print("\nCheck of country and region values:")
        print(check_table.to_string(index=False, formatters={
                                    "Total pixels": "{:,.0f}".format,
                                    "Zero pixels": "{:,.0f}".format,
                                    "Zero %": "{:.2%}".format,
                                    "Non-zero pixels": "{:,.0f}".format,
                                    "Non-zero %": "{:.2%}".format}))
        print()

        # Check: what is the value at a known location?
        # e.g. pixel at lon=0 should be somewhere in Africa/Europe, not ocean
        print("Value at lon=0 center:", countries[900, 1800])   # row 900 = equator, col 1800 = lon 0 in -180:180
        print("Value at lon=180 center:", countries[900, 3599]) # col 3599 = lon 180

    return tiff_file


"""
Assign ocean pixels of the GADM country/region raster to the nearest country/region,
but only where a gridded dataset (xarray) actually holds values.

Paste `assign_ocean_pixels_to_nearest_region` into the module that defines `create_GADM_region_raster`
(or import that function here). Assumes the raster is in a geographic CRS (lon/lat degrees, e.g. EPSG:4326)
and that the dataset uses the same longitude convention as the raster (-180..180 for the GADM raster).
"""

def create_GADM_region_raster_profile(project_dir: Path, profile: str, model: str, SSP_base:str, base_year:int=2020):

    import downscaling.settings_downscaling as settings_downscaling
    from tools.general_functions import PRINT_COLORS, apply_root_json
    import downscaling.read_process_grid_data as process_grid_data

    local_log.info(f"{PRINT_COLORS["green"]}Creating GADM region raster for profile: {profile}, model: {model}, "
                   f"SSP_base: {SSP_base}, base_year: {base_year}{PRINT_COLORS["end"]}")

    dir_processed = project_dir / "data" / "processed" / "GADM"
    dir_processed.mkdir(parents=True, exist_ok=True)

    sources = settings_downscaling.SOURCE_PROFILES[profile]
    source_POP = sources["source_POP"]
    version_POP = sources["version_POP"]
    source_GDP = sources["source_GDP"]
    version_GDP = sources["version_GDP"]
    source_EM = sources["source_EM"]
    version_EM = sources["version_EM"]

    varname_GDP = settings_downscaling.varname_GDP
    varname_POP = settings_downscaling.varname_POP
    varname_EM = settings_downscaling.varname_EM
    varname_gdp_per_capita = settings_downscaling.varname_gdp_per_capita
    varname_em_per_gdp_ppp = settings_downscaling.varname_em_per_gdp_ppp

    unit_POP = settings_downscaling.unit_POP
    unit_GDP = settings_downscaling.unit_GDP_PPP
    unit_EM = settings_downscaling.unit_EM

    settings_file = project_dir / "downscaling" / "settings_data_locations.json"
    with open(settings_file, "r") as f:
        data_files = json.load(f)
    data_files = apply_root_json(data_files, data_files["data_root"])

    coarse_factor_POP, coarse_factor_GDP, coarse_factor_EM, \
    res_min_POP, res_min_GDP, res_min_EM = process_grid_data.get_coarsening_factors(population_source=sources["source_POP"],
                                                                                    gdp_source=sources["source_GDP"],
                                                                                    emissions_source=sources["source_EM"])
    resolution_minutes = max(res_min_POP, res_min_GDP, res_min_EM)
    local_log.info(f"{PRINT_COLORS["yellow"]}Using resolution_minutes={resolution_minutes} for GADM rasterization and ocean pixel assignment{PRINT_COLORS["end"]}")

    match (profile):
        # GDP (PPP) data sources
        case "base_run":
            dir_2UP = Path(data_files["grid"]["run"]["dir_population_2UP_GHSL_2024_M3_run"])
            local_log.info(f"Reading 2UP population from {dir_2UP}")
            xr_population, _, df_POP_grid_sum = process_grid_data.read_process_grid_data_socioeconomic(dir_processed=dir_processed, varname=varname_POP, source=source_POP,
                                                                                                        version=version_POP, SSP_base=SSP_base, coarse_factor=coarse_factor_POP,
                                                                                                        unit=unit_POP, save=False, check=False, log=local_log)
            dir_Murakami = Path(data_files["grid"]["run"]["dir_gdp_ppp_Murakami_version_2021_1_run"])
            local_log.info(f"Reading Murakami 2021_1 GDP (PPP) from {dir_Murakami}")
            xr_gdp_ppp, _, df_GDP_grid_sum = process_grid_data.read_process_grid_data_socioeconomic(dir_processed=dir_processed, varname=varname_GDP, source=source_GDP,
                                                                                   version=version_GDP, SSP_base=SSP_base, coarse_factor=coarse_factor_GDP,
                                                                                   unit=unit_GDP, save=False, check=False, log=local_log)
            dir_EDGAR = Path(data_files["grid"]["run"]["dir_emissions_EDGAR_2024_run"])
            local_log.info(f"Reading EDGAR 2024 emissions from {dir_EDGAR}")
            xr_emissions, _ = process_grid_data.read_process_grid_data_EM(dir_processed=dir_processed, varname=varname_EM,
                                                                          unit=unit_EM, source=source_EM, version=version_EM,
                                                                          base_year=base_year, coarse_factor=coarse_factor_EM,
                                                                          save=False, check=False,
                                                                          log=local_log)
        case "sensitivity_3":
            xr_population = xr.DataArray()
            xr_gdp_ppp = xr.DataArray()
            xr_emissions = xr.DataArray()
        case "sensitivity_4":
            xr_population = xr.DataArray()
            xr_gdp_ppp = xr.DataArray()
            xr_emissions = xr.DataArray()
        case "sensitivity_5":
            xr_population = xr.DataArray()
            xr_gdp_ppp = xr.DataArray()
            xr_emissions = xr.DataArray()
        case _:
            raise ValueError(f"{PRINT_COLORS["red"]}Unknown profile: {profile}{PRINT_COLORS["end"]}")

    da_POP = xr_population[varname_POP].copy()
    tiff_file_POP = assign_ocean_pixels_to_nearest_region(project_dir, das=[da_POP], model=model,
                                                          resolution_minutes=resolution_minutes, x_dim="x", y_dim="y",
                                                          max_distance_km=100, label=f"{profile}_POP")
    da_GDP = xr_gdp_ppp[varname_GDP].copy()
    local_log.info(f"Saved population filled GADM region raster to {tiff_file_POP}")
    tiff_file_GDP = assign_ocean_pixels_to_nearest_region(project_dir, das=[da_GDP], model=model,
                                                          resolution_minutes=resolution_minutes, x_dim="x", y_dim="y",
                                                          max_distance_km=100, label=f"{profile}_GDP")

    local_log.info(f"Saved GDP filled GADM region raster to {tiff_file_GDP}")
    da_EM = xr_emissions[varname_EM].copy()
    tiff_file_EM = assign_ocean_pixels_to_nearest_region(project_dir, das=[da_EM], model=model,
                                                         resolution_minutes=resolution_minutes, x_dim="x", y_dim="y",
                                                         max_distance_km=100, label=f"{profile}_EM")
    local_log.info(f"Saved emissions filled GADM region raster to {tiff_file_EM}")
    # all three datasets are processed, now save the filled rasters to the processed directory
    tiff_file_POP_GDP_EM = assign_ocean_pixels_to_nearest_region(project_dir, das=[da_POP, da_GDP, da_EM], model=model,
                                                                 resolution_minutes=resolution_minutes, x_dim="x", y_dim="y",
                                                                 max_distance_km=100, label=f"{profile}_POP_GDP_EM")
    local_log.info(f"Saved emissions filled GADM region raster to {tiff_file_POP_GDP_EM}")


def assign_ocean_pixels_to_nearest_region(project_dir: Path, das: list[xr.DataArray] | xr.DataArray, model: str = "IMAGE",
                                          resolution_minutes: float = 0.5, x_dim: str = "x", y_dim: str = "y",
                                          max_distance_km: float = None, label: str = "POP_GDP_EM") -> Path:
    """
    Create the GADM country/region raster with `create_GADM_region_raster` and derive a second raster in which
    ocean pixels (country ID 0) that hold data in `da` take the country ID and region number of the nearest
    pixel that has a region assigned.

    Parameters:
        project_dir, model, resolution_minutes : passed on to `create_GADM_region_raster`
        da : xarray.DataArray with the gridded data (extra dims such as time are collapsed)
        x_dim, y_dim : names of the longitude/latitude dimensions in `da`
        max_distance_km : optional cap; ocean pixels further away than this stay ocean
        label : suffix of the new file name, e.g. "filled_EM" to keep one raster per dataset

    Returns:
        Path of the adjusted 2-band GeoTIFF (band 1 = country IDs, band 2 = region numbers).
        The raster made by `create_GADM_region_raster` is kept unchanged.
    """

    if isinstance(das, xr.DataArray):
        das = [das]

    # 1. create the country/region raster and read both bands
    tiff_file = Path(create_GADM_region_raster(project_dir, model=model, resolution_minutes=resolution_minutes, label=""))
    tiff_file_out = tiff_file.with_name(f"{tiff_file.stem}_{label}.tif")
    with rasterio.open(tiff_file) as src:
        countries, regions = src.read(1), src.read(2)
        profile, transform = src.profile, src.transform
    height, width = countries.shape
    xs = transform.c + (np.arange(width) + 0.5) * transform.a     # longitudes of pixel centres
    ys = transform.f + (np.arange(height) + 0.5) * transform.e    # latitudes of pixel centres

    # 2. mask on the raster grid: True where at least one dataset holds a non-zero, non-NaN value
    on_grid_any = np.zeros((height, width), dtype=bool)
    for da in das:
        has_data = da.notnull() & (da != 0)
        other_dims = [d for d in has_data.dims if d not in (x_dim, y_dim)]
        if other_dims:
            has_data = has_data.any(dim=other_dims)
        has_data = has_data.sortby([x_dim, y_dim])
        on_grid = (has_data.sel({x_dim: xr.DataArray(xs, dims="x_raster"), y_dim: xr.DataArray(ys, dims="y_raster")},
                                method="nearest")
                   .transpose("y_raster", "x_raster")
                   .values)
        in_extent = []  # raster pixels outside the extent of this dataset never count as data
        for coords_raster, coords_data in ((xs, has_data[x_dim].values), (ys, has_data[y_dim].values)):
            half_cell = abs(float(np.diff(coords_data).mean())) / 2 if len(coords_data) > 1 else 0.0
            in_extent.append((coords_raster >= coords_data.min() - half_cell) & (coords_raster <= coords_data.max() + half_cell))
        on_grid &= in_extent[1][:, None] & in_extent[0][None, :]
        on_grid_any |= on_grid
        print(f"{da.name}: ocean pixels with data: {np.count_nonzero(on_grid & (countries == 0)):,}")
        del has_data, on_grid

    # 3. ocean pixels that need a country/region (target), and the coastal pixels they may take it from (source)
    target = (countries == 0) & on_grid_any # & in_extent[1][:, None] & in_extent[0][None, :]
    source = regions > 0
    coast = source & ~binary_erosion(source, structure=np.ones((3, 3), dtype=bool))
    rows_tgt, cols_tgt = np.nonzero(target)
    rows_src, cols_src = np.nonzero(coast)

    # 4. nearest source pixel by great-circle distance (unit vectors on the sphere, so the dateline is handled)
    n_filled = 0
    if len(rows_tgt) > 0 and len(rows_src) > 0:
        lon_rad = np.radians(np.concatenate([xs[cols_src], xs[cols_tgt]]))
        lat_rad = np.radians(np.concatenate([ys[rows_src], ys[rows_tgt]]))
        xyz = np.column_stack([np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)])
        chord, nearest = cKDTree(xyz[:len(rows_src)]).query(xyz[len(rows_src):], workers=-1)
        distance_km = 2 * np.arcsin(np.clip(chord / 2, 0, 1)) * 6371.0088
        keep = np.ones(len(nearest), dtype=bool) if max_distance_km is None else distance_km <= max_distance_km
        rows_fill, cols_fill = rows_tgt[keep], cols_tgt[keep]
        rows_from, cols_from = rows_src[nearest[keep]], cols_src[nearest[keep]]
        countries[rows_fill, cols_fill] = countries[rows_from, cols_from]
        regions[rows_fill, cols_fill] = regions[rows_from, cols_from]
        n_filled = int(keep.sum())
    print(f"Ocean pixels with data: {len(rows_tgt):,}, reassigned to nearest region: {n_filled:,}")

    # 5. save the adjusted raster with the same CRS, transform and compression
    with rasterio.open(tiff_file_out, "w", **profile) as dst:
        dst.write(countries, 1)
        dst.write(regions, 2)
    print(f"Adjusted raster saved to: {tiff_file_out}")

    # 6. save the same two bands as NetCDF next to the GeoTIFF
    nc_file_out = tiff_file_out.with_suffix(".nc")
    ds_out = xr.Dataset({"country_id_GADM": (("y", "x"), countries), "region_number": (("y", "x"), regions)},
                        coords={"y": ys, "x": xs})
    ds_out = ds_out.rio.write_crs(profile["crs"])
    ds_out.to_netcdf(nc_file_out, mode="w", engine="netcdf4",
                     encoding={var: {"zlib": True, "complevel": 4} for var in ds_out.data_vars})
    print(f"Adjusted raster saved to: {nc_file_out}")

    return tiff_file_out
