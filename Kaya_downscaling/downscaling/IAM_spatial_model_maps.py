from pathlib import Path
import json
from typing import Tuple

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

colour_red = "\033[91m"
colour_green = "\033[92m"
colour_yellow = "\033[93m"
color_end = "\033[0m"

def GADM_vector_to_raster(input_dir:Path, output_dir:Path, resolution_degrees: float = 5, plot:bool=False) -> Tuple[Path, pd.DataFrame, pd.DataFrame]:
    """
    Read in GADM countries file and convert vector data to raster format using rasterio.

    Parameters
    project_dir : str - Project directory path
    resolution_degrees : float - Resolution in degrees. Default 1/120 corresponds to 0.5 arc-minutes
    plot : bool - Whether to create a plot of the countries
    """
    print(f"\nConverting GADM vector data to raster format with resolution {resolution_degrees:.6f} degrees ({60 * resolution_degrees:.2f} arc-minutes)...")

    # Read in
    file_GADM_countries = input_dir / "gadm_410.gpkg"
    print(f"Reading GADM countries from: {file_GADM_countries}")
    countries = gpd.read_file(file_GADM_countries)
    # remove countries not needed
    #mask_countries_excluded = ~countries["GID_0"].str.match(r"^(X|Z\d{2})") # also exclude 'XKO' (Kosovo)
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
    if plot:
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
    # pd.DataFrame(list(iso_to_id.items()), columns=["ISO", "id"]).to_csv(f"{dir_GADM}/iso_to_id_mapping.csv", sep=";", index=False)
    # id_to_iso = {i: iso for iso, i in iso_to_id.items()}
    # pd.DataFrame(list(id_to_iso.items()), columns=["id", "ISO"]).to_csv(f"{dir_GADM}/id_to_iso_mapping.csv", sep=";", index=False)
    countries["iso_id"] = countries["GID_0"].map(iso_to_id)

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
    axes[1,0].set_title("Band 2 — IMAGE Region Numbers")
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

def create_GADM_region_raster(project_dir:Path, model:str="IMAGE", resolution_minutes:float=0.5, plot=False):
    '''
    Creates a raster and netcdf file with GADM country and model region codes
    Steps:
    1. Check if raster file with GADM country codes already exists for the specified resolution, if not create it by converting GADM vector data to raster format using convert_GIS.GADM_vector_to_raster
    2. Convert GADM raster to rioxarray and retrieve transform and crs using rasterio
    3. Add model region numbers to GADM raster based on country-region mapping for the specified model
    4. Extend region numbers to nearest neighbour to fill small gaps in GADM raster (e.g. small islands)
    5. Save GADM raster with country and region codes to netcdf and tiff files
    '''
    dir_GADM = project_dir / "data" / "processed" / "GADM"
    print(f"PROJECT_DIR: {project_dir}")
    print(f"dir_GADM: {dir_GADM}")
    dir_GADM.mkdir(parents=True, exist_ok=True)

    settings_file = project_dir / "downscaling" / "settings_data_locations.json"
    with open(settings_file, "r") as f:
        settings = json.load(f)
        data_dir_GADM = Path(settings["GADM"]["dir_GADM_single"])

    print("Creating GADM raster file for regions...")

    id_to_iso = pd.DataFrame()
    # 1. check if file with GADM raster countries exists
    res_min_file_end = f"{resolution_minutes:.2f}".replace(".", "_")
    iso_GADM_raster_file = f"{dir_GADM}/iso_codes_raster_{res_min_file_end}.tif"
    print(f"Checking if GADM raster file exists at: {iso_GADM_raster_file}")
    if not Path(iso_GADM_raster_file).exists():
        print(f"Reading in GADM raster with resolution {resolution_minutes} arc minutes file for countries: {iso_GADM_raster_file}")
        raster_file, df_iso_to_id, df_id_to_iso = GADM_vector_to_raster(data_dir_GADM, dir_GADM, resolution_degrees = resolution_minutes/60, plot=False)
        #df_iso_to_id = pd.DataFrame(list(iso_to_id.items()), columns=["ISO", "id"])
    else:
        print(f"GADM raster with resolution {resolution_minutes} arc minutes file already exists at: {iso_GADM_raster_file}, skipping creation.")
        df_iso_to_id = pd.read_csv(f"{dir_GADM}/id_to_iso_mapping.csv", sep=";")

    # 2. convert to rioxarray and rasterio dataset and add model region numbers
    # Open GADM raster file
    #ds_GADM_raster = xr.open_dataset(iso_GADM_raster_file, decode_coords="all")
    ds_GADM_raster = rxr.open_rasterio(iso_GADM_raster_file)
    ds_GADM_raster = ds_GADM_raster.squeeze("band", drop=True)
    #ds_GADM_raster = ds_GADM_raster.rename({"band_data":"country_id_GADM"})
    ds_GADM_raster = ds_GADM_raster.to_dataset(name="country_id_GADM")
    print(f"\nGADM raster dataset: {ds_GADM_raster}")

    # retrieve transform and crs using rasterio which is more reliable than using rioxarray attributes, which can be incorrect after processing steps
    with rasterio.open(iso_GADM_raster_file) as src:
        transform = src.transform
        crs = src.crs

    # add country_ID_GADM code from GADM
    df_model_GADM_region_code_number = pd.DataFrame()
    if model == "IMAGE":
        country_to_region_file = project_dir / "data" / "input" / "models" / "IMAGE" / "country_to_regions.csv"
        df_model_coutry_to_region = pd.read_csv(country_to_region_file, sep=";") # ISO3
        df_model_coutry_to_region.rename(columns={"Country name": "country_name_model"}, inplace=True)
        df_model_coutry_to_region.loc[df_model_coutry_to_region["ISO3"]=="GRL", "Region code"] = "WEU" # change GRL region code to WEU
        # correct "HKG" and "MAC" ISO3 codes in GADM
        df_iso_to_id_IMAGE = df_iso_to_id.copy()
        new_rows = pd.DataFrame({"id": [None, None],
                                 "ISO": ["HKG", "MAC"],
                                 "NAME": ["Hong Kong", "Macau"]})
        df_iso_to_id_IMAGE = pd.concat([df_iso_to_id_IMAGE, new_rows], ignore_index=True)
        df_model_GADM_region_code = pd.merge(df_model_coutry_to_region, df_iso_to_id_IMAGE, left_on="ISO3", right_on="ISO", how="outer")
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
        region_numbers_file = project_dir / "data" / "input" / "models" / "IMAGE" / "image_region_numbers.csv"
        df_region_numbers = pd.read_csv(region_numbers_file, sep=",")
        df_model_GADM_region_code_number = pd.merge(df_model_GADM_region_code, df_region_numbers, left_on="Region code", right_on="IMAGE region", how="left")
        df_model_GADM_region_code_number.drop(columns=["Region code", "IMAGE region", "country_name_GADM", "ISO3_GADM", "ISO3_model"], inplace=True)
        df_model_GADM_region_code_number.rename(columns={"id": "country_id_GADM",  "IMAGE number": "model_region_number"}, inplace=True)
        df_model_GADM_region_code_number["model_region_number"] = df_model_GADM_region_code_number["model_region_number"].fillna(0).astype(np.int8)
        # add ocean with region number 0 to mapping
        ocean_row = pd.DataFrame({"country_id_GADM": [0], "country_name_model": ["Ocean"], "model_region_number": [0]})
        df_model_GADM_region_code_number = pd.concat([df_model_GADM_region_code_number, ocean_row],ignore_index=True)
        df_model_GADM_region_code_number.rename(columns={"model_region_number": "region_number"}, inplace=True)
        df_model_GADM_region_code_number["country_id_GADM"] = (pd.to_numeric(df_model_GADM_region_code_number["country_id_GADM"], errors="coerce")
                                                                             .fillna(0)
                                                                             .astype(np.int16))
        df_model_GADM_region_code_number.rename(columns={"IMAGE_region_number": "region_number"}, inplace=True)
        df_model_GADM_region_code_number.to_csv(f"{dir_GADM}/IMAGE_GADM_country_to_region_codes.csv", sep=";", index=False)

    print("\nMerging GADM raster with model region numbers...")
    # add region numbers to GADM raster
    country_to_region = df_model_GADM_region_code_number.set_index("country_id_GADM")["region_number"].to_dict()
    ds_GADM_raster["region_number"] = xr.full_like(ds_GADM_raster["country_id_GADM"], fill_value=0, dtype=np.int16)
    ds_GADM_raster["region_number"] = xr.apply_ufunc(np.vectorize(lambda x: 0 if np.isnan(x) else country_to_region.get(x, 0)),
                                                     ds_GADM_raster["country_id_GADM"],
                                                     output_dtypes=[np.int16])

    # save to netcdf and tiff
    print(f"CRS: {ds_GADM_raster.rio.crs}")
    print(f"\nSaving GADM raster to {colour_green}netcdf {color_end}file in {dir_GADM} with region numbers...")
    ds_GADM_raster.to_netcdf(f"{dir_GADM}/IMAGE_GADM_regions_raster_{res_min_file_end}_arcmin.nc", mode="w", engine="netcdf4")
    print(f"\nSaving GADM raster to {colour_yellow}tiff {color_end}file in {dir_GADM} with region numbers...")
    data = np.stack([ds_GADM_raster["country_id_GADM"].values, ds_GADM_raster["region_number"].values])
    tiff_file = f"{dir_GADM}/IMAGE_GADM_regions_raster_{res_min_file_end}_arcmin.tif"
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
    dir_input = Path(f"{dir_GADM}/IMAGE_GADM_regions_raster_{res_min_file_end}_arcmin.tif")
    dir_fig = Path(f"{dir_GADM}/figures")
    dir_fig.mkdir(parents=True, exist_ok=True)
    #plot_maps.plot_coast_checks(dir_input, dir_fig, f"_{resolution_minutes:.2f}")
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
