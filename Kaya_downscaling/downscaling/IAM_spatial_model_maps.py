from pathlib import Path

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from pathlib import Path
import geopandas as gpd
import xarray as xr

def _fill_nearest_neighbour(da: xr.DataArray, max_fill_pixels: int = 0,
                             strip_size: int = 1000, overlap: int = 500) -> xr.DataArray:
    """
    Fills invalid pixels (0 or NaN) with the value of the nearest valid pixel.
    Processes in horizontal strips to avoid memory issues with large global arrays.
    da: xr.DataArray with values to fill (0 or NaN treated as invalid)
    max_fill_pixels: maximum fill distance in pixels; 0 means unlimited
    strip_size: number of rows per strip (tune to fit available memory)
    overlap: number of extra rows added above/below each strip to avoid edge artefacts
    returns: xr.DataArray with filled values, same dtype as input
    """
    data = da.values.copy()
    original_dtype = data.dtype
    data = data.astype(np.float32)  # float needed for NaN checks
    mask_valid = (data != 0) & ~np.isnan(data)

    if np.all(mask_valid):
        return da

    h, w = data.shape
    result = data.copy()

    for y_start in range(0, h, strip_size):
        y_end = min(y_start + strip_size, h)

        # Extend strip with overlap on both sides
        y0 = max(0, y_start - overlap)
        y1 = min(h, y_end + overlap)

        strip = data[y0:y1, :]
        strip_mask_valid = mask_valid[y0:y1, :]

        if not np.any(strip_mask_valid):
            continue
        if np.all(strip_mask_valid):
            result[y_start:y_end, :] = strip[y_start - y0 : y_end - y0, :]
            continue

        indices = distance_transform_edt(~strip_mask_valid, return_distances=False, return_indices=True)
        assert isinstance(indices, np.ndarray), "Expected ndarray from distance_transform_edt"
        filled_strip = strip[tuple(indices)]

        if max_fill_pixels > 0:
            distances = distance_transform_edt(~strip_mask_valid)
            assert isinstance(distances, np.ndarray), "Expected ndarray from distance_transform_edt"
            too_far = ~strip_mask_valid & (distances > max_fill_pixels)
            filled_strip[too_far] = strip[too_far]

        # Write back only the non-overlapping centre rows
        result[y_start:y_end, :] = filled_strip[y_start - y0 : y_end - y0, :]

    return xr.DataArray(result.astype(original_dtype),dims=da.dims, coords=da.coords)

def GADM_vector_to_raster(input_dir:Path, output_dir:Path, resolution_degrees: float = 5):
    """
    Convert GADM vector data to raster format using rasterio.

    Parameters:
    output_dir : str - Directory path for output files
    input_dir : str - Directory path where GADM input data is stored
    resolution_degrees : float - Resolution in degrees. Default 1/120 corresponds to 0.5 arc-minutes
    """
    # Read in
    file_GADM_countries = input_dir / "gadm_410.gpkg"
    print(f"Reading GADM countries from: {file_GADM_countries}")
    countries = gpd.read_file(file_GADM_countries)

    # Save check file
    check_countries = pd.DataFrame(countries.drop(columns="geometry"))
    check_countries.to_csv(f"{output_dir}/GADM_countries.csv", sep=";", index=False)

    # Convert
    print("Converting GADM vector data to raster format...")
    # Map ISO codes to unique integer IDs (raster pixels need numeric values)
    iso_codes = countries["GID_0"].unique()
    iso_to_id = {iso: i + 1 for i, iso in enumerate(iso_codes)}  # Start from 1, reserve 0 for NoData
    pd.DataFrame(list(iso_to_id.items()), columns=["ISO", "id"]).to_csv(f"{output_dir}/iso_to_id_mapping.csv", sep=";", index=False)
    id_to_iso = {i: iso for iso, i in iso_to_id.items()}
    pd.DataFrame(list(id_to_iso.items()), columns=["id", "ISO"]).to_csv(f"{output_dir}/id_to_iso_mapping.csv", sep=";", index=False)
    countries["iso_id"] = countries["GID_0"].map(iso_to_id)

    print(f"\nMapped {len(iso_codes)} unique ISO codes to integer IDs")
    print(countries[["GID_0", "iso_id"]].head())

    # Save the mapping for later reference
    mapping_df = pd.DataFrame({"GID_0": iso_to_id.keys(), "iso_id": iso_to_id.values()})

    # Define output filename
    resolution_minutes_str = f"{60 * resolution_degrees:.2f}".replace(".", "_")
    print(f"Resolution: {60 * resolution_degrees:.2f} arc-minutes ({resolution_degrees:.6f} degrees)")
    raster_file = f"{output_dir}/iso_codes_raster_{resolution_minutes_str}.tif"

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

    rasterized = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=nodata_value,
        dtype=np.int16,
        all_touched=True  # Set True if you want all pixels touched by polygons
    )

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

    return raster_file, iso_to_id, id_to_iso

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

def create_GADM_region_raster(input_GADM_dir: Path, input_model_dir:Path,  output_dir:Path, model:str="IMAGE", resolution_minutes:float=0.5) -> Tuple[Path, Path]:

    '''
    This function creates a raster file from GADM vector data using the specified resolution (in minutes) with country IDs and region numbers for the specified model
    The results is saved to both NetCDF and GeoTIFF formats.
    pre: file gadm_410.gpkg is downloaded from https://gadm.org/, more specifically from
        https://gadm.org/download_world.html and stored in the input directory
    input_GADM_dir: directory where GADM vector data the file gadm_410.gpkg is stored (e.g. "data/input/GADM")
    input_model_dir: directory where files "country_to_regions.csv" and  "image_region_numbers.csv" are stored (e.g. "data/input/models/IMAGE")
    output_dir: directory where converted GADM raster file will be stored (e.g. "data/processed/GADM")
    resolution_minutes: resolution of the output raster file in arc minutes (e.g. 0.5 for 30 arc seconds, 6 for 6 arc minutes)
    '''
    # TO DO --> make this model agnostic (now only for IMAGE)

    print("Creating GADM raster file for regions...")
    id_to_iso = pd.DataFrame()
    # 1. check if file with GADM raster countries exists
    res_min_file_end = f"{resolution_minutes:.2f}".replace(".", "_")
    iso_GADM_raster_file = f"{output_dir}/iso_codes_raster_{res_min_file_end}.tif"
    print(f"Checking if GADM raster file exists at: {iso_GADM_raster_file}")
    if not Path(iso_GADM_raster_file).exists():
        print(f"Reading in GADM raster with resolution {resolution_minutes} arc minutes file for countries: {iso_GADM_raster_file}")
        raster_file, iso_to_id, id_to_iso = GADM_vector_to_raster(input_GADM_dir, output_dir, resolution_degrees = resolution_minutes/60)
        df_iso_to_id = pd.DataFrame(list(iso_to_id.items()), columns=["ISO", "id"])
    else:
        print(f"GADM raster with resolution {resolution_minutes} arc minutes file already exists at: {iso_GADM_raster_file}, skipping creation.")
        df_iso_to_id = pd.read_csv(f"{output_dir}/id_to_iso_mapping.csv", sep=";")

    # 2. convert to xarray and rasterio dataset
    # Open GADM raster file
    ds_GADM_raster = xr.open_dataset(iso_GADM_raster_file, decode_coords="all")
    ds_GADM_raster = ds_GADM_raster.rename({"band_data":"country_id_GADM"})
    ds_GADM_raster = ds_GADM_raster.squeeze("band")
    print(ds_GADM_raster)

    # Assign proper geographic coordinates before any further processing
    resolution_degrees = resolution_minutes / 60
    n_x = ds_GADM_raster.sizes["x"]
    n_y = ds_GADM_raster.sizes["y"]
    x_coords = np.linspace(-180 + resolution_degrees / 2, 180 - resolution_degrees / 2, n_x)
    y_coords = np.linspace(90 - resolution_degrees / 2, -90 + resolution_degrees / 2, n_y)
    ds_GADM_raster = ds_GADM_raster.assign_coords(x=x_coords, y=y_coords)
    ds_GADM_raster = ds_GADM_raster.rio.write_crs("EPSG:4326")

    # add country_ID_GADM code from GADM
    df_model_GADM_region_code_number = pd.DataFrame()
    if model == "IMAGE":
        country_to_region_file = input_model_dir / "country_to_regions.csv"
        #df_model_coutry_to_region = pd.read_csv(f"{output_dir.name}/data/input/models/IMAGE/country_to_regions.csv", sep=";") # ISO3
        df_model_coutry_to_region = pd.read_csv(country_to_region_file, sep=";") # ISO3
        df_model_coutry_to_region.loc[df_model_coutry_to_region["ISO3"]=="GRL", "Region code"] = "WEU" # change GRL region code to WEU
        # correct "HKG" and "MAC" ISO3 codes in GADM
        df_iso_to_id_IMAGE = df_iso_to_id.copy()
        new_rows = pd.DataFrame({"id": [None, None],
                                 "ISO": ["HKG", "MAC"]})
        df_iso_to_id_IMAGE = pd.concat([df_iso_to_id_IMAGE, new_rows], ignore_index=True)
        df_model_GADM_region_code = pd.merge(df_model_coutry_to_region, df_iso_to_id_IMAGE, left_on="ISO3", right_on="ISO", how="outer")
        df_model_GADM_region_code.rename(columns={"ISO3": "ISO3_model", "ISO": "ISO3_GADM"}, inplace=True) # TO DO --> check missing ISO3 codes between model/GADM
        # print ISO3_model codes that are missing in GADM
        missing_ISO3_GADM = df_model_GADM_region_code[df_model_GADM_region_code["ISO3_GADM"].isna()]["ISO3_model"].unique()
        if len(missing_ISO3_GADM) > 0:
            print(f"Warning: The following ISO3 codes from the model are missing in GADM and will be assigned a region number of 0:")
            print(missing_ISO3_GADM)
        # print ISO3_GADM codes that are missing in model
        missing_ISO3_model = df_model_GADM_region_code[df_model_GADM_region_code["ISO3_model"].isna()]["ISO3_GADM"].unique()
        if len(missing_ISO3_model) > 0:
            print(f"Warning: The following ISO3 codes from GADM are missing in the model and will be ignored:")
            print(missing_ISO3_model)

        # map to region numbers
        #df_region_numbers = pd.read_csv(f"{output_dir.name}/data/input/models/IMAGE/image_region_numbers.csv", sep=",")
        region_numbers_file = input_model_dir / "image_region_numbers.csv"
        df_region_numbers = pd.read_csv(region_numbers_file, sep=",")
        df_model_GADM_region_code_number = pd.merge(df_model_GADM_region_code, df_region_numbers, left_on="Region code", right_on="IMAGE region", how="left")
        df_model_GADM_region_code_number.drop(columns=["Country name", "Region code", "IMAGE region"], inplace=True)
        df_model_GADM_region_code_number.rename(columns={"id": "country_id_GADM", "IMAGE number": "IMAGE_region_number"}, inplace=True)
        df_model_GADM_region_code_number["IMAGE_region_number"] = df_model_GADM_region_code_number["IMAGE_region_number"].fillna(0).astype(np.int8)
        df_model_GADM_region_code_number["country_id_GADM"] = df_model_GADM_region_code_number["country_id_GADM"].fillna(0).infer_objects(copy=False).astype(np.int16)
        df_model_GADM_region_code_number.rename(columns={"IMAGE_region_number": "region_number"}, inplace=True)
        df_model_GADM_region_code_number.to_csv(f"{output_dir}/IMAGE_GADM_country_to_region_codes.csv", sep=";", index=False)

    print(df_model_GADM_region_code_number)
    print("\nMerging GADM raster with model region numbers...")

    # add region numbers to GADM raster
    country_to_region = df_model_GADM_region_code_number.set_index("country_id_GADM")["region_number"].to_dict()
    # ds_GADM_raster["region_number"] = xr.full_like(ds_GADM_raster["country_id_GADM"], fill_value=0, dtype=np.int16)
    # ds_GADM_raster["region_number"] = xr.apply_ufunc(np.vectorize(lambda x: 0 if np.isnan(x) else country_to_region.get(x, 0)),
    #                                                  ds_GADM_raster["country_id_GADM"],
    #                                                  output_dtypes=[np.int16])

    ds_GADM_raster["region_number"] = _map_country_ids_to_region_numbers(
        ds_GADM_raster["country_id_GADM"], country_to_region
    )


    print(ds_GADM_raster)
    print(f"regions: {np.unique(ds_GADM_raster["region_number"].values)}")

    # extend region numbers to nearest neighbour to fill small gaps in GADM raster (e.g. small islands)
    # This only applies for country_id_GADM that are present in the model (i.e. with region_number > 0)
    ds_GADM_raster["region_number"]   = _fill_nearest_neighbour(ds_GADM_raster["region_number"],   max_fill_pixels=0)
    ds_GADM_raster["country_id_GADM"] = _fill_nearest_neighbour(ds_GADM_raster["country_id_GADM"], max_fill_pixels=0)

    # save to netcdf and tiff
    print(f"CRS: {ds_GADM_raster.rio.crs}")
    print(f"\nSaving GADM raster to netcdf file in {output_dir} with region numbers...")
    raster_nc_file = output_dir / f"IMAGE_GADM_regions_raster_{res_min_file_end}_arcmin.nc"

    ds_GADM_raster.load()
    print(f"CRS: {ds_GADM_raster.rio.crs}")
    print(f"\nSaving GADM raster to netcdf file in {output_dir} with region numbers...")
    raster_nc_file = output_dir / f"IMAGE_GADM_regions_raster_{res_min_file_end}_arcmin.nc"
    # add chunking and compression to the netcdf write to keep memory manageable during the write:
    encoding = {"country_id_GADM": {"chunksizes": (256, 256), "zlib": True, "complevel": 4}, "region_number":   {"chunksizes": (256, 256), "zlib": True, "complevel": 4}}

    ds_GADM_raster.to_netcdf(raster_nc_file, mode="w", engine="netcdf4", encoding=encoding)
    print(f"\nSaving GADM raster to tiff file in {output_dir} with region numbers...")
    data = np.stack([ds_GADM_raster["country_id_GADM"].values, ds_GADM_raster["region_number"].values])
    ds_GADM_raster.close()
    raster_tif_file = output_dir / f"IMAGE_GADM_regions_raster_{res_min_file_end}_arcmin.tif"
    with rasterio.open(
        raster_tif_file,
        "w",
        driver="GTiff",
        height=data.shape[1], width=data.shape[2],
        count=2,
        dtype=data.dtype,
        crs=ds_GADM_raster.rio.crs,
        transform=ds_GADM_raster.rio.transform(),
        compress="LZW",
        predictor=2,  # Improves compression for integer data
        tiled=True,   # Better for large files
        all_touched=True,
        blockxsize=256, blockysize=256) as dst: dst.write(data)

    return raster_tif_file, raster_nc_file
