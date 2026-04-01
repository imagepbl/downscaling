import warnings
import shutil
import logging
from datetime import datetime
import re
import calendar
import cftime
import json

from typing import Tuple, Optional

import numpy as np
import pandas as pd

from dask.diagnostics.progress import ProgressBar
import gc

from matplotlib import pyplot as plt
from matplotlib import use as plt_use
import seaborn as sns

from osgeo import gdal
import rasterio
import rasterio.plot
from rasterio.transform import from_bounds
from rasterio.errors import NotGeoreferencedWarning
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from affine import Affine

import xarray as xr
import rioxarray as rxr
from pathlib import Path

import tqdm
from tools.functions_logging import init_logging
from tools.general_functions import PRINT_COLORS
from .settings_downscaling_cities import NL_bbox, lon_MidAtlantic, lat_MidAtlantic, lon_Amsterdam, lat_Amsterdam
from downscaling.settings_resolution import DATASETS

gdal.UseExceptions()

progress_bar = ProgressBar()
progress_bar.register()

chunks = 512  # chunk size for dask operations

local_log, dummy_log = init_logging("log", "log/reading_data")

def compare_two_raster_files(project_dir:Path, src1:DatasetReader, src2: DatasetReader, rtol=1e-5, atol=1e-8, save_not_close_map=False, chunk_size = 1024, log: logging.Logger=local_log) -> dict:

    metadata_match = {
        "crs_match": src1.crs == src2.crs,
        "shape_match": src1.shape == src2.shape,
        "bounds_match": src1.bounds == src2.bounds,
        "resolution_match": src1.res == src2.res,
        "count_match": src1.count == src2.count
    }

    # If shapes don't match, can't compare pixel values
    if not metadata_match["shape_match"]:
        print(f"{PRINT_COLORS['red']}Shapes don't match - cannot compare pixel values{PRINT_COLORS['end']}")
        print(f"shape 1: {src1.shape}, shape 2: {src2.shape}")
        exit()

    # Read data as xarray DataArrays with chunking for memory efficiency
    # Create coordinates
    y_coords = np.linspace(src1.bounds.top, src1.bounds.bottom, src1.height)
    x_coords = np.linspace(src1.bounds.left, src1.bounds.right, src1.width)
    data1 = src1.read(out_shape=(src1.count, src1.height, src1.width))
    data2 = src2.read(out_shape=(src2.count, src2.height, src2.width))

    # Convert to xarray DataArrays
    raster1 = xr.DataArray(
        data1,
        coords={
            "band": np.arange(1, src1.count + 1),
            "y": y_coords,
            "x": x_coords
        },
        dims=["band", "y", "x"]
    )
    raster1.rio.write_crs(src1.crs, inplace=True)

    raster2 = xr.DataArray(
        data2,
        coords={
            "band": np.arange(1, src2.count + 1),
            "y": y_coords,
            "x": x_coords
        },
        dims=["band", "y", "x"]
    )
    raster2.rio.write_crs(src2.crs, inplace=True)

    # Chunk the arrays for memory-efficient processing
    raster1 = raster1.chunk({"x": chunk_size, "y": chunk_size})
    raster2 = raster2.chunk({"x": chunk_size, "y": chunk_size})

    # Check if arrays are close (memory-efficient for chunked data)
    # are_close = (xr.apply_ufunc(np.isclose, raster1, raster2, kwargs={"rtol": rtol, "atol": atol, "equal_nan": True}, dask="allowed")
    #             .all()
    #             .compute())

    close_map = xr.apply_ufunc(np.isclose, raster1, raster2, kwargs={"rtol": rtol, "atol": atol, "equal_nan": True}, dask="allowed")
    are_close = close_map.all().compute()
    not_close_map = (~close_map).astype(np.uint8).compute()
    if save_not_close_map:
        not_close_map.rio.write_crs(src1.crs, inplace=True)
        file_path = project_dir / "not_close_map.tif"
        not_close_map.rio.to_raster(file_path, dtype="uint8")

    difference = raster2 - raster1
    has_any_diff = (difference != 0).any().compute()

    # If there are differences, compute statistics
    diff_stats = None
    if has_any_diff:
        diff_stats = {
            "min": float(difference.min().compute()),
            "max": float(difference.max().compute()),
            "mean": float(difference.mean().compute()),
            "std": float(difference.std().compute()),
            "num_different_pixels": int((difference != 0).sum().compute())
        }
        log.info(f"metadata: {metadata_match}")
        log.info(f"can_compare_values: {True}")
        log.info(f"are_close: {bool(are_close)}")
        log.info(f"has_any_differences: {bool(has_any_diff)}")
        log.info(f"difference_stats: {diff_stats}")
        log.info(f"tolerance_used: {{'rtol': rtol, 'atol': atol}}")
    else:
        print(f"{PRINT_COLORS["green"]}All pixel values are close within the specified tolerance{PRINT_COLORS["end"]}")

    return {
        "metadata": metadata_match,
        "can_compare_values": True,
        "are_close": bool(are_close),
        "has_any_differences": bool(has_any_diff),
        "difference_stats": diff_stats,
        "tolerance_used": {"rtol": rtol, "atol": atol}
    }

def ensure_rio_dimension_order(da):
    """
    Ensure DataArray dimensions are in rioxarray-compatible order.
    Spatial dimensions (y, x or lat, lon) should be last.
    """
    dims = list(da.dims)

    # Identify spatial dimensions
    spatial_y = None
    spatial_x = None

    for dim in dims:
        if dim in ["y", "lat", "latitude"]:
            spatial_y = dim
        elif dim in ["x", "lon", "longitude"]:
            spatial_x = dim

    if spatial_y is None or spatial_x is None:
        print(f"Warning: Could not identify spatial dimensions in {dims}")
        return da

    # Build new dimension order: non-spatial dims first, then spatial
    non_spatial = [d for d in dims if d not in [spatial_y, spatial_x]]
    new_order = non_spatial + [spatial_y, spatial_x]

    # Check if reordering is needed
    if tuple(new_order) == da.dims:
        return da

    print(f"Reordering dimensions from {da.dims} to {tuple(new_order)}")

    return da.transpose(*new_order)


def read_in_tiff_to_rio(data_dir: Path, glob_pattern: str, search_pattern: str, varname: str, year_check:int, log: logging.Logger) -> xr.Dataset:
    """
    Read TIFF files and combine them into a NetCDF while preserving all metadata including CRS.
    """

    files = sorted(data_dir.glob(glob_pattern))
    log.info(f"Found {len(files)} files matching {glob_pattern}")

    data_list = []
    years = []

    # Store metadata from the first file (assuming all files have consistent metadata)
    first_file_metadata = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        for i, f in enumerate(files):
            #log.info(f"Reading file: {f.name}")
            year = int(re.search(search_pattern, f.name).group(1))
            #if year >= base_year:
            years.append(year)

            da = rxr.open_rasterio(f, chunks={"x": "auto", "y": "auto"})
            da = da.squeeze(drop=True)
            da = da.chunk("auto")
            da = ensure_rio_dimension_order(da)

            # Store metadata from first file
            if i == 0:
                first_file_metadata = {
                    'crs': da.rio.crs,
                    'transform': da.rio.transform(),
                    'nodata': da.rio.nodata, # informational only, not used internally
                    'encoding': da.encoding.copy(),
                    'attrs': da.attrs.copy(),
                    'unit': da.attrs.get('unit', None)
                }
                log.info(f"NoData: {first_file_metadata['nodata']}")
                log.info(f"CRS: {first_file_metadata['crs']}")
                log.info(f"Transform: {first_file_metadata['transform']}")
                arc_seconds, arc_minutes, arc_degrees = calculate_resolution(da)
                log.info(f"Resolution of annual data: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.2f} arc degrees")
            if year == year_check:
                check_nan_locations(f, varname, year_check, log)
            data_list.append(da)

    # Combine along time dimension
    # --> data becomes CF format (CF-compliant geospatial NetCDF)
    # https://cfconventions.org/
    # This means:
    # - missing data = NaN
    # - nodata is not a raster attribute
    # - _FillValue is only for writing to disk
    ds_rxr = xr.concat(data_list, dim="time")
    ds_rxr = ds_rxr.assign_coords(time=years)
    ds_rxr = ds_rxr.to_dataset(name=varname)

    # Restore CRS information
    # From this moment on:
    # - internal missing data = NaN
    # - rio.nodata is irrelevant
    # - all math works correctly
    # - memory and reload behave identically
    # What should not be moved forward:
    # - 'encoding': da.encoding.copy(),       # ❌ remove
    # - 'spatial_ref': da.spatial_ref,         # ❌ remove
    # - 'spatial_ref_attrs': da.spatial_ref.attrs.copy()  # ❌ remov

    # Handle nodata values
    nodata = first_file_metadata['nodata']
    print(f"Changing from tiff to xarray dataset for variable: {varname}, therefore nodata value ({nodata}) is set to NaN internally")
    if nodata is not None:
        ds_rxr[varname] = ds_rxr[varname].where(ds_rxr[varname] != nodata)
        ds_rxr.attrs["source_nodata"] = nodata
    else:
        ds_rxr.attrs["source_nodata"] = np.nan
    # Restore CRS if available
    if first_file_metadata['crs'] is not None:
        ds_rxr = ds_rxr.rio.write_crs(first_file_metadata['crs'])
        ds_rxr = ds_rxr.rio.write_coordinate_system()
    # Restore transform if available
    if first_file_metadata['transform'] is not None:
        ds_rxr = ds_rxr.rio.write_transform(first_file_metadata['transform'])
    ds_rxr.attrs.update({
        'title': f'Combined {varname} data from TIFF files',
        'source': f'Combined from {len(files)} TIFF files',
        #'crs': str(first_file_metadata['crs']) if first_file_metadata['crs'] else 'Unknown',
        'original_transform': str(first_file_metadata['transform']) if first_file_metadata['transform'] else 'Unknown',
        #'original_nodata': first_file_metadata['nodata'] if first_file_metadata['nodata'] is not None else 'None'
    })

    return ds_rxr

def check_data_locations(da: xr.DataArray, year_check:int) -> list:
    info_lines = []

    info_lines.append(f"\nCheck data locations:")
    info_lines.append(f"Data read with rioxarray: {da}")
    info_lines.append(f"Tests:")

    # check if 'time' coordinate exists, extract data for 2020
    if "time" in da.coords:
        ds_Amsterdam = da.sel(time=year_check, y=lat_Amsterdam, x=lon_Amsterdam, method="nearest")
        ds_Atlantic = da.sel(time=year_check, y=lat_MidAtlantic, x=lon_MidAtlantic, method="nearest")
    else:
        ds_Amsterdam = da.sel(y=lat_Amsterdam, x=lon_Amsterdam, method="nearest")
        ds_Atlantic = da.sel(y=lat_MidAtlantic, x=lon_MidAtlantic, method="nearest")

    # add locations
    info_lines.append(f"\tAmsterdam: {ds_Amsterdam.values:,.1f}")
    info_lines.append(f"\tMidAtlantic: {ds_Atlantic.values:,.1f}")

    # NL --> sum values within bbox
    if da.y[0] < da.y[-1]:
        # Ascending y coordinates (south to north)
        y_slice = slice(NL_bbox["south"], NL_bbox["north"])
    else:
        # Descending y coordinates (north to south)
        y_slice = slice(NL_bbox["north"], NL_bbox["south"])

    x_slice = slice(NL_bbox["west"], NL_bbox["east"])

    info_lines.append(f"\tNetherlands: y (lat): {(NL_bbox['south'], NL_bbox['north'])}, "
                      f"x (lon): {(NL_bbox['west'], NL_bbox['east'])}")

    if "time" in da.coords:
        ds_test = da.sel(time=year_check, x=x_slice, y=y_slice).sum(skipna=True)
    else:
        ds_test = da.sel(x=x_slice, y=y_slice).sum(skipna=True)
    info_lines.append(f"\tNetherlands: {ds_test.values:,.1f}")

    return info_lines

def calculate_resolution(da: xr.DataArray, input_unit="decimal_degrees") -> tuple:
    """
    Calculate the spatial resolution of a raster file or rioxarray DataArray.

    Parameters:
        data (str or rxr.DataArray): Path to raster file or a rioxarray DataArray.
        input_unit (str): Unit of input coordinates (default: "decimal_degrees").

    Returns:
        tuple: (arc_seconds, arc_minutes)
    """

    if isinstance(da, str):
        da = rxr.open_rasterio(da)
        if not type(da) is xr.DataArray:
            log.info(f"Error: Expected a single DataArray, got {type(da)}")
            exit()
    else:
        da = da

    # list_info.append(f"Calculate resolution for DataArray:")
    # list_info.append(f"CRS: {da.rio.crs}")

    x_res, y_res = da.rio.resolution()
    x_res = abs(x_res)
    y_res = abs(y_res)

    if x_res != y_res:
        print(f"Warning: Non-uniform resolution detected ({x_res} vs {y_res})")

    arc_seconds = x_res * 3600
    arc_minutes = x_res * 60
    arc_degrees = x_res

    return arc_seconds, arc_minutes, arc_degrees

def print_transform(transform):
    list_info = []
    list_info.append(f"Affine Transform:")
    list_info.append(f"|{transform.a:.4f}, {transform.b:.4f}, {transform.c:.4f}|")
    list_info.append(f"|{transform.d:.4f}, {transform.e:.4f}, {transform.f:.4f}|")
    list_info.append(f"|{transform.g:.4f}, {transform.h:.4f}, {transform.i:.4f}|")

    return list_info

def print_info_rasterio(src: DatasetReader, band:int=1, log: logging.Logger=local_log) -> float:
    raster = src.read(band)

    print("=== Rasterio dataset parameters ===")
    print(f"\tcrs: {src.crs}")
    print(f"\tNodata: {src.nodata}")
    print(f"\tData type: {src.dtypes[band - 1]}")
    print(f"\tBounds (west, south, east, north): {src.bounds}")
    print(f"\tWidth: {src.width}, Height: {src.height}")
    print(f"\tNumber of bands: {src.count}")
    print(f"\tPixel size: {src.res}")
    print(f"\tRaster dimensions: {src.width} x {src.height} pixels")
    print(f"\tNumber of bands: {src.count}")
    print(f"\tunit: {src.units}")
    print(f"\tunit: {src.crs.linear_units}")
    print(f"mask_flag_enums: {src.mask_flag_enums}")
    unit = src.crs.linear_units

    # Get nodata value from band if not specified
    print(f"Check no data in band {band}")
    nodata = src.nodatavals[band - 1]
    print(f"NoData value: {nodata}")
    _FillValue = src.tags(band).get('_FillValue')
    print(f"NoData value: {_FillValue}")

    # Get GeoTransform equivalent
    transform = src.transform
    print("GeoTransform:")
    print(f"\t{transform.c} ({unit}(: x-coordinate of the upper-left corner of the upper-left pixel.")
    print(f"\t{transform.a} ({unit}(: w-e pixel resolution / pixel width.")
    print(f"\t{transform.b} ({unit}(: row rotation (typically zero).")
    print(f"\t{transform.f} ({unit}(: y-coordinate of the upper-left corner of the upper-left pixel.")
    print(f"\t{transform.d} ({unit}(: column rotation (typically zero).")
    print(f"\t{transform.e} ({unit}(: n-s pixel resolution / pixel height (negative value for a north-up image).")

    # check min/max values
    print(f"Min value: {raster.min()}")
    print(f"Max value: {raster.max()}")
    print(f"Unique values: {np.unique(raster)}")
    print("===========================")

    return nodata

def print_info_rioxarray(da: xr.DataArray, log: logging.Logger=local_log):
    log.info("=== xr parameters ===")
    log.info(f"encoding:")
    [log.info(f"  - {key}: {value}") for key, value in da.encoding.items()]
    log.info(f"attrs:")
    [log.info(f"  - {key}: {value}") for key, value in da.attrs.items()]
    log.info(f"coords:")
    [log.info(f"\t  - {key}: {value}") for key, value in da.coords.items()]
    if hasattr(da, 'spatial_ref'):
        [log.info(f"  - {key}: {value}") for key, value in da.spatial_ref.attrs.items()]
    else:
        log.info(f"  - No spatial_ref attribute found")

    log.info(f"Shape: {da.values.shape}")
    log.info(f"Type: {da.values.dtype}")
    log.info(f"Dims: {da.dims}")
    log.info(f"Sizes: {da.sizes}")

    log.info("=== rxr rio parameters ===")
    # if da.rio.nodata exists
    if hasattr(da.rio, 'nodata'):
        log.info(f"da.rio.nodata: {da.rio.nodata}")
    else:
        log.info(f"da.rio.nodata: <not defined>")

    rio_attrs = [attr for attr in dir(da.rio) if not attr.startswith("_")]
    for attr in rio_attrs:
        try:
            value = getattr(da.rio, attr)
            # Check if it's a method or a value
            if callable(value):
                pass
            else:
                log.info(f"\t\t{attr}: {value}")
        except Exception as e:
            log.info(f"{attr}: <error accessing: {e}>")

def coarsen_save_rio_xarray(ds_rxr: xr.Dataset, factor: float, zero_to_nan:bool, save: bool, save_type: str, chunks_size_save: int,
                            varname:str, filepath: Path, aggregation_methods: dict[str, str], log: logging.Logger=local_log) -> xr.Dataset:
        # set_chunks_method is "optimal" or "auto"
        # save_type is "netcdf" or "zar", "both" or "none"

        current_dir = Path().cwd()
        log.info(f"Current directory: {current_dir}")
        aggregation_method = aggregation_methods.get(varname, "")


        if factor == 1:
                log.info("Factor is 1, directly saving to netcdf, without coarsening")
                ds_coarsened = ds_rxr
                ds_coarsened = ds_coarsened.chunk({"time": -1, "y": "auto", "x":  "auto"})
                if zero_to_nan:
                    ds_coarsened[varname] = ds_coarsened[varname].where(ds_coarsened[varname] != 0, np.nan)
                if save:
                    ds_coarsened.to_netcdf(filepath)
        else:
            # 0. init
            #da_rxr = ds_rxr[varname]
            da = ds_rxr[varname]

            # 1. coarsen
            if factor > 1: # downsampling
                factor_int = int(factor)
                is_int_factor = (factor == factor_int)

                log.info(f"1. Coarsen")
                if is_int_factor:
                    # When you coarsen with xarray, you're essentially downsampling by combining multiple cells into one, so the spatial resolution decreases (cells become larger) by the coarsening factor.
                    # Each new cell represents an area that was previously covered by factor² original cells.
                    # Factor 2: 0.5 × 1 = 1.0 minutes
                    # Factor 10: 0.5 × 10 = 5.0 minutes
                    # When using min_count in coarsen(), you must also set coord_func, otherwise xarray tries to apply min_count to coordinate means and crashes.
                    # log.info(f"\tRio data before coarsening: {ds_rxr.rio.crs}")

                    # coarsen.mean and coarsen.sum did not work as they resulted in nan values to become zero
                    if aggregation_method == "sum":
                        summed = da.coarsen(y=factor_int, x=factor_int, boundary="trim").reduce(np.nansum)
                        count = da.coarsen(y=factor_int, x=factor_int, boundary="trim").count()
                        da_coarse = summed.where(count > 0)
                    elif aggregation_method == "mean":
                        da_coarse = da.coarsen(y=factor_int, x=factor_int, boundary="trim").mean(skipna=True)
                    else:
                        log.info(f"Unknown aggregation_method: {aggregation_method}")
                        exit()
                    ds_coarsened = da_coarse.to_dataset(name=varname)
                    ds_coarsened = ds_coarsened.chunk({"time": -1, "y": "auto", "x": "auto"})

                    # 3. Update transform
                    log.info(f"3. Update transform")
                    # You need to update the affine transform after coarsening so the pixel size matches the new resolution.
                    old_transform = ds_rxr.rio.transform()
                    log.info("Old transform:")
                    info = print_transform(ds_rxr.rio.transform())
                    log.info(info)
                    new_transform = Affine(old_transform.a * factor_int, old_transform.b, old_transform.c,
                                        old_transform.d, old_transform.e * factor_int, old_transform.f)
                    ds_coarsened = ds_coarsened.rio.write_transform(new_transform)
                    log.info(f"Factor: {factor_int}")
                    log.info("New transform:")
                    info = print_transform(ds_coarsened.rio.transform())
                    log.info(info)
                else: # factor is float
                    # Use reproject() for non-integer factors (e.g., 1.2, 2.5, etc.)
                    log.warning(f"Factor {factor} is not an integer, using reproject instead of coarsen")
                    log.info(f"1. Coarsen by float factor {factor} (using reproject)")

                    # Calculate new resolution
                    if hasattr(ds_rxr.rio, 'resolution'):
                        current_res_x, current_res_y = ds_rxr.rio.resolution()
                    else:
                        current_res_x = abs(float(da.x[1] - da.x[0]))
                        current_res_y = abs(float(da.y[1] - da.y[0]))
                    new_res_x = abs(current_res_x) * factor
                    new_res_y = abs(current_res_y) * factor
                    log.info(f"Current resolution: x={current_res_x}, y={current_res_y}")
                    log.info(f"New resolution: x={new_res_x}, y={new_res_y}")

                    # Choose resampling method based on aggregation
                    if aggregation_method == "sum":
                        # For sum, use average to preserve totals (values per unit area)
                        resampling_method = Resampling.average
                        log.info("Using average resampling for coarsening (sum aggregation)")
                    elif aggregation_method == "mean":
                        resampling_method = Resampling.average
                        log.info("Using average resampling for coarsening (mean aggregation)")
                    else:
                        log.info(f"Unknown aggregation_method: {aggregation_method}, defaulting to average")
                        resampling_method = Resampling.average

                    crs = ds_rxr.rio.crs

                    # Ensure chunking for memory efficiency
                    if not hasattr(da.data, 'chunks'):
                        da = da.chunk({"time": -1, "y": 500, "x": 500})

                    # Reproject handles the transform automatically
                    da_coarsened = da.rio.reproject(
                        dst_crs=crs,
                        resolution=(new_res_x, new_res_y),
                        resampling=resampling_method
                    )

                    ds_coarsened = da_coarsened.to_dataset(name=varname)
                    ds_coarsened = ds_coarsened.chunk({"time": -1, "y": "auto", "x": "auto"})

                    # Transform is already set by reproject - just log it
                    log.info(f"3. Transform (automatically set by reproject)")
                    log.info("New transform:")
                    info = print_transform(ds_coarsened.rio.transform())
                    log.info(info)
            elif 0 < factor < 1: # upsampling
                log.info(f"Factor < 1: Upsampling by factor {factor}")
                log.info(f"1. Upsample by factor {factor}")
                if hasattr(ds_rxr.rio, 'resolution'):
                    current_res_x, current_res_y = ds_rxr.rio.resolution()
                else:
                    current_res_x = abs(float(da.x[1] - da.x[0]))
                    current_res_y = abs(float(da.y[1] - da.y[0]))
                new_res_x = current_res_x * factor
                new_res_y = current_res_y * factor
                log.info(f"Current resolution: x={current_res_x}, y={current_res_y}")
                log.info(f"New resolution: x={new_res_x}, y={new_res_y}")

                # 2. Reproject with interpolation
                if aggregation_method == "sum":
                    resampling_method = Resampling.bilinear
                    log.info("Using bilinear interpolation for upsampling (sum aggregation)")
                elif aggregation_method == "mean":
                    resampling_method = Resampling.bilinear
                    log.info("Using bilinear interpolation for upsampling (mean aggregation)")
                else:
                    log.info(f"Unknown aggregation_method: {aggregation_method}, defaulting to bilinear")
                    resampling_method = Resampling.bilinear

                # Reproject handles the transform automatically
                crs = ds_rxr.rio.crs
                da_upsampled = da.rio.reproject(dst_crs=crs,resolution=(new_res_x, new_res_y),resampling=resampling_method)
                ds_coarsened = da_upsampled.to_dataset(name=varname)
                ds_coarsened = ds_coarsened.chunk({"time": -1, "y": "auto", "x": "auto"})

                # 3. Transform is already set by reproject - just log it
                log.info(f"3. Transform (automatically set by reproject)")
                log.info("New transform:")
                info = print_transform(ds_coarsened.rio.transform())
                log.info(info)
            else:
                raise ValueError(f"Invalid factor: {factor}. Factor must be > 0.")

            # 4. Capture CRS & nodata
            log.info(f"4. Capture CRS & nodata")
            # When you coarsen, you also drop extra coordinates from y and x (since .coarsen() trims edges if they’re not divisible by the factor).
            # After that, it’s a good idea to carry over the CRS and nodata info so the file is fully georeferenced if you save it later.
            # write_crs() --> defines what CRS you are using
            # write_transform()	defines where pixels are
            # write_coordinate_system()	ensures CF metadata is complete

            if da.rio.crs is not None:
                ds_coarsened[varname] = (ds_coarsened[varname]
                            .rio.write_crs(da.rio.crs))
                ds_coarsened = ds_coarsened.rio.write_coordinate_system()
                log.info(f"\tRio data after CRS: {ds_coarsened[varname].rio.crs}")
                log.info(f"\tDataset CRS: {ds_coarsened.rio.crs}")
            else:
                log.warning("WARNING: Input data has no CRS! Setting to EPSG:4326")

            # Compare resolutions
            arc_seconds_start, arc_minutes_start, arc_degrees_start = calculate_resolution(ds_rxr[varname])
            arc_seconds_end, arc_minutes_end, arc_degrees_end = calculate_resolution(ds_coarsened[varname])
            log.info(f"Type: {type(ds_rxr)}")
            log.info(f"Type: {ds_rxr[varname].dtype}")
            log.info(f"Shape change: {ds_rxr[varname].shape} --> {ds_coarsened[varname].shape}")
            log.info(f"Resolution change: {arc_seconds_start:.1f}->{arc_seconds_end:.1f} arcsec, {arc_minutes_start:.2f}->{arc_minutes_end:.2f} arcmin, {arc_degrees_start:.2f}->{arc_degrees_end:.2f} arcdeg")
            log.info(f"Expected factor: {factor}, Actual factor: {arc_seconds_end/arc_seconds_start:.1f}")

            # 5. Set zero to nan
            if zero_to_nan:
                log.info(f"5. Set zero to nan")
                ds_coarsened[varname] = ds_coarsened[varname].where(ds_coarsened[varname] != 0, np.nan)

            # 6. save to disk
            if save:
                match save_type:
                    case "netcdf":
                        save_netcdf = True
                        save_zarr = False
                    case "zarr":
                        save_netcdf = False
                        save_zarr = True
                    case "both":
                        save_netcdf = True
                        save_zarr = True
                    case "none":
                        save_netcdf = False
                        save_zarr = False
                    case _:
                        log.info(f"Unknown save_type: {save_type}")
                        exit()

                # 6.1. zarr
                log.info(f"5.1 Save to zarr")
                log.info(f"\tChunks: {ds_coarsened.chunks}")
                if save_zarr:
                    chunks_zarr = chunks_size_save
                    ds_coarsened_zarr = ds_coarsened.chunk({"time": -1, "y": chunks_zarr, "x": chunks_zarr})
                    log.info(f"\tSaving to zarr: {filepath}.zarr")
                    zarr_file = filepath / ".zarr"
                    ds_coarsened_zarr.to_zarr(
                        zarr_file,
                        mode="w",
                        compute=True,                 # triggers compute
                        encoding={f"{varname}": {"dtype": "float32"}} ) # smaller + faster

                # 6.2. netcdf
                log.info(f"5.2 Save to netcdf")
                # NetCDF works better with smaller, time-contiguous chunks
                if save_netcdf:
                    log.info(f"\tSaving to netcdf: {filepath}")
                    #ds_coarsened_netcdf = ds_coarsened.chunk({"time": -1, "y": chunks_netcdf, "x": chunks_netcdf})
                    ds_coarsened_netcdf = ds_coarsened
                    if filepath.exists():
                        # delete file
                        filepath.unlink()

                    #--------------------------------
                    # ADD VERIFICATION HERE - Check what we're about to save
                    log.info(f"\t=== PRE-SAVE VERIFICATION ===")
                    log.info(f"\tCRS before save: {ds_coarsened_netcdf[varname].rio.crs}")
                    info = print_transform(ds_coarsened_netcdf.rio.transform())
                    log.info(info)
                    log.info(f"\tNoData before save: {ds_coarsened_netcdf[varname].rio.nodata}")
                    log.info(f"\tHas spatial_ref: {'spatial_ref' in ds_coarsened_netcdf.coords}")
                    if 'spatial_ref' in ds_coarsened_netcdf.coords:
                        log.info(f"\tspatial_ref attrs: {ds_coarsened_netcdf.spatial_ref.attrs.keys()}")
                    log.info(f"\tCurrent encoding before save:")
                    for key, value in ds_coarsened_netcdf[varname].encoding.items():
                        log.info(f"\t  - {key}: {value}")
                    #--------------------------------

                    # determine chunc size
                    # Get actual dimensions
                    actual_shape = ds_coarsened_netcdf[varname].shape
                    n_time = actual_shape[0] if len(actual_shape) > 2 else 1
                    n_y = actual_shape[-2]
                    n_x = actual_shape[-1]
                    # Calculate safe chunk sizes (don't exceed dimension size)
                    chunks_netcdf = chunks_size_save
                    chunk_time = -1
                    chunk_y = min(chunks_netcdf, n_y)
                    chunk_x = min(chunks_netcdf, n_x)

                    # Define encoding for NetCDF variable
                    #  xarray writes missing data via _FillValue when saving. In fact, by default xarray sets _FillValue = NaN for floats,
                    # but you should override this explicitly for a consistent numeric nodata.
                    nodata_value = ds_coarsened_netcdf.attrs.get("source_nodata", None)
                    if nodata_value is not None:
                        # Remove _FillValue from attrs to avoid conflict
                        if "_FillValue" in ds_coarsened_netcdf[varname].attrs:
                            del ds_coarsened_netcdf[varname].attrs["_FillValue"]
                        # Add to encoding as _FillValue
                        #encoding[varname]["_FillValue"] = nodata_value
                    encoding={
                        #"spatial_ref": {"dtype": "int32"},
                        f"{varname}": {
                            #"chunksizes": (1, chunks_netcdf, chunks_netcdf),  # NetCDF internal chunks to match
                            "dtype": "float32",
                            "zlib": True,
                            "complevel": 3,
                            "_FillValue": nodata_value,
                            "chunksizes": (n_time, chunk_y, chunk_x)
                            }
                        }
                    # Add spatial_ref to encoding
                    if hasattr(ds_coarsened_netcdf['spatial_ref'], 'encoding'):
                        ds_coarsened_netcdf['spatial_ref'].encoding.clear()
                    encoding["spatial_ref"] = {
                        "dtype": "int32",
                        "_FillValue": None}  # spatial_ref should not have fill value
                    if "_FillValue" in ds_coarsened_netcdf[varname].attrs:
                        del ds_coarsened_netcdf[varname].attrs["_FillValue"]
                        # Clear existing encoding that might conflict
                        for var in ds_coarsened_netcdf.data_vars:
                            if hasattr(ds_coarsened_netcdf[var], 'encoding'):
                                # Keep only essential encoding items
                                essential_keys = {'_FillValue', 'dtype'}
                                ds_coarsened_netcdf[var].encoding = {
                                    k: v for k, v in ds_coarsened_netcdf[var].encoding.items()
                                    if k in essential_keys}
                        for coord in ds_coarsened_netcdf.coords:
                            if hasattr(ds_coarsened_netcdf[coord], 'encoding'):
                                if coord == 'spatial_ref':
                                    ds_coarsened_netcdf[coord].encoding.clear()
                                else:
                                    # For other coords, keep minimal encoding
                                    essential_keys = {'dtype'}
                                    ds_coarsened_netcdf[coord].encoding = {
                                        k: v for k, v in ds_coarsened_netcdf[coord].encoding.items()
                                        if k in essential_keys}
                    # Save to netcdf
                    ds_coarsened_netcdf.to_netcdf(filepath, mode="w", encoding=encoding)
                    log.info(f"\tFile saved to: {filepath}")

                    #--------------------------------
                    # ADD VERIFICATION AFTER SAVE - Read it back immediately
                    log.info(f"\t=== POST-SAVE VERIFICATION ===")
                    log.info(f"\t=== saved data ===")
                    ds_verify = xr.open_dataset(filepath, decode_coords="all")
                    log.info(f"\tCRS after reload: {ds_verify.rio.crs}")
                    info = print_transform(ds_verify.rio.transform())
                    log.info(info)
                    log.info(f"\tNoData from rio.nodata: {ds_verify[varname].rio.nodata}") # netcdf does not store rio.nodata
                    log.info(f"\tNoData from attrs: {ds_verify.attrs.get("source_nodata", None)}")
                    log.info(f"\t_FillValue from encoding: {ds_verify[varname].encoding.get('_FillValue')}")
                    log.info(f"\tCoordinates in file: {list(ds_verify.coords.keys())}")
                    ds_verify.close()

                    # VERIFICATION of returned dataset
                    log.info(f"\t=== returned data ===")
                    log.info(f"\tCRS of returned dataset: {ds_coarsened[varname].rio.crs}")
                    info = print_transform(ds_coarsened.rio.transform())
                    log.info(info)
                    log.info(f"\tNoData of returned dataset from rio.nodata: {ds_coarsened[varname].rio.nodata}")
                    log.info(f"\tNoData from attrs: {ds_coarsened_netcdf.attrs.get("source_nodata", None)}")
                    log.info(f"\t_FillValue of returned dataset from encoding: {ds_coarsened[varname].encoding.get('_FillValue')}")
                    #--------------------------------

        return ds_coarsened

def calc_total_sum_rio_xarray(da:xr.DataArray, year: int, log:logging.Logger=local_log):
    # Calculate total sum and count for xarray DataArray
    da_count = da.sel(time=year)
    print(f"Calculating total sum for DataArray with shape: {da_count.shape} and dtype: {da_count.dtype}")
    total_sum = da_count.sum(skipna=True).compute()
    log.info(f"Total Sum of valid cells: {total_sum:,.2f}")
    print(f"Total Sum of valid cells: {total_sum:,.2f}")

def count_values_rio_xarray(ds:xr.Dataset, varname:str, year: int, log:logging.Logger=local_log):
    # Count statistics for xarray DataArray
    print(f"Counting values for DataArray with shape: {ds[varname].sel(time=year).shape} and dtype: {ds[varname].dtype}")

    stats_data = []
    data = ds[varname].sel(time=year).values
    num_cells = data.size

    #log.info(f"Total number of cells: {num_cells}")
    # Count total values
    stats_data.append(['Total', num_cells, f"{num_cells/num_cells*100:.2f}%"])
    # Count zero, negative, positive values
    zero_count = (data == 0).sum()
    stats_data.append(['Zero', zero_count, f"{zero_count/num_cells*100:.2f}%"])
    neg_count = (data < 0).sum()
    stats_data.append(['Negative', neg_count, f"{neg_count/num_cells*100:.2f}%"])
    pos_count = (data > 0).sum()
    stats_data.append(['Positive', pos_count, f"{pos_count/num_cells*100:.2f}%"])
    # Count NaN values
    nan_count = np.isnan(data).sum()
    stats_data.append(['NaN', nan_count, f"{nan_count/num_cells*100:.2f}%"])
    # Count unexpected nodata value
    nodata_value = ds.attrs.get("source_nodata", None)
    print(f"{PRINT_COLORS["yellow"]}Check for counting values, source_nodata value from attributes: {nodata_value}{PRINT_COLORS["end"]}")
    nodata_value_count = (data == nodata_value).sum()
    stats_data.append([f'Nodata', nodata_value_count, f"{nodata_value_count/num_cells*100:.2f}%"])

    # Create DataFrame
    df = pd.DataFrame(stats_data, columns=['Statistic', 'Count', 'Percentage'])
    df['Count'] = df['Count'].apply(lambda x: f"{int(x):,}" if isinstance(x, (int, float)) else x)
    log.info("Count Statistics:")
    log.info("\n" + df.to_string(index=False))

def calc_total_sum_count_average(src:DatasetReader, chunk_size=512) -> Tuple[float|None, int|None, float|None]:
    '''
    Calculate total sum, count of valid cells, and average value of a raster file using rasterio in chunks.'''
    total_sum = total_count = 0.0
    for row in range(0, src.height, chunk_size):
        for col in range(0, src.width, chunk_size):
            window = Window(col, row,
                            min(chunk_size, src.width - col),
                            min(chunk_size, src.height - row))

            chunk = src.read(1, window=window, masked=True)

            #.size = total array size (including masked/NoData elements)
            #.count() = number of non-masked elements only
            valid_data = chunk[chunk != 0]
            if valid_data.count() > 0:
                total_sum += float(valid_data.sum())
                total_count += valid_data.count()

    sum = total_sum
    count = int(total_count)
    average = total_sum / total_count if total_count > 0 else None

    return sum, count, average

def print_src_info_rasterio(src:DatasetReader, log: logging.Logger=local_log):
    # Print default tags
    log.info("=== Default tags ===")
    for key, value in src.tags().items():
        value_str = str(value)
        if '\n' in value_str:
            value_str = value_str.replace('\n', ' ')
        log.info(f"\t\t - {key}: {value_str}")

    # Print band-level tags
    for i in range(1, src.count + 1):
        band_tags = src.tags(i)
        if band_tags:
            log.info(f"=== Band {i} tags ===")
            for key, value in band_tags.items():
                value_str = str(value)
                if '\n' in value_str:
                    value_str = value_str.replace('\n', ' ')
                log.info(f"\t\t - {key}: {value_str}")

    # Print all metadata domain tags
    for domain in src.tag_namespaces():
        tags = src.tags(ns=domain)
        if tags:
            log.info(f"=== {domain} ===")
            for key, value in tags.items():
                value_str = str(value)
                if '\n' in value_str:
                    value_str = value_str.replace('\n', ' ')
                log.info(f"\t\t - {key}: {value_str}")

    # Print profile
    log.info("=== Profile ===")
    for key, value in src.profile.items():
        value_str = str(value)
        if '\n' in value_str:
            value_str = value_str.replace('\n', ' ')
        log.info(f"\t\t - {key}: {value_str}")

    # nodata check
    log.info("=== NoData Check ===")
    nodata = src.nodata  # Changed from src.rio.nodata
    if nodata is not None:
        log.info(f"NoData value: {nodata}")
        data = src.read(1)  # Read first band
        has_nodata = (data == nodata).any()
        log.info(f"Contains nodata cells ixels: {has_nodata}")
    else:
        log.info("No NoData value defined")
    # Check for NaN values
    data = src.read(1) if 'data' not in locals() else data
    has_nan = np.isnan(data).any()
    log.info(f"Contains NaN cells: {has_nan}")

def check_nan_locations(filepath:Path, varname: str, year: int, log: logging.Logger=local_log):

    """
    Sample 5 known ocean locations to inspect the nodata value.
    """
    ocean_points = [
        {"lon": -140.0, "lat":   0.0},  # Central Pacific
        {"lon":  -30.0, "lat":  15.0},  # Central Atlantic
        {"lon":   70.0, "lat": -20.0},  # Indian Ocean
        {"lon": -170.0, "lat": -30.0},  # South Pacific
        {"lon":   10.0, "lat": -40.0},  # South Atlantic
    ]

    #if filepath.endswith(".tif") or filepath.endswith(".tiff"):
    if filepath.suffix in {".tif", ".tiff"}:
        da = xr.open_dataarray(filepath, engine="rasterio")
    elif filepath.suffix == ".nc":
        if varname is None:
            raise ValueError("varname must be provided for .nc files.")
        ds = xr.open_dataset(filepath)
        da = ds[varname]
        if year is not None and "time" in da.dims:
            da = da.sel(time=str(year), method="nearest")
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Expected .tif, .tiff, or .nc.")

    log.info(f"Dimensions: {da.dims}")
    log.info(f"Coordinates: {list(da.coords)}")

    lon_dim = "x" if "x" in da.dims else "longitude" if "longitude" in da.dims else "lon"
    lat_dim = "y" if "y" in da.dims else "latitude" if "latitude" in da.dims else "lat"

    for point in ocean_points:
        value = da.sel(
            {lon_dim: point["lon"], lat_dim: point["lat"]},
            method="nearest"
        )
        log.info(f"lon={point['lon']}, lat={point['lat']} --> {value.values}")

def check_values_xr_dataarray(da:xr.DataArray, year=None, inlc_inf=False, log: logging.Logger=local_log):

    # check for year/time coordinate
    if year is None or 'time' not in da.coords:
        log.info(f"DataArray info:")
    else:
        log.info(f"DataArray info for year {year}:")
        da_year = da.sel(time=year)
    # print values
    log.info(f"Data shape: {da.shape}")
    # Get minimum value that's not -inf
    log.info(f"Data min: {da.min().values:,.0f}")
    log.info(f"Data max: {da.max().values:,.0f}")

    min_no_inf = np.min(da.values[~np.isinf(da.values) & ~np.isnan(da.values)]) if (~np.isinf(da.values)).any() else np.nan
    log.info(f"Data min (no -inf): {min_no_inf:,.0f}")
    max_no_inf = np.max(da.values[~np.isinf(da.values) & ~np.isnan(da.values)]) if (~np.isinf(da.values)).any() else np.nan
    log.info(f"Data max (no inf): {max_no_inf:,.0f}")

    log.info(f"Number of NaN values: {np.isnan(da.values).sum():,.0f}")
    log.info(f"Number of zero values: {(da.values == 0).sum():,.0f}")
    log.info(f"Number of negative values: {(da.values < 0).sum():,.0f}")
    log.info(f"Number of positive values: {(da.values > 0).sum():,.0f}")
    if inlc_inf:
        log.info(f"Number of -inf values: {(da.values == -np.inf).sum():,.0f}")
        log.info(f"Number of +inf values: {(da.values == np.inf).sum():,.0f}")

def check_values_tiff(filepath:Path, band=1, inlc_inf=False, log: logging.Logger=local_log):
    """
    Check statistical values for a TIFF file using GDAL.

    Parameters:
    -----------
    tiff_path : str
        Path to the TIFF file
    band : int
        Band number to analyze (1-indexed, default=1)
    nodata : float, optional
        NoData value to track separately
    inlc_inf : bool
        Whether to include infinity counts
    """

    log.info(f"Checking values for TIFF file: {filepath}, band: {band}, include_inf: {inlc_inf}")

    # with rasterio.open(file) as src:
    #     if src is None:
    #         log.info(f"Error: Could not open file {file}")
    #         return None

    # 1. read in with rasterio
    src = rasterio.open(filepath)
    if src is None:
        log.info(f"Error: Could not open file {filepath}")
        return None

    print_src_info_rasterio(src, log)

    unit = "<undefined unit>"
    if src.crs:
        nodata = print_info_rasterio(src, band, log)

        # 2. read in with rioxarray
        da = rxr.open_rasterio(filepath)
        da = da.squeeze(drop=True)
        log.info("\n***************************rioxarray info***************************")
        print_info_rioxarray(da, log)
        log.info("\n***************************Resolution calculation***************************")
        arc_seconds, arc_minutes, arc_degrees = calculate_resolution(da)
        log.info(f"Resolution: {arc_seconds:.1f} arcsec, {arc_minutes:.2f} arcmin, {arc_degrees:.5f} arcdeg")

        # Check encoding rioxarray
        if hasattr(da, 'encoding'):
            log.info(f"\nencoding:")
            encoding_dict = da.encoding
            for key, value in encoding_dict.items():
                if '\n' in str(value):
                    value = str(value).replace('\n', ' ')
                log.info(f"\t\t - {key}: {value}")
        else:
            log.info(f"\n\tNo encoding information found.")

        # 3. Statistics based on rasterio
        log.info("\n=== Statistics ===")

        # Get dimensions
        cols = src.width
        rows = src.height
        num_cells = rows * cols

        log.info(f"Number of grid cells: {num_cells:,.0f}")
        log.info(f"Data shape: ({rows}, {cols})")

        # Read the data as array (band is 1-indexed)
        data = src.read(band)

        # Basic statistics
        min_value = np.nanmin(data)
        max_value = np.nanmax(data)

        log.info(f"Data min: {min_value:,.0f}")
        log.info(f"Data max: {max_value:,.0f}")

        # Handle infinities
        if min_value == -np.inf or max_value == np.inf:
            finite_mask = np.isfinite(data)
            if min_value == -np.inf:
                min_no_inf = np.min(data[finite_mask])
                log.info(f"Data min (no -inf): {min_no_inf:,.0f}")
            if max_value == np.inf:
                max_no_inf = np.max(data[finite_mask])
                log.info(f"Data max (no inf): {max_no_inf:,.0f}")

        # Count statistics
        stats_data = []
        if nodata is not None:
            nodata_count = (data == nodata).sum()
            stats_data.append(['NoData values', nodata_count, f"{nodata_count/num_cells*100:.2f}%"])

        nan_count = np.isnan(data).sum()
        stats_data.append(['NaN values', nan_count, f"{nan_count/num_cells*100:.2f}%"])
        zero_count = (data == 0).sum()
        stats_data.append(['Zero values', zero_count, f"{zero_count/num_cells*100:.2f}%"])
        neg_count = (data < 0).sum()
        stats_data.append(['Negative values', neg_count, f"{neg_count/num_cells*100:.2f}%"])
        pos_count = (data > 0).sum()
        stats_data.append(['Positive values', pos_count, f"{pos_count/num_cells*100:.2f}%"])

        # Excluding nodata
        if nodata is not None and not np.isnan(nodata):
            zero_count_excl_nodata = ((data == 0) & (data != nodata)).sum()
            stats_data.append(['Zero values (excl. nodata)', zero_count_excl_nodata, f"{zero_count_excl_nodata/num_cells*100:.2f}%"])
            neg_count_excl_nodata = ((data < 0) & (data != nodata)).sum()
            stats_data.append(['Negative values (excl. nodata)', neg_count_excl_nodata, f"{neg_count_excl_nodata/num_cells*100:.2f}%"])
            pos_count_excl_nodata = ((data > 0) & (data != nodata)).sum()
            stats_data.append(['Positive values (excl. nodata)', pos_count_excl_nodata, f"{pos_count_excl_nodata/num_cells*100:.2f}%"])
        else:
            stats_data.append(['Zero values (excl. nodata)', 'N/A', 'N/A'])
            stats_data.append(['Negative values (excl. nodata)', 'N/A', 'N/A'])
            stats_data.append(['Positive values (excl. nodata)', 'N/A', 'N/A'])

        # Infinity values
        if inlc_inf:
            neg_inf_count = (data == -np.inf).sum()
            stats_data.append(['-inf values', neg_inf_count, f"{neg_inf_count/num_cells*100:.2f}%"])

            pos_inf_count = (data == np.inf).sum()
            stats_data.append(['+inf values', pos_inf_count, f"{pos_inf_count/num_cells*100:.2f}%"])

        # Create DataFrame
        df = pd.DataFrame(stats_data, columns=['Statistic', 'Count', 'Percentage'])
        df['Count'] = df['Count'].apply(lambda x: f"{int(x):,}" if isinstance(x, (int, float)) else x)
        log.info("Count Statistics:")
        log.info("\n" + df.to_string(index=False))

        # Add summary statistics
        sum, cnt, avg = calc_total_sum_count_average(src)
        log.info(f"Count of number of cells (not nodata/zero): {cnt}")
        log.info(f"Sum of global values: {sum}")
        log.info(f"Average per cell (excluding nodata and zero): {avg}")

def get_coarsening_factors(population_source: str, gdp_source: str, emissions_source: str) -> Tuple[int|float, int|float, int|float, float, float, float]:
    """
    Determines the coarsening factor for each dataset relative to the lowest
    resolution in minutes (largest cell size) among the three selected datasets.

    Parameters
    - population_source : str
        Source of the population dataset (e.g. '2UP').
    - gdp_source : str
        Source of the GDP dataset (e.g. 'Wang' or 'Murakami').
    - emissions_source : str
        Source of the emissions dataset (e.g. 'EDGAR' or 'CMIP').
    Returns: Tuple[int|float, int|float, int|float]
        A tuple of coarsening factors for population, GDP, and emissions.
        The coarsening factor is an integer if it divides evenly,
        otherwise a float.
    """

    dataset_lookup = {(entry["source"], entry["variable"]): entry for entry in DATASETS}
    variable_options = {
        "Population": [source for source, variable in dataset_lookup.keys() if variable == "Population"],
        "GDP":        [source for source, variable in dataset_lookup.keys() if variable == "GDP"],
        "Emissions":  [source for source, variable in dataset_lookup.keys() if variable == "Emissions"],
    }

    for dataset_source, expected_variable in [(population_source, "Population"), (gdp_source, "GDP"), (emissions_source, "Emissions")]:
        if (dataset_source, expected_variable) not in dataset_lookup:
            raise ValueError(
                f"Dataset '{dataset_source}' not found for variable '{expected_variable}' in settings_resolution. "
                f"Available {expected_variable} datasets: {variable_options[expected_variable]}."
            )

    resolutions = {
        population_source: dataset_lookup[(population_source, "Population")]["resolution"]["degrees"],
        gdp_source:        dataset_lookup[(gdp_source, "GDP")]["resolution"]["degrees"],
        emissions_source:  dataset_lookup[(emissions_source, "Emissions")]["resolution"]["degrees"],
    }

    lowest_resolution = max(resolutions.values())

    def to_factor(res):
        factor = lowest_resolution / res
        return int(factor) if factor == int(factor) else round(factor, 10)

    coarse_factor_POP = to_factor(resolutions[population_source])
    coarse_factor_GDP = to_factor(resolutions[gdp_source])
    coarse_factor_EM  = to_factor(resolutions[emissions_source])

    res_min_POP = float(dataset_lookup[(population_source, "Population")]["resolution"]["minutes"])
    res_min_GDP = float(dataset_lookup[(gdp_source, "GDP")]["resolution"]["minutes"])
    res_min_EM = float(dataset_lookup[(emissions_source, "Emissions")]["resolution"]["minutes"])

    return coarse_factor_POP, coarse_factor_GDP, coarse_factor_EM, res_min_POP, res_min_GDP, res_min_EM

def get_parameters_SE(process_data:bool=False, varname="Population", source:str="2UP", version="GHSL_2024_M3", SSP_base="SSP2", log: logging.Logger=local_log) -> Tuple[Path, str, str, str, str, str, float]:
    # init
    end_filename = ""
    glob_pattern = ""
    search_pattern = ""
    test_filename = ""
    mult_factor = 1.0
    data_dir = Path(".")  # default to current directory if not set
    rxr_filename = ""

    # Read the JSON file
    with open("downscaling/settings_data_locations.json", "r") as f:
        data_files = json.load(f)
    data_original = data_files["grid"]["original"]
    data_run = data_files["grid"]["run"]

    match varname:
        case "Population":
            match source:
                case "2UP":
                    # 2UP population data
                    # https://www.pbl.nl/sites/default/files/downloads/pbl-2018-Towards-an-urban-preview_3255.pdf
                    # 1 init settings population data
                    if process_data:
                        DIR_MAPPING_POP_2UP = {
                            "GHSL_2024_M1": data_original["dir_population_2UP_GHSL_M1_original"],
                            "GHSL_2024_M3": data_original["dir_population_2UP_GHSL_M3_original"],
                            "M1": data_original["dir_population_2UP_M1_original"],
                            "M3": data_original["dir_population_2UP_M3_original"],
                            }
                    else:
                        DIR_MAPPING_POP_2UP = {
                            "GHSL_2024_M1": f"{data_run["dir_population_2UP_GHSL_M1_run"]}/{SSP_base}",
                            "GHSL_2024_M3": f"{data_run["dir_population_2UP_GHSL_M3_run"]}/{SSP_base}",
                            "M1": f"{data_run["dir_population_2UP_M1_run"]}/{SSP_base}",
                            "M3": f"{data_run["dir_population_2UP_M3_run"]}/{SSP_base}",
                            }
                    TXT_MAPPING_2UP = {
                        "GHSL_2024_M1": "var1",
                        "GHSL_2024_M3": "var1",
                        "M1": "",
                        "M3": "var1",
                    }
                    if version in DIR_MAPPING_POP_2UP:
                        #data_dir = DIR_MAPPING_POP_2UP[version]
                        data_dir = Path(DIR_MAPPING_POP_2UP[version])
                        model = version.split('_')[-1]  # Gets "M1" or "M3"
                        txt = TXT_MAPPING_2UP[version]
                        test_filename = f"{model}{txt}_{SSP_base}_2020_tpop.tif"
                        rxr_filename = f"population_2UP_{version}_{SSP_base}"
                        end_filename="tpop"
                        glob_pattern=rf"M?{txt}_{SSP_base}_????_tpop.tif"
                        search_pattern = rf"M?{txt}_{SSP_base}_(\d{{4}})_tpop\.tif"
                        mult_factor = 1
                    else:
                        raise ValueError(f"Unknown version: {version} for getting parameters {source} population")
                case "Wang":
                    # Wang et al. (2022) population data
                    # https://figshare.com/articles/dataset/Projecting_1_km-grid_population_distributions_from_2020_to_2100_globally_under_shared_socioeconomic_pathways/19608594/3
                    if process_data:
                        DIR_MAPPING_POP_Wang = {
                            #"version_1": dir_population_Wang_v1,
                            "version_2": data_original["dir_population_Wang_v2_original"],
                            "version_3": data_original["dir_population_Wang_v3_original"],
                        }
                    else:
                        DIR_MAPPING_POP_Wang = {
                            #"version_1": dir_population_Wang_v1,
                            "version_2": data_run["dir_population_Wang_v2_run"],
                            "version_3": data_run["dir_population_Wang_v3_run"],
                        }
                    if version in DIR_MAPPING_POP_Wang:
                        #data_dir = f"{DIR_MAPPING_POP_Wang[version]}/{SSP_base}"
                        data_dir = Path(DIR_MAPPING_POP_Wang[version]) / SSP_base
                        test_filename = f"{SSP_base}_2020.tif"
                        rxr_filename = f"population_Wang_{version}"
                        end_filename = ""
                        glob_pattern = f"{SSP_base}_????.tif"
                        search_pattern = f"{SSP_base}_(\\d{{4}}).tif"
                        mult_factor = 1
                    else:
                        raise ValueError(f"Unknown version: {version} for getting parameters {source} population")
                case "Zhuang":
                    if process_data:
                        DIR_MAPPING_POP_Zhuang = {
                            "version_1": data_original["dir_population_Zhuang_v1_original"],
                        }
                    else:
                        DIR_MAPPING_POP_Zhuang = {
                            "version_1": data_run["dir_population_Zhuang_v1_run"],
                        }
                    if version in DIR_MAPPING_POP_Zhuang:
                        #data_dir = f"{DIR_MAPPING_POP_Zhuang[version]}/{SSP_base}"
                        data_dir = Path(DIR_MAPPING_POP_Zhuang[version]) / SSP_base
                        test_filename = f"2020.tif"
                        rxr_filename = f"population_Zhuang_{version}"
                        end_filename = ""
                        glob_pattern = f"????.tif"
                        search_pattern = f"(\\d{{4}}).tif"
                        mult_factor = 1
                    else:
                        raise ValueError(f"Unknown version: {version} for getting parameters {source} population")
                case "Murakami":
                    if process_data:
                        DIR_MAPPING_POP_Murakami = {
                            "version_2021_1": data_original["dir_population_Murakami_v2021_1_original"],
                        }
                    else:
                        DIR_MAPPING_POP_Murakami = {
                            "version_2021_1": data_run["dir_population_Murakami_v2021_1_run"],
                        }
                    if version in DIR_MAPPING_POP_Murakami:
                        #data_dir = f"{DIR_MAPPING_POP_Murakami[version]}/{SSP_base}"
                        data_dir = Path(DIR_MAPPING_POP_Murakami[version]) / SSP_base
                        match SSP_base:
                            case "SSP1": SSP_str = "p1"
                            case "SSP2": SSP_str = "p2"
                            case "SSP3": SSP_str = "p3"
                            case _:
                                raise ValueError(f"Unknown SSP_base: {SSP_base} for Murakami population data")
                        test_filename = f"{SSP_str}_2020.tif"
                        rxr_filename = f"population_Murakami_{version}_{SSP_base}"
                        end_filename = ""
                        glob_pattern = f"{SSP_str}_????.tif"
                        search_pattern = f"{SSP_str}_(\\d{{4}}).tif"
                        mult_factor = 1e6
                case "COMPASS":
                    if process_data:
                        DIR_MAPPING_POP_COMPASS = {
                            "version_2": data_original["dir_population_COMPASS_v2_original"],
                        }
                    else:
                        DIR_MAPPING_POP_COMPASS = {
                            "version_2": f"{data_run["dir_population_COMPASS_v2_run"]}/{SSP_base}",
                        }
                    if version in DIR_MAPPING_POP_COMPASS:
                        data_dir = Path(DIR_MAPPING_POP_COMPASS[version])
                        if process_data:
                            test_filename = f"Population_count_2020_6min.tif"
                        else:
                            test_filename = f"Population_count_2020_6min_SSP2_not_harm.tif"
                        rxr_filename = f"population_COMPASS_{version}_{SSP_base}"
                        end_filename = ""
                        glob_pattern = f"Population_count_????_6min_{SSP_base}_not_harm.tif"
                        search_pattern = f"Population_count_(\\d{{4}})_6min_{SSP_base}_not_harm.tif"
                case _:
                    log.info(f"Error: Unknown source {source}, exiting")
                    exit()
        case "GDP|PPP":
                match source:
                    case "Wang":
                        # Wang et al. (2022) GDP data
                        # https://zenodo.org/records/7898409
                        if process_data:
                            DIR_MAPPING = {
                                "version_7": data_original["dir_gdp_ppp_Wang_v7_original"],
                            }
                        else:
                            DIR_MAPPING = {
                                "version_7": data_run["dir_gdp_ppp_Wang_v7_run"],
                            }
                        if version in DIR_MAPPING:
                            #data_dir = f"{DIR_MAPPING[version]}/{SSP_base}"
                            data_dir = Path(DIR_MAPPING[version]) / SSP_base
                            test_filename = f"GDP2020_{SSP_base}.tif"
                            rxr_filename = f"gdp_ppp_Wang_{version}_{SSP_base}"
                            end_filename = ""
                            glob_pattern = f"GDP????_{SSP_base}.tif"
                            search_pattern = f"GDP(\\d{{4}})_{SSP_base.lower()}.tif"
                            mult_factor = 1
                        else:
                            raise ValueError(f"Unknown version: {version} for getting parameters {source} GDP (PPP)")
                    case "Murakami":
                        if process_data:
                            DIR_MAPPING_POP_Murakami = {
                            "version_2021_1": data_original["dir_gdp_ppp_Murakami_v2021_1_original"],
                            }
                        else:
                            DIR_MAPPING_POP_Murakami = {
                                "version_2021_1": data_run["dir_gdp_ppp_Murakami_v2021_1_run"],
                            }
                        if version in DIR_MAPPING_POP_Murakami:
                            #data_dir = f"{DIR_MAPPING_POP_Murakami[version]}/{SSP_base}"
                            data_dir = Path(DIR_MAPPING_POP_Murakami[version]) / SSP_base
                            test_filename = f"gdp2020.tif"
                            rxr_filename = f"gdp_ppp_Murakami_{version}_{SSP_base}"
                            end_filename = ""
                            glob_pattern = f"gdp????.tif"
                            search_pattern = f"gdp(\\d{{4}}).tif"
                            mult_factor = 1.0 #1e9
                    case "COMPASS":
                        if process_data:
                            DIR_MAPPING_POP_COMPASS = {
                                "version_2": data_original["dir_gdp_ppp_COMPASS_v2_original"],
                            }
                        else:
                            DIR_MAPPING_POP_COMPASS = {
                                "version_2": f"{data_run["dir_gdp_ppp_COMPASS_v2_run"]}/{SSP_base}",
                            }
                        if version in DIR_MAPPING_POP_COMPASS:
                            data_dir = Path(DIR_MAPPING_POP_COMPASS[version])
                            if process_data:
                                test_filename = f"GDP_2020_6min.tif"
                            else:
                                test_filename = f"GDP_2020_6min_SSP2_not_harm.tif"
                            rxr_filename = f"GDP_COMPASS_{version}_{SSP_base}"
                            end_filename = ""
                            glob_pattern = f"GDP_????_6min_{SSP_base}_not_harm.tif"
                            search_pattern = f"GDP_(\\d{{4}})_6min_{SSP_base}_not_harm.tif"
                    case _:
                        log.info(f"Error: Unknown source {source}, exiting")
                        exit()
        case _:
            log.info(f"Error: Unknown variable name {varname}, exiting")
            rxr_filename = None
            exit()

    return data_dir, rxr_filename, end_filename, glob_pattern, search_pattern, test_filename, mult_factor

def update_GIS_parameters(varname: str, source: str, version: str, SSP_base, data_dir:Path, glob_pattern:str, band:int=1, log: logging.Logger=local_log) -> Path:
    """
    Add CRS and geotransform information to a TIFF file based on WGS84 LatLong specifications.

    Parameters:
    -----------
    input_tiff_path : str
        Path to the input TIFF file without CRS
    output_tiff_path : str, optional
        Path for the output TIFF file with CRS. If None, overwrites the input file.
    """
    output_tiff_dir = Path(".")  # default to current directory if not set
    with open("downscaling/settings_data_locations.json", "r") as f:
        data_files = json.load(f)
    data_processed = data_files["grid"]["processed"]

    match varname:
        case "Population":
            match source:
                case "2UP":
                    output_tiff_dir = Path(data_processed["dir_population_processed"]) / "2UP" / f"processed_{version}" / SSP_base
                    output_tiff_dir.mkdir(parents=True, exist_ok=True)
                    files = sorted(data_dir.glob(glob_pattern))
                    log.info(f"Found {len(files)} files matching {glob_pattern}")
                    log.info(f"Updating GIS meta data for files in {data_dir} to {output_tiff_dir}")

                    for i, f in enumerate(tqdm.tqdm(files, desc="Updating GIS meta data")):
                        input_tiff_path = data_dir / f.name
                        output_tiff_path = output_tiff_dir / f.name

                        # Update profile
                        with rasterio.open(input_tiff_path, "r") as src:
                            data = src.read(band)
                            profile = src.profile.copy()

                        # Set/update profile parameters
                        crs = CRS.from_epsg(4326)
                        cell_size = (0.5/60)
                        x_min = -180.0
                        y_max = 90.0
                        transform = Affine(
                            cell_size, 0.0, x_min,
                            0.0, -cell_size, y_max
                        )

                        profile.update({
                            "crs": crs,
                            "transform": transform,
                            "nodata": -9999
                        })

                        # Write output with updated profile
                        Path(output_tiff_path).unlink(missing_ok=True)
                        with rasterio.open(output_tiff_path, "w", **profile) as dst:
                            dst.write(data, band)
                case "Wang":
                    # see https://zenodo.org/records/7898409
                    output_tiff_dir = Path(data_processed["dir_population_processed"]) / "Wang" / f"processed_{version}" / SSP_base
                    output_tiff_dir.mkdir(parents=True, exist_ok=True)
                    files = sorted(data_dir.glob(glob_pattern))
                    log.info(f"Found {len(files)} files matching {glob_pattern}")

                    for i, f in enumerate(tqdm.tqdm(files, desc="Updating CRS")):
                        input_tiff_path = data_dir / f.name
                        output_tiff_path = output_tiff_dir / f.name

                        # Update metadata in the new file
                        if input_tiff_path != output_tiff_path:
                            shutil.copy(input_tiff_path, output_tiff_path)
                            with rasterio.open(output_tiff_path, "r") as src:
                                data = src.read(band)
                                profile = src.profile.copy()
                            # Original bounds: BoundingBox(left=-0.4916666667, bottom=-18719.5, right=43199.5083333333, top=0.5)
                            transform = from_bounds(-180.0, -90.0, 180.0, 90.0, src.width, src.height)
                            profile.update({
                                "crs": CRS.from_epsg(4326),
                                "transform": transform
                            })
                            Path(output_tiff_path).unlink(missing_ok=True)
                            with rasterio.open(output_tiff_path, "w", **profile) as dst:
                                dst.write(data, band)
                        else:
                            log.info(f"Input and output paths are the same: {input_tiff_path}, skipping copy and update.")
                case "Zhuang":
                    output_tiff_dir = Path(data_processed["dir_population_processed"]) / "Zhuang" / f"processed_{version}" / SSP_base
                    output_tiff_dir.mkdir(parents=True, exist_ok=True)
                    files = sorted(data_dir.glob(glob_pattern))
                    log.info(f"Found {len(files)} files matching {glob_pattern}")

                    for i, f in enumerate(tqdm.tqdm(files, desc="Updating CRS from EPSG:6933 to EPSG:4326")):
                        input_tiff_path = data_dir / f.name
                        output_tiff_path = output_tiff_dir / f.name
                        with rasterio.open(input_tiff_path) as src:
                            # 1: Convert density → population in EPSG:6933
                            pixel_width = abs(src.transform.a)
                            pixel_height = abs(src.transform.e)
                            pixel_area_km2 = (pixel_width * pixel_height) / 1e6
                            data = src.read(1, masked=True)

                            # density to population
                            data = data * pixel_area_km2
                            nodata = src.nodata
                            fill_value = float(nodata) if nodata is not None else np.nan

                            # 2: Reproject to EPSG:4326
                            dst_crs = "EPSG:4326"

                            # 3. Create full grid --> cover full lat range
                            # EPSG:6933 = World Cylindrical Equal Area
                            # Key properties:
                            # - projected (meters)
                            # - equal-area
                            # - not defined for the poles
                            # - valid latitude range is limited
                            # - This projection cannot represent ±90° latitude, therefore results is y in [83:-55]
                            # Extending raster (y-coordiate) to [90,-90] by writing the reprojected image into a larger global raster using a window and leaving everything else as nodata
                            res_minutes = 0.5
                            target_res_degree = res_minutes/60 #0.008333333333333333  # degrees = 30 arc-seconds
                            xmin, xmax = -180.0, 180.0
                            ymin, ymax = -90.0, 90.0
                            global_width = int((xmax - xmin) / target_res_degree)
                            global_height = int((ymax - ymin) / target_res_degree)
                            global_transform = from_origin(xmin, ymax, target_res_degree, target_res_degree)

                            reprojected_population = np.full((global_height, global_width), fill_value, dtype=np.float32)
                            reproject(
                                source=data.filled(nodata),
                                destination=reprojected_population,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                src_nodata=nodata,
                                dst_transform=global_transform,
                                dst_crs=dst_crs,
                                dst_nodata=nodata,
                                resampling=Resampling.sum
                            )

                            # 5. Write into full global raster
                            kwargs = src.meta.copy()
                            kwargs.update({
                                "crs": dst_crs,
                                "transform": global_transform,
                                "width": global_width,
                                "height": global_height,
                                "nodata": nodata,
                                "dtype": "float32",
                                "compress": "deflate",
                                "predictor": 2,
                                "zlevel": 4,
                            })
                            with rasterio.open(output_tiff_path, "w", **kwargs) as dst:
                                dst.write(reprojected_population, 1)
                case "Murakami":
                    output_tiff_dir = Path(data_processed["dir_population_processed"]) / "Murakami" / f"processed_{version}" / SSP_base
                    output_tiff_dir.mkdir(parents=True, exist_ok=True)
                    files = sorted(data_dir.glob(glob_pattern))
                    log.info(f"Found {len(files)} files matching {glob_pattern}")

                    for i, f in enumerate(tqdm.tqdm(files, desc="Copying files...")):
                        input_tiff_path = data_dir / f.name
                        output_tiff_path = output_tiff_dir / f.name

                        # Update metadata in the new file
                        if input_tiff_path != output_tiff_path:
                            shutil.copy(input_tiff_path, output_tiff_path)
                        else:
                            log.info(f"Input and output paths are the same: {input_tiff_path}, skipping copy and update.")
                case "COMPASS":
                    output_tiff_dir = Path(data_processed["dir_population_processed"]) / "COMPASS" / f"processed_{version}" / SSP_base
                    output_tiff_dir.mkdir(parents=True, exist_ok=True)
                    files = sorted(data_dir.glob(glob_pattern))
                    log.info(f"Found {len(files)} files matching {glob_pattern}")

                    for i, f in enumerate(tqdm.tqdm(files, desc="Copying files...")):
                        input_tiff_path = data_dir / f.name
                        output_tiff_path = output_tiff_dir / f.name

                        # Update metadata in the new file
                        if input_tiff_path != output_tiff_path:
                            shutil.copy(input_tiff_path, output_tiff_path)

                    # copy historic file to similar format as the future files (with SSP2_not_harm in the name)
                    file_path_src = data_dir
                    file_path_dest = output_tiff_dir
                    for src in file_path_src.glob("*.tif"):
                        if "ssp" not in src.name:
                            shutil.copy2(src, file_path_dest / src.name)

                    # rename the historic file to have similar format as the future files (with SSP2_not_harm in the name)
                    for src in file_path_dest.glob("*.tif"):
                        if "ssp" not in src.name:
                            year = src.stem.split("_")[2]  # thrid element
                            new_name = f"Population_count_{year}_6min_SSP2_not_harm.tif"
                            src.replace(file_path_dest / new_name)
                case _:
                    log.info(f"Error: No update parameters are given for updating source {source}, exiting")
        case "GDP|PPP":
            match source:
                case "Wang":
                    #output_tiff_dir = f"{data_processed["dir_gdp_ppp_processed"]}/Wang/processed_{version}/{SSP_base}"
                    output_tiff_dir = Path(data_processed["dir_gdp_ppp_processed"]) / "Wang" / f"processed_{version}" / SSP_base
                    output_tiff_dir.mkdir(parents=True, exist_ok=True)
                    files = sorted(data_dir.glob(glob_pattern))
                    log.info(f"Found {len(files)} files matching {glob_pattern}")

                    for i, f in enumerate(tqdm.tqdm(files, desc=f"Updating CRS for {varname}-{source}-{version}")):
                        input_tiff_path = data_dir / f.name
                        output_tiff_path = output_tiff_dir / f.name
                        # Update metadata in the new file
                        if input_tiff_path != output_tiff_path:
                            shutil.copy(input_tiff_path, output_tiff_path)
                            with rasterio.open(output_tiff_path, "r+") as src:
                                data = src.read(band)
                                profile = src.profile.copy()
                            profile.update({"nodata": 0})
                            output_tiff_path.unlink(missing_ok=True)
                            with rasterio.open(output_tiff_path, "w", **profile) as dst:
                                dst.write(data, band)
                case "Murakami":
                    #output_tiff_dir = f"{data_processed["dir_gdp_ppp_processed"]}/Murakami/processed_{version}/{SSP_base}"
                    output_tiff_dir = Path(data_processed["dir_gdp_ppp_processed"]) / "Murakami" / f"processed_{version}" / SSP_base
                    if not output_tiff_dir.exists():
                        output_tiff_dir.mkdir(parents=True, exist_ok=True)
                    files = sorted(data_dir.glob(glob_pattern))
                    log.info(f"Found {len(files)} files matching {glob_pattern}")

                    for i, f in enumerate(tqdm.tqdm(files, desc="Copying files ...")):
                        input_tiff_path = data_dir / f.name
                        output_tiff_path = output_tiff_dir / f.name

                        # Update metadata in the new file
                        if input_tiff_path != output_tiff_path:
                            shutil.copy(input_tiff_path, output_tiff_path)
                            with rasterio.open(output_tiff_path, "r") as src:
                                data = src.read(band)
                                profile = src.profile.copy()
                                profile["crs"] = CRS.from_epsg(4326)  # Add here
                            Path(output_tiff_path).unlink(missing_ok=True)
                            with rasterio.open(output_tiff_path, "w", **profile) as dst:
                                dst.write(data, band)
                        else:
                            log.info(f"Input and output paths are the same: {input_tiff_path}, skipping copy and update.")
                case "COMPASS":
                    output_tiff_dir = Path(data_processed["dir_gdp_ppp_processed"]) / "COMPASS" / f"processed_{version}" / SSP_base
                    output_tiff_dir.mkdir(parents=True, exist_ok=True)
                    files = sorted(data_dir.glob(glob_pattern))
                    log.info(f"Found {len(files)} files matching {glob_pattern}")

                    for i, f in enumerate(tqdm.tqdm(files, desc="Copying files...")):
                        input_tiff_path = data_dir / f.name
                        output_tiff_path = output_tiff_dir / f.name

                        # Update metadata in the new file
                        if input_tiff_path != output_tiff_path:
                            shutil.copy(input_tiff_path, output_tiff_path)

                    # copy historic file to similar format as the future files (with SSP2_not_harm in the name)
                    file_path_src = data_dir
                    file_path_dest = output_tiff_dir
                    for src in file_path_src.glob("*.tif"):
                        if "ssp" not in src.name:
                            shutil.copy2(src, file_path_dest / src.name)
                    log.info(f"Error: No update parameters are given for updating source {source}, exiting")

                    # rename the historic file to have similar format as the future files (with SSP2_not_harm in the name)
                    for src in file_path_dest.glob("*.tif"):
                        if "ssp" not in src.name:
                            year = src.stem.split("_")[1]  # second element
                            new_name = f"GDP_{year}_6min_SSP2_not_harm.tif"
                            src.replace(file_path_dest / new_name)
        case _:
            log.info(f"Error: No update parameters are given for updating variable {varname}, exiting")
            output_tiff_dir = Path(".")

    return output_tiff_dir

def get_seconds_per_month(ds):
    """
    Returns seconds per month, accounting for leap years.
    Checks the calendar type first: cftime.DatetimeNoLeap always uses 28 days
    for February regardless of year, while standard/gregorian calendars respect
    actual leap years.
    """
    first_time = ds["time"].values[0]
    year = first_time.year

    if isinstance(first_time, cftime.DatetimeNoLeap):
        # No-leap calendar: February is always 28 days
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        # Standard/gregorian calendar: check actual leap year
        days_per_month = [calendar.monthrange(year, m)[1] for m in range(1, 13)]

    return np.array(days_per_month) * 86400  # seconds per month

def compute_cell_area(ds):
    """
    Computes grid cell area in m2 for a regular lat/lon grid.
    Area varies with latitude: A = (R * delta_lat_rad) * (R * cos(lat_rad) * delta_lon_rad)
    """
    R = 6371000  # Earth radius in meters

    delta_lat_rad = np.deg2rad(0.1)
    delta_lon_rad = np.deg2rad(0.1)
    lat_rad = np.deg2rad(ds["lat"])

    cell_area = (R * delta_lat_rad) * (R * np.cos(lat_rad) * delta_lon_rad)

    return cell_area  # shape: (lat,), broadcasts over lon


def compute_annual_emissions(ds):
    """
    Computes total annual CO2 emissions per cell (kg)
    by weighting monthly mean flux (kg m-2 s-1) by seconds per month,
    then multiplying by cell area (m2).
    """
    seconds_per_month = get_seconds_per_month(ds)
    weights = xr.DataArray(
        seconds_per_month,
        dims="time",
        coords={"time": ds["time"]}
    )

    cell_area = compute_cell_area(ds)

    # kg m-2 s-1 * s * m2 = kg
    da_annual = (ds["CO2_em_anthro"] * weights).sum(dim="time") * cell_area

    year = ds["time"].values[0].year
    da_annual = da_annual.assign_attrs(
        {"units": "kg", "long_name": f"CO2 Annual Total Emissions {year}"}
    )

    return da_annual

def pre_process_data_socioeconomic(varname:str="Population", source:str="2UP", version:str="GHSL_2024_M3", SSP_base:str="SSP2", copy=False, log: logging.Logger=local_log):

    #log, log = init_logging(f"log_pre_process_{varname.replace("|", "_")}_{source}_{version}", "log/reading_data")

    #-------------------------------------------------------------------------------------------------------------------
    log.info("\n\n********************************PROCESS DATA****************************************************************************")
    log.info("\n\n**********************************PROCESS DATA**************************************************************************")
    log.info(f"variable: {varname}, source: {source}, version: {version}")
    log.info(f"variable: {varname}, source: {source}, version: {version}")
    log.info("************************************************************************************************************")
    log.info("************************************************************************************************************")

    (data_dir_original, rxr_filename, end_filename, glob_pattern, search_pattern, test_filename_original, mult_factor) = get_parameters_SE(process_data=True, varname=varname, source=source, version=version, SSP_base=SSP_base, log=log)
    (data_dir_run, dummy1, dummy2,dummy3, dummy4, test_filename_run, mult_factor) = get_parameters_SE(process_data=False, varname=varname, source=source, version=version, SSP_base=SSP_base, log=log)

    log.info(f"\n\n------------check_values_tiff (before update)----------------------------------------------------------------------------")
    log.info(f"{PRINT_COLORS["green"]}Original data directory: {data_dir_original}{PRINT_COLORS["end"]}")
    test_file_path_original = data_dir_original / test_filename_original
    log.info(f"Checking info original file: {test_file_path_original}")
    check_values_tiff(test_file_path_original, band=1, inlc_inf=False, log=log)

    # 1 process SE data
    if "processed" in data_dir_original.parts:
        response = input(f"{PRINT_COLORS['yellow']}The directory '{data_dir_original}' contains 'processed'. Do you want switch 'process data' off? (y/n): {PRINT_COLORS['end']}")
        if response.lower() in ["y", "yes"]:
            process_data = False
            log.info(f"{PRINT_COLORS['yellow']}Switching 'process_data' to False since 'processed' is in the data directory.{PRINT_COLORS['end']}")

    # 2 read SE data
    data_dir_processed = update_GIS_parameters(varname, source, version, SSP_base, data_dir_original, glob_pattern, 1, log)
    log.info(f"{PRINT_COLORS["green"]}Updated data directory: {data_dir_processed}{PRINT_COLORS["end"]}")
    log.info(f"\n\n------------check_values_tiff (after update)----------------------------------------------------------------------------")
    test_file_path_update = data_dir_processed / test_filename_run
    log.info(f"Checking info updated file: {test_file_path_update}")
    check_values_tiff(test_file_path_update, band=1, inlc_inf=False, log=log)

    # 3. Copy processed data to run directory
    if copy:
        log.info(f"\n\n------------copy processed data to run directory----------------------------------------------------------------------------")
        if data_dir_run is None:
            log.info(f"No run directory specified, skipping copy step.")
        elif data_dir_run.resolve() == data_dir_processed.resolve():
            log.info(f"Processed data directory is the same as run directory, so no copy needed.")
        else:
            data_dir_run.mkdir(parents=True, exist_ok=True)
            files = sorted(data_dir_processed.glob("*"))
            for f in tqdm.tqdm(files, desc="Copying processed data to run directory"):
                if f.is_file():
                    run_tiff_path = data_dir_run / f.name
                    shutil.copy(f, run_tiff_path)
                    log.info(f"Copied {f} to {run_tiff_path}")

def _read_in_nc(data_dir_original_source:Path, glob_pattern:str, search_pattern:str, varname_source:str, varname_processed:str, log: logging.Logger):
    # read in nc files
    files = sorted(data_dir_original_source.glob(glob_pattern))
    log.info(f"Found {len(files)} files matching {glob_pattern}")
    # Store metadata from the first file (assuming all files have consistent metadata)
    data_list = []
    year_list = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        for i, f in enumerate(files):
            ds = xr.open_dataset(f)
            #da = rxr.open_rasterio(f, chunks={"x": "auto", "y": "auto"})
            year = int(re.search(search_pattern, f.name).group(1))
            data_list.append(ds[varname_source])
            #data_list.append(da)  # rioxarray opens with a default band dimension, so we take the first band
            year_list.append(year)
    xr_data = xr.concat(data_list, dim="time")
    xr_data = xr_data.assign_coords(time=year_list)
    xr_data = xr_data.to_dataset(name=varname_processed)
    arc_seconds, arc_minutes, arc_degrees = calculate_resolution(xr_data[varname_processed])
    log.info(f"Resolution: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.2f} arc degrees")

    return xr_data

def pre_process_data_emissions(varname:str="Emissions|CO2|Excl. shipping, aviation, AFOLU", source:str="CEDS_CMIP7", version:str="2025_04_18", copy=False, log: logging.Logger=local_log):

    #log, log = init_logging(f"log_pre_process_{varname.replace("|", "_")}_{source}_{version}", "log/reading_data")

    #-------------------------------------------------------------------------------------------------------------------
    log.info("\n\n********************************PROCESS DATA****************************************************************************")
    log.info("\n\n**********************************PROCESS DATA**************************************************************************")
    log.info(f"variable: {varname}, source: {source}, version: {version}")
    log.info(f"variable: {varname}, source: {source}, version: {version}")
    log.info("************************************************************************************************************")
    log.info("************************************************************************************************************")

    with open("downscaling/settings_data_locations.json", "r") as f:
        data_files = json.load(f)
    data_dir_original = data_files["grid"]["original"]
    data_dir_run = data_files["grid"]["run"]
    data_dir_processed = Path(data_files["grid"]["processed"]["dir_emissions_processed"])
    data_dir_run_source = None
    data_dir_original_source = None
    data_dir_processed_source = None

    df_total = pd.DataFrame()
    summary_data = {"Year": [], "Value": [], "Unit": []}
    match source:
        case "EDGAR":
            match version:
                case "2024":
                    varname_EDGAR = "emissions"
                    year_check = 2020
                    # ==> EDGAR data
                    # https://edgar.jrc.ec.europa.eu/dataset_ghg2024#p2
                    data_dir_original_source = Path(data_dir_original["dir_emissions_EDGAR_2024_original"])
                    data_dir_run_source = Path(data_dir_run["dir_emissions_EDGAR_2024_run"])
                    data_dir_processed_source = data_dir_processed / source / version
                    data_dir_processed_source.mkdir(parents=True, exist_ok=True)

                    # read_CO2_grid_files (CO2 total, CO2 shipping, CO2 aviation as shiping and aviation are subtracted from total CO2)
                    # 1. Read in tiff files and create rioxarray
                    # 1a. Total CO2
                    varname_CO2 = "emissions_CO2"
                    glob_pattern_CO2 = f"EDGAR_{version}_GHG_CO2_????_TOTALS_emi.nc"
                    search_pattern_CO2 = f"EDGAR_{version}_GHG_CO2_(\\d{{4}})_TOTALS_emi\\.nc"
                    xr_emissions_CO2 = _read_in_nc(data_dir_original_source, glob_pattern_CO2, search_pattern_CO2, varname_EDGAR, varname_CO2, log)
                    xr_emissions_CO2.to_netcdf(data_dir_processed_source / f"EDGAR_{version}_GHG_CO2_1970_2020_TOTALS_emi.nc")

                    # 1b. Total CO2 Shipping
                    # EDGAR_{version}_GHG_CO2_1970_TNR_Ship_emi.nc
                    glob_pattern_CO2_shipping = f"EDGAR_{version}_GHG_CO2_????_TNR_Ship_emi.nc"
                    search_pattern_CO2_shipping = f"EDGAR_{version}_GHG_CO2_(\\d{{4}})_TNR_Ship_emi\\.nc"
                    varname_CO2_shipping = "emissions_CO2_shipping"
                    # read in nc files
                    #warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
                    xr_emissions_CO2_shipping = _read_in_nc(data_dir_original_source, glob_pattern_CO2_shipping, search_pattern_CO2_shipping, varname_EDGAR, varname_CO2_shipping, log)
                    xr_emissions_CO2_shipping.to_netcdf(data_dir_processed_source / f"EDGAR_{version}_GHG_CO2_1970_2020_TNR_Ship_emi.nc")

                    # 1c.Total Aviation
                    # Aviation climbing&descent
                    # EDGAR_2024_GHG_CO2_1970_TNR_Aviation_CDS_emi.nc
                    glob_pattern_CO2_aviation_CDS = f"EDGAR_{version}_GHG_CO2_????_TNR_Aviation_CDS_emi.nc"
                    search_pattern_CO2_aviation_CDS = f"EDGAR_{version}_GHG_CO2_(\\d{{4}})_TNR_Aviation_CDS_emi\\.nc"
                    varname_CO2_aviation_CDS = "emissions_CO2_aviation_CDS"
                    xr_emissions_CO2_aviation_CDS = _read_in_nc(data_dir_original_source, glob_pattern_CO2_aviation_CDS, search_pattern_CO2_aviation_CDS, varname_EDGAR, varname_CO2_aviation_CDS, log)                    # Aviation climbing&descent
                    #                EDGAR_{version}_GHG_CO2_1970_TNR_Aviation_CRS_emi.nc
                    glob_pattern_CO2_aviation_CRS = f"EDGAR_{version}_GHG_CO2_????_TNR_Aviation_CRS_emi.nc"
                    search_pattern_CO2_aviation_CRS = f"EDGAR_{version}_GHG_CO2_(\\d{{4}})_TNR_Aviation_CRS_emi\\.nc"
                    varname_CO2_aviation_CRS = "emissions_CO2_aviation_CRS"
                    xr_emissions_CO2_aviation_CRS = _read_in_nc(data_dir_original_source, glob_pattern_CO2_aviation_CRS, search_pattern_CO2_aviation_CRS, varname_EDGAR, varname_CO2_aviation_CRS, log)                         # Aviation Aviation landing&takeoff
                    # EDGAR_{version}_GHG_CO2_1970_TNR_Aviation_LTO_emi.nc
                    glob_pattern_CO2_aviation_LTO = f"EDGAR_{version}_GHG_CO2_????_TNR_Aviation_LTO_emi.nc"
                    search_pattern_CO2_aviation_LTO = f"EDGAR_{version}_GHG_CO2_(\\d{{4}})_TNR_Aviation_LTO_emi\\.nc"
                    varname_CO2_aviation_LTO = "emissions_CO2_aviation_LTO"
                    xr_emissions_CO2_aviation_LTO = _read_in_nc(data_dir_original_source, glob_pattern_CO2_aviation_LTO, search_pattern_CO2_aviation_LTO, varname_EDGAR, varname_CO2_aviation_LTO, log)
                    # Aviation supersonic
                    # EDGAR_{version}_GHG_CO2_1970_TNR_Aviation_SPS_emi.nc
                    glob_pattern_CO2_aviation_SPS = f"EDGAR_{version}_GHG_CO2_????_TNR_Aviation_SPS_emi.nc"
                    search_pattern_CO2_aviation_SPS = f"EDGAR_{version}_GHG_CO2_(\\d{{4}})_TNR_Aviation_SPS_emi\\.nc"
                    varname_CO2_aviation_SPS = "emissions_CO2_aviation_SPS"
                    xr_emissions_CO2_aviation_SPS = _read_in_nc(data_dir_original_source, glob_pattern_CO2_aviation_SPS, search_pattern_CO2_aviation_SPS, varname_EDGAR, varname_CO2_aviation_SPS, log)

                    # 2. Combine into single dataset
                    varname_CO2_aviation = "emissions_CO2_aviation"
                    xr_emissions_CO2_aviation = xr.Dataset({varname_CO2_aviation: xr_emissions_CO2_aviation_CDS[varname_CO2_aviation_CDS] +
                                                                xr_emissions_CO2_aviation_CRS[varname_CO2_aviation_CRS] +
                                                                xr_emissions_CO2_aviation_LTO[varname_CO2_aviation_LTO] #+
                                                                # data_emissions_CO2_aviation_SPS_rxr["emissions_CO2_aviation_SPS"] # exclude, as it only has data until 2003
                                                                })
                    xr_emissions_CO2_aviation.to_netcdf(data_dir_processed_source / f"EDGAR_{version}_GHG_CO2_1970_2020_TNR_Aviation_emi.nc")
                    xr_emissions_CO2_excl_bunkers = xr.Dataset({varname: xr_emissions_CO2[varname_CO2] - xr_emissions_CO2_shipping[varname_CO2_shipping] - xr_emissions_CO2_aviation[varname_CO2_aviation]})
                    xr_emissions_CO2_excl_bunkers.to_netcdf(data_dir_processed_source / f"EDGAR_{version}_GHG_CO2_1970_2020_excl_bunkers_emi.nc")

                    # save processed data
                    filename_EM_EDGAR_processed = f"Emissions_CO2_Excl_shipping_aviation_AFOLU.nc"
                    ds_file_path = data_dir_processed_source / filename_EM_EDGAR_processed
                    arc_seconds, arc_minutes, arc_degrees = calculate_resolution(xr_emissions_CO2_excl_bunkers[varname])
                    log.info(f"Resolution of annual data: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.2f} arc degrees")
                    xr_emissions_CO2_excl_bunkers.to_netcdf(ds_file_path)

        case "CEDS_CMIP7":
            match version:
                case "2025_04_18":
                    data_dir_original_source = Path(data_dir_original["dir_emissions_CEDS_CMIP7_original"])
                    data_dir_run_source = Path(data_dir_run["dir_emissions_CEDS_CMIP7_run"])
                    data_dir_processed_source = data_dir_processed / source / version
                    data_dir_processed_source.mkdir(parents=True, exist_ok=True)

                    # calc total CO2 excl bunkers
                    # 0: Agriculture; 1: Energy; 2: Industrial; 3: Transportation; 4: Residential, Commercial, Other; 5: Solvents production and application; 6: Waste; 7: International Shipping

                    # loop through data dir
                    glob_pattern = "CO2-em-anthro_input4MIPs_emissions_*.nc"
                    files = sorted(Path(data_dir_original_source).glob(glob_pattern))

                    #varname = "Emissions|CO2|Excl. shipping, aviation, AFOLU"
                    for i, file in enumerate(files):
                        print(f"Processing {file.name}...")
                        match = re.search(r"_gr_(\d{4})", file.name)
                        if match:
                            year = int(match.group(1))
                            # sum over sector 1, 2, 3, 4, 5, (exclude sector 0,6, 7)
                            ds = xr.open_dataset(file)
                            #varname = list(ds.data_vars)[0]
                            # 0: AGR Non-combustion agricultural sector
                            # 1: ENE Energy transformation and extraction
                            # 2: IND Industrial combustion and processes
                            # 3: TRA Surface Transportation (Road, Rail, Other)
                            # 4: RCO Residential, commercial, and other
                            # 5: SLV Solvents
                            # 6: WST Waste disposal and handling
                            # 7: SHP International shipping (including VOCs from oil tanker loading/leakage)
                            da_annual_excl_bunkers = compute_annual_emissions(ds).sel(sector=slice(1, 5)).sum(dim="sector")
                            total_gton_excl_bunkers = da_annual_excl_bunkers.sum(dim=["lat", "lon"]).values
                            log.info(f"{year}: {total_gton_excl_bunkers/1e12:.2f} Gton CO2 excl bunkers")
                            summary_data["Year"].append(f"{year:.0f}")
                            summary_data["Value"].append(f"{total_gton_excl_bunkers/10e12:,.2f}")
                            summary_data["Unit"].append("GtCO2")
                            da_annual_excl_bunkers.name = varname
                            da_annual_excl_bunkers = da_annual_excl_bunkers / 1000
                            da_annual_excl_bunkers.attrs["unit"] = "tonnes CO2/year"
                            da_annual_excl_bunkers = da_annual_excl_bunkers.expand_dims(time=[pd.Timestamp(f"{year}")])
                            da_file_path = data_dir_processed_source / f"CO2-em-anthro_annual_excl_bunkers_{year:}.nc"
                            arc_seconds, arc_minutes, arc_degrees = calculate_resolution(da_annual_excl_bunkers)
                            print(f"Resolution of annual data: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.2f} arc degrees")
                            da_annual_excl_bunkers.to_netcdf(da_file_path)

                            df_total = pd.DataFrame(summary_data)
                            if not df_total.empty:
                                df_file_path = data_dir_processed_source / f"annual_emissions_excl_bunkers_summary.csv"
                                df_total.to_csv(df_file_path, sep=";", index=False)
                                print(df_total.to_string(index=False))
                        else:
                            print(f"Skipping {file.name}: no year found in filename")

    # 3. Copy processed data to run directory
    if copy:
        log.info(f"\n\n------------copy processed data to run directory----------------------------------------------------------------------------")
        if data_dir_run_source is None:
            log.info(f"No run directory specified, skipping copy step.")
        elif data_dir_run_source.resolve() == data_dir_processed_source.resolve():
            log.info(f"Processed data directory is the same as run directory, so no copy needed.")
        else:
            data_dir_run_source.mkdir(parents=True, exist_ok=True)
            files = sorted(data_dir_processed_source.glob("*"))
            for f in tqdm.tqdm(files, desc="Copying processed data to run directory"):
                if f.is_file():
                    run_tiff_path = data_dir_run_source / f.name
                    shutil.copy(f, run_tiff_path)
                    log.info(f"Copied {f} to {run_tiff_path}")

def read_process_grid_data_socioeconomic(dir_processed:Path, varname="Population", source:str="2UP", version="GHSL_2024_M3", SSP_base="SSP2",
                                         coarse_factor:float=1, unit:str="", save: bool=False, check: bool=False, log: logging.Logger=local_log) -> Tuple[xr.Dataset, Path]:
    '''
    Read in grid data files for population, GDP, and CO2 emissions
    Parameters:
    ----------
    project_dir : str | Path
        The root directory of the project
    Returns:
    ----------
    Filename_population, filename_GDP, filename_CO2
    '''

    #log, log = init_logging(f"log_read_{varname.replace("|", "_")}_{source}_{version}", "log/reading_data")
    year_check = 2020
    base_year = 2015

    #-------------------------------------------------------------------------------------------------------------------
    log.info("\n\n*************************************RUN socio-economice DATA*********************************************************")
    log.info("\n\n***************************************RUN DATA*********************************************************************")
    log.info(f"variable: {varname}, source: {source}, version: {version}, coarse_factor: {coarse_factor}")
    log.info(f"variable: {varname}, source: {source}, version: {version}, coarse_factor: {coarse_factor}")
    log.info("************************************************************************************************************")
    log.info("************************************************************************************************************")

    # init
    rxr_SE = None

    # 1. Read population data
    # retrieve parameters
    nodata=np.nan
    (data_dir, rxr_filename, end_filename,
     glob_pattern, search_pattern,
     test_filename, factor) = get_parameters_SE(process_data=False, varname=varname, source=source, version=version, SSP_base=SSP_base, log=log)

    # Check population data for the year 2020
    log.info(f"\n\n************before coarsening***************************************************************************")
    log.info(f"\n\n------------check_values_tiff----------------------------------------------------------------------------")
    test_file_path = data_dir / test_filename
    check_values_tiff(test_file_path, band=1, inlc_inf=False, log=log)

    # read SE data
    log.info(f"\nReading {varname} data from {data_dir} with pattern {glob_pattern}")
    log.info(f"\n\n------------read_in_tiff_to_rio----------------------------------------------------------------------------")
    #warnings.filterwarnings("ignore", category=NotGeoreferencedWarning) # Ignore not georeferenced warnings, will be fixed later
    rxr_SE = read_in_tiff_to_rio(data_dir, glob_pattern, search_pattern, varname, year_check, log)

    log.info(f"Before coarsening:")
    log.info(f"rio nodata: {PRINT_COLORS["yellow"]}{rxr_SE[varname].rio.nodata}{PRINT_COLORS["end"]}")
    log.info(f"_FillValue: {PRINT_COLORS["yellow"]}{rxr_SE[varname].encoding.get("_FillValue")}{PRINT_COLORS["end"]}")
    log.info(f"crs: {PRINT_COLORS["yellow"]}{rxr_SE.rio.crs}{PRINT_COLORS["end"]}")
    info = print_transform(rxr_SE.rio.transform())
    log.info(f"transform: {PRINT_COLORS["yellow"]}{info}{PRINT_COLORS["end"]}")
    log.info(f"{varname} data:{rxr_SE}")
    da = rxr_SE[varname].isel(time=0)

    log.info(f"\n\n------------calculate_resolution----------------------------------------------------------------------------")
    arc_seconds, arc_minutes, arc_degrees = calculate_resolution(da, input_unit="decimal_degrees")
    log.info(f"{varname} data resolution: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.2f} arc degrees")

    if check:
        # count number of cells for the year 2020: total, zero, positive, negative and nan values
        log.info(f"\n\n------------check_values_rio_xarray before coarsening--------------------------------------------------------------------")
        count_values_rio_xarray(rxr_SE, varname, year=year_check, log=log)
        calc_total_sum_rio_xarray(rxr_SE[varname], year=year_check, log=log)

    # check bounds and coordinates
    info_lines = []
    info_lines.append(f"\nBounds: {da.rio.bounds()}")
    info_lines.append(f"\nCoordinates:")
    for name, coord in da.coords.items():
        info_lines.append(f"\t{name}, values: {coord.values.min():.2f} to {coord.values.max():.2f}")
    log.info(info_lines)

    # 4 coarsen SE data
    log.info(f"\n\n------------coarsen_save_rio_xarray----------------------------------------------------------------------------")
    rxr_filepath = dir_processed / f"{rxr_filename}_cf_{coarse_factor}.nc"
    rxr_SE_coarsened = coarsen_save_rio_xarray(rxr_SE,
                                                factor=coarse_factor,
                                                zero_to_nan=True,
                                                save=save,
                                                save_type="netcdf",
                                                chunks_size_save=chunks,
                                                varname=varname,
                                                filepath=rxr_filepath,
                                                aggregation_methods={varname: "sum"},
                                                log=log)
    log.info(f"\n\n************after coarsening***************************************************************************")
    log.info(f"After coarsening:")
    #log.info(f"_FillValue: {PRINT_COLORS["yellow"]}{rxr_SE_coarsened[varname].encoding.get('_FillValue')}{PRINT_COLORS["end"]}")
    #log.info(f"nodata: {PRINT_COLORS["yellow"]}{rxr_SE_coarsened[varname].rio.nodata}{PRINT_COLORS["end"]}")  #netcdf does not store nodata in rio attributes
    log.info(f"attrs nodata (stored, not used): {PRINT_COLORS["yellow"]}{rxr_SE_coarsened.attrs.get("source_nodata", None)}{PRINT_COLORS["end"]}")
    log.info(f"crs: {PRINT_COLORS["yellow"]}{rxr_SE_coarsened.rio.crs}{PRINT_COLORS["end"]}")
    info = print_transform(rxr_SE_coarsened.rio.transform())
    log.info(f"transform: {PRINT_COLORS["yellow"]}{info}{PRINT_COLORS["end"]}")

    da = rxr_SE_coarsened[varname].isel(time=0)
    arc_seconds, arc_minutes, arc_degrees = calculate_resolution(da, input_unit="decimal_degrees")
    log.info("************************************************************************************************************")
    log.info(f"{varname} data resolution: {arc_seconds:.1f} arc seconds, {arc_minutes:.1f} arc minutes, {arc_degrees:.2f} arc degrees")

    rxr_SE_coarsened[varname].attrs["unit"] = unit

    if check:
        # count number of cells for the year 2020: total, zero, positive, negative and nan values
        log.info(f"\n\n------------check_values_rio_xarray after coarsening--------------------------------------------------------------------")
        count_values_rio_xarray(rxr_SE_coarsened, varname, year=year_check, log=log)
        calc_total_sum_rio_xarray(rxr_SE_coarsened[varname], year=year_check, log=log)

    return rxr_SE_coarsened, rxr_filepath

def _clean_dataset_for_netcdf(ds: xr.Dataset) -> tuple[xr.Dataset, dict]:
    """
    Removes internal NetCDF4 attrs from variables and moves encoding-related
    attrs to the encoding dict. Returns the cleaned dataset and an encoding dict
    ready to pass to to_netcdf().
    """

    INTERNAL_NETCDF4_ATTRS = {"_NCProperties", "_Netcdf4Coordinates", "ChunkSizes"}
    ENCODING_ATTRS = {"_FillValue", "scale_factor", "add_offset", "dtype"}

    encoding = {}

    for var in list(ds.data_vars) + list(ds.coords):
        var_attrs = ds[var].attrs
        var_encoding = dict(ds[var].encoding)

        for attr in INTERNAL_NETCDF4_ATTRS:
            var_attrs.pop(attr, None)

        for attr in ENCODING_ATTRS:
            if attr in var_attrs:
                var_encoding[attr] = var_attrs.pop(attr)

        encoding[var] = var_encoding

    return ds, encoding

def read_process_grid_data_EM(dir_processed:Path, varname="Emissions_CO2_Excl_shipping_aviation_AFOLU", unit="tonnes CO2/year", source:str="EDGAR", version="2024",
                              base_year=2020, coarse_factor:float=1, save:bool=False, check:bool=False, log: logging.Logger=local_log) -> Tuple[xr.Dataset, Path]:
    # Read CO2 emissions data

    #log, log = init_logging("log_read_EM", "log/reading_data")

    log.info("\n\n*************************************RUN emissions DATA*********************************************************")
    log.info("\n\n***************************************RUN DATA*********************************************************************")
    log.info(f"variable: {varname}, source: {source}, version: {version}, coarse_factor: {coarse_factor}")
    log.info(f"variable: {varname}, source: {source}, version: {version}, coarse_factor: {coarse_factor}")
    log.info("************************************************************************************************************")
    log.info("************************************************************************************************************")

    rxr_filepath = Path(".")
    ds_emissions_CO2_excl_bunkers_coarsened = xr.Dataset()

    with open("downscaling/settings_data_locations.json", "r") as f:
        data_files = json.load(f)
    data_run = data_files["grid"]["run"]

    data_dir_EM = Path(".")
    match source:
        case "EDGAR":
            varname_EDGAR = "emissions"
            year_check = 2020
            # ==> EDGAR data
            # https://edgar.jrc.ec.europa.eu/dataset_ghg2024#p2
            #data_dir_EM = f"{project_dir}/data/input/emissions/{source}/{version}/emissions_grid"
            match version:
                case "2024":
                    data_dir_EM = Path(data_run["dir_emissions_EDGAR_2024_run"])
                    ds_emissions_CO2_excl_bunkers = xr.open_dataset(data_dir_EM / f"Emissions_CO2_Excl_shipping_aviation_AFOLU.nc", chunks={"x": "auto", "y": "auto"})

                    # change georeferencing of dataset
                    # The function rxr.open_rasterio() is designed for reading raster formats like GeoTIFF.
                    # For NetCDF files, plain xr.open_dataset() is the cleaner entry point, and rioxarray is then used only for the spatial operations it is actually needed for.
                    ds_emissions_CO2_excl_bunkers = ds_emissions_CO2_excl_bunkers.rename({"lat": "y", "lon": "x"})
                    da = ds_emissions_CO2_excl_bunkers[varname]
                    # n_x = da.sizes["x"]
                    # n_y = da.sizes["y"]
                    # x_res = 360.0 / n_x
                    # y_res = 180.0 / n_y
                    # lon = np.linspace(-180.0 + x_res / 2, 180.0 - x_res / 2, n_x)
                    # lat = np.linspace(90.0 - y_res / 2, -90.0 + y_res / 2, n_y)
                    # da = da.assign_coords(x=lon, y=lat)
                    # da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
                    da = da.rio.write_crs("EPSG:4326")
                    da.attrs["crs"] = "EPSG:4326"
                    da = da.rio.write_transform()
                    ds_emissions_CO2_excl_bunkers = da.to_dataset(name=varname)
                    ds_emissions_CO2_excl_bunkers, encoding = _clean_dataset_for_netcdf(ds_emissions_CO2_excl_bunkers)
                    ds_emissions_CO2_excl_bunkers.attrs["unit"] = "tonnes CO2/year"
                    #ds_emissions_CO2_excl_bunkers.to_netcdf(data_dir_EM / f"EDGAR_{version}_GHG_CO2_1970_2020_TOTALS_emi.nc", encoding=encoding)

                    log.info(f"Before coarsening:")
                    log.info(f"rio nodata: {PRINT_COLORS["yellow"]}{ds_emissions_CO2_excl_bunkers[varname].rio.nodata}{PRINT_COLORS["end"]}")
                    log.info(f"_FillValue: {PRINT_COLORS["yellow"]}{ds_emissions_CO2_excl_bunkers[varname].encoding.get("_FillValue")}{PRINT_COLORS["end"]}")
                    log.info(f"crs: {PRINT_COLORS["yellow"]}{ds_emissions_CO2_excl_bunkers.rio.crs}{PRINT_COLORS["end"]}")
                    info = print_transform(ds_emissions_CO2_excl_bunkers.rio.transform())
                    log.info(f"transform: {PRINT_COLORS["yellow"]}{info}{PRINT_COLORS["end"]}")

                    if check:
                        # count number of cells for the year 2020: total, zero, positive, negative and nan values
                        log.info(f"\n\n------------check_values_rio_xarray before coarsening--------------------------------------------------------------------")
                        count_values_rio_xarray(ds_emissions_CO2_excl_bunkers, varname, year=year_check, log=log)
                        calc_total_sum_rio_xarray(ds_emissions_CO2_excl_bunkers[varname], year=year_check, log=log)

                    # 3. coarsen and save file
                    log.info("Factor is 1, directly saving to netcdf, without coarsening")
                    rxr_filepath = dir_processed / f"emissions_CO2_excl_shipping_aviation_{source}_{version}_hist_cf_{coarse_factor}.nc"
                    ds_emissions_CO2_excl_bunkers_coarsened = coarsen_save_rio_xarray(ds_emissions_CO2_excl_bunkers,
                                                                                      factor=coarse_factor,
                                                                                      save=save,
                                                                                      zero_to_nan=True,
                                                                                      save_type="netcdf",
                                                                                      chunks_size_save=chunks,
                                                                                      varname=varname,
                                                                                      filepath=rxr_filepath,
                                                                                      aggregation_methods={varname: "sum"},
                                                                                      log=log)
                    ds_emissions_CO2_excl_bunkers_coarsened[varname].attrs["unit"] = unit
                    if check:
                        # count number of cells for the year 2020: total, zero, positive, negative and nan values
                        log.info(f"\n\n------------check_values_rio_xarray after coarsening--------------------------------------------------------------------")
                        if ds_emissions_CO2_excl_bunkers_coarsened is not None:
                            count_values_rio_xarray(ds_emissions_CO2_excl_bunkers_coarsened, varname, year=year_check, log=log)
                            calc_total_sum_rio_xarray(ds_emissions_CO2_excl_bunkers_coarsened[varname], year=year_check, log=log)
        case "CEDS_CMIP7":
            match version:
                case "2025_04_18":
                    year_check = 2020
                    data_dir_EM = Path(data_run["dir_emissions_CEDS_CMIP7_v2025_run"])
                    rxr_filepath = Path(".")
                    #rxr_filepath = f"{data_dir_EM}/CO2-em-anthro_annual_excl_bunkers_2020.nc"
                    glob_pattern = f"CO2-em-anthro_annual_excl_bunkers_????.nc"
                    search_pattern = f"CO2-em-anthro_annual_excl_bunkers_(\\d{{4}}).nc"


                    files = sorted(data_dir_EM.glob(glob_pattern))
                    log.info(f"Found {len(files)} files matching {glob_pattern}")

                    data = []
                    years = []

                    for i, f in enumerate(files):
                        #ds_emissions_CO2_excl_bunkers_coarsened = read_in_tiff_to_rio(data_dir_EM, glob_pattern, search_pattern, varname, year_check, log)
                        #da_file = data_dir_EM / f"CO2-em-anthro_annual_excl_bunkers_{base_year}.nc"
                        da_data = xr.open_dataarray(f)
                        da_data.attrs["unit"] = unit
                        year = pd.Timestamp(da_data.time.values[0]).year
                        da_data = da_data.assign_coords(time=[year])
                        data.append(da_data)
                        years.append(year)
                    ds_emissions_CO2_excl_bunkers_coarsened = xr.concat(data, dim="time")
                    ds_emissions_CO2_excl_bunkers_coarsened = ds_emissions_CO2_excl_bunkers_coarsened.assign_coords(time=years)
                    ds_emissions_CO2_excl_bunkers_coarsened = ds_emissions_CO2_excl_bunkers_coarsened.rename({k: v for k, v in {"lat": "y", "lon": "x"}.items() if k in ds_emissions_CO2_excl_bunkers_coarsened.dims})
                    ds_emissions_CO2_excl_bunkers_coarsened = ds_emissions_CO2_excl_bunkers_coarsened.to_dataset(name=varname)
        case _:
            log.info(f"Error: Unknown source {source}, exiting")
            data_emissions_CO2_excl_bunkers_rxr_coarsened = None
            rxr_filepath = Path(".")

    return ds_emissions_CO2_excl_bunkers_coarsened, rxr_filepath

def read_processed_grid_data(data_dir: Path, file: Optional[Path], varname: str,
                             SSP_base: str, source: str, version: str, coarse_factor: int|float) -> xr.Dataset | None:

        # check files
        varname_read = varname.replace("|", "_")
        print(f"Checking input files in read_processed_IPAT_grid_data for {varname_read}...")
        if file is None:
            file = data_dir / f"{varname_read}_{source}_{version}_{SSP_base}_cf_{coarse_factor}.nc"
            if not file.exists():
                raise FileNotFoundError(f"{varname_read} file not found: {file}")
        print(f"Using file: {file}")

        # read in data if not already read in
        print(f"Reading {varname_read} data from file...")
        with xr.open_dataset(file, decode_coords="all") as rxr_IPAT_factor:
            print(f"Type: {type(rxr_IPAT_factor)}")

        return rxr_IPAT_factor

def main():
    pass
    # log, log = init_logging("log_main", "log/reading_data")

    # project_dir = Path(__file__).parent
    # log.info(f"Project directory: {project_dir}")

    # # population
    # save_POP = True
    # rxr_pop, f_pop = read_process_grid_data_socioeconomic(str(project_dir), source="2UP", coarse_factor=12, save=save_POP)
    # if save_POP:
    #     log.info(f"Population data saved to: {f_pop}")

    # # GDP
    # save_GDP_PPP = True
    # rxr_GDP_PPP, f_GDP_PPP = read_process_grid_data_socioeconomic(str(project_dir), source="Wang", coarse_factor=12, save=save_GDP_PPP)
    # if save_GDP_PPP:
    #     log.info(f"GDP data saved to: {f_GDP_PPP}")

    # # CO2 emissions
    # rxr_EM, f_EM = read_process_grid_data_EM(str(project_dir), source="EDGAR", coarse_factor=1, save=True)
    # log.info(f"CO2 emissions data saved to: {f_EM}")


if __name__ == "__main__":
    main()
