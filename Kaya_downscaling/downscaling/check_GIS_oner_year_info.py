import rasterio
#from rasterio.crs import CRS
#from rasterio.transform import from_bounds
#from affine import Affine

import json
import shutil

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import rasterio
import rioxarray as rxr

# print crs info

def print_info_rasterio_attributes(rasterio_file_path:str, band:int=1):

    # The 0 values represent nodata regions.
    # The mask array shows the 255 values that indicate valid data regions.
    print(("Rasterio attributes: "))
    with rasterio.open(rasterio_file_path, "r") as src:

        print("=== Default tags ===")
        for key, value in src.tags().items():
            value_str = str(value)
            if '\n' in value_str:
                value_str = value_str.replace('\n', ' ')
            print(f"\t\t - {key}: {value_str}")

        # Print band-level tags
        for i in range(1, src.count + 1):
            band_tags = src.tags(i)
            if band_tags:
                print(f"=== Band {i} tags ===")
                for key, value in band_tags.items():
                    value_str = str(value)
                    if '\n' in value_str:
                        value_str = value_str.replace('\n', ' ')
                    print(f"\t\t - {key}: {value_str}")

        # Print all metadata domain tags
        for domain in src.tag_namespaces():
            tags = src.tags(ns=domain)
            if tags:
                print(f"=== {domain} ===")
                for key, value in tags.items():
                    value_str = str(value)
                    if '\n' in value_str:
                        value_str = value_str.replace('\n', ' ')
                    print(f"\t\t - {key}: {value_str}")

        # Print profile
        print("=== Profile ===")
        for key, value in src.profile.items():
            value_str = str(value)
            if '\n' in value_str:
                value_str = value_str.replace('\n', ' ')
            print(f"\t\t - {key}: {value_str}")

def print_info_rasterio(rasterio_file_path:str, band:int=1):

    print(("Rasterio info: "))
    with rasterio.open(rasterio_file_path, "r") as src:
        # info
        print("\n=== General Info ===")
        print(f"shape: {src.shape}")
        print(f"count: {src.count}")
        print(f"dtypes: {src.dtypes}")
        print(f"nodatavals: {src.nodatavals}")
        print(f"nodata: {src.nodata}")

        # NODATA
        print("\n=== NoData Check ===")
        # The 0 values represent nodata regions, the mask array shows the 255 and -9999 values that indicate valid data regions.
        nodata = src.nodata
        # calculate number of cells with nodata value
        data = src.read(band)  # Read first band
        num_nodata_cells = np.sum(data == nodata)
        print(f"Number of cells with nodata value ({nodata}): {num_nodata_cells:,}")
        num_zero = np.sum(data == 0)
        print(f"Number of cells with zero value (0): {num_zero:,}")

        # Check for NaN values
        data = src.read(1) if 'data' not in locals() else data
        has_nan = np.isnan(data).any()
        print(f"Contains NaN cells: {has_nan}")

        # CRS
        print("\n=== CRS and Spatial Info ===")
        if src.crs is not None:
            print(f"CRS: {src.crs}")
            print(f"Transform: {src.transform}")
            print(f"Width: {src.width}")
            print(f"Height: {src.height}")
            print(f"Bounds: {src.bounds}")
            print(f"Resolution: {src.res}")
            print(f"Affine: {src.transform}")
        else :
            print(f"crs: {src.crs}")

def print_info_rioxarray_attributes(rxr_file_path:str):

    print(("rioxarray attributes: "))
    # Opren with rioxarray to check attributes and encoding
    da = rxr.open_rasterio(rxr_file_path)

    for attr, value in da.attrs.items():
        print(f"Attribute: {attr} = {value}")

    for enc_attr, enc_value in da.encoding.items():
        print(f"Encoding: {enc_attr} = {enc_value}")

def print_info_rioxarray(rxr_file_path:str):

    print("rioxarray info: ")
    # Opren with rioxarray to check attributes and encoding
    da = rxr.open_rasterio(rxr_file_path)
    print(da)

    nodata = da.rio.nodata
    print(f"Nodata value: {nodata}")
    fill_value = da.attrs.get('_FillValue', None)
    print(f"_FillValue: {fill_value}")
    fill_value_encoding = da.encoding.get('_FillValue', None)
    print(f"_FillValue (from encoding): {fill_value_encoding}")

def calc_GIS_stats_rasterio(file_path:str, coarse_factor=1, band:int=1, incl_inf:bool=False):
    # The 0 values represent nodata regions.
    # The mask array shows the 255 values that indicate valid data regions.
    with rasterio.open(file_path, "r") as src:

        # 1. NODATA MASK
        print(f"shape: {src.shape}")
        print(f"count: {src.count}")
        print(f"dtypes: {src.dtypes}")
        print(f"nodatavals: {src.nodatavals}")
        print(f"nodata: {src.nodata}")
        out_shape = (src.height // coarse_factor, src.width // coarse_factor)
        msk = src.read_masks(band, out_shape=out_shape)

        numt_total = msk.size
        num_zeros = np.sum(msk == 0)
        num_nodatas = np.sum(msk == src.nodata)
        num_other = msk.size - num_zeros - num_nodatas

        print()

        df_nodata = pd.DataFrame(
            [   {"Metric": "total pixels", "Value": f"{numt_total:,}"},
                {"Metric": "zeros (nodata)", "Value": f"{int(num_zeros):,}"},
                {"Metric": "num_num_nodatas", "Value": f"{int(num_nodatas):,}"},
                {"Metric": "num_other", "Value": f"{int(num_other):,}"},
            ]
        )
        print(df_nodata.to_string(index=False))

        # 2. STATISTICS
        # Read data
        data = src.read(band)

        cols = src.width
        rows = src.height
        num_cells = rows * cols

        # Get nodata value from band if not specified
        nodata = src.nodatavals[band - 1]  # band is 1-indexed in your code

        # Get _FillValue from band tags
        band_tags = src.tags(band)

        _FillValue = None
        _FillValue = band_tags.get('_FillValue', None)

        min_value = np.nanmin(data)
        min_value_excl_zero_nodata = np.nanmin(data[(data != 0) & (data != nodata)]) if nodata is not None else np.nanmin(data[data != 0])
        max_value = np.nanmax(data)


        # Handle infinities
        if min_value == -np.inf or max_value == np.inf:
            finite_mask = np.isfinite(data)
            if min_value == -np.inf:
                min_no_inf = np.min(data[finite_mask])
            if max_value == np.inf:
                max_no_inf = np.max(data[finite_mask])

        # Count statistics
        stats_data = []
        nodata_count = None
        if nodata is not None:
            nodata_count = (data == nodata).sum()

        nan_count = np.isnan(data).sum()
        zero_count = (data == 0).sum()
        neg_count = (data < 0).sum()
        pos_count = (data > 0).sum()

        # Excluding nodata
        zero_count_excl_nodata = None; neg_count_excl_nodata = None; pos_count_excl_nodata = None
        if nodata is not None and not np.isnan(nodata):
            zero_count_excl_nodata = ((data == 0) & (data != nodata)).sum()
            neg_count_excl_nodata = ((data < 0) & (data != nodata)).sum()
            pos_count_excl_nodata = ((data > 0) & (data != nodata)).sum()

        # Infinity values
        neg_inf_count = None; pos_inf_count = None
        if incl_inf:
            neg_inf_count = (data == -np.inf).sum()
            stats_data.append(['-inf values', neg_inf_count, f"{neg_inf_count/num_cells*100:.2f}%"])

            pos_inf_count = (data == np.inf).sum()
            stats_data.append(['+inf values', pos_inf_count, f"{pos_inf_count/num_cells*100:.2f}%"])

        print(f"\nData min: {min_value:,.0f}")
        print(f"Data min (not zero/nodata): {min_value_excl_zero_nodata:,.0f}")
        print(f"Data max: {max_value:,.0f}")
        print(f"NoData value: {nodata}")

        print()
        df_stat = pd.DataFrame(
            [   {"Metric": "min_value", "Value": f"{min_value:,}"},
            {"Metric": "max_value", "Value": f"{max_value:,}"},
            {"Metric": "nodata", "Value": f"{nodata if nodata is not None else 'N/A'}"},
            {"Metric": "_FillValue", "Value": f"{_FillValue if _FillValue is not None else 'N/A'}"},

            {"Metric": "nodata_count", "Value": f"{nodata_count:,}" if nodata_count is not None else 'N/A'},
            {"Metric": "nan_count", "Value": f"{nan_count:,}"},
            {"Metric": "zero_count", "Value": f"{zero_count:,}"},
            {"Metric": "neg_count", "Value": f"{neg_count:,}"},
            {"Metric": "pos_count", "Value": f"{pos_count:,}"},

            {"Metric": "zero_count_excl_nodata", "Value": f"{zero_count_excl_nodata:,}" if zero_count_excl_nodata is not None else 'N/A'},
            {"Metric": "neg_count_excl_nodata", "Value": f"{neg_count_excl_nodata:,}" if neg_count_excl_nodata is not None else 'N/A'},
            {"Metric": "pos_count_excl_nodata", "Value": f"{pos_count_excl_nodata:,}" if pos_count_excl_nodata is not None else 'N/A'},

            {"Metric": "neg_inf_count", "Value": f"{neg_inf_count:,}" if neg_inf_count is not None else 'N/A'},
            {"Metric": "pos_inf_count", "Value": f"{pos_inf_count:,}" if pos_inf_count is not None else 'N/A'},
            ]
        )
        print(df_stat.to_string(index=False))

def plot_GIS_mask_rasterio(file_path:str, coarse_factor=1, band:int=1):

    with rasterio.open(file_path, "r") as src:
        out_shape = (src.height // coarse_factor, src.width // coarse_factor)
        msk = src.read_masks(band, out_shape=out_shape)

        # Plot the mask
        plt.imshow(msk, cmap="Blues_r")
        plt.colorbar(label="Mask value")
        plt.title(f"Band {band} Mask")
        plt.show()

