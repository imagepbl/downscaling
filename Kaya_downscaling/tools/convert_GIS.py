
import argparse
import subprocess
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
import geopandas as gpd
import xarray as xr
import tkinter as tk
from tkinter import filedialog

def convert_IMAGE_regions_netcdf_to_tiff():

    ds_IMAGE_regions = xr.open_dataset("data/input/models/IMAGE/GREG.nc")
    ds_IMAGE_regions = ds_IMAGE_regions.rename({"latitude": "lat", "longitude": "lon"})
    print(f"Dataset opened: {ds_IMAGE_regions}")
    print(ds_IMAGE_regions)
    # get variable from dataset
    var_regions = list(ds_IMAGE_regions.variables)[0]
    print(f"Variable in dataset: {var_regions}")
    print(f"Variable type: {type(var_regions)}")
    output_file = "data/processed/IMAGE_GREG.tif"
    convert_netcdf_to_tiff(ds_IMAGE_regions, var_regions, output_file)

def convert_netcdf_to_tiff(ds: xr.Dataset, variable_name: str, output_file: str):
    # Open the NetCDF file

    # Select the variable you want (replace "variable_name" with your actual variable)
    data_array = ds[variable_name]

    # Set spatial dimensions and CRS if not already set
    data_array = data_array.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    data_array = data_array.rio.write_crs("EPSG:4326")  # Adjust CRS as needed

    # Export to GeoTIFF
    data_array.rio.to_raster(output_file)

def GADM_vector_to_raster(project_dir:str, dir_GADM: str, resolution_degrees: float = 1/120, plot: bool = False):
    """
    Convert GADM vector data to raster format using rasterio.

    Parameters
    ----------
    project_dir : str
        Project directory path
    resolution_degrees : float
        Resolution in degrees. Default 1/120 corresponds to 0.5 arc-minutes
    plot : bool
        Whether to create a plot of the countries
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from pathlib import Path
    import geopandas as gpd
    import pandas as pd
    import numpy as np

    # Read in
    file_GADM_countries = f"{project_dir}/data/input/GADM/single/gadm_410.gpkg"
    print(f"Reading GADM countries from: {file_GADM_countries}")
    countries = gpd.read_file(file_GADM_countries)

    # Save check file
    check_countries = pd.DataFrame(countries.drop(columns="geometry"))
    check_countries.to_csv(f"{dir_GADM}/GADM_countries.csv", sep=";", index=False)

    # Plot
    if plot:
        import matplotlib.pyplot as plt
        print("Plotting GADM countries...")
        fig, ax = plt.subplots(figsize=(12, 8))
        countries.plot(ax=ax, edgecolor="black", facecolor="lightblue", linewidth=0.5)
        ax.set_title("GADM Countries")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.savefig(Path(project_dir) / "figures/GADM_countries.png", dpi=300)
        plt.close()

    # Convert
    print("Converting GADM vector data to raster format...")
    # Map ISO codes to unique integer IDs (raster pixels need numeric values)
    iso_codes = countries["GID_0"].unique()
    iso_to_id = {iso: i + 1 for i, iso in enumerate(iso_codes)}  # Start from 1, reserve 0 for NoData
    pd.DataFrame(list(iso_to_id.items()), columns=["ISO", "id"]).to_csv(f"{dir_GADM}/iso_to_id_mapping.csv", sep=";", index=False)
    id_to_iso = {i: iso for iso, i in iso_to_id.items()}
    pd.DataFrame(list(id_to_iso.items()), columns=["id", "ISO"]).to_csv(f"{dir_GADM}/id_to_iso_mapping.csv", sep=";", index=False)
    countries["iso_id"] = countries["GID_0"].map(iso_to_id)

    print(f"\nMapped {len(iso_codes)} unique ISO codes to integer IDs")
    print(countries[["GID_0", "iso_id"]].head())

    # Save the mapping for later reference
    mapping_df = pd.DataFrame({"GID_0": iso_to_id.keys(), "iso_id": iso_to_id.values()})

    # Define output filename
    resolution_minutes_str = f"{60 * resolution_degrees:.2f}".replace(".", "_")
    print(f"Resolution: {60 * resolution_degrees:.2f} arc-minutes ({resolution_degrees:.6f} degrees)")
    raster_file = f"{project_dir}/data/processed/GADM/iso_codes_raster_{resolution_minutes_str}.tif"

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

def convert_gpkg_to_shapefile():
    # # settings
    # data_dir = "Z:/cold_data_storage/users/roelfsemam/data_downscaling"
    # file_open = "global_population_and_gdp.gpkg"
    # file_path_open = f"{data_dir}/{file_open}"
    root = tk.Tk()
    root.withdraw()

    file_path_open = filedialog.askopenfilename(
        title="Select a GeoPackage file",
        filetypes=[
            ("GeoPackage files", "*.gpkg"),
            ("All files", "*.*")
        ]
    )
    stem_open = Path(file_path_open).stem
    dir_open = Path(file_path_open).parent
    file_path_save = f"{dir_open}/{stem_open}.shp"

    # read in file
    print(f"Reading file: {file_path_open}")
    gdf = gpd.read_file(file_path_open)

    # Convert .gpkg to shapefile and Earth Engine object
    print(f"Converting to shapefile and Earth Engine object, saving to: {file_path_save}")
    gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(file_path_save)

def add_crs_to_raster(data_dir: str, tiff_file: str, crs: str = "EPSG:4326", bounds: tuple = (-180, -90, 180, 90)):
    """Add CRS and transform to a raster file and save to add_crs subdirectory."""
    print(f"Adding CRS {crs} to raster file: {tiff_file}")

    # Create the new directory if it doesn't exist
    new_dir = Path(data_dir) / "add_crs"
    new_dir.mkdir(parents=True, exist_ok=True)

    # Read the original file
    input_path = Path(data_dir) / tiff_file
    with rasterio.open(input_path) as src:
        dst_crs = src.crs
        dst_transform = src.transform
        print(f"Current CRS: {dst_crs}")
        print(f"Current transform: {dst_transform}")

        # Read the data
        data = src.read()
        meta = src.meta.copy()

        # Update CRS if needed
        if dst_crs is None:
            meta.update({"crs": CRS.from_string(crs)})
            print(f"CRS added: {meta['crs']}")
        else:
            print("Raster already has a CRS, copying as-is.")

        # Update transform if needed and bounds are provided
        if bounds is not None and (dst_transform is None or dst_transform == rasterio.Affine.identity()):
            west, south, east, north = bounds
            width = src.width
            height = src.height
            transform = from_bounds(west, south, east, north, width, height)
            meta.update({"transform": transform})
            print(f"Transform added: {transform}")
        elif dst_transform is None or dst_transform == rasterio.Affine.identity():
            print("WARNING: No transform in file and no bounds provided. Transform not set.")
        else:
            print("Raster already has a transform, copying as-is.")

        # Write to new file with updated metadata
        output_path = new_dir / tiff_file
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(data)

    print(f"File saved to: {output_path}\n")


def update_crs_in_tiff_files(data_dir: str, crs: str = "EPSG:4326", bounds: tuple = (-180, -90, 180, 90)):
    """Loop through all TIFF files in data_dir and add CRS."""

    data_dir_path = Path(data_dir)
    tiff_files = list(data_dir_path.glob("*.tif")) + list(data_dir_path.glob("*.TIF"))
    print(f"Found {len(tiff_files)} TIFF files in {data_dir}\n")

    for tiff_path in tiff_files:
        tiff_filename = tiff_path.name
        add_crs_to_raster(data_dir, tiff_filename, crs=crs, bounds=bounds)

def check(dir:str):
    with rasterio.open(dir) as src:
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")
        print(f"Bounds: {src.bounds}")
        print(f"Width: {src.width}, Height: {src.height}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GIS conversion tools") # add_help=True by default
    #parser.add_argument("-g", "--process", metavar="copy", choices=["copy", "no_copy"], help="process datasets and 'copy' to run folder or 'no_copy'")
    #parser.add_argument("-d", "--downscale", action="store_true", help="downscale emissions")
    parser.add_argument("-g", "--gpkg_to_shapefile", action="store_true", help="convert GeoPackage to shapefile")
    parser.add_argument("-i", "--IMAGE_regions_netcdf_to_tiff", action="store_true", help="convert IMAGE regions NetCDF to TIFF")
    # include directory argument for add_crs_to_tiff_files
    parser.add_argument("-a", "--add_crs_to_tiff_files", action="store_true", help="adds CRS to TIFF files in a directory")
    parser.add_argument("-d", "--directory", type=str, help="directory containing TIFF files to process")
    parser.add_argument("-c", "--crs", type=str, help="css to add to TIFF files, e.g. 'EPSG:4326'")
    parser.add_argument("-b", "--bounds", type=str, help="tuple with bounds (west, south, east, north) for setting transform if missing, e.g. '-180,-90,180,90'")
    # check (add -d option for directory)
    parser.add_argument("-x", "--check", action="store_true", help="check")

    arguments = parser.parse_args()
    print(f"Arguments provided: {arguments}")
    try:
        if hasattr(arguments, 'gpkg_to_shapefile') and arguments.gpkg_to_shapefile is True:
            convert_gpkg_to_shapefile()
        if hasattr(arguments, 'IMAGE_regions_netcdf_to_tiff') and arguments.IMAGE_regions_netcdf_to_tiff is True:
            convert_IMAGE_regions_netcdf_to_tiff()
                # if no arguments, print message
        if hasattr(arguments, 'add_crs_to_tiff_files') and arguments.add_crs_to_tiff_files is True:
        # pixi run python tools/convert_GIS.py -a -d "Z:\cold_data_storage\users\roelfsemam\data_downscaling\population\2UP\TowardsAnUrbanPreview_2024_GHSL2014_M3\results\LatLong_World\tpop" -c "EPSG:4326" -b "(-180, -90, 180, 90)"
            if arguments.directory is None or arguments.bounds is None:
                parser.error("--add_crs_to_tiff_files requires --directory and --bounds to be specified")
            update_crs_in_tiff_files(arguments.directory, arguments.crs, tuple(arguments.bounds))
        if not any(vars(arguments).values()):
            print("No arguments provided. Use -h or --help for more information.")
        if hasattr(arguments, 'check') and arguments.check is True:
            if arguments.directory is None:
                    parser.error("--check requires --directory")
            check(arguments.directory)
    except Exception as e:
        print(f"An error occurred: {e}")

