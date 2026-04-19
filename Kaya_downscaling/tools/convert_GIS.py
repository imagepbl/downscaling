
import argparse
import subprocess
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from pyproj import Transformer
from shapely.ops import transform as shapely_transform

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

def GADM_vector_to_raster(project_dir:Path, data_dir:Path, resolution_degrees: float = 1/120, plot: bool = False):
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
    file_GADM_countries = data_dir / "gadm_410.gpkg"
    print(f"Reading GADM countries from: {file_GADM_countries}")
    countries = gpd.read_file(file_GADM_countries)

    dir_GADM = project_dir / "data" / "processed" / "GADM"
    dir_GADM.mkdir(parents=True, exist_ok=True)

    # Save check file
    check_countries = pd.DataFrame(countries.drop(columns="geometry"))
    check_countries.to_csv(f"{project_dir}/processed/GADM/GADM_countries.csv", sep=";", index=False)

    # Plot
    if plot:
        import matplotlib.pyplot as plt
        print("Plotting GADM countries...")
        fig, ax = plt.subplots(figsize=(12, 8))
        countries.plot(ax=ax, edgecolor="black", facecolor="lightblue", linewidth=0.5)
        ax.set_title("GADM Countries")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.savefig(project_dir / "figures/GADM_countries.png", dpi=300)
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

def _area_m2(geometry, crs_from="EPSG:4326"):
    """Return area of a shapely geometry in square metres using an equal-area projection."""
    transformer = Transformer.from_crs(
        crs_from,
        "EPSG:6933",  # WGS 84 / NSIDC EASE-Grid 2.0 — global equal-area
        always_xy=True,
    )
    projected = shapely_transform(transformer.transform, geometry)
    return projected.area

def gadm_levels_to_csv(gpkg_file_path: Path, output_dir: Path) -> Path:
    """
    Read all six GADM 4.1 admin levels from a local GeoPackage and write
    the attribute data (excluding geometry) to a single CSV file.

    Each row in the output CSV has a 'level' column indicating which admin
    level it came from (0-5). Columns not present in a given level are
    filled with NaN.

    Parameters:
    gpkg_path : Path - Path to the local gadm_410-levels.gpkg file.
    output_dir : Path - Directory where the output CSV will be saved.

    Returns:
    Path - Path to the saved CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_levels = []

    level_names = {0: "Country" , 1: "State_Province", 2: "County_District", 3: "Commune_Municipality", 4: "Sub-municipal_1", 5: "Sub-municipal_2"}

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

def get_city_polygon(gpkg_path: Path, iso3: str, city_name: str, output_dir: Path, search_levels: list = None, output_format: str = "gpkg") -> gpd.GeoDataFrame:
    """
    Find and save the polygon of a city/town from a local GADM 4.1 levels GeoPackage.

    Cities and towns are typically found at admin levels 2-4, which varies
    by country. For example, Dutch municipalities (gemeenten) are at level 2.
    The function searches all levels in search_levels and returns all matches.

    Parameters:
    gpkg_path : Path - Path to the local gadm_410-levels.gpkg file.
    iso3 : str - ISO 3166-1 alpha-3 country code to filter by (e.g. 'NLD', 'DEU').
    city_name : str - Name of the city/town to search for (case-insensitive).
    output_dir : Path - Directory where the output file will be saved.
    search_levels : list, optional - Admin levels to search. Defaults to [2, 3, 4].
    output_format : str - Output format for saving: 'gpkg' or 'shp'. Defaults to 'gpkg'.

    Returns:
    gpd.GeoDataFrame - GeoDataFrame with matching rows and a 'found_at_level' column added.

    Raises:
    ValueError - If the city is not found at any of the searched levels.
    """
    if search_levels is None:
        search_levels = [2, 3, 4]

    results = []

    for level in search_levels:
        layer = f"ADM_{level}"
        try:
            gdf = gpd.read_file(gpkg_path, layer=layer)
        except Exception as e:
            print(f"Could not read layer {layer}: {e}")
            continue

        # Filter to the requested country first to reduce memory pressure
        if "GID_0" not in gdf.columns:
            print(f"Column 'GID_0' not found in layer {layer}, skipping.")
            continue
        gdf = gdf[gdf["GID_0"] == iso3.upper()]

        name_col = f"NAME_{level}"
        if name_col not in gdf.columns:
            print(f"Column {name_col} not found in layer {layer}, skipping.")
            continue

        match = gdf[gdf[name_col].str.lower() == city_name.lower()].copy()
        if not match.empty:
            match["found_at_level"] = level
            results.append(match)
            print(f"Found '{city_name}' in layer {layer} ({len(match)} feature(s)).")

    if not results:
        raise ValueError(
            f"'{city_name}' not found for country '{iso3}' at levels {search_levels}. "
            f"Check the spelling or try different levels (e.g. search_levels=[1, 2, 3, 4, 5])."
        )

    gdf_result = gpd.GeoDataFrame(pd.concat(results, ignore_index=True))

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = city_name.replace(" ", "_")
    if output_format == "gpkg":
        out_path = output_dir / f"{iso3}_{safe_name}.gpkg"
        gdf_result.to_file(out_path, driver="GPKG")
    elif output_format == "shp":
        out_path = output_dir / f"{iso3}_{safe_name}.shp"
        gdf_result.to_file(out_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    print(f"Saved city polygon to: {out_path}")
    return gdf_result

def get_us_city_polygon(tiger_dir: Path, city_name: str, output_dir: Path, output_format: str = "gpkg") -> gpd.GeoDataFrame:
    """
    Find and save the polygon of a US city/town from TIGER/Line place shapefiles.

    Searches all state-level place shapefiles found in tiger_dir (files matching
    the pattern tl_*_place.shp). Each file covers one state. The NAME column
    contains the bare place name (e.g. 'New York', 'Carrboro') and NAMELSAD
    contains the name with type appended (e.g. 'New York city', 'Carrboro town').

    Parameters:
    tiger_dir : Path - Directory containing one or more TIGER/Line place shapefiles
        (e.g. tl_2025_36_place.shp, tl_2025_37_place.shp).
    city_name : str - Name of the city/town to search for (case-insensitive).
        Use the bare name without type suffix, e.g. 'New York' not 'New York city'.
    output_dir : Path - Directory where the output file will be saved.
    output_format : str - Output format: 'gpkg' or 'shp'. Defaults to 'gpkg'.

    Returns:
    gpd.GeoDataFrame - GeoDataFrame with matching rows and a 'source_file' column added.
    """

    shapefiles = sorted(tiger_dir.glob("tl_*_place.shp"))
    if not shapefiles:
        raise FileNotFoundError(
            f"No TIGER place shapefiles (tl_*_place.shp) found in: {tiger_dir}"
        )
    print(f"Found {len(shapefiles)} TIGER place shapefile(s) in {tiger_dir}")

    results = []

    for shp_path in shapefiles:
        print(f"Searching {shp_path.name}...")
        gdf = gpd.read_file(shp_path)

        if "NAME" not in gdf.columns:
            print(f"  Column 'NAME' not found in {shp_path.name}, skipping.")
            continue

        match = gdf[gdf["NAME"].str.lower() == city_name.lower()].copy()
        if not match.empty:
            match["source_file"] = shp_path.name
            results.append(match)
            print(
                f"  Found '{city_name}' in {shp_path.name} "
                f"({len(match)} feature(s)): {match['NAMELSAD'].tolist()}"
            )

    if not results:
        raise ValueError(
            f"'{city_name}' not found in any TIGER place shapefile in {tiger_dir}. "
            f"Check the spelling or make sure the correct state file is downloaded."
        )

    gdf_result = gpd.GeoDataFrame(pd.concat(results, ignore_index=True))

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = city_name.replace(" ", "_")
    if output_format == "gpkg":
        out_path = output_dir / f"USA_{safe_name}.gpkg"
        gdf_result.to_file(out_path, driver="GPKG")
    elif output_format == "shp":
        out_path = output_dir / f"USA_{safe_name}.shp"
        gdf_result.to_file(out_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    print(f"Saved city polygon to: {out_path}")
    return gdf_result

def calculate_emissions_in_polygon(da: xr.DataArray, polygon: gpd.GeoDataFrame, city_name: str) -> dict:
    """
    Calculate emission statistics within a city/town polygon, with fractional
    pixel coverage for cells that are partially inside and partially outside.

    For each raster cell, the fraction of the cell area covered by the polygon
    is computed via geometric intersection (shapely). This fraction is used as
    a weight when computing the weighted mean, and to compute a fractional sum
    (i.e. each cell contributes proportionally to how much of it lies inside
    the polygon). Fully outside cells (coverage = 0) are excluded entirely.

    Parameters:
    da : xr.DataArray - DataArray to calculate emissions for. Should already
        be spatially subsetted (e.g. via clip_box) and have a CRS set.
        Must have 'x'/'y' or 'lon'/'lat' dimensions.
    polygon : gpd.GeoDataFrame - GeoDataFrame with the city/town polygon.
        Must be in EPSG:4326 or will be reprojected to match da.
    city_name : str - Name of the city/town, used for logging only.

    Returns:
    dict - Dictionary with keys:
        'city'             : city name
        'sum_weighted'     : total emissions (in unit used for emissions) in the city
        'sum_full'         : total tonnes (in unit used for emissions) in the city fully covered pixels only (original emission units)
        'mean_per_m2'      : average emissions ((in unit used for emissions) per square metre in the city
        'min'              : min emission value among pixels with any coverage
        'max'              : max emission value among pixels with any coverage
        'n_pixels_full'    : number of pixels fully inside polygon (coverage = 1)
        'n_pixels_partial' : number of pixels partially inside polygon (0 < coverage < 1)
        'n_pixels_any'     : total pixels with any coverage (full + partial)
    """
    from shapely.geometry import box as shapely_box

    empty_result = {
        "city": city_name,
        "sum_weighted": float("nan"),
        "sum_full": float("nan"),
        "mean_per_m2": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
        "n_pixels_full": 0,
        "n_pixels_partial": 0,
        "n_pixels_any": 0,
    }

    if polygon is None or polygon.empty:
        print(f"Warning: No polygon available for '{city_name}', skipping emission calculation.")
        return empty_result

    try:
        # Ensure CRS is set on the DataArray
        if da.rio.crs is None:
            da = da.rio.write_crs("EPSG:4326")

        # Reproject polygon to match DataArray CRS if needed
        if polygon.crs is None:
            polygon = polygon.set_crs("EPSG:4326")
        elif polygon.crs.to_epsg() != da.rio.crs.to_epsg():
            polygon = polygon.to_crs(da.rio.crs)

        # Get x/y coordinate arrays and resolution
        x_dim = "x" if "x" in da.dims else "lon"
        y_dim = "y" if "y" in da.dims else "lat"
        x_coords = da[x_dim].values
        y_coords = da[y_dim].values
        res_x = abs(float(x_coords[1] - x_coords[0]))
        res_y = abs(float(y_coords[1] - y_coords[0]))

        # Get the 2D emission values (squeeze out any extra dims)
        values_2d = da.values.squeeze()

        # Union of all polygon parts into one geometry for intersection
        poly_union = polygon.geometry.union_all()

        # Iterate over all cells and compute fractional coverage
        weighted_values_sum = []
        weighted_values_mean = []
        weights_mean = []
        full_values = []
        all_values = []
        n_full = 0
        n_partial = 0

        for row_idx, cy in enumerate(y_coords):
            for col_idx, cx in enumerate(x_coords):
                val = float(values_2d[row_idx, col_idx])
                if not np.isfinite(val):
                    continue

                cell = shapely_box(cx - res_x / 2, cy - res_y / 2, cx + res_x / 2, cy + res_y / 2)
                if not poly_union.intersects(cell):
                    continue

                intersection = poly_union.intersection(cell)
                cell_area_m2 = _area_m2(cell)
                intersection_area_m2 = _area_m2(intersection)
                coverage = intersection_area_m2 / cell_area_m2  # dimensionless 0-1

                if coverage <= 0:
                    continue

                weighted_values_sum.append(val * coverage)           # coverage-fraction weighted
                weighted_values_mean.append(val * intersection_area_m2)  # physical area weighted
                weights_mean.append(intersection_area_m2)
                all_values.append(val)

                if coverage >= 1.0:
                    full_values.append(val)
                    n_full += 1
                else:
                    n_partial += 1

        if not weighted_values_sum:
            print(f"Warning: No valid pixels within polygon for '{city_name}'.")
            return empty_result

        weighted_vals_sum_arr = np.array(weighted_values_sum)
        weighted_vals_mean_arr = np.array(weighted_values_mean)
        weights_mean_arr = np.array(weights_mean)
        all_vals_arr = np.array(all_values)
        total_intersection_area_m2 = float(np.sum(weights_mean_arr))

        if n_full == 0:
            print(f"Warning: No fully covered pixels found for '{city_name}'. "
                  f"The city likely does not cover any complete raster pixels at this resolution "
                  f"({res_x:.4f} x {res_y:.4f} degrees). Use 'sum_weighted' instead of 'sum_full'.")

        result = {"city": city_name,
                  "sum_weighted":  float(np.sum(weighted_vals_sum_arr)),
                  "sum_full": float(np.sum(full_values)) if full_values else 0.0, # total emissions in city
                  "mean_per_m2":  float(np.sum(weighted_vals_sum_arr)) / total_intersection_area_m2,  # tonnes/m²
                  "min": float(np.min(all_vals_arr)),
                  "max": float(np.max(all_vals_arr)),
                  "n_pixels_full": n_full,
                  "n_pixels_partial": n_partial,
                  "n_pixels_any": n_full + n_partial}
        print(f"  Emissions in '{city_name}': "
              f"sum_weighted={result['sum_weighted']:,.4f}, "
              f"mean_per_m2={result['mean_per_m2']:,.4f}, "
              f"full_pixels={n_full}, partial_pixels={n_partial}")
        return result

    except Exception as e:
        print(f"Warning: Could not calculate emissions in polygon for '{city_name}': {e}")
        return empty_result

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

