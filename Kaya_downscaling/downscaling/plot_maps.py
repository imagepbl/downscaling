from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import xarray as xr
import rioxarray as rxr
import rasterio
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from rasterio.transform import from_bounds

from tools.general_functions import replace_punctuation_in_filenames
import downscaling.read_process_IAM_data as process_IAM_data
import downscaling.process_IPAT_factors as process_IPAT_factors

def save_to_grid_tiff(dir_processed:Path,
                      xr_grid:xr.Dataset, varname:str,
                      add_text: str,
                      years:list,
                      model:str, scenario:str,
                      include_model_in_filename:bool=True) -> None:

    ys = xr_grid["y"].values
    xs = xr_grid["x"].values
    height = len(ys)
    width = len(xs)

    res_x = float(xs[1] - xs[0])
    res_y = float(ys[1] - ys[0])
    west  = float(xs.min()) - res_x / 2
    east  = float(xs.max()) + res_x / 2
    south = float(ys.min()) - abs(res_y) / 2
    north = float(ys.max()) + abs(res_y) / 2

    transform = from_bounds(west=west, south=south, east=east, north=north,
                            width=width, height=height)

    for year in years:
        data = xr_grid[varname].sel(time=year).values.astype(np.float32)
        # Ensure data is ordered north-to-south to match the transform
        if ys[0] < ys[-1]:
            data = data[::-1, :]
        if include_model_in_filename:
            stem = f"{replace_punctuation_in_filenames(varname)}{add_text}_{model}_{scenario}"
        else:
            stem = f"{replace_punctuation_in_filenames(varname)}{add_text}_{scenario}"
        print(f"Saving {varname} for year {year} to GeoTIFF: {stem}_{year}.tif")
        tif_file = dir_processed / f"{stem}_{year}.tif"
        with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"):
            with rasterio.open(
                tif_file,
                "w",
                driver="GTiff",
                height=data.shape[0], width=data.shape[1],
                count=1,
                dtype=data.dtype,
                crs=xr_grid.rio.crs,
                transform=transform, #xr_grid.rio.transform(),
                #compress="LZW",
                compress="ZSTD",
                zstd_level=1,  # 1 = fastest, 22 = best compression. For large files, 1 is usually the right trade-off.
                predictor=3 if np.issubdtype(data.dtype, np.floating) else 2,
                tiled=True,   # Better for large files
                blockxsize=256, blockysize=256) as dst: dst.write(data, 1)


def plot_coast_checks(gadm_tif_path: Path, output_path: Path, add_text:str="", resolution_minutes: float = 0.5) -> None:
    """
    For each CHECK_REGIONS bounding box, plot country_id_GADM and region_number
    side by side with coastlines overlaid, and save to output_dir.

    Parameters
    ----------
    gadm_tif_path : str
        Path to the IMAGE_GADM_regions_raster_*.tif file with 2 bands:
        band 1 = country_id_GADM, band 2 = region_number.
    output_dir : str
        Directory where the PNG files will be saved.
    resolution_minutes : float
        Resolution in arc minutes, used only for the figure title.
    """
    # (min_lon, max_lon, min_lat, max_lat), label
    CHECK_REGIONS = {
        "Indonesia_Strait_of_Malacca": (99.0,  109.0, -6.0,  7.0),
        "Philippines":                  (116.0, 127.0,  5.0, 20.0),
        "Japan_Seto_Inland_Sea":        (130.0, 136.0, 32.0, 36.0),
        "Greece_Aegean":                (22.0,  30.0,  36.0, 42.0),
        "Denmark_Wadden_Sea":           (7.0,   13.0,  54.0, 58.0),
        "Peru_Lima_Coast":              (-80.0, -74.0, -14.0, -10.0)}

    print(f"Reading GADM raster: {gadm_tif_path}")
    # Open lazily — only clip regions are loaded into memory
    ds = rxr.open_rasterio(gadm_tif_path, chunks="auto", lock=False)

    for region_name, (min_lon, max_lon, min_lat, max_lat) in CHECK_REGIONS.items():
        print(f"Plotting {region_name} ...")

        # Clip to bounding box — loads only this small tile into memory
        ds_clip = ds.rio.clip_box(minx=min_lon, maxx=max_lon, miny=min_lat, maxy=max_lat).compute()  # bring clipped data into memory

        band_country = ds_clip.sel(band=1).values.astype(float)
        band_region  = ds_clip.sel(band=2).values.astype(float)

        # Mask 0 (ocean / unassigned) so it shows as white
        band_country = np.where(band_country == 0, np.nan, band_country)
        band_region  = np.where(band_region  == 0, np.nan, band_region)

        lon_coords = ds_clip.x.values
        lat_coords = ds_clip.y.values
        extent = [lon_coords.min(), lon_coords.max(),
                  lat_coords.min(), lat_coords.max()]

        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": proj})
        fig.suptitle(f"{region_name.replace('_', ' ')}  —  {resolution_minutes} arc-min resolution",
            fontsize=13, fontweight="bold")

        titles    = ["Band 1: country_id_GADM", "Band 2: region_number"]
        data_arrs = [band_country, band_region]
        cmaps     = ["tab20b", "tab20c"]

        for ax, title, data, cmap in zip(axes, titles, data_arrs, cmaps):
            ax.set_extent(extent, crs=proj)

            im = ax.imshow(data, origin="upper", extent=extent, transform=proj, cmap=cmap, interpolation="nearest")

            # Coastlines and borders for visual reference
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black")
            ax.add_feature(cfeature.BORDERS,   linewidth=0.5, edgecolor="grey",
                           linestyle="--")
            ax.add_feature(cfeature.OCEAN,      facecolor="lightcyan", alpha=0.3)

            ax.gridlines(draw_labels=True, linewidth=0.4, color="grey",
                         alpha=0.6, linestyle=":")
            ax.set_title(title, fontsize=11)

            # Colourbar
            plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.06,
                         fraction=0.04, label="ID / region number")

            # Legend patch for unassigned pixels
            unassigned_patch = mpatches.Patch(
                facecolor="white", edgecolor="grey", label="0 = unassigned"
            )
            ax.legend(handles=[unassigned_patch], loc="lower left", fontsize=8)

        plt.tight_layout()
        save_path = output_path / f"archipelago_check{add_text}_{region_name}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

        # Free clipped data explicitly
        del ds_clip, band_country, band_region

    print("All checks saved.")


def plot_IPAT_summary(dir_processed:Path, add_txt:str,
                      xr_population_hist:xr.Dataset, xr_gdp_ppp_hist:xr.Dataset, xr_emissions_hist:xr.Dataset,
                      xr_population_proj:xr.Dataset, xr_gdp_ppp_proj:xr.Dataset, xr_emissions_proj:xr.Dataset,
                      varname_POP:str="Population", varname_GDP:str="GDP|PPP", varname_EM:str="Emissions|CO2|Excl. shipping, aviation, AFOLU",
                      years_hist:int|list=2020, years_projection:int|list=[2030, 2050],
                      coarse_factor:int=10) -> bool:

    # check if data exists for years
    # Normalize to lists to handle single int input
    if isinstance(years_hist, int):
        years_hist = [years_hist]
    if isinstance(years_projection, int):
        years_projection = [years_projection]
    xr_population_hist = xr_population_proj.reindex(method="nearest", tolerance=0.01)
    xr_gdp_ppp_hist = xr_gdp_ppp_proj.reindex(method="nearest", tolerance=0.01)
    xr_emissions_hist = xr_emissions_proj.reindex(method="nearest", tolerance=0.01)
    checks = {
        "xr_population_hist": (xr_population_hist, years_hist),
        "xr_gdp_ppp_hist":    (xr_gdp_ppp_hist,    years_hist),
        "xr_emissions_hist":  (xr_emissions_hist,   years_hist),
        "xr_population_proj": (xr_population_proj,  years_projection),
        "xr_gdp_ppp_proj":    (xr_gdp_ppp_proj,     years_projection),
        "xr_emissions_proj":  (xr_emissions_proj,   years_projection),
    }
    all_ok = True
    for name, (ds, years) in checks.items():
        available = ds.time.values.tolist()
        missing = [y for y in years if y not in available]
        if missing:
            print(f"WARNING: {name} is missing the following years: {missing}")
            all_ok = False
        else:
            print(f"OK: all years found in {name}")

    if not all_ok:
        #raise ValueError("Some requested years are missing from the datasets. See warnings above.")
        print("Some requested years are missing from the datasets. See warnings above.")
        return False
    else:
        # plot figures for population, gdp, and emissions for historical and projection years
        xr_population = xr.concat([xr_population_hist.sel(time=years_hist), xr_population_proj.sel(time=years_projection)], dim="time")
        xr_gdp_ppp    = xr.concat([xr_gdp_ppp_hist.sel(time=years_hist), xr_gdp_ppp_proj.sel(time=years_projection)],    dim="time")
        xr_emissions  = xr.concat([xr_emissions_hist.sel(time=years_hist), xr_emissions_proj.sel(time=years_projection)],  dim="time")
        years = years_hist + years_projection
        variables = {
            varname_POP: xr_population[varname_POP],
            varname_GDP: xr_gdp_ppp[varname_GDP],
            varname_EM: xr_emissions[varname_EM]
        }

        n_years = len(years)
        n_vars = len(variables)
        varnames = list(variables.keys())
        das_coarsened = {}
        for varname, da in variables.items():
            print(f"Coarsening variable: {varname}, type: {type(da)}")
            da_coarsened = (da
                            .sel(time=years)
                            .where(da > 0)
                            .coarsen(x=coarse_factor, y=coarse_factor, boundary="trim").sum())
            da_coarsened = da_coarsened.chunk({"y": "auto", "x": "auto"})
            das_coarsened[varname] = da_coarsened

        fig, ax = plt.subplots(
            figsize=(12 * n_vars, 4 * n_years),
            nrows=n_years,
            ncols=n_vars,
            squeeze=False
        )

        for i, year in enumerate(years):
            for j, varname in enumerate(varnames):
                print(f"Plotting year: {year} ({i+1}/{n_years}), variable: {varname} ({j+1}/{n_vars})")
                da_year = das_coarsened[varname].sel(time=year)

                # Replace the regular Axes cell with a Mercator GeoAxes
                fig.delaxes(ax[i, j])
                ax_map = fig.add_subplot(n_years, n_vars, i * n_vars + j + 1, projection=ccrs.Mercator())
                plot_Mercator_projection(da_year, ax=ax_map, transform="log", title=f"{varname}\n{year}\n{add_txt}")
                ax_map.set_xlabel("Longitude")
                ax_map.set_ylabel("Latitude")

        plt.tight_layout()

        combined_varnames = "_".join(varnames).replace('|', '_').replace(' ', '_')
        fig_path = dir_processed / "figures" / f"IPAT_summary_{combined_varnames}{add_txt}_cf_{coarse_factor}.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")

        return True

def plot_hist_map(dir_processed:Path, xr_year_plot:xr.Dataset, scenario:str, varname:str, year:int):

    # Create a regular axis for the histogram and a GeoAxes for the map (required by Cartopy)
    fig = plt.figure(figsize=(12, 4))
    ax_hist = fig.add_subplot(1, 2, 1)
    ax_map = fig.add_subplot(1, 2, 2, projection=ccrs.Mercator())

    data_year = xr_year_plot[varname].sel(time=year)
    p001 = float(data_year.quantile(0.01))
    p99 = float(data_year.quantile(0.99))

    xr_year_plot[varname].sel(time=year).plot.hist(ax=ax_hist, bins=100, color="lightblue", edgecolor="black", range=(min(0, p001), p99))
    ax_hist.set_title(f"Histogram of CO2 emissions for scenario {scenario} and year {year}")
    ax_hist.set_xlabel("CO2 emissions")
    ax_hist.set_ylabel("Frequency")

    # Plot on a Mercator GeoAxes
    plot_Mercator_projection(xr_year_plot[varname].sel(time=year), ax=ax_map, title="CO2 emissions")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")

    varname_save = varname.replace("|", "_").replace(" ", "_")
    png_file = dir_processed / "figures" / f"hist_map_{varname_save}_{year}.png"
    plt.savefig(png_file, dpi=300, bbox_inches="tight")

def plot_map(dir_processed:str, da:xr.DataArray, varname:str, add_txt:str, years:list, coarse_factor:int=10):
    # # Create data for plot

    #years_plot = [2020, 2030, 2050]
    # years_plot = [2020]
    # First coarsen to avoid memory errors
    print(f"{type(da)}")
    da_plot_coarsened_plot = (da
                            .sel(time=years)
                            .where(da > 0)
                            .coarsen(x=coarse_factor, y=coarse_factor, boundary="trim").sum())
    da_plot_coarsened_plot = da_plot_coarsened_plot.chunk({"y": "auto", "x": "auto"})

    # Plot the map
    n_years = len(years)
    fig, ax = plt.subplots(figsize=(12, 4*n_years), nrows=n_years, squeeze=False)

    for i, year in enumerate(years):
        print(f"Plotting year: {year}, index: {i+1} of {n_years}")
        da_plot_coarsened_plot_year = da_plot_coarsened_plot.sel(time=year)
        # Plot on a Mercator GeoAxes
        fig.delaxes(ax[i,0])  # remove the regular Axes in that cell
        # rows, cols, index
        ax_map = fig.add_subplot(n_years, 1, i+1, projection=ccrs.Mercator())
        plot_Mercator_projection(da_plot_coarsened_plot_year, ax=ax_map, transform="log", title=f"{varname}\n{year}")
        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")

    plt.tight_layout()

    plt.savefig(f"{dir_processed}/figures/map_{varname.replace("|", "_").replace(" ", "_")}{add_txt}_cf_{coarse_factor}.png", dpi=300, bbox_inches="tight")

def plot_hist(dir_processed: Path, ds:xr.Dataset, scenario:str, varname:str, add_txt:str, year:int):


    # convert to dataframe and sort
    da = ds[varname].sel(time=year)
    df_sorted = da.to_dataframe(name=varname).reset_index().sort_values(by=varname)
    df_sorted.rename(columns={varname: "value"}, inplace=True)
    df_sorted["value"] /= 10**6  # convert to millions for better readability

    plt.figure(figsize=(8,6))
    sns.histplot(data=df_sorted, x="value", bins=50, log_scale=(True, False))
    plt.title(f"Histogram (logscale) for scenario {scenario} of {varname} for {year}")
    unit = da.attrs.get("unit", "") if da.attrs else ""
    plt.xlabel(f"{varname}" + (f" ({unit})" if unit else ""))
    plt.ylabel("Count (log scale)")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.12f}'))
    plt.xticks(rotation=45)
    fig_file = dir_processed / "figures" / f"hist_{varname.replace('|', '_').replace(' ', '_')}{add_txt}_{year}.png"
    fig_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_file, dpi=300)

def plot_model_region_country_raster(ds_GADM_raster: xr.Dataset, save_file_path:str, save=False):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(20, 6),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    country_data: xr.DataArray = ds_GADM_raster["country_id_GADM"]
    country_data.plot(
        ax=axes[0],
        transform=ccrs.PlateCarree(),
        cmap="tab20",
        add_colorbar=True
    ) # type: ignore
    axes[0].coastlines()
    axes[0].set_title("Country ID (GADM)")

    region_data: xr.DataArray = ds_GADM_raster["country_id_GADM"]
    region_data.plot(
        ax=axes[1],
        transform=ccrs.PlateCarree(),
        cmap="tab20",
        add_colorbar=True
    ) # type: ignore
    axes[1].coastlines()
    axes[1].set_title("Region Number")

    plt.tight_layout()
    if save:
        plt.savefig(save_file_path, dpi=300)
    plt.show()

def plot_boxplot_per_region(project_dir:Path, dir_processed:Path, file_IAM_model_region_numbers:str,
                            xr_data: xr.Dataset, varname:str,
                            model:str, scenario:str,
                            years:list):

    regions, regions_mapping = process_IAM_data.get_regions(project_dir, model, file_IAM_model_region_numbers)
    lookup = np.vectorize(lambda x: regions_mapping.get(x, "unknown"))
    unit = xr_data[varname].attrs.get("unit", "") if xr_data[varname].attrs else ""

    # Apply lookup to the region_number variable and assign as new variable
    xr_data["region_name"] = xr.apply_ufunc(
        lookup,
        xr_data["region_number"],
        dask="parallelized",
        output_dtypes=[str],
    )

    n = len(years)
    fig, axes = plt.subplots(1, len(years), figsize=(n*5, 6), sharey=True)

    for ax, year in zip(axes, years):
        # Select year — keeps rest of data out of memory
        ds_year = xr_data.sel(time=year)

        # Flatten spatial dimensions to 1D arrays
        em_flat = ds_year[varname].values.flatten()
        region_flat = ds_year["region_name"].values.flatten()

        # Remove NaN: float check for EM, None/"nan" string check for region_name
        valid_mask = ~np.isnan(em_flat) & (region_flat != None) & (region_flat != "nan")
        em_valid = em_flat[valid_mask]
        region_valid = region_flat[valid_mask]  # already strings, no .astype(int)

        # Group EM values per region
        unique_regions = np.unique(region_valid)
        data_by_region = [em_valid[region_valid == r] for r in unique_regions]
        labels = unique_regions.tolist()  # already strings, no str() conversion needed

        ax.boxplot(data_by_region, labels=labels, patch_artist=True, showfliers=False)
        ax.set_title(str(year), fontsize=14)
        ax.set_xlabel("Region", fontsize=12)
        ax.tick_params(axis="x", rotation=90, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

        # Free memory after each year
        del ds_year, em_flat, region_flat, em_valid, region_valid

    axes[0].set_ylabel(unit, fontsize=12)
    fig.suptitle(f"{varname} per grid cell per model region for scenario {scenario}", y=1.02, fontsize=16)
    plt.tight_layout()

    save_file = dir_processed / "figures" / f"boxplot_per_region_{varname.replace('|', '_').replace(' ', '_')}_{model}_{scenario}.png"
    plt.savefig(save_file, dpi=300, bbox_inches="tight")

def plot_urban_emissions_per_region(project_dir:Path, df_urban_emissions:pd.DataFrame):
    # This function can be implemented similarly to plot_boxplot_per_region, but using the percentage_class variable from the emissions_urban_regional_sums dataframe.
    # It would create boxplots of the percentage of emissions in urban areas per region, for each year.
    pass

def plot_comparison_IAM_grid(project_dir:Path, dir_processed:Path, read_processed_emissions:bool,
                             df_IAM:pd.DataFrame, xr_grid: xr.Dataset, xr_IAM_regions_grid:xr.Dataset, varname: str,
                             model: str, scenario: str, years_downscaling:list,
                             file_IAM_model_region_numbers:str):


    regions, regions_mapping = process_IAM_data.get_regions(project_dir, model, file_IAM_model_region_numbers)
    #xr_IAM_regions_grid_downscaling = xr_IAM_regions_grid.reindex_like(xr_grid, method="nearest")
    #xr_IAM_regions_grid_downscaling = xr_IAM_regions_grid_downscaling.assign_coords(late=xr_grid.y, longitude=xr_grid.x)

    plot_dir = dir_processed / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    file_process_emissions_IAM = plot_dir / f"IAM_{model}_{scenario}_emissions_IAM.csv"
    file_processed_emissions_grid = plot_dir / f"regional_sums_emissions_{model}_{scenario}.nc"

    if read_processed_emissions or not file_process_emissions_IAM.exists():
        df_emissions, xr_emissions_regional_sums = process_IPAT_factors.calc_urban_regional_emissions(xr_grid, varname, df_IAM, years_downscaling)
        df_emissions.to_csv(file_process_emissions_IAM, sep=";", index=False)
        xr_emissions_regional_sums.to_netcdf(file_processed_emissions_grid, mode="w", engine="h5netcdf")
    else:
        df_emissions = pd.read_csv(file_process_emissions_IAM, sep=";")
        xr_emissions_regional_sums = xr.open_dataset(file_processed_emissions_grid)

    varname_IAM = f"{varname}_IAM"
    varname_grid = f"{varname}_grid_summed"
    df_emissions_plot = df_emissions[["region_number", "year", varname_grid, varname_IAM]].copy()
    df_emissions_plot.rename(columns={varname_grid: "value_grid", varname_IAM: "value_IAM"}, inplace=True)
    # add World
    df_emissions_plot_World = df_emissions_plot.groupby("year").sum().reset_index()
    df_emissions_plot_World["region_number"] = 28
    df_emissions_plot = pd.concat([df_emissions_plot, df_emissions_plot_World], ignore_index=True)
    df_regions = pd.DataFrame.from_dict(regions_mapping, orient="index").reset_index()
    df_regions.rename(columns={'index': 'region_number'}, inplace=True)
    df_regions.columns = ["region_number", "region_name"]
    df_emissions_plot = pd.merge(df_emissions_plot, df_regions, on="region_number", how="left")
    df_emissions_plot = df_emissions_plot.melt(id_vars=["region_number", "region_name", "year"], value_vars=["value_grid", "value_IAM"], var_name="variable", value_name="value")

    # plot
    fig, axs = plt.subplots(nrows=6, ncols=5, figsize=(16, 6))
    sns.set_theme(style="whitegrid", font_scale=1.5)
    g = sns.relplot(data=df_emissions_plot, x="year", y="value", col="region_name", hue="variable",  style="variable", col_wrap=4, kind="line", linewidth=2.5, markers={"value_grid": "o", "value_IAM": "s"}, facet_kws={"sharey": False})
    for ax in g.axes.flat:
        current_ymin, current_ymax = ax.get_ylim()
        ax.set_ylim(min(0, current_ymin), current_ymax)
        ax.set_ylabel("")  # remove individual y-axis labels

    g.fig.supylabel("Emissions", x=0.02)  # single centered label

    save_file = dir_processed / "figures" / f"comparison_IAM_grid_{varname.replace('|', '_').replace(' ', '_')}_{model}_{scenario}.png"
    plt.savefig(save_file, dpi=300, bbox_inches="tight")

def plot_factors_GDP_POP(project_dir:Path, ds_population:None|xr.Dataset, ds_gdp_ppp:None|xr.Dataset, ds_gdp_per_pop:None|xr.Dataset,
                         year:int=2020, coarsen:int=10):

    names = []
    varnames = []
    xr_datasets = []
    if not ds_population is None:
        da_population_2020 = ds_population['Population'].sel(time=2020)
        names.append("ds_POP_2UP_GHSL_M3_2020")
        varnames.append("Population")
        xr_datasets.append(ds_population.sel(time=year))
    if not ds_gdp_ppp is None:
        da_gdp_ppp_2020 = ds_gdp_ppp['GDP|PPP'].sel(time=2020)
        names.append("ds_GDP_PPP_Wang_v7_2020")
        varnames.append("GDP|PPP")
        xr_datasets.append(ds_gdp_ppp.sel(time=year))
    if not ds_gdp_per_pop is None:
        da_gdp_per_pop_2020 = ds_gdp_per_pop.sel(time=2020)
        names.append("ds_GDP_PPP_Wang_v7_2UP_GHSL_M3_2020")
        varnames.append("GDP|PPP per capita")
        xr_datasets.append(ds_gdp_per_pop.sel(time=year))

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.15, wspace=0.15)

    for i, ds in enumerate(xr_datasets):
        da = ds[varnames[i]]
        da = da.where((da > 0) & (da != da.rio.nodata))  # set zero, negative, and nodata values to NaN
        da.name = names[i]

        print("plot_factors_GDP_POP:")
        print(f"\nDataset: {da.name}")
        print(f"masked: {np.ma.is_masked(da.values)}")
        print(f"da.rio.nodata: {da.rio.nodata}")
        print(f"da.attr nodata {da.attrs.get('nodata')}")
        print(f"da.encoding _FillValue: {da.encoding.get('_FillValue')}")
        print(f"da.attr _FillValue{da.attrs.get('_FillValue')}")
        ax_map = fig.add_subplot(gs[i], projection=ccrs.Mercator())
        plot_Mercator_projection_update(da, ax=ax_map, coarsen=coarsen, transform="log", title=f"{da.name}",
                                 cbar_shrink=0.6, cbar_aspect=20, cbar_pad=0.1)
    plt.show()
    fig_path = project_dir / "figures" / f"pop_gdp_ppp_grid_cf_{coarsen}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

def plot_Mercator_projection(da, *, ax=None, coarsen=12, transform="linear", show=False, title=None,
                             cbar_shrink=0.6, cbar_aspect=20, cbar_pad=0.05,
                             vmin=None, vmax=None, add_polygon=None):
    """
    Plot an xarray.DataArray on a Mercator map using xarray's .plot().
    - Creates no figure on import; only when called.
    - Supports "log" or "linear" transform for color scale.
    - Only calls plt.show() if show=True.
    - If ax is provided, draws into it (no new figure created).

    - cbar_shrink : float, default=0.6 --> scale factor for colorbar length (0.6 = 60% of original size)
    - cbar_aspect : int, default=20 --> ratio of colorbar length to width (higher = thinner)
    - cbar_pad : float, default=0.05 --> Distance between axes and colorbar

    Returns (fig, ax, mappable).

    """
    # Check and rename dimensions BEFORE coarsening

    # Rename lon/lat to x/y if needed (BEFORE coarsening)
    if "lon" in da.coords and "lat" in da.coords:
        da = da.rename({"lon": "x", "lat": "y"})
        print(f"Renamed lon/lat to x/y")
    elif "lon" in da.dims and "lat" in da.dims:
        da = da.rename({"lon": "x", "lat": "y"})
        print(f"Renamed lon/lat dimensions to x/y")

    # Verify we have x and y dimensions
    if "x" not in da.dims or "y" not in da.dims:
        raise ValueError(f"DataArray must have 'x' and 'y' dimensions (or 'lon' and 'lat'). Found: {da.dims}")

    # Now coarsen with correct dimension names
    da_coarsened = da.coarsen(x=coarsen, y=coarsen, boundary="trim").mean()

    # Normalise longitudes if needed
    if (da_coarsened.x > 180).any():
        da_coarsened = da_coarsened.assign_coords(x=((da_coarsened.x + 180) % 360) - 180).sortby("x")

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.Mercator())
        created_fig = True
    else:
        fig = ax.figure

    if transform == "log":
        #norm_plot = norm = mcolors.LogNorm()
        # Get valid positive values for log scale
        valid_data = da_coarsened.values[np.isfinite(da_coarsened.values) & (da_coarsened.values > 0)]
        if len(valid_data) == 0:
            print(f"Warning: No valid positive data for log transform, switching to linear")
            #norm_plot = mcolors.Normalize()
            norm_plot = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            _vmin = vmin if vmin is not None else np.percentile(valid_data, 2)
            _vmax = vmax if vmax is not None else np.percentile(valid_data, 98)
            if _vmin <= 0:
                _vmin = np.min(valid_data[valid_data > 0])
            norm_plot = mcolors.LogNorm(vmin=_vmin, vmax=_vmax)
    elif transform == "linear":
        # norm_plot = norm = mcolors.Normalize()
        norm_plot = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        print(f"Unknown transform: {transform}, defaulting to linear")
        norm_plot = norm = mcolors.Normalize()

    # Enhanced cbar_kwargs with size control
    cbar_kwargs = {
        "label": da_coarsened.name,
        "shrink": cbar_shrink,
        "aspect": cbar_aspect,
        "pad": cbar_pad
    }

    pm = da_coarsened.plot( #.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        cbar_kwargs=cbar_kwargs,
        add_colorbar=True,
        robust=True,
        rasterized=True,
        infer_intervals=True,
        norm=norm_plot
    )

    ax.coastlines(linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.right_labels = False
    gl.top_labels = False

        # Draw polygon boundary if provided
    if add_polygon is not None:
        if add_polygon.empty:
            print(f"Warning: polygon GeoDataFrame is empty, skipping polygon drawing.")
        else:
            ax.add_geometries(add_polygon.geometry, crs=ccrs.PlateCarree(), facecolor="none", edgecolor="red", linewidth=1.5, zorder=5)

    if title:
        ax.set_title(title)

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax, pm

def plot_Mercator_projection_original(da, *, ax=None, coarsen=12, transform="linear", show=False, title=None,
                             cbar_shrink=0.6, cbar_aspect=20, cbar_pad=0.05):
    """
    Plot an xarray.DataArray on a Mercator map using xarray's .plot().
    - Creates no figure on import; only when called.
    - Supports "log" or "linear" transform for color scale.
    - Only calls plt.show() if show=True.
    - If ax is provided, draws into it (no new figure created).

    - cbar_shrink : float, default=0.6 --> scale factor for colorbar length (0.6 = 60% of original size)
    - cbar_aspect : int, default=20 --> ratio of colorbar length to width (higher = thinner)
    - cbar_pad : float, default=0.05 --> Distance between axes and colorbar

    Returns (fig, ax, mappable).

    """
    da_coarsened = da.coarsen(x=coarsen, y=coarsen, boundary='trim').mean()

    # rename if coordinates are lon and lat
    if 'lon' in da_coarsened.coords and 'lat' in da_coarsened.coords:
        da_coarsened = da_coarsened.rename({'lon': 'x', 'lat': 'y'})
    # Normalise longitudes if needed
    if (da_coarsened.x > 180).any():
        da_coarsened = da_coarsened.assign_coords(x=((da_coarsened.x + 180) % 360) - 180).sortby("x")

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.Mercator())
        created_fig = True
    else:
        fig = ax.figure

    if transform == "log":
        norm_plot = norm = mcolors.LogNorm()
    elif transform == "linear":
        norm_plot = norm = mcolors.Normalize()
    else:
        print(f"Unknown transform: {transform}, defaulting to linear")
        norm_plot = norm = mcolors.Normalize()

    # Enhanced cbar_kwargs with size control
    cbar_kwargs = {
        "label": da_coarsened.name,
        "shrink": cbar_shrink,
        "aspect": cbar_aspect,
        "pad": cbar_pad
    }

    pm = da_coarsened.plot(
    #pm = da_coarsened.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        #cbar_kwargs={"label": da_coarsened.name},
        cbar_kwargs=cbar_kwargs,
        add_colorbar=True,
        robust=True,
        rasterized=True,
        infer_intervals=False,
        norm=norm_plot
    )

    ax.coastlines(linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.right_labels = False
    gl.top_labels = False


    if title:
        ax.set_title(title)

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax, pm
