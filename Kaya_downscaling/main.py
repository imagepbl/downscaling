import sys
import os
from pathlib import Path
import downscaling.IAM_spatial_model_maps as IAM_maps

"""
Configure GDAL and PROJ data directories for the active Python environment.

The environment root is derived from `sys.executable`, ensuring that both GDAL and PROJ use data files from the same environment (e.g. Pixi/Conda). The paths
to the GDAL and PROJ data folders are then constructed relative to this root(`Library/share/gdal` and `Library/share/proj`).

The environment variables `GDAL_DATA`, `PROJ_LIB`, and `PROJ_DATA` are set so that the underlying native libraries can locate their required resource files.
In addition, `pyproj.datadir.set_data_dir` is used to explicitly direct PROJ tothe correct data directory at runtime.
"""
env_root = os.path.dirname(sys.executable)

gdal_path = os.path.join(env_root, 'Library', 'share', 'gdal')
proj_path = os.path.join(env_root, 'Library', 'share', 'proj')

os.environ['GDAL_DATA'] = gdal_path
os.environ['PROJ_LIB'] = proj_path
os.environ['PROJ_DATA'] = proj_path

from pyproj import datadir
datadir.set_data_dir(proj_path)

print("Check GDAL and PROJ data directories:")
print("\tGDAL_DATA location:", gdal_path)
print("\tPROJ_LIB location:", proj_path)
print("\tproj.db exists:", os.path.exists(os.path.join(proj_path, 'proj.db')))
print("\tpyproj data dir:", datadir.get_data_dir())

import argparse
import downscaling.downscaling as downscaling

if __name__ == "__main__":
    '''

    -Process data ('copy' to run folder or 'no_copy)
    python run main.py --process copy --ssp_baseline SSP2
    python run main.py --process no_copy --ssp_baseline SSP2

    -Create GADM raster for countries
    python run main.py --create_GADM_raster --resolution 6.00

    -Compare to raster files
    pixi run python main.py --compare

    -Downscaling emissions to grid level
    **********************************************
    INPUT PROFILE
    **********************************************
    'First round' (2UP, Wang, EDGAR)
    'Second round' (2UP, Murakami, EDGAR)
    'Third round' (2UP, Murakami, CEDS_CMIP7)
    'Fourth round' (Zhuang, Murakami, CEDS_CMIP7)
    'Fifth round' (COMPASS, COMPASS, CEDS_CMIP7)

    --Downscale POPULATION
    pixi run python main.py --downscale_population --scenario ELV-SSP2-CP --model IMAGE --profile %profile% --emissions net
    pixi run python main.py --downscale_population --scenario ELV-SSP2-1150F --model IMAGE --profile %profile% --emissions net

    --Downscale NET EMISSIONS
    pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile %profile% --emissions net
    pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile first_round --emissions net
    pixi run python main.py --downscale_emissions --scenario ELV-SSP2-1150F --model IMAGE --profile  %profile% --emissions net

    --Downscale GROSS EMISSIONS
    pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile %profile% --emissions gross
    pixi run python main.py --downscale_emissions --scenario ELV-SSP2-1150F --model IMAGE --profile %profile% --emissions gross

    -Plot results
    python run main.py --plot --scenario ELV-SSP2-CP --model IMAGE --profile %profile% --emissions net
    python run main.py --plot --scenario ELV-SSP2-CP --model IMAGE --profile %profile% --emissions net --global_min 0 --global_max 100

    -Upload results to Google Earth Engine
    python run main.py --upload --scenario ELV-SSP2-CP --model IMAGE --profile %profile%

    '''
    project_dir = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(description="Downscaling emissions to grid level") # add_help=True by default
    parser.add_argument("--process", metavar="copy", choices=["copy", "no_copy"], help="process datasets and 'copy' to run folder or 'no_copy'")
    parser.add_argument("--ssp_baseline", type=str, help="baseline scenario from SSP")

    parser.add_argument("--create_GADM_raster", action="store_true", help="create GADM raster file for countries")
    parser.add_argument("--resolution", type=str, help="Resolution for GADM raster in minutes")

    parser.add_argument("--plot", action="store_true", help="plot results")

    parser.add_argument("--downscale_population", action="store_true", help="downscale population")
    parser.add_argument("--downscale_gdp_ppp", action="store_true", help="downscale GDP (PPP)")
    parser.add_argument("--downscale_emissions", action="store_true", help="downscale emissions")
    parser.add_argument("--scenario", type=str, help="Scenario to downscale for (e.g. ELV-SSP2-CP)")
    parser.add_argument("--model", type=str, help="Model from which scenario input is used (e.g. IMAGE, REMIND")
    parser.add_argument("--profile", type=str, help="Settings for input files")
    parser.add_argument("--emissions", type=str, help="net" or "gross")

    parser.add_argument("--global_min", type=str, help="minimum for plot range emissions")
    parser.add_argument("--global_max", type=str, help="maximum for plot range emissions")

    parser.add_argument("--upload", action="store_true", help="Upload results to Google Earth Engine")

    parser.add_argument("--compare", action="store_true", help="Compare two raster files")

    arguments = parser.parse_args()
    print(f"Arguments provided: {arguments}")
    if hasattr(arguments, 'process') and arguments.process is not None:
        if arguments.ssp_baseline is None:
            parser.error("--processing requires a SSP baseline scenario to be specified with --ssp_base")
        if arguments.process == "copy":
            downscaling.process_datasets(project_dir, arguments.ssp_baseline, copy=True)
        else:
            downscaling.process_datasets(project_dir, arguments.ssp_baseline, copy=False)
    if hasattr(arguments, 'create_GADM_raster') and arguments.create_GADM_raster is True:
        if arguments.resolution is None:
            parser.error("--create_GADM_raster requires a resolution to be specified with --resolution")
        IAM_maps.create_GADM_region_raster(project_dir, "IMAGE", float(arguments.resolution), True)
    if hasattr(arguments, 'downscale_population') and arguments.downscale_population is True:
        if arguments.scenario is None or arguments.profile is None:
            parser.error("--scenario requires a scenario to be specified and/or --profile requires a profile to be specified")
        downscaling.downscale_SE_data(project_dir, "Population", arguments.scenario, arguments.model, False, arguments.profile)
    if hasattr(arguments, 'downscale_gdp_ppp') and arguments.downscale_gdp_ppp is True:
        if arguments.scenario is None or arguments.profile is None:
            parser.error("--scenario requires a scenario to be specified and/or --profile requires a profile to be specified")
        downscaling.downscale_SE_data(project_dir, "GDP|PPP", arguments.scenario, arguments.model, True, arguments.profile)
    if hasattr(arguments, 'downscale_emissions') and arguments.downscale_emissions is True:
        if arguments.scenario is None or arguments.profile is None or arguments.emissions is None:
            parser.error("--scenario requires a scenario to be specified and/or --profile requires a profile to be specified")
        if arguments.emissions not in ["net", "gross"]:
            parser.error("--emissions requires a value of 'net' or 'gross'")
        elif arguments.emissions == "net":
            net_emissions = True
        else:
            net_emissions = False
        downscaling.downscale_emissions(project_dir, arguments.scenario, arguments.model, arguments.profile, net_emissions)
    if hasattr(arguments, 'plot') and arguments.plot is True:
        if arguments.scenario is None or arguments.profile is None:
            parser.error("--scenario requires a scenario to be specified and/or --profile requires a profile to be specified")
        if arguments.emissions not in ["net", "gross"]:
            parser.error("--emissions requires a value of 'net' or 'gross'")
        elif arguments.emissions == "net":
            net_emissions = True
        else:
            net_emissions = False
        if arguments.global_min is None or arguments.global_max is None:
            downscaling.plot_results(arguments.scenario, "IMAGE", arguments.profile, net_emissions, None, None)
        else:
            downscaling.plot_results(arguments.scenario, "IMAGE", arguments.profile, net_emissions, float(arguments.global_min), float(arguments.global_max))
    if hasattr(arguments, 'upload') and arguments.upload is True:
        if arguments.scenario is None or arguments.profile is None:
            parser.error("--upload requires a scenario to be specified and/or --profile requires a profile to be specified")
        downscaling.upload_to_GEE(arguments.scenario, "IMAGE", arguments.profile)
    if hasattr(arguments, 'compare') and arguments.compare is True:
        downscaling.compare_two_raster_files()

    # if no arguments, print message
    if not any(vars(arguments).values()):
        print("No arguments provided. Use -h or --help for more information.")

