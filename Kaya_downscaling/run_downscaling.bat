@echo off

REM **********************************************

REM **********************************************
REM DOWNLOAD SCENARIOMIP DATA
REM **********************************************

REM pre: - account on IIASA database is needed
REM      - ixmp4 login <username>
REM      - give password
REM      - see: https://pyam-iamc.readthedocs.io/en/stable/api/iiasa.html
REM The <model_name> should be replaced with the actual model name, e.g., IMAGE_ScenarioMIP, REMIND_ScenarioMIP, etc.

REM 1. Download ScenarioMIP emissions, population, and GDP data for the specified model
REM Copy result to data/processed/models/<model_name>/ into the appropriate SSP directories.
REM pixi run python main.py --download_emissions --model IMAGE_ScenarioMIP
REM pixi run python main.py --download_emissions --model REMIND_ScenarioMIP

REM 2. Retrieve country and regions for R10 grouping from the IAMC common.yaml file and GADM level-0 country names
REM For each ScenarioMIP model used, copy country/region files from 'data/input/Models/' to 'data/input/Models/<model_name>/' as
REM - <model_name>_country_to_regions.csv
REM - <model_name>_region_numbers.csv
REM pixi run python main.py --download_regions

REM CREATE REGION RASTER FILE
REM **********************************************
REM Uses thes copied versions (to appropriate locations): <model_name>_region_numbers.csv and <model_name>_country_to_regions.csv
REM pixi run python main.py --create_GADM_raster --model IMAGE_ScenarioMIP --resolution 6.00
REM pixi run python main.py --create_GADM_raster --model REMIND_ScenarioMIP --resolution 6.00

REM CREATE URBAN CLASSIFICATION FILE
REM **********************************************
REM This will create a file with the urban classification for each region.
REM pixi run python main.py --process_urban_classification

@echo off
REM **********************************************
REM PROCESS GRID DATA
REM **********************************************

REM pixi run python main.py --process_grid_data_source --driver Emissions    --source EDGAR       --version 2024
REM pixi run python main.py --process_grid_data_source --driver Emissions    --source CEDS_CMIP7  --version 2025_04_18

REM pixi run python main.py --process_grid_data_source --driver Population   --source 2UP         --version GHSL_2024_M3      --ssp_baseline SSP2
REM pixi run python main.py --process_grid_data_source --driver GDP_PPP      --source Murakami    --version version_2021_1    --ssp_baseline SSP2
REM pixi run python main.py --process_grid_data_source --driver Population   --source Zhuang      --version version_1         --ssp_baseline SSP2
REM pixi run python main.py --process_grid_data_source --driver Population   --source COMPASS     --version version_2         --ssp_baseline SSP2
REM pixi run python main.py --process_grid_data_source --driver GDP_PPP      --source COMPASS     --version version_2         --ssp_baseline SSP2

REM pixi run python main.py --process_grid_data_source --driver Population   --source 2UP         --version GHSL_2024_M3      --ssp_baseline SSP1
REM pixi run python main.py --process_grid_data_source --driver GDP_PPP      --source Murakami    --version version_2021_1    --ssp_baseline SSP1
REM pixi run python main.py --process_grid_data_source --driver Population   --source Zhuang      --version version_1         --ssp_baseline SSP1
REM pixi run python main.py --process_grid_data_source --driver Population   --source COMPASS     --version version_2         --ssp_baseline SSP1
REM pixi run python main.py --process_grid_data_source --driver GDP_PPP      --source COMPASS     --version version_2         --ssp_baseline SSP1

REM pixi run python main.py --process_grid_data_source --driver Population   --source 2UP         --version GHSL_2024_M3      --ssp_baseline SSP3
REM pixi run python main.py --process_grid_data_source --driver GDP_PPP      --source Murakami    --version version_2021_1    --ssp_baseline SSP3
REM pixi run python main.py --process_grid_data_source --driver Population   --source Zhuang      --version version_1         --ssp_baseline SSP3
REM pixi run python main.py --process_grid_data_source --driver Population   --source COMPASS     --version version_2         --ssp_baseline SSP3
REM pixi run python main.py --process_grid_data_source --driver GDP_PPP      --source COMPASS     --version version_2         --ssp_baseline SSP3

@echo on
REM **********************************************************************************************************************************
REM **********************************************
REM DOWNSCALE EMISSIONS
REM **********************************************

REM NET EMISSIONS

REM Second round (2UP, Murakami, EDGAR)
REM Base run
REM pixi run python main.py --downscale_emissions --scenario "IMAGE 3.4_Medium - SSP2" --model IMAGE_ScenarioMIP --profile base_run --ssp_baseline SSP2 --emissions net
REM pixi run python main.py --downscale_emissions --scenario "IMAGE 3.4_Low - SSP2" --model IMAGE_ScenarioMIP --profile base_run --ssp_baseline SSP2 --emissions net

REM Senstivity 1 (baseline SSP1)
REM pixi run python main.py --downscale_emissions --scenario "IMAGE 3.4_Medium - SSP1" --model IMAGE_ScenarioMIP --profile base_run --ssp_baseline SSP1 --emissions net
REM pixi run python main.py --downscale_emissions --scenario "IMAGE 3.4_Low - SSP1" --model IMAGE_ScenarioMIP --profile base_run --ssp_baseline SSP1 --emissions net

REM Senstivity 2 (model REMIND_ScenarioMIP)
REM pixi run python main.py --downscale_emissions --scenario "REMIND-MAgPIE 3.5-4.11_Medium - SSP2" --model REMIND_ScenarioMIP --profile base_run --ssp_baseline SSP2 --emissions net
REM pixi run python main.py --downscale_emissions --scenario "REMIND-MAgPIE 3.5-4.11_Low - SSP2" --model REMIND_ScenarioMIP --profile base_run --ssp_baseline SSP2 --emissions net

REM Senstivity 3 (source emissions: CEDS_CMIP7)
pixi run python main.py --downscale_emissions --scenario "IMAGE 3.4_Medium - SSP2" --model IMAGE_ScenarioMIP --profile sensitivity_3 --ssp_baseline SSP2 --emissions net
REM pixi run python main.py --downscale_emissions --scenario "IMAGE 3.4_Medium - SSP2" --model IMAGE_ScenarioMIP --profile sensitivity_3 --ssp_baseline SSP2 --emissions net

REM Senstivity 4 (source population: COMPASS)
REM pixi run python main.py --downscale_emissions --scenario "IMAGE 3.4_Medium - SSP2" --model REMIND_ScenarioMIP --profile sensitivity_4 --ssp_baseline SSP2 --emissions net
REM pixi run python main.py --downscale_emissions --scenario "IMAGE 3.4_Medium - SSP2" --model IMAGE_ScenarioMIP --profile sensitivity_4 --ssp_baseline SSP2 --emissions net

REM Senstivity 5 (source GDP: COMPASS)
REM pixi run python main.py --downscale_emissions --scenario "IMAGE 3.4_Medium - SSP2" --model IMAGE_ScenarioMIP --profile sensitivity_5 --ssp_baseline SSP2 --emissions net
REM pixi run python main.py --downscale_emissions --scenario "IMAGE 3.4_Medium - SSP2" --model IMAGE_ScenarioMIP --profile sensitivity_5 --ssp_baseline SSP2 --emissions net

REM GROSS EMISSIONS
REM pixi run python main.py --downscale_emissions --scenario "IMAG 3.4_Medium - SSP2" --model IMAGE_ScenarioMIP --profile base_run --ssp_baseline SSP2 --emissions gross
REM pixi run python main.py --downscale_emissions --scenario "IMAG 3.4_Low - SSP2" --model IMAGE_ScenarioMIP --profile base_run --ssp_baseline SSP2 --emissions gross

@echo off
REM **********************************************
REM PLOT
REM **********************************************

REM pixi run python main.py --plot --scenario "IMAGE 3.4_Medium - SSP2" --model IMAGE_ScenarioMIP --profile base_run --emissions net
REM pixi run python main.py --plot --scenario "IMAGE 3.4_Low - SSP2" --model IMAGE_ScenarioMIP --profile base_run --emissions net

REM **********************************************
REM UPLOAD
REM **********************************************

REM pixi run python main.py --upload --scenario "IMAGE 3.4_Medium - SSP2" --model IMAGE_ScenarioMIP --profile base_run
REM pixi run python main.py --upload --scenario "IMAGE 3.4_Low - SSP2" --model IMAGE_ScenarioMIP --profile base_run
