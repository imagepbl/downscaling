@echo off

REM **********************************************
REM CREATE REGION RASTER FILE
REM **********************************************
REM pixi run python main.py --create_GADM_raster --model IMAGE_ScenarioMIP --resolution 6.00
REM pixi run python main.py --create_GADM_raster --model REMIND_ScenarioMIP --resolution 6.00

REM **********************************************
REM DOWNLOAD SCENARIOMIP DATA
REM **********************************************

REM pre: - account on IIASA database is needed
REM      - ixmp4 login <username>
REM      - give password
REM      - see: https://pyam-iamc.readthedocs.io/en/stable/api/iiasa.html
REM The <model_name> should be replaced with the actual model name, e.g., IMAGE_ScenarioMIP, REMIND_ScenarioMIP, etc.

REM Retrieve country and regions for R10 grouping from the IAMC common.yaml file and GADM level-0 country names
REM For each ScenarioMIP model used, copy country/region files from 'data/input/Models/' to 'data/input/Models/<model_name>/' as
REM - <model_name>_country_to_regions.csv
REM - <model_name>_region_numbers.csv
REM pixi run python -m downscaling.download_ScenarioMIP --download_IAMC_region_R10

REM Copy result to data/processed/models/<model_name>/ into the appropriate SSP directories.
REM pixi run python -m downscaling.download_ScenarioMIP --download_emissions --model IMAGE_ScenarioMIP
REM pixi run python -m downscaling.download_ScenarioMIP --download_emissions --model REMIND_ScenarioMIP

REM **********************************************
REM INPUT PROFILE
REM **********************************************
REM 'First round' (2UP, Wang, EDGAR)
REM 'Second round' (2UP, Murakami, EDGAR)
REM 'Third round' (2UP, Murakami, CEDS_CMIP7)
REM 'Fourth round' (Zhuang, Murakami, CEDS_CMIP7)
REM 'Fifth round' (COMPASS, COMPASS, CEDS_CMIP7)

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
REM pixi run python main.py --downscale_emissions --scenario "IMAGE 3.4_Low - SSP2.xlsx" --model IMAGE --profile second_round --emissions net
REM pixi run python main.py --downscale_emissions --scenario "IMAGE 3.4_Medium - SSP2.xlsx" --model IMAGE --profile second_round --emissions net

REM pixi run python main.py --plot --scenario ELV-SSP2-CP --model IMAGE --profile second_round --emissions net
REM pixi run python main.py --plot --scenario ELV-SSP2-1150F --model IMAGE --profile second_round --emissions net

REM Third round (2UP, Murakami, CEDS)
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile third_round --emissions net
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-1150F --model IMAGE --profile third_round --emissions net

REM pixi run python main.py --plot --scenario ELV-SSP2-CP --model IMAGE --profile third_round --emissions net
REM pixi run python main.py --plot --scenario ELV-SSP2-1150F --model IMAGE --profile third_round --emissions net

REM Fourth round (Zhuang, Murakami, CEDS_CMIP7)
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile fourth_round --emissions net
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-1150F --model IMAGE --profile fourth_round --emissions net

REM pixi run python main.py --plot --scenario ELV-SSP2-CP --model IMAGE --profile fourth_round --emissions net
REM pixi run python main.py --plot --scenario ELV-SSP2-1150F --model IMAGE --profile fourth_round --emissions net

REM Fifth round (COMPASS, COMPASS, CEDS_CMIP7)
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile fifth_round --emissions net
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-1150F --model IMAGE --profile fifth_round --emissions net

REM pixi run python main.py --plot --scenario ELV-SSP2-CP --model IMAGE --profile fifth_round --emissions net
REM pixi run python main.py --plot --scenario ELV-SSP2-1150F --model IMAGE --profile fifth_round --emissions net

REM GROSS EMISSIONS
REM pixi run python main.py --downscale_emissions --scenario IMAGE3.4_Medium_SSP2 --model IMAGE_ScenarioMIP --profile %profile% --emissions gross
REM pixi run python main.py --downscale_emissions --scenario IMAGE3.4_Low_SSP2 --model IMAGE_ScenarioMIP --profile %profile% --emissions gross


@echo on
REM **********************************************
REM PLOT
REM **********************************************

REM pixi run python main.py --plot --scenario ELV-SSP2-CP --model IMAGE --profile first_round --emissions net
REM pixi run python main.py --plot --scenario ELV-SSP2-1150F --model IMAGE --profile first_round --emissions net

REM pixi run python main.py --plot --scenario ELV-SSP2-CP --model IMAGE --profile second_round --emissions net
REM pixi run python main.py --plot --scenario ELV-SSP2-1150F --model IMAGE --profile second_round --emissions net

REM pixi run python main.py --plot --scenario ELV-SSP2-CP --model IMAGE --profile third_round --emissions net
REM pixi run python main.py --plot --scenario ELV-SSP2-1150F --model IMAGE --profile third_round --emissions net

REM pixi run python main.py --plot --scenario ELV-SSP2-CP --model IMAGE --profile fourth_round --emissions net
REM pixi run python main.py --plot --scenario ELV-SSP2-1150F --model IMAGE --profile fourth_round --emissions net



REM **********************************************
REM UPLOAD
REM **********************************************

REM pixi run python main.py --upload --scenario ELV-SSP2-CP --model IMAGE --profile second_round
REM pixi run python main.py --upload --scenario ELV-SSP2-1150F --model IMAGE --profile second_round
