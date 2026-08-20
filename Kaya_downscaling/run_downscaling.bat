REM TO DO: make possible to choose between create raster file and run downscaling population or emissions (including scenarios and net/gross emissions) and plot results. Also add option to run sensitivities.
@echo off
REM set profile=%1

REM **********************************************
REM CREATE REGION RASTER FILE
REM **********************************************
REM pixi run python main.py --create_GADM_raster --resolution 6.00
REM pixi run python main.py --create_GADM_raster --resolution 0.50

REM **********************************************
REM INPUT PROFILE
REM **********************************************
REM 'First round' (2UP, Wang, EDGAR)
REM 'Second round' (2UP, Murakami, EDGAR)
REM 'Third round' (2UP, Murakami, CEDS_CMIP7)
REM 'Fourth round' (Zhuang, Murakami, CEDS_CMIP7)
REM 'Fifth round' (COMPASS, COMPASS, CEDS_CMIP7)

REM POPULATION
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile %profile% --emissions net
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-1150F --model IMAGE --profile %profile% --emissions net

REM NET EMISSIONS
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile %profile% --emissions net
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-1150F --model IMAGE --profile %profile% --emissions net

REM GROSS EMISSIONS
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile %profile% --emissions gross
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-1150F --model IMAGE --profile %profile% --emissions gross

@echo on
REM **********************************************************************************************************************************
REM **********************************************
REM SENSITIVITIES
REM **********************************************

REM First round (2UP, Wang, EDGAR)
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile first_round --emissions net
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-1150F --model IMAGE --profile first_round --emissions net

REM Second round (2UP, Murakami, EDGAR)
pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile second_round --emissions net
pixi run python main.py --downscale_emissions --scenario ELV-SSP2-1150F --model IMAGE --profile second_round --emissions net

REM Second round (2UP, Murakami, EDGAR)
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile second_round_v2 --emissions net
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-1150F --model IMAGE --profile second_round_v2 --emissions net

REM Third round (2UP, Murakami, CEDS)
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile third_round --emissions net
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-1150F --model IMAGE --profile third_round --emissions net

REM Fourth round (Zhuang, Murakami, CEDS_CMIP7)
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile fourth_round --emissions net
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-1150F --model IMAGE --profile fourth_round --emissions net

REM Fifth round (COMPASS, COMPASS, CEDS_CMIP7)
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-CP --model IMAGE --profile fifth_round --emissions net
REM pixi run python main.py --downscale_emissions --scenario ELV-SSP2-1150F --model IMAGE --profile fifth_round --emissions net

@echo off
REM **********************************************
REM DOWNSCALE_POPULATION
REM **********************************************

REM pixi run python main.py --downscale_population --scenario ELV-SSP2-CP --model IMAGE --profile third_round
REM pixi run python main.py --downscale_population --scenario ELV-SSP2-1150F --model IMAGE --profile third_round

REM **********************************************
REM DOWNSCALE_GDP_PPP
REM **********************************************
REM pixi run python main.py --downscale_gdp_ppp --scenario ELV-SSP2-CP --model IMAGE --profile third_round

@echo off
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

REM pixi run python main.py --plot --scenario ELV-SSP2-CP --model IMAGE --profile fifth_round --emissions net
REM pixi run python main.py --plot --scenario ELV-SSP2-1150F --model IMAGE --profile fifth_round --emissions net

REM **********************************************
REM UPLOAD
REM **********************************************

REM pixi run python main.py --upload --scenario ELV-SSP2-CP --model IMAGE --profile second_round
REM pixi run python main.py --upload --scenario ELV-SSP2-1150F --model IMAGE --profile second_round
