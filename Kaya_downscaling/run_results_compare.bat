REM Compare harmonised .tif between second and third version		
pixi run python main.py --compare_tifs --dir1 "Z:/cold_data_storage/users/roelfsemam/downscaling_share/data_second_version/data/processed/base_run/IMAGE_ScenarioMIP_IMAGE 3.4_Medium - SSP2_conv_year_2150_net" ^
                                     --dir2 "data/processed/base_run_downscale_SE/IMAGE_ScenarioMIP_IMAGE 3.4_Medium - SSP2_conv_year_2150_net" ^
                                     --filename "Emissions_CO2_Excl_shipping_aviation_AFOLU_harmonised_IMAGE_ScenarioMIP_IMAGE 3.4_Medium - SSP2" ^
                                     --label "second_third_harm_tif"

REM Compare unharmonised .nc between second and third version
pixi run python main.py --compare_ncs --dir1 "Z:/cold_data_storage/users/roelfsemam/downscaling_share/data_second_version/data/processed/base_run/IMAGE_ScenarioMIP_IMAGE 3.4_Medium - SSP2_conv_year_2150_net" ^
                                      --dir2 "data/processed/base_run_downscale_SE/IMAGE_ScenarioMIP_IMAGE 3.4_Medium - SSP2_conv_year_2150_net" ^
									  --filename1 "Emissions_CO2_Excl_shipping_aviation_AFOLU_unharmonised_SSP2.nc" ^
								      --filename2 "Emissions_CO2_Excl_shipping_aviation_AFOLU_unharmonised_SSP2.nc" ^
                                      --label "second_third_unharm"

REM Compare harmonised .nc between second and third version									  
pixi run python main.py --compare_ncs --dir1 "Z:/cold_data_storage/users/roelfsemam/downscaling_share/data_second_version/data/processed/base_run/IMAGE_ScenarioMIP_IMAGE 3.4_Medium - SSP2_conv_year_2150_net" ^
                                     --dir2 "data/processed/base_run_downscale_SE/IMAGE_ScenarioMIP_IMAGE 3.4_Medium - SSP2_conv_year_2150_net" ^
                                     --filename1 "Emissions_CO2_Excl_shipping_aviation_AFOLU_harmonised_SSP2.nc" ^
									  --filename2 "Emissions_CO2_Excl_shipping_aviation_AFOLU_harmonised_SSP2.nc" ^
                                      --label "second_third_harm_nc"

REM Compare unharmonised and harmonised .nc from same run
pixi run python main.py --compare_ncs --dir1 "Z:/cold_data_storage/users/roelfsemam/downscaling_share/data_second_version/data/processed/base_run/IMAGE_ScenarioMIP_IMAGE 3.4_Medium - SSP2_conv_year_2150_net" ^
                                     --dir2 "Z:/cold_data_storage/users/roelfsemam/downscaling_share/data_second_version/data/processed/base_run/IMAGE_ScenarioMIP_IMAGE 3.4_Medium - SSP2_conv_year_2150_net" ^
                                     --filename1 "Emissions_CO2_Excl_shipping_aviation_AFOLU_harmonised_SSP2.nc" ^
									--filename2 "Emissions_CO2_Excl_shipping_aviation_AFOLU_unharmonised_SSP2.nc"	^
									--label "third_unharm_harm_nc"									  