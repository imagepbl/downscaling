profile --> combination of population, gdp, and emissions source files

run_downscaling.bat --> enables to run different profiles

main.py --> enables running
            - Process data --> processes GIS and IAM data to convert them into the format used for the analysis
            - Create GADM raster for countries --> creates a raster file based on the downloaded GADM polygons,
                this is used to determine the country for each grid (resolution in minutes needs to be defined manually
                as for example '0.50' (0.5 minute), or '6.00' for 6 minutes)
            - Compare to raster files --> compares raster files on a few characteristics
            - Downscalign population to grid level based on selected profile
            - Downscaling emissions to grid level --> downscales emissions based on selected profile
                                                      - net emissions --> including negative emissions
                                                      - postiive emissions --> excluding negative emissions
            - Plot results
            - Upload results to Google Earth Engine

Downscalig in main.py uses different settings files:
- settings_data_locations.json --> contains directories where GIS data is stored, and where the program 'R' is located on disk
- settings_downscaling_cities.py
- settings_downscaling.py --> defines the profiles (data sources for POP, GDP, and EM), defines process/check flags
- settings_models.json --> settings such as unit conversions and file locations for individual IAMs/models (e.g. IMAGE)
