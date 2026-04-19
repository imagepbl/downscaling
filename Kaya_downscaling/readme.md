
# 1. Pixi Environment Guide

This workspace uses pixi, which is a package manager and the README provides instructions for setting up and using pixi, a powerful environment and package manager.

To use this, do at least these things
- Install pixi (see 'Installation' below)
- Execute command 'pixi install'
- Use .pixi\envs\default\python.exe as your interpreter

Now, running a python script can best be done with 'pixi python <file>.py
Also, installing a new conda pacakge with 'pixi add <conda package>
    and a pip package with 'pixi add --pypi <pypi package>'

The `pixi.toml` file defines the project configuration:

## Additional Resources

- [Pixi Documentation](https://pixi.sh/docs)
- [GitHub Repository](https://github.com/prefix-dev/pixi)


# 2. IMAGE modelling

The country_to_regions.csv file that gives the model regions and included countries and image_region_numbers.csv that maps the region names and numbers can be found at:
https://github.com/imagepbl/geospatial-metadata/tree/main/mappings
