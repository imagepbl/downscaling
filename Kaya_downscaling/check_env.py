import os
import sys

print("CONDA_PREFIX:", repr(os.environ.get("CONDA_PREFIX")))
print("PROJ_DATA:   ", repr(os.environ.get("PROJ_DATA")))
print("PROJ_LIB:    ", repr(os.environ.get("PROJ_LIB")))
print("GDAL_DATA:   ", repr(os.environ.get("GDAL_DATA")))
print("sys.prefix:  ", sys.prefix)

import pyproj
print("pyproj data dir:", pyproj.datadir.get_data_dir())

import rasterio
print("rasterio GDAL_DATA:", rasterio._env.get_gdal_data())

import pyogrio
print("pyogrio version:", pyogrio.__version__, pyogrio.__gdal_version_string__)
