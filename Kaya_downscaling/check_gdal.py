import os

print(os.environ.get("GDAL_DATA", "NOT SET"))
print(os.environ.get("PROJ_LIB", "NOT SET"))

print(os.path.exists(os.environ.get("GDAL_DATA", "")))
print(os.path.exists(os.environ.get("PROJ_LIB", "")))
