# Clean internal netCDF4 attrs right after reading
from pathlib import Path
import xarray as xr

internal_attrs = ["_NCProperties", "_Netcdf4Coordinates", "ChunkSizes", "scale_factor", "add_offset"]

def clean_read_in_netcdf(ds:xr)->xr:
    for var in ds.data_vars:
        for attr in internal_attrs:
            ds[var].attrs.pop(attr, None)
    return ds
