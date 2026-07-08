from pathlib import Path
from tools import convert_GIS

import os
from pathlib import Path

from pyproj import datadir
proj_path = datadir.get_data_dir()
os.environ["PROJ_LIB"] = datadir.get_data_dir()
os.environ["PROJ_DATA"] = proj_path

convert_GIS.gadm_levels_to_csv(Path("Z:/cold_data_storage/users/roelfsemam/surdrive_UU/NSA/Downscaling_share/data_downscaling/GADM/geopackage/gadm_410-levels.gpkg"), Path("data/processed/GADM"))
