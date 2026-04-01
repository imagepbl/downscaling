"""
Project runtime setup.

Ensures GDAL / PROJ can always find proj.db,
independent of platform, drive letter, or Pixi location.
"""

from pathlib import Path
import os
from pyproj import datadir


def configure_proj():
    # Get PROJ data directory from pyproj
    proj_path = Path(datadir.get_data_dir())

    # Safety check (helps debugging if env is broken)
    proj_db = proj_path / "proj.db"
    if not proj_db.exists():
        raise RuntimeError(
            f"PROJ database not found at: {proj_db}"
        )

    # Set environment variable for GDAL
    os.environ["PROJ_LIB"] = str(proj_path)


# Run automatically on import
configure_proj()
