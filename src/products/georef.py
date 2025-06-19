# Hardcode some parameters
from pyproj import CRS

PROJ4_MODIS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
MODIS_CRS = CRS.from_proj4(PROJ4_MODIS)
