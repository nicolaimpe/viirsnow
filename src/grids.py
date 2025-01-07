from typing import Tuple
from affine import Affine
import numpy as np
from rasterio.enums import Resampling
from pyproj import CRS
from rasterio.transform import from_origin

DEFAULT_CRS_PROJ = 32631
DEFAULT_CRS = DEFAULT_CRS_PROJ
OUTPUT_GRID_RES = 250  # m
OUTPUT_GRID_X0, OUTPUT_GRID_Y0 = 0, 5400000
OUPUT_GRID_X_SIZE, OUPUT_GRID_Y_SIZE = 4200, 3300
RESAMPLING = Resampling.nearest


class DefaultGrid:
    def __init__(self) -> None:
        self.crs = CRS.from_epsg(DEFAULT_CRS)
        self.resolution = OUTPUT_GRID_RES
        self.xmin = OUTPUT_GRID_X0 + OUTPUT_GRID_RES / 2
        self.ymax = OUTPUT_GRID_Y0 + OUTPUT_GRID_RES / 2
        self.width = OUPUT_GRID_X_SIZE
        self.height = OUPUT_GRID_Y_SIZE

    @property
    def xmax(self):
        return self.xmin + self.width * self.resolution

    @property
    def ymin(self):
        return self.ymax - self.height * self.resolution

    @property
    def xcoords(self) -> np.array:
        return np.arange(self.xmin, self.xmax, self.resolution)

    @property
    def ycoords(self) -> np.array:
        return np.arange(self.ymin, self.ymax, self.resolution)

    @property
    def affine(self) -> Affine:
        return from_origin(self.xmin - OUTPUT_GRID_RES / 2, self.ymax - OUTPUT_GRID_RES / 2, self.resolution, self.resolution)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)
