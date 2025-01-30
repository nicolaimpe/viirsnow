from typing import Tuple
from affine import Affine
import numpy as np
from rasterio.enums import Resampling
from pyproj import CRS
from rasterio.transform import from_origin
from dataclasses import dataclass

DEFAULT_CRS_PROJ = 32631
DEFAULT_CRS = DEFAULT_CRS_PROJ
OUTPUT_GRID_RES = 250  # m
OUTPUT_GRID_X0, OUTPUT_GRID_Y0 = 0, 5400000
OUPUT_GRID_X_SIZE, OUPUT_GRID_Y_SIZE = 4200, 3300
RESAMPLING = Resampling.nearest


@dataclass
class Grid:
    crs: CRS
    resolution: float
    x0: float
    y0: float
    width: int
    height: int

    @property
    def xmin(self):
        return self.x0 + self.resolution / 2

    @property
    def ymax(self):
        return self.y0 - self.resolution / 2

    @property
    def xmax(self):
        return self.xmin + (self.width - 1) * self.resolution

    @property
    def ymin(self):
        return self.ymax - (self.height - 1) * self.resolution

    @property
    def xend(self):
        return self.xmax + self.resolution / 2

    @property
    def yend(self):
        return self.ymin - self.resolution / 2

    @property
    def extent_llx_lly_urx_ury(self):
        return self.x0, self.yend, self.xend, self.y0

    @property
    def xcoords(self) -> np.array:
        return np.arange(self.xmin, self.xmax + self.resolution, self.resolution)

    @property
    def ycoords(self) -> np.array:
        return np.arange(self.ymin, self.ymax + self.resolution, self.resolution)

    @property
    def affine(self) -> Affine:
        return from_origin(self.x0, self.y0, self.resolution, self.resolution)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)


@dataclass
class DefaultGrid(Grid):
    crs = CRS.from_epsg(DEFAULT_CRS)
    resolution = OUTPUT_GRID_RES
    x0 = OUTPUT_GRID_X0
    y0 = OUTPUT_GRID_Y0
    width = OUPUT_GRID_X_SIZE
    height = OUPUT_GRID_Y_SIZE
