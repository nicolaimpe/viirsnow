from typing import Self, Tuple
from affine import Affine
import numpy as np
from rasterio.enums import Resampling
from pyproj import CRS
from rasterio.transform import from_origin
from dataclasses import dataclass
import xarray as xr


DEFAULT_CRS_PROJ = 32631
DEFAULT_CRS = DEFAULT_CRS_PROJ
OUTPUT_GRID_RES = 375  # m
OUTPUT_GRID_X0, OUTPUT_GRID_Y0 = 0, 5400000
OUPUT_GRID_X_SIZE, OUPUT_GRID_Y_SIZE = 2800, 2200
RESAMPLING = Resampling.nearest


class Grid:
    def __init__(self, resolution: float, x0: float, y0: float, width: int, height: int, crs: CRS | None = None) -> None:
        self.crs = crs
        self.resolution = resolution
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height

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
        return self.x0 + self.width * self.resolution

    @property
    def yend(self):
        return self.y0 - self.height * self.resolution

    @property
    def extent_llx_lly_urx_ury(self):
        return self.x0, self.yend, self.xend, self.y0

    @property
    def xcoords(self) -> np.array:
        return np.linspace(self.xmin, self.xmax, self.width)

    @property
    def ycoords(self) -> np.array:
        return np.linspace(self.ymax, self.ymin, self.height)

    @property
    def affine(self) -> Affine:
        return from_origin(self.x0, self.y0, self.resolution, self.resolution)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    # @classmethod
    # def extract_from_dataset(cls, dataset: xr.Dataset) -> Self:
    #     """Be very careful"""
    #     ds_crs = dataset.data_vars["spatial_ref"].attrs["spatial_ref"]
    #     dims = ("lat", "lon") if ds_crs.is_geographic else ("y", "x") if ds_crs.is_projected else None
    #     y_coords, x_coords = dataset.coords[dims[0]], dataset.coords[dims[1]]
    #     width, height = len(x_coords), len(y_coords)
    #     resolution = np.abs(x_coords[-1] - x_coords[0]) / (width - 1)
    #     return cls(
    #         crs=ds_crs,
    #         resolution=resolution,
    #         x0=x_coords[0] - resolution / 2,
    #         y0=y_coords[0] + resolution / 2,
    #         width=width,
    #         height=height,
    #     )


class DefaultGrid(Grid):
    def __init__(self) -> None:
        super().__init__(
            crs=CRS.from_epsg(DEFAULT_CRS),
            resolution=OUTPUT_GRID_RES,
            x0=OUTPUT_GRID_X0,
            y0=OUTPUT_GRID_Y0,
            width=OUPUT_GRID_X_SIZE,
            height=OUPUT_GRID_Y_SIZE,
        )


class DefaultGrid_1km(Grid):
    def __init__(self) -> None:
        super().__init__(
            crs=CRS.from_epsg(DEFAULT_CRS),
            resolution=1000,
            x0=OUTPUT_GRID_X0,
            y0=OUTPUT_GRID_Y0,
            width=1050,
            height=825,
        )
