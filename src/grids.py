from typing import Tuple

import numpy as np
import pyproj
import xarray as xr
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from products.georef import modis_crs

DEFAULT_CRS_PROJ = 32631
DEFAULT_CRS = DEFAULT_CRS_PROJ
OUTPUT_GRID_RES = 375  # m
OUTPUT_GRID_X0, OUTPUT_GRID_Y0 = 0, 5400000
OUPUT_GRID_X_SIZE, OUPUT_GRID_Y_SIZE = 2800, 2200
RESAMPLING = Resampling.nearest


class GeoGrid:
    def __init__(
        self, resolution: float, x0: float, y0: float, width: int, height: int, crs: CRS | None = None, name: str | None = None
    ) -> None:
        self.crs = crs
        self.resolution = resolution
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.name = name

    """
    x0,y0,xend,yend pixel corners
    xmin, ymin, xmax, ymax pixel centers
    width, height, number of pixel columns/rows

    x0,  y0 ------------------------------................--------------xend, y0
        |               |               |                   |               |
        |       .       |       .       | ................  |      .        |
        |   xmin,ymax   |               |                   |   xmax,ymax   |
        |               |               |                   |               |
        |---------------------------------.................------------------
        |               |               |                   |               |
        |               |               |                   |               |
        |

    
    """

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

    @property
    def xarray_coords(self) -> xr.Coordinates:
        # dims = dim_name(self.crs)
        dims = ("y", "x")
        return xr.Coordinates({dims[0]: self.ycoords, dims[1]: self.xcoords})

    def bounds_projected_to_epsg(self, to_epsg: int | str):
        transformer = Transformer.from_crs(crs_from=self.crs, crs_to=CRS.from_epsg(to_epsg), always_xy=True)
        return transformer.transform_bounds(*self.extent_llx_lly_urx_ury)

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


class UTM375mGrid(GeoGrid):
    def __init__(self) -> None:
        super().__init__(
            crs=CRS.from_epsg(DEFAULT_CRS),
            resolution=OUTPUT_GRID_RES,
            x0=OUTPUT_GRID_X0,
            y0=OUTPUT_GRID_Y0,
            width=OUPUT_GRID_X_SIZE,
            height=OUPUT_GRID_Y_SIZE,
            name="UTM_375m",
        )


class SIN375mGrid(GeoGrid):
    def __init__(self) -> None:
        super().__init__(
            crs=modis_crs,
            resolution=370.650173222222,
            x0=-420000,
            y0=5450000,
            width=3500,
            height=2600,
            name="SIN_375m",
        )


class UTM1kmGrid(GeoGrid):
    def __init__(self) -> None:
        super().__init__(
            crs=CRS.from_epsg(DEFAULT_CRS),
            resolution=1000,
            x0=OUTPUT_GRID_X0,
            y0=OUTPUT_GRID_Y0,
            width=1050,
            height=825,
            name="UTM_1km",
        )


# def dim_name(crs: pyproj.CRS) -> Tuple[str, str]:
#     if crs.is_geographic:
#         return ("lat", "lon")
#     elif crs.is_projected:
#         return ("y", "x")


def georef_netcdf(data_array: xr.DataArray | xr.Dataset, crs: pyproj.CRS) -> xr.Dataset | xr.Dataset:
    """
    The strict minimum to georeference in netCDF convention

    Turn a DataArray into a Dataset  for which the GDAL driver (GDAL and QGIS) is able to read the georeferencing
    https://github.com/pydata/xarray/issues/2288
    https://gis.stackexchange.com/questions/230093/set-projection-for-netcdf4-in-python
    """

    # dims = dim_name(crs=crs)
    data_array.coords["y"].attrs["axis"] = "Y"
    data_array.coords["x"].attrs["axis"] = "X"
    data_array.attrs["grid_mapping"] = "spatial_ref"

    georeferenced = data_array.assign_coords(coords={"spatial_ref": 0})
    georeferenced.coords["spatial_ref"].attrs["spatial_ref"] = crs.to_wkt()

    return georeferenced


def georef_netcdf_rioxarray(data_array: xr.DataArray | xr.Dataset, crs: pyproj.CRS) -> xr.Dataset | xr.DataArray:
    """
    Turn a DataArray into a Dataset  for which the GDAL driver (GDAL and QGIS) is able to read the georeferencing
    https://github.com/pydata/xarray/issues/2288
    https://gis.stackexchange.com/questions/230093/set-projection-for-netcdf4-in-python
    """

    return data_array.rio.write_crs(crs).rio.write_coordinate_system()
