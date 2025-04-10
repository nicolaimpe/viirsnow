from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pyproj
import rasterio
import rasterio.enums
import xarray as xr
from affine import Affine
from rasterio.features import rasterize

from grids import GeoGrid, dim_name, georef_data_array


def gdf_to_binary_mask(gdf: gpd.GeoDataFrame, grid: GeoGrid) -> xr.DataArray:
    gdf = gdf.to_crs(grid.crs)
    transform = grid.affine

    # Prepare geometries for rasterization
    shapes = [(geom, 1) for geom in gdf.geometry]  # Assign a value of 1 to all polygons

    # Rasterize
    binary_mask = rasterize(
        shapes,
        out_shape=(grid.height, grid.width),
        transform=transform,
        fill=0,  # Background value
        dtype="uint8",
    )

    dims = dim_name(grid.crs)
    binary_mask_data_array = xr.DataArray(
        data=binary_mask,
        dims=(dims[0], dims[1]),
        coords={dims[0]: (dims[0], grid.ycoords), dims[1]: (dims[1], grid.xcoords)},
    )
    out = georef_data_array(binary_mask_data_array, grid.crs)

    return out


def reproject_dataset(
    dataset: xr.Dataset,
    new_crs: pyproj.CRS,
    new_resolution: float | None = None,
    resampling: rasterio.enums.Resampling | None = None,
    nodata: int | float | None = None,
    transform: Affine | None = None,
    shape: Tuple[int, int] | None = None,
) -> xr.DataArray:
    # Wrap rioxarray reproject_dataset so that it's typed

    # Rioxarray reproject nearest by default
    dims = dim_name(new_crs)
    return dataset.rio.reproject(
        dst_crs=new_crs,
        resolution=new_resolution,
        resampling=resampling,
        transform=transform,
        nodata=nodata,
        shape=shape,
    ).rename({"x": dims[1], "y": dims[0]})


def reproject_using_grid(
    dataset: xr.Dataset,
    output_grid: GeoGrid,
    nodata: int | float | None = None,
    resampling_method: rasterio.enums.Resampling | None = None,
) -> xr.Dataset:
    dataset_reprojected = reproject_dataset(
        dataset=dataset,
        shape=output_grid.shape,
        transform=output_grid.affine,
        new_crs=output_grid.crs,
        resampling=resampling_method,
        nodata=nodata,
    )

    return dataset_reprojected


def extract_netcdf_coords_from_rasterio_raster(raster: rasterio.DatasetReader) -> Dict[str, npt.NDArray]:
    transform = raster.transform

    x_scale, x_off, y_scale, y_off = transform.a, transform.c, transform.e, transform.f
    # transform origin is half pixel away from first pixel point
    x0, y0 = x_off + x_scale / 2, y_off + y_scale / 2

    n_cols, n_rows = raster.width, raster.height
    # for GDAL the UL corener is the UL corner of the image while for xarray is the center of the upper left pixel
    # Compensate for it
    x_coord = np.arange(n_cols) * x_scale + x0
    y_coord = np.arange(n_rows) * y_scale + y0
    dims = dim_name(raster.crs)
    return {dims[0]: y_coord, dims[1]: x_coord}


def to_rioxarray(dataset: xr.Dataset) -> xr.DataArray:
    return dataset.rio.write_crs(dataset.data_vars["spatial_ref"].attrs["spatial_ref"])


def mask_dataarray_with_vector_file(data_array: xr.DataArray, roi_file: str, output_grid: GeoGrid, fill_value: int = 255):
    roi_mask = gdf_to_binary_mask(gdf=gpd.read_file(roi_file), grid=output_grid)
    masked = data_array.values * roi_mask.data_vars["binary_mask"].values
    masked[roi_mask.data_vars["binary_mask"].values == 0] = fill_value
    data_array[:] = masked
    data_array.rio.write_nodata(fill_value, inplace=True)
    return data_array
