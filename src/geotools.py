from typing import Tuple
import rasterio.enums
import xarray as xr
import numpy as np
import geopandas as gpd
import rasterio
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import numpy as np
import pyproj


def gdf_to_binary_mask(gdf: gpd.GeoDataFrame, out_resolution: float, out_crs: pyproj.CRS) -> xr.Dataset:
    gdf = gdf.to_crs(out_crs)
    # Define the output raster properties
    bounds = gdf.total_bounds  # Get the bounds of the shapefile
    minx, miny, maxx, maxy = bounds

    coord_x = np.arange(minx, maxx + out_resolution, out_resolution)
    coord_y = np.arange(miny, maxy + out_resolution, out_resolution)
    # Compute the raster shape
    width = len(coord_x)
    height = len(coord_y)

    # Define the transform
    transform = from_origin(minx, maxy, out_resolution, out_resolution)
    # Prepare geometries for rasterization
    shapes = [(geom, 1) for geom in gdf.geometry]  # Assign a value of 1 to all polygons

    # Rasterize
    binary_mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,  # Background value
        dtype="uint8",
    )

    dims = dim_name(out_crs)
    binary_mask_data_array = xr.DataArray(
        data=binary_mask,
        dims=(dims[0], dims[1]),
        coords={dims[0]: (dims[0], np.flip(coord_y)), dims[1]: (dims[1], coord_x)},
    )
    return georef_data_array(binary_mask_data_array, "binary_mask", crs=out_crs)


def dim_name(crs: pyproj.CRS) -> Tuple[str, str]:
    if crs.is_geographic:
        return ("lat", "lon")
    elif crs.is_projected:
        return ("y", "x")


def georef_data_array(data_array: xr.DataArray, data_array_name: str, crs: pyproj.CRS) -> xr.Dataset:
    """
    Turn a DataArray into a Dataset  for which the GDAL driver (GDAL and QGIS) is able to read the georeferencing
    https://github.com/pydata/xarray/issues/2288
    https://gis.stackexchange.com/questions/230093/set-projection-for-netcdf4-in-python
    """

    dims = dim_name(crs=crs)
    data_array.coords[dims[0]].attrs["axis"] = "Y"
    data_array.coords[dims[1]].attrs["axis"] = "X"
    data_array.attrs["grid_mapping"] = "spatial_ref"

    crs_variable = xr.DataArray(0)
    crs_variable.attrs["spatial_ref"] = crs.to_wkt()

    georeferenced_dataset = xr.Dataset({data_array_name: data_array, "spatial_ref": crs_variable})

    return georeferenced_dataset


def create_empty_grid_from_roi(
    roi: gpd.GeoDataFrame | gpd.GeoSeries,
    output_grid_resolution: float,
    fill_value: np.uint8,
    crs: pyproj.CRS,
    data_array_name: str,
    overwrite_x_origin: float | None = None,
    overwrite_y_origin: float | None = None,
) -> xr.Dataset:
    minx, miny, maxx, maxy = roi.total_bounds
    x_origin = minx if overwrite_x_origin is None else overwrite_x_origin
    y_origin = miny if overwrite_y_origin is None else overwrite_y_origin
    dims = dim_name(crs=crs)
    empty_data_array = xr.DataArray(
        fill_value,
        dims=dims,
        coords={
            dims[0]: np.arange(y_origin, maxy, output_grid_resolution),
            dims[1]: np.arange(x_origin, maxx, output_grid_resolution),
        },
    )
    empty_data_array = georef_data_array(empty_data_array, data_array_name, crs)
    return empty_data_array


def reproject_dataset(
    dataset: xr.Dataset,
    new_crs: pyproj.CRS,
    new_resolution: float | None = None,
    resampling: rasterio.enums.Resampling | None = None,
    fill_value: int | float = None,
) -> xr.Dataset:
    # Rioxarray reproject nearest by default
    dims = dim_name(new_crs)
    # rioxarray_dataset = dataset.rio.write_crs(dataset.data_vars["spatial_ref"].attrs["spatial_ref"])
    return (
        dataset.rio.write_crs(dataset.data_vars["spatial_ref"].attrs["spatial_ref"])
        .rio.reproject(
            dst_crs=new_crs,
            resolution=new_resolution,
            resampling=resampling,
            nodata=fill_value,
        )
        .rename({"x": dims[1], "y": dims[0]})
    )


def to_rioxarray(dataset: xr.Dataset) -> xr.Dataset:
    return dataset.rio.write_crs(dataset.data_vars["spatial_ref"].attrs["spatial_ref"])


def find_nearest_bounds_for_selection(dataset: xr.Dataset, other: xr.Dataset) -> Tuple[str, str, str, str]:
    """To be used very carefully."""
    dataset, other = to_rioxarray(dataset), to_rioxarray(other)
    if dataset.rio.crs != other.rio.crs or dataset.rio.crs is None:
        raise ValueError("Expected both Data Arrays to be georeferenced in the same CRS")
    dims_data_array = dim_name(dataset.rio.crs)
    x, y = dims_data_array[1], dims_data_array[0]

    xmin = dataset.coords[x].values[np.argmin(np.abs(dataset.coords[x].values - other.coords[x].min().values))]
    xmax = dataset.coords[x].values[np.argmin(np.abs(dataset.coords[x].values - other.coords[x].max().values))]
    ymin = dataset.coords[y].values[np.argmin(np.abs(dataset.coords[y].values - other.coords[y].min().values))]
    ymax = dataset.coords[y].values[np.argmin(np.abs(dataset.coords[y].values - other.coords[y].max().values))]
    return xmin, xmax, ymin, ymax
