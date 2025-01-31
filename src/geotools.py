from typing import Dict, Tuple
from affine import Affine
import rasterio.enums
import xarray as xr
import numpy as np
import geopandas as gpd
import rasterio
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import pyproj
from grids import Grid
import numpy.typing as npt


def gdf_to_binary_mask(gdf: gpd.GeoDataFrame, grid: Grid) -> xr.Dataset:
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
        coords={dims[0]: (dims[0], np.flip(grid.ycoords)), dims[1]: (dims[1], grid.xcoords)},
    )
    out = georef_data_array(binary_mask_data_array, "binary_mask", crs=grid.crs)

    return out


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
    nodata: int | float = None,
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


"""Some functions in drafts.ipynb to push before bad stuff happens

from pyresample import geometry,AreaDefinition, kd_tree
from geotools import georef_data_array
from grids import Grid
from fractional_snow_cover import nasa_ndsi_snow_cover_to_fraction
import xarray as xr
import numpy as np
from pyproj import CRS

#output_grid = Grid(crs=CRS.from_epsg(32631),resolution=375, x0=0, y0=5400000,width=2800, height=2200)
PROJ4_MODIS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
modis_crs = CRS.from_proj4(PROJ4_MODIS)
output_grid = Grid(crs=modis_crs,resolution=370.65017322, x0=0, y0=5559752.598333,width=3000, height=3000)

fn = 'VNP10.A2023340.1200.002.2023340210716.nc'
l2_geoloc = xr.open_dataset(f'/home/imperatoren/work/VIIRS_S2_comparison/data/V10/{fn}',group='/GeolocationData')
l2_data = xr.open_dataset(f'/home/imperatoren/work/VIIRS_S2_comparison/data/V10/{fn}',group='/SnowData').data_vars['NDSI_Snow_Cover']

lonsModif = np.ma.masked_array(l2_geoloc.data_vars['longitude'])	
latsModif = np.ma.masked_array(l2_geoloc.data_vars['latitude'])
swath_def = geometry.SwathDefinition(lons=lonsModif, lats=latsModif)

area_def = AreaDefinition(area_id='France',description='France CMS bounding box', proj_id='France',projection=output_grid.crs,width=output_grid.width, height=output_grid.height,area_extent=output_grid.extent_llx_lly_urx_ury)
reprojected_l2_data = kd_tree.resample_gauss(source_geo_def=swath_def,sigmas=25000, data=l2_data.values, target_geo_def=area_def, radius_of_influence=100000, fill_value=254, nprocs=1)

reprojected_l2_data = nasa_ndsi_snow_cover_to_fraction(reprojected_l2_data)
reprojected_l2_data_array = xr.DataArray(data=reprojected_l2_data, coords={'y': np.flip(output_grid.ycoords), 'x': output_grid.xcoords})

reprojected_l2_dataset = georef_data_array(reprojected_l2_data_array, data_array_name='snow_cover', crs=output_grid.crs)
reprojected_l2_dataset.to_netcdf('../output_folder/nasa_l2_reprojected_sin_gauss_fsc.nc')

---------------------------------------------------------------------------------------------------------
import pyproj
from geotools import georef_data_array, dim_name, reproject_dataset
from fractional_snow_cover import nasa_ndsi_snow_cover_to_fraction
import xarray as xr

filepath = '/home/imperatoren/work/VIIRS_S2_comparison/data/V10A1/VNP10A1/VNP10A1.A2023340.h18v04.002.2023341095458.h5'
PROJ4_MODIS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
#PROJ4_VIIRS="+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
modis_crs = pyproj.CRS.from_proj4(PROJ4_MODIS)
dims = dim_name(crs=modis_crs)
dataset_grid = xr.open_dataset(filepath, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D", engine="netcdf4")
ndsi_snow_cover = xr.open_dataset(
    filepath, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields", engine="netcdf4"
).data_vars["NDSI_Snow_Cover"]
y_coords = dataset_grid.coords["YDim"].values- 375/2
x_coords = dataset_grid.coords["XDim"].values+ 375/2

ndsi_snow_cover = ndsi_snow_cover.rename({"XDim": dims[1], "YDim": dims[0]}).assign_coords(
    coords={dims[0]: y_coords , dims[1]: x_coords}
)

georef_dataset = georef_data_array(data_array=ndsi_snow_cover, data_array_name="snow_cover", crs=modis_crs)

georef_dataset.data_vars["snow_cover"][:] = nasa_ndsi_snow_cover_to_fraction(
    georef_dataset.data_vars["snow_cover"].values
)

georef_dataset.to_netcdf('../output_folder/test_v10_geoloc_modif_half_pixel.nc')

"""
