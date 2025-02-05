from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr

from geotools import dim_name, extract_netcdf_coords_from_rasterio_raster, gdf_to_binary_mask, georef_data_array
from grids import Grid
from logger_setup import default_logger as logger
from products.classes import METEOFRANCE_CLASSES, NASA_CLASSES
from products.georef import modis_crs
from reprojections import reprojection_l3_nasa_to_grid


def create_spatial_l3_nasa_composite(
    day_files: List[str], output_grid: Grid | None = None, roi_file: str | None = None
) -> xr.Dataset | None:
    day_data_arrays = []
    dims = dim_name(crs=modis_crs)
    for filepath in day_files:
        # try:
        logger.info(f"Processing product {Path(filepath).name}")

        product_grid_data_variable = xr.open_dataset(filepath, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D", engine="netcdf4")
        bin_size = xr.open_dataset(filepath, engine="netcdf4").attrs["CharacteristicBinSize"]
        nasa_l3_grid = Grid(
            resolution=bin_size,
            x0=product_grid_data_variable.coords["XDim"][0].values,
            y0=product_grid_data_variable.coords["YDim"][0].values,
            width=len(product_grid_data_variable.coords["XDim"]),
            height=len(product_grid_data_variable.coords["YDim"]),
        )
        ndsi_snow_cover = xr.open_dataset(
            filepath, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields", engine="netcdf4"
        ).data_vars["NDSI_Snow_Cover"]

        ndsi_snow_cover = ndsi_snow_cover.rename({"XDim": dims[1], "YDim": dims[0]}).assign_coords(
            coords={dims[0]: nasa_l3_grid.ycoords, dims[1]: nasa_l3_grid.xcoords}
        )

        day_data_arrays.append(georef_data_array(data_array=ndsi_snow_cover, data_array_name="NDSI_Snow_Cover", crs=modis_crs))

    merged_day_dataset = xr.combine_by_coords(day_data_arrays, data_vars="minimal").astype(np.uint8)
    # merged_day_dataset.data_vars["NDSI_Snow_Cover"].attrs.pop("valid_range")
    # merged_day_dataset.to_netcdf("./output_folder/test_merged.nc")

    if output_grid is None:
        raise NotImplementedError(
            "Output grid must be specified or a way to extract the output grid from the NASA products has to be implemented."
        )

    day_dataset_reprojected = reprojection_l3_nasa_to_grid(nasa_dataset=merged_day_dataset, output_grid=output_grid)

    if roi_file is not None:
        roi_mask = gdf_to_binary_mask(
            gdf=gpd.read_file(roi_file),
            grid=output_grid,
        )

        masked = day_dataset_reprojected.data_vars["NDSI_Snow_Cover"].values * roi_mask.data_vars["binary_mask"].values
        masked[roi_mask.data_vars["binary_mask"].values == 0] = NASA_CLASSES["fill"][0]
        day_dataset_reprojected.data_vars["NDSI_Snow_Cover"][:] = masked

    # Apparently need to pop this attribute for correct encoding...not like it took me two hours to understand this :')
    # In practice, when a valid range attribute is encoded, a GDal driver reading the NetCDF will set all values outside this
    # range to NaN.
    # Since valid range in the V10A1 collection is {0,100}, i.e. the NDSI range, all other pixels (clouds, lakes etc.) are set to NaN
    # and that's not useful for the anamysis
    day_dataset_reprojected.data_vars["NDSI_Snow_Cover"].attrs.pop("valid_range")
    # If we want not to encode the fill value like nodata
    # reprojected.data_vars["snow_cover"].attrs.pop("_FillValue")

    return day_dataset_reprojected


def create_temporal_l2_composite_meteofrance(daily_files: List[str]) -> xr.Dataset:
    day_dataset = georef_data_array(
        xr.DataArray(day_data.astype(np.uint8), coords=extract_netcdf_coords_from_rasterio_raster(first_image_raster)),
        data_array_name="snow_cover",
        crs=first_image_raster.crs,
    )

    return day_dataset


def create_temporal_l2_composite_meteofrance(daily_files: List[str]) -> xr.Dataset:
    logger.info(f"Reading file {daily_files[0]}")
    first_image_raster = rasterio.open(daily_files[0])
    day_data = first_image_raster.read(1)

    for day_file in daily_files[1:]:
        logger.info(f"Reading file {day_file}")
        new_acquisition = rasterio.open(day_file).read(1)

        no_data_mask = day_data == METEOFRANCE_CLASSES["nodata"]
        day_data = np.where(no_data_mask, new_acquisition, day_data)

        cloud_mask_old = day_data == METEOFRANCE_CLASSES["clouds"]

        cloud_mask_new = new_acquisition == METEOFRANCE_CLASSES["clouds"]
        nodata_mask_new = new_acquisition == METEOFRANCE_CLASSES["nodata"]
        no_observation_mask_new = cloud_mask_new | nodata_mask_new
        observation_mask_new = no_observation_mask_new == False
        new_observations_mask = cloud_mask_old & observation_mask_new
        day_data = np.where(new_observations_mask, new_acquisition, day_data)

    day_dataset = georef_data_array(
        xr.DataArray(day_data.astype(np.uint8), coords=extract_netcdf_coords_from_rasterio_raster(first_image_raster)),
        data_array_name="snow_cover",
        crs=first_image_raster.crs,
    )

    return day_dataset
