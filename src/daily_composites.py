from datetime import timedelta
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
from products.filenames import get_datetime_from_viirs_meteofrance_filepath, get_datetime_from_viirs_nasa_filepath
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


def match_daily_snow_cover_and_geometry_meteofrance(daily_snow_cover_files: List[str], daily_geometry_files: List[str]):
    if len(daily_snow_cover_files) != len(daily_geometry_files):
        logger.warning("Different number of files between geometry and snow cover.")
    output_snow_cover_files, output_geometry_files = [], []
    for snow_cover_file in sorted(daily_snow_cover_files):
        snow_cover_datetime = get_datetime_from_viirs_meteofrance_filepath(snow_cover_file)
        for geometry_file in daily_geometry_files:
            geometry_datetime = get_datetime_from_viirs_nasa_filepath(geometry_file)
            if np.abs(snow_cover_datetime - geometry_datetime) < timedelta(seconds=3600):
                output_snow_cover_files.append(snow_cover_file)
                output_geometry_files.append(geometry_file)
    return output_snow_cover_files, output_geometry_files


def create_temporal_l2_composite_meteofrance(daily_snow_cover_files: List[str], daily_geometry_files: List[str]) -> xr.Dataset:
    """Create a L3 daily composite form daily L2 swath views using a sensor zenith angle criterion.
    For each pixel we select the "best" observation, i.e. the observation with smaller zenith angle.
    We also make the choice of retrieving some "non-optimal" information.
    If the "best" observation is cloud covered (more generally invalid), we take the other observation,
    even if it has been done at a very high sensor zenith angle.
    This will recover some invalid pixels but at the same time probably introduces false detections
    (more generally "bad" observations)
    """

    # This is to account for the fact that NASA and Météo-France L2 come from two very different pipelines
    # Metadata are different (i.e. observation times) and it's possible that there is a different number of daily files
    # This funciton is here to filter this but ideally in a future iteration where sensor zenith angle will
    # be given in the L2 Météo-France this will be useless
    daily_snow_cover_files, daily_geometry_files = match_daily_snow_cover_and_geometry_meteofrance(
        daily_snow_cover_files, daily_geometry_files
    )

    # Check that we can suppose to be on the very same grid
    if xr.open_dataset(daily_geometry_files[0]).coords.equals(
        xr.Coordinates(extract_netcdf_coords_from_rasterio_raster(rasterio.open(daily_snow_cover_files[0])))
    ):
        raise ValueError(
            "The coordinates of snow cover and geometry datasets should be the same for the composite algorithm to run"
        )

    ################# Sorry for this section :`)
    # Read data and assemble in a numpy temporally ordered array
    snow_cover_daily_images = np.array([rasterio.open(file).read(1) for file in daily_snow_cover_files])
    view_angles_daily_array = np.array(
        [xr.open_dataset(file, mask_and_scale=True).data_vars["sensor_zenith"].values for file in daily_geometry_files]
    )
    view_angles_daily_array = np.ma.masked_array(view_angles_daily_array, np.isnan(view_angles_daily_array))

    invalid_masks = snow_cover_daily_images > METEOFRANCE_CLASSES["water"][0]

    # Take best observation
    best_observation_index = np.nanargmin(view_angles_daily_array, axis=0)
    best_observation_angle = np.min(view_angles_daily_array, axis=0)
    n_obs, height, width = snow_cover_daily_images.shape

    snow_cover_best_observation = snow_cover_daily_images[
        best_observation_index, np.arange(height)[:, None], np.arange(width)
    ]  # Numpy advanced indexing for selecting for each pixel the best observation index

    # In this part we recover observations taken at a worse zenith angle if in the best observation composite the pixel is invalid
    out_snow_cover = snow_cover_best_observation
    out_view_angle = best_observation_angle
    invalid_mask_best_observation = snow_cover_best_observation > METEOFRANCE_CLASSES["water"][0]
    for idx in range(n_obs):
        out_snow_cover = np.where(
            invalid_masks[idx]
            < invalid_mask_best_observation,  # pixels that are marked as invalid in the best observation but not in another observation
            snow_cover_daily_images[idx],
            out_snow_cover,
        )
        # Replace data also for view zenith angle
        out_view_angle = np.where(
            invalid_masks[idx] < invalid_mask_best_observation, view_angles_daily_array[idx], out_view_angle
        )

    # Some boilerplate code to make it compliant with xarray and GDAL drivers...hopefully will change in future iterations
    output_coords = xr.open_dataset(daily_geometry_files[0]).coords
    meteofrance_crs = rasterio.open(daily_snow_cover_files[0]).crs

    dims = dim_name(meteofrance_crs)
    day_dataset = georef_data_array(
        xr.DataArray(out_snow_cover, dims=dims, coords=output_coords),
        data_array_name="fractional_snow_cover",
        crs=meteofrance_crs,
    )
    day_dataset = day_dataset.assign({"sensor_zenith": xr.DataArray(out_view_angle, dims=dims, coords=output_coords)})

    return day_dataset


def create_temporal_l2_naive_composite_meteofrance(daily_files: List[str]) -> xr.Dataset:
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
