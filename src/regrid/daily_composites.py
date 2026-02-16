from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import rasterio
import rioxarray
import xarray as xr
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
from geospatial_grid.gsgrid import GSGrid

from logger_setup import default_logger as logger
from products.classes import METEOFRANCE_ARCHIVE_CLASSES, NASA_CLASSES
from products.filenames import get_datetime_from_viirs_nasa_filepath


def match_daily_snow_cover_and_geometry_meteofrance(daily_snow_cover_files: List[str], daily_geometry_files: List[str]):
    if len(daily_snow_cover_files) != len(daily_geometry_files):
        logger.warning("Different number of files between geometry and snow cover.")
    output_snow_cover_files, output_geometry_files = [], []
    for snow_cover_file in sorted(daily_snow_cover_files):
        snow_cover_datetime = datetime.strptime(Path(snow_cover_file).name[:13], "%Y%m%d_%H%M%S")
        for geometry_file in daily_geometry_files:
            geometry_datetime = datetime.strptime(Path(geometry_file).name[:13], "%Y%m%d_%H%M%S")
            if np.abs(snow_cover_datetime - geometry_datetime) < timedelta(seconds=60):
                output_snow_cover_files.append(snow_cover_file)
                output_geometry_files.append(geometry_file)
    return output_snow_cover_files, output_geometry_files


def create_temporal_composite_meteofrance_single_platform(
    daily_snow_cover_files: List[str], daily_geometry_files: List[str]
) -> xr.Dataset:
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

    # daily_snow_cover_files, daily_geometry_files = (
    #     match_daily_snow_cover_and_geometry_meteofrance(
    #         daily_snow_cover_files, daily_geometry_files
    #     )
    # )

    daily_snow_cover_files.sort()
    daily_geometry_files.sort()

    # Read data and assemble in a numpy temporally ordered array
    snow_cover_daily_images = np.array([rasterio.open(file).read(1) for file in daily_snow_cover_files])

    # Solar zenith angle no observation are encoded as "0", which is also a possible physical value of the incidence angle
    # This is problematic for the algorithm so we correct it.
    # Maybe in the operational product these nodata values will be encoded differently?

    # Compose a view zenith angle data array
    view_angles = [rasterio.open(file).read(1).astype("float32") for file in daily_geometry_files]

    view_angles_daily_array = np.ma.masked_array(view_angles, view_angles == 255.0)

    # Sort by view angle
    view_angle_sorting_index = np.argsort(view_angles, axis=0)
    rearrenged_snow_cover = np.take_along_axis(snow_cover_daily_images, view_angle_sorting_index, axis=0)
    rearrenged_view_angle = np.take_along_axis(view_angles_daily_array, view_angle_sorting_index, axis=0)

    snow_cover_best_observation = rearrenged_snow_cover[0, :]
    best_observation_angle = rearrenged_view_angle[0, :]

    ## In this part we recover observations taken at a worse zenith angle if in the best observation composite the pixel is invalid
    # Intitialize
    out_snow_cover = snow_cover_best_observation
    out_view_angle = best_observation_angle

    # Invalid observations
    invalid_masks = rearrenged_snow_cover > METEOFRANCE_ARCHIVE_CLASSES["water"][0]
    invalid_mask_out_snow_cover = out_snow_cover > METEOFRANCE_ARCHIVE_CLASSES["water"][0]

    for idx in range(snow_cover_daily_images.shape[0]):
        # pixels that are marked as invalid in the best observation but not in another observation
        pixels_to_reverse_mask = invalid_masks[idx] < invalid_mask_out_snow_cover
        out_snow_cover = np.where(pixels_to_reverse_mask, rearrenged_snow_cover[idx], out_snow_cover)
        # Replace data also for view zenith angle and platform
        out_view_angle = np.where(pixels_to_reverse_mask, rearrenged_view_angle[idx], out_view_angle)

        invalid_mask_out_snow_cover = out_snow_cover > METEOFRANCE_ARCHIVE_CLASSES["water"][0]

    # Here we output in netcdf for export but it can be changed
    sample_data = (
        xr.open_dataset(daily_snow_cover_files[0], decode_cf=True).data_vars["band_data"].sel(band=1).drop_vars("band")
    )
    day_dataset = xr.Dataset(
        {
            "snow_cover_fraction": xr.DataArray(out_snow_cover, dims=sample_data.dims, coords=sample_data.coords),
            "sensor_zenith_angle": xr.DataArray(out_view_angle, dims=sample_data.dims, coords=sample_data.coords),
        }
    ).rio.write_crs(sample_data.rio.crs)

    return day_dataset


def create_temporal_composite_meteofrance_multiplatform(
    daily_snow_cover_files: List[str], daily_geometry_files: List[str]
) -> xr.Dataset:
    """Create a L3 daily composite form daily L2 swath views using a sensor zenith angle criterion.
    For each pixel we select the "best" observation, i.e. the observation with smaller zenith angle.
    We also make the choice of retrieving some "non-optimal" information.
    If the "best" observation is cloud covered (more generally invalid), we take the other observation,
    even if it has been done at a very high sensor zenith angle.
    This will recover some invalid pixels but at the same time probably introduces false detections
    (more generally "bad" observations)
    """

    # daily_snow_cover_files, daily_geometry_files = (
    #     match_daily_snow_cover_and_geometry_meteofrance(
    #         daily_snow_cover_files, daily_geometry_files
    #     )
    # )
    daily_snow_cover_files.sort()
    daily_geometry_files.sort()

    # This is to account for the fact that NASA and Météo-France L2 come from two very different pipelines
    # Metadata are different (i.e. observation times) and it's possible that there is a different number of daily files
    # This funciton is here to filter this but ideally in a future iteration where sensor zenith angle will
    # be given in the L2 Météo-France this will be useless
    # Read data and assemble in a numpy temporally ordered array
    snow_cover_daily_images = np.array([rasterio.open(file).read(1) for file in daily_snow_cover_files])
    view_angles = [
        xr.open_dataset(file, mask_and_scale=False).data_vars["band_data"].sel(band=1).drop_vars("band")
        for file in daily_geometry_files
    ]

    platform_array = np.zeros_like(snow_cover_daily_images)
    for idx, file in enumerate(daily_snow_cover_files):
        if "npp" in file:
            platform_array[idx, :] = 1
        elif "noaa20" in file:
            platform_array[idx, :] = 2
        elif "noaa21" in file:
            platform_array[idx, :] = 3
        else:
            raise NotImplementedError
    # Solar zenith angle no observation are encoded as "0", which is also a possible physical value of the incidence angle
    # This is problematic for the algorithm so we correct it.
    # Maybe in the operational product these nodata values will be encoded differently?

    # Compose a view zenith angle data array
    view_angles_daily_array = np.array(view_angles)

    # Sort by view angle
    view_angle_sorting_index = np.argsort(view_angles, axis=0)
    rearrenged_snow_cover = np.take_along_axis(snow_cover_daily_images, view_angle_sorting_index, axis=0)
    rearrenged_view_angle = np.take_along_axis(view_angles_daily_array, view_angle_sorting_index, axis=0)
    rearrenged_platform = np.take_along_axis(platform_array, view_angle_sorting_index, axis=0)

    snow_cover_best_observation = rearrenged_snow_cover[0, :]
    best_observation_angle = rearrenged_view_angle[0, :]
    best_platform = rearrenged_platform[0, :]

    ## In this part we recover observations taken at a worse zenith angle if in the best observation composite the pixel is invalid
    # Intitialize
    out_snow_cover = snow_cover_best_observation
    out_view_angle = best_observation_angle
    out_platform = best_platform

    # Invalid observations
    invalid_masks = rearrenged_snow_cover > METEOFRANCE_ARCHIVE_CLASSES["water"][0]
    invalid_mask_out_snow_cover = out_snow_cover > METEOFRANCE_ARCHIVE_CLASSES["water"][0]

    for idx in range(snow_cover_daily_images.shape[0]):
        # pixels that are marked as invalid in the best observation but not in another observation
        pixels_to_reverse_mask = invalid_masks[idx] < invalid_mask_out_snow_cover
        out_snow_cover = np.where(pixels_to_reverse_mask, rearrenged_snow_cover[idx], out_snow_cover)
        # Replace data also for view zenith angle and platform
        out_view_angle = np.where(pixels_to_reverse_mask, rearrenged_view_angle[idx], out_view_angle)
        out_platform = np.where(pixels_to_reverse_mask, rearrenged_platform[idx], out_platform)
        invalid_mask_out_snow_cover = out_snow_cover > METEOFRANCE_ARCHIVE_CLASSES["water"][0]

    # Here we output in netcdf for export but it can be changed
    sample_data = (
        xr.open_dataset(daily_snow_cover_files[0], decode_cf=True).data_vars["band_data"].sel(band=1).drop_vars("band")
    )
    day_dataset = xr.Dataset(
        {
            "snow_cover_fraction": xr.DataArray(out_snow_cover, dims=sample_data.dims, coords=sample_data.coords),
            "sensor_zenith_angle": xr.DataArray(out_view_angle, dims=sample_data.dims, coords=sample_data.coords),
            "platform": xr.DataArray(out_platform, dims=sample_data.dims, coords=sample_data.coords),
        }
    ).rio.write_crs(sample_data.rio.crs)

    day_dataset.attrs["platform_encoding_values"] = ["1", "2", "3"]
    day_dataset.attrs["platform_encoding_platforms"] = ["SNPP", "JPSS1", "JPSS2"]
    return day_dataset


def create_temporal_l2_naive_composite_meteofrance(daily_files: List[str]) -> xr.Dataset:
    logger.info(f"Reading file {daily_files[0]}")
    first_image_raster = rasterio.open(daily_files[0])
    day_data = first_image_raster.read(1)

    for day_file in daily_files[1:]:
        logger.info(f"Reading file {day_file}")
        new_acquisition = rasterio.open(day_file).read(1)

        no_data_mask = day_data == METEOFRANCE_ARCHIVE_CLASSES["nodata"]
        day_data = np.where(no_data_mask, new_acquisition, day_data)

        cloud_mask_old = day_data == METEOFRANCE_ARCHIVE_CLASSES["clouds"]

        cloud_mask_new = new_acquisition == METEOFRANCE_ARCHIVE_CLASSES["clouds"]
        nodata_mask_new = new_acquisition == METEOFRANCE_ARCHIVE_CLASSES["nodata"]
        no_observation_mask_new = cloud_mask_new | nodata_mask_new
        observation_mask_new = no_observation_mask_new == False
        new_observations_mask = cloud_mask_old & observation_mask_new
        day_data = np.where(new_observations_mask, new_acquisition, day_data)

    affine = first_image_raster.transform
    coords = GSGrid(
        x0=affine.c,
        y0=affine.f,
        resolution=(affine.b, affine.e),
        width=first_image_raster.width,
        height=first_image_raster.height,
    ).xarray_coords

    day_dataset = georef_netcdf_rioxarray(
        xr.DataArray(day_data.astype(np.uint8), coords=coords),
        data_array_name="snow_cover",
        crs=first_image_raster.crs,
    )

    return day_dataset
