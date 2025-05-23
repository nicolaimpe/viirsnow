from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import rasterio
import rioxarray
import xarray as xr

from geotools import extract_netcdf_coords_from_rasterio_raster
from grids import GeoGrid, georef_netcdf, georef_netcdf_rioxarray
from harmonisation.reprojections import resample_s2_to_grid
from logger_setup import default_logger as logger
from products.classes import METEOFRANCE_CLASSES, NASA_CLASSES, S2_CLASSES
from products.filenames import get_datetime_from_viirs_nasa_filepath
from products.georef import modis_crs
from reductions.completeness import mask_of_pixels_in_range


def create_spatial_l3_nasa_composite(daily_snow_cover_files: List[str]) -> xr.DataArray:
    day_data_arrays = []
    dims = ("y", "x")
    for filepath in daily_snow_cover_files:
        # try:
        logger.info(f"Processing product {Path(filepath).name}")

        product_grid_data_variable = xr.open_dataset(filepath, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D", engine="netcdf4")
        bin_size = xr.open_dataset(filepath, engine="netcdf4").attrs["CharacteristicBinSize"]
        nasa_l3_grid = GeoGrid(
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

        day_data_arrays.append(georef_netcdf(data_array=ndsi_snow_cover, crs=modis_crs))

    merged_day_dataset = (
        xr.combine_by_coords(day_data_arrays, data_vars="minimal", fill_value=NASA_CLASSES["fill"][0])
        .astype(np.uint8)
        .data_vars["NDSI_Snow_Cover"]
    ).rio.write_nodata(NASA_CLASSES["fill"][0])

    return merged_day_dataset


def create_spatial_s2_composite(day_files: List[str], output_grid: GeoGrid) -> xr.Dataset:
    day_data_array = xr.DataArray(np.uint8(S2_CLASSES["nodata"][0]), coords=output_grid.xarray_coords)
    for filepath in day_files:
        logger.info(f"Processing product {Path(filepath).name}")
        s2_image = rioxarray.open_rasterio(filepath)
        s2_image = s2_image.sel(band=1).drop_vars("band")
        s2_resampled_image = resample_s2_to_grid(s2_dataset=s2_image, output_grid=output_grid)
        day_data_array = day_data_array.where(day_data_array != S2_CLASSES["nodata"][0], s2_resampled_image)
    day_dataset = xr.Dataset({"snow_cover_fraction": day_data_array.rio.write_crs(output_grid.crs)})
    return day_dataset


def create_spatial_s2_composite_sca(day_files: List[str], output_grid: GeoGrid) -> xr.Dataset:
    day_data_array = xr.DataArray(np.uint8(S2_CLASSES["nodata"][0]), coords=output_grid.xarray_coords)
    for filepath in day_files:
        logger.info(f"Processing product {Path(filepath).name}")
        s2_image = rioxarray.open_rasterio(filepath)
        s2_image = s2_image.sel(band=1).drop_vars("band")
        high_fsc_mask = mask_of_pixels_in_range(range=range(51, 101), data_array=s2_image)
        low_fsc_mask = mask_of_pixels_in_range(range=range(1, 51), data_array=s2_image)
        s2_image = s2_image.where(1 - high_fsc_mask, 100)
        s2_image = s2_image.where(1 - low_fsc_mask, 0)
        s2_resampled_image = resample_s2_to_grid(s2_dataset=s2_image, output_grid=output_grid)
        day_data_array = day_data_array.where(day_data_array != S2_CLASSES["nodata"][0], s2_resampled_image)
    day_dataset = xr.Dataset({"snow_cover_fraction": day_data_array})
    return georef_netcdf(day_dataset, crs=output_grid.crs)


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


def create_temporal_composite_meteofrance(daily_snow_cover_files: List[str], daily_geometry_files: List[str]) -> xr.Dataset:
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

    ################# Sorry for this section :`)

    # Read data and assemble in a numpy temporally ordered array
    snow_cover_daily_images = np.array([rasterio.open(file).read(1) for file in daily_snow_cover_files])
    view_angles_daily_array = np.array(
        [
            xr.open_dataset(file, mask_and_scale=True).data_vars["band_data"].sel(band=1).drop_vars("band").values
            for file in daily_geometry_files
        ]
    )
    view_angles_daily_array = np.ma.masked_array(view_angles_daily_array, np.isnan(view_angles_daily_array))
    # View angles Météo-France encoded on half degree
    view_angles_daily_array = view_angles_daily_array / 2

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

    day_dataset = georef_netcdf(
        xr.DataArray(day_data.astype(np.uint8), coords=extract_netcdf_coords_from_rasterio_raster(first_image_raster)),
        data_array_name="snow_cover",
        crs=first_image_raster.crs,
    )

    return day_dataset


def create_temporal_l3_naive_composite_nasa(daily_data_arrays: List[xr.DataArray]) -> xr.Dataset:
    first_day_data = daily_data_arrays[0]
    day_data = first_day_data.values

    for day_data_array in daily_data_arrays:
        new_acquisition = day_data_array.values

        no_data_mask = day_data == NASA_CLASSES["fill"]
        day_data = np.where(no_data_mask, new_acquisition, day_data)

        cloud_mask_old = day_data == NASA_CLASSES["clouds"]

        cloud_mask_new = new_acquisition == NASA_CLASSES["clouds"]
        nodata_mask_new = new_acquisition == NASA_CLASSES["fill"]
        no_observation_mask_new = cloud_mask_new | nodata_mask_new
        observation_mask_new = no_observation_mask_new == False
        new_observations_mask = cloud_mask_old & observation_mask_new
        day_data = np.where(new_observations_mask, new_acquisition, day_data)

    day_dataset = georef_netcdf_rioxarray(
        xr.DataArray(day_data.astype(np.uint8), coords=first_day_data.coords),
        crs=first_day_data.rio.crs,
    )

    return day_dataset


def match_daily_snow_cover_and_geometry_nasa(daily_snow_cover_files: List[str], daily_geometry_files: List[str]):
    if len(daily_snow_cover_files) != len(daily_geometry_files):
        logger.warning("Different number of files between geometry and snow cover.")
    output_snow_cover_files, output_geometry_files = [], []
    for snow_cover_file in sorted(daily_snow_cover_files):
        snow_cover_datetime = get_datetime_from_viirs_nasa_filepath(snow_cover_file)
        for geometry_file in daily_geometry_files:
            geometry_datetime = get_datetime_from_viirs_nasa_filepath(geometry_file)
            # Granule timestamps should be matching...we put a tight threshold of 1 minute
            if np.abs(snow_cover_datetime - geometry_datetime) < timedelta(seconds=60):
                output_snow_cover_files.append(snow_cover_file)
                output_geometry_files.append(geometry_file)
    return output_snow_cover_files, output_geometry_files


def create_temporal_composite_nasa(daily_snow_cover_files: List[str], daily_geometry_files: List[str]) -> xr.Dataset:
    """Create a L3 daily composite form daily L2 swath views using a sensor zenith angle criterion.
    For each pixel we select the "best" observation, i.e. the observation with smaller zenith angle.
    We also make the choice of retrieving some "non-optimal" information.
    If the "best" observation is cloud covered (more generally invalid), we take the other observation,
    even if it has been done at a very high sensor zenith angle.
    This will recover some invalid pixels but at the same time probably introduces false detections
    (more generally "bad" observations)
    """

    # Check that we can suppose to be on the very same grid
    if not xr.open_dataset(daily_geometry_files[0]).coords.equals(xr.open_dataset(daily_snow_cover_files[0]).coords):
        raise ValueError(
            "The coordinates of snow cover and geometry datasets should be the same for the composite algorithm to run"
        )

    ################# Sorry for this section :`)
    # Read data and assemble in a numpy temporally ordered array
    ndsi_snow_cover_daily_images = np.array(
        [xr.open_dataset(file, mask_and_scale=True).data_vars["NDSI_Snow_Cover"].values for file in daily_snow_cover_files]
    )
    view_angles_daily_array = np.array(
        [xr.open_dataset(file, mask_and_scale=True).data_vars["sensor_zenith"].values for file in daily_geometry_files]
    )
    view_angles_daily_array = np.ma.masked_array(view_angles_daily_array, np.isnan(view_angles_daily_array))

    # Inland water is considered valid observation
    invalid_masks = ~(
        (ndsi_snow_cover_daily_images <= NASA_CLASSES["snow_cover"][-1])
        | (ndsi_snow_cover_daily_images == NASA_CLASSES["water"][0])
    )

    # Take best observation
    best_observation_index = np.nanargmin(view_angles_daily_array, axis=0)
    best_observation_angle = np.min(view_angles_daily_array, axis=0)
    n_obs, height, width = ndsi_snow_cover_daily_images.shape

    snow_cover_best_observation = ndsi_snow_cover_daily_images[
        best_observation_index, np.arange(height)[:, None], np.arange(width)
    ]  # Numpy advanced indexing for selecting for each pixel the best observation index

    # In this part we recover observations taken at a worse zenith angle if in the best observation composite the pixel is invalid
    out_ndsi_snow_cover = snow_cover_best_observation
    out_view_angle = best_observation_angle

    invalid_mask_best_observation = ~(
        (snow_cover_best_observation <= NASA_CLASSES["snow_cover"][-1])
        | (snow_cover_best_observation == NASA_CLASSES["water"][0])
    )
    for idx in range(n_obs):
        out_ndsi_snow_cover = np.where(
            invalid_masks[idx]
            < invalid_mask_best_observation,  # pixels that are marked as invalid in the best observation but not in another observation
            ndsi_snow_cover_daily_images[idx],
            out_ndsi_snow_cover,
        )

        # Replace data also for view zenith angle
        out_view_angle = np.where(
            invalid_masks[idx] < invalid_mask_best_observation, view_angles_daily_array[idx], out_view_angle
        )

    sample_data = rioxarray.open_rasterio(daily_snow_cover_files[0]).data_vars["NDSI_Snow_Cover"].sel(band=1).drop_vars("band")

    day_dataset = xr.Dataset(
        {
            "NDSI_Snow_Cover": xr.DataArray(out_ndsi_snow_cover, dims=("y", "x"), coords=sample_data.coords),
            "sensor_zenith_angle": xr.DataArray(out_view_angle, dims=("y", "x"), coords=sample_data.coords),
        }
    ).rio.write_crs(sample_data.rio.crs)

    return day_dataset
