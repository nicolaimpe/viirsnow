import re
from metrics import WinterYear
from glob import glob
import xarray as xr
import numpy as np
from pathlib import Path


import numpy as np
import numpy.typing as npt
from typing import Dict, List
from datetime import datetime
import os
import geopandas as gpd
import rasterio
from products import METEOFRANCE_CLASSES
from geotools import georef_data_array, gdf_to_binary_mask, reproject_dataset, dim_name, find_nearest_bounds_for_selection
from logger_setup import default_logger as logger
import pyproj

platforms_products_dict = {"SuomiNPP": "SNPP"}
OUT_GRID_CRS = 32631
OUT_GRID_RES = 250  # m
RESAMPLING_METHOD = rasterio.enums.Resampling.nearest
FILL_VALUE = METEOFRANCE_CLASSES["fill"][0]
# def extract_netcdf_latlon_from_gdal_raster(gdal_dataset_object: gdal.Dataset) -> Dict[str, npt.NDArray]:
#     transform = gdal_dataset_object.GetGeoTransform()
#     x_off, x_scale, _, y_off, _, y_scale = transform
#     n_cols, n_rows = gdal_raster.RasterXSize, gdal_dataset_object.RasterYSize
#     # for GDAL the UL corener is the UL corner of the image while for xarray is the center of the upper left pixel
#     # Compensate for it
#     longitudes = np.arange(n_cols) * x_scale + x_off + x_scale / 2
#     latitudes = np.arange(n_rows) * y_scale + y_off + y_scale / 2
#     return {LAT: latitudes, LON: longitudes}


def extract_netcdf_coords_from_rasterio_raster(raster: rasterio.DatasetReader) -> Dict[str, npt.NDArray]:
    transform = raster.transform

    x_scale, x_off, y_scale, y_off = transform.a, transform.c, transform.e, transform.f
    n_cols, n_rows = raster.width, raster.height
    # for GDAL the UL corener is the UL corner of the image while for xarray is the center of the upper left pixel
    # Compensate for it
    x_coord = np.arange(n_cols) * x_scale + x_off
    y_coord = np.arange(n_rows) * y_scale + y_off
    dims = dim_name(raster.crs)
    return {dims[0]: y_coord, dims[1]: x_coord}


def timestamp_to_datetime(observation_timestamp: str) -> datetime:
    return datetime.strptime(observation_timestamp, f"%Y%m%d%H%M%S")


def get_viirs_timestamp_from_filepath(filepath: str) -> str:
    return Path(filepath).name.split(".")[1].split("_")[0]


def get_dateteime_from_viirs_filepath(filepath: str) -> str:
    return timestamp_to_datetime(get_viirs_timestamp_from_filepath(filepath))


def get_daily_filenames_per_platform(platform: str, day: datetime, viirs_data_folder: str) -> List[str] | None:
    return glob(f"{viirs_data_folder}/VIIRS{day.year}/*_{platforms_products_dict[platform]}_*{day.strftime("%Y%m%d")}*.LT")


def create_composite_meteofrance(daily_files: List[str], roi_file: str | None = None) -> xr.Dataset:
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

    if roi_file is not None:
        roi_mask = gdf_to_binary_mask(
            gdf=gpd.read_file(roi_file), out_resolution=first_image_raster.transform.a, out_crs=first_image_raster.crs
        )

    day_dataset = georef_data_array(
        xr.DataArray(day_data.astype(np.uint8), coords=extract_netcdf_coords_from_rasterio_raster(first_image_raster)),
        data_array_name="snow_cover",
        crs=first_image_raster.crs,
    )

    xmin, xmax, ymin, ymax = find_nearest_bounds_for_selection(dataset=day_dataset, other=roi_mask)
    dims = dim_name(pyproj.CRS(day_dataset.data_vars["spatial_ref"].attrs["spatial_ref"]))
    masked_dataset = day_dataset.sel({dims[1]: slice(xmin, xmax), dims[0]: slice(ymax, ymin)})
    masked = masked_dataset.data_vars["snow_cover"].values * roi_mask.data_vars["binary_mask"].values
    masked[roi_mask.data_vars["binary_mask"].values == 0] = FILL_VALUE
    masked_dataset.data_vars["snow_cover"][:] = masked

    day_dataset_reprojected = reproject_dataset(
        dataset=masked_dataset, new_crs=pyproj.CRS(OUT_GRID_CRS), new_resolution=OUT_GRID_RES, resampling=RESAMPLING_METHOD
    )
    return day_dataset_reprojected


def create_meteofrance_time_series(
    year: WinterYear,
    viirs_data_folder: str,
    output_folder: str,
    roi_shapefile: str | None = None,
    platform: str = "SuomiNPP",
):
    outpaths = []
    for day in year.iterate_days():
        logger.info(f"Processing day {day}")
        # if day.day > 5:
        #     break
        daily_files = get_daily_filenames_per_platform(platform=platform, day=day, viirs_data_folder=viirs_data_folder)
        meteofrance_composite = create_composite_meteofrance(daily_files=daily_files, roi_file=roi_shapefile)
        meteofrance_composite = meteofrance_composite.expand_dims(time=[day])
        outpath = f"{output_folder}/{day.strftime("%Y%m%d")}.nc"
        outpaths.append(outpath)
        meteofrance_composite.to_netcdf(outpath)

    time_series = xr.open_mfdataset(outpaths)
    output_name = Path(f"{output_folder}/WY_{year.from_year}_{year.to_year}_{platform}_meteofrance_time_series.nc")
    time_series.to_netcdf(
        output_name, encoding={"time": {"calendar": "gregorian", "units": f"days since {str(year.from_year)}-10-01"}}
    )
    [os.remove(file) for file in outpaths]
    return


# # Get all the dates of the series
# observation_times = []
# for filepath in viirs_data_filepaths:
#     observation_times.append(get_dateteime_from_viirs_filepath(filepath))
# observation_times = sorted(observation_times)

# # Get the geographic transform
# gdal_raster = gdal.Open(viirs_data_filepaths[0])
# latitudes_longitudes_dict = extract_netcdf_latlon_from_gdal_raster(gdal_raster)


# # Initilize a time series on the whole dataset
# time_series = xr.DataArray(
#     np.zeros((len(observation_times), gdal_raster.RasterYSize, gdal_raster.RasterXSize), dtype="uint8"),
#     dims=["time", LAT, LON],
#     coords={
#         "time": observation_times,  # The new time instant as a coordinate
#         LAT: latitudes_longitudes_dict[LAT],
#         LON: latitudes_longitudes_dict[LON],
#     },
# )

if __name__ == "__main__":
    # User inputs
    year = WinterYear(2023, 2024)
    month = "*"  # regex
    folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/EOFR62"
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/snow_cover_extent_analysis"
    roi_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/vectorial/massifs/massifs.shp"

    create_meteofrance_time_series(
        year=year, viirs_data_folder=folder, output_folder=output_folder, roi_shapefile=roi_shapefile
    )
    # # Loop over all the files to populate the time series
    # for count, filepath in enumerate(viirs_data_filepaths):
    #     logger.info(
    #         f"Processing image {str(count+1)}/{str(len(viirs_data_filepaths))}, file name {os.path.basename(filepath)}"
    #     )
    #     gdal_raster = gdal.Open(filepath)
    #     observation_time = get_dateteime_from_viirs_filepath(filepath)
    #     time_series.loc[dict(time=observation_time)] = gdal_raster.GetRasterBand(1).ReadAsArray()

    #     # print('number of pixels with 210 value: ', np.sum(gdal_raster.GetRasterBand(1).ReadAsArray() == 210))
    #     # print('number of pixels with 215 value: ', np.sum(gdal_raster.GetRasterBand(1).ReadAsArray() == 215))

    # # Export to netCDF. Compression is specified here.
    # georef_time_series = georef_data_array(
    #     data_array=time_series, data_array_name="snow_cover", crs_wkt=gdal.Open(viirs_data_filepaths[0]).GetProjection()
    # )
    # georef_time_series.coords["time"].attrs["long_name"] = "hydrological day"
    # outpath = f"{output_folder}/{str(year)}_{month}_meteofrance_time_series.nc"
    # logger.info(f"Save to {outpath} and compress (it might take a while)")
    # georef_time_series.to_netcdf(
    #     outpath, encoding={"snow_cover": {"zlib": True}, "time": {"calendar": "gregorian", "units": "days since 2016-10-01"}}
    # )
