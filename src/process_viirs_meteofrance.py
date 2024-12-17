import re
from metrics import WinterYear
from glob import glob
import xarray as xr
import numpy as np
from pathlib import Path


import numpy as np
import netCDF4 as nc
import netCDF4 as nc
import numpy.typing as npt
from typing import Dict, List
from datetime import datetime
import os
import logging
import geopandas as gpd
import rasterio
from products import METEOFRANCE_CLASSES
from geotools import georef_data_array,gdf_to_binary_mask, reproject_dataset

# Module configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

platforms_products_dict = {"SuomiNPP": "SNPP"}
OUT_GRID_CRS = 32631
OUT_GRID_RES = 250  # m

# def extract_netcdf_latlon_from_gdal_raster(gdal_dataset_object: gdal.Dataset) -> Dict[str, npt.NDArray]:
#     transform = gdal_dataset_object.GetGeoTransform()
#     x_off, x_scale, _, y_off, _, y_scale = transform
#     n_cols, n_rows = gdal_raster.RasterXSize, gdal_dataset_object.RasterYSize
#     # for GDAL the UL corener is the UL corner of the image while for xarray is the center of the upper left pixel
#     # Compensate for it
#     longitudes = np.arange(n_cols) * x_scale + x_off + x_scale / 2
#     latitudes = np.arange(n_rows) * y_scale + y_off + y_scale / 2
#     return {LAT: latitudes, LON: longitudes}


def extract_netcdf_latlon_from_rasterio_raster(rasterio_dataset) -> Dict[str, npt.NDArray]:
    transform = rasterio_dataset.window_transform
    x_off, x_scale, _, y_off, _, y_scale = transform
    n_cols, n_rows = gdal_raster.RasterXSize, gdal_dataset_object.RasterYSize
    # for GDAL the UL corener is the UL corner of the image while for xarray is the center of the upper left pixel
    # Compensate for it
    longitudes = np.arange(n_cols) * x_scale + x_off + x_scale / 2
    latitudes = np.arange(n_rows) * y_scale + y_off + y_scale / 2
    return {LAT: latitudes, LON: longitudes}


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
    day_data = first_image_raster.read()

    for day_file in daily_files[1:]:
        no_data_mask = day_data == METEOFRANCE_CLASSES["nodata"]
        clouds_mask = day_data == METEOFRANCE_CLASSES["clouds"]

        day_data[no_data_mask or clouds_mask] = rasterio.open(day_file).read()

    if roi_file is not None:
        gdf_to_binary_mask()
        
    
    day_data_array = xr.DataArray(day_data, coords=extract_netcdf_latlon_from_rasterio_raster(first_image_raster))
    day_dataset = georef_data_array(day_data_array)
    reproject_dataset()
    return 


def create_meteofrance_time_series(
    year: WinterYear,
    viirs_data_folder: str,
    output_folder: str,
    roi_shapefile: str | None = None,
    platform: str = "SuomiNPP",
):
    outpaths = []
    for day in year.iterate_days():
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
    year = WinterYear(2023,2024)
    month = "*"  # regex
    folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/EOFR62"
    output_folder = "./output_folder/completeness_analysis"
    roi_shapefile = 

    create_meteofrance_time_series(year=year,viirs_data_folder=folder,output_folder=output_folder, roi_shapefile=)
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
