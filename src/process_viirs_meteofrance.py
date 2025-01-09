import re

from affine import Affine
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
from geotools import (
    extract_netcdf_coords_from_rasterio_raster,
    georef_data_array,
    gdf_to_binary_mask,
    reproject_dataset,
    to_rioxarray,
)
from logger_setup import default_logger as logger
import pyproj
from grids import RESAMPLING, DefaultGrid


PLATFORMS_PRODUCT_DICT = {"SuomiNPP": "SNPP"}
GRID = DefaultGrid()


def get_datetime_from_viirs_filepath(filepath: str) -> str:
    # This function is not use for the moment...to see whether to take it out
    def timestamp_to_datetime(observation_timestamp: str) -> datetime:
        return datetime.strptime(observation_timestamp, "%Y%m%d%H%M%S")

    return timestamp_to_datetime(Path(filepath).name.split(".")[1].split("_")[0])


def get_daily_filenames_per_platform(platform: str, day: datetime, viirs_data_folder: str) -> List[str] | None:
    return glob(f"{viirs_data_folder}/VIIRS{day.year}/*_{PLATFORMS_PRODUCT_DICT[platform]}_*{day.strftime("%Y%m%d")}*.LT")


def create_composite_meteofrance(daily_files: List[str], roi_file: str | None = None) -> xr.Dataset:
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

    day_dataset_reprojected = reproject_dataset(
        dataset=to_rioxarray(day_dataset),
        shape=GRID.shape,
        transform=GRID.affine,
        new_crs=pyproj.CRS(GRID.crs),
        resampling=RESAMPLING,
        fill_value=METEOFRANCE_CLASSES["fill"][0],
    )

    if roi_file is not None:
        roi_mask = gdf_to_binary_mask(gdf=gpd.read_file(roi_file), grid=GRID)
        masked = day_dataset_reprojected.data_vars["snow_cover"].values * roi_mask.data_vars["binary_mask"].values
        masked[roi_mask.data_vars["binary_mask"].values == 0] = METEOFRANCE_CLASSES["fill"]
        day_dataset_reprojected.data_vars["snow_cover"][:] = masked

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
        daily_files = get_daily_filenames_per_platform(platform=platform, day=day, viirs_data_folder=viirs_data_folder)
        if len(daily_files) == 0:
            logger.warning(f"No data fuond in date {day}")
            continue
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


if __name__ == "__main__":
    # User inputs
    year = WinterYear(2023, 2024)
    folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/EOFR62"
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/cms_workshop"
    roi_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/vectorial/massifs/massifs.shp"

    create_meteofrance_time_series(
        year=year, viirs_data_folder=folder, output_folder=output_folder, roi_shapefile=roi_shapefile
    )
