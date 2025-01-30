import os
from typing import List
import xarray as xr
import glob
import geopandas as gpd
from metrics import WinterYear
from pathlib import Path
import rioxarray
from datetime import datetime
import numpy as np
from geotools import gdf_to_binary_mask, reproject_dataset, georef_data_array
from grids import DefaultGrid
from rasterio.enums import Resampling
from viirsnow.products.classes import S2_CLASSES
from logger_setup import default_logger as logger

GRID = DefaultGrid()


def get_all_s2_files_of_winter_year(s2_folder: str, winter_year: WinterYear) -> List[str]:
    s2_files = glob.glob(str(Path(s2_folder).joinpath(f"FSC_*{winter_year.from_year}1[0-2]*/*FSCOG.tif")))
    s2_files.extend(glob.glob(str(Path(s2_folder).joinpath(f"FSC_*{winter_year.to_year}0[1-9]*/*FSCOG.tif"))))
    return sorted(s2_files)


def s2_filename_to_datetime(s2_file: str):
    observation_timestamp = Path(s2_file).name.split("_")[1]
    observation_datetime = datetime.strptime(observation_timestamp[:8], "%Y%m%d")
    return observation_datetime


def add_time_dim(data_array: xr.DataArray) -> xr.DataArray:
    return data_array.expand_dims(time=[s2_filename_to_datetime(s2_file=data_array.encoding["source"])])


def read_aggregate_s2(s2_filename: str) -> xr.Dataset:
    # 250m resolution FSC from FSCOG S2 product with a "zombie" nodata mask
    s2_in_image = rioxarray.open_rasterio(s2_filename)

    # Validity "zombie mask": wherever there is at least one non valid pixel, the output grid pixel is set as invalid (<-> cloud)
    s2_validity_mask = reproject_dataset(
        s2_in_image,
        new_crs=GRID.crs,
        resampling=Resampling.max,
        fill_value=S2_CLASSES["nodata"][0],
        transform=GRID.affine,
        shape=GRID.shape,
    )

    # Aggregate the dataset at 250 m
    s2_aggregated = reproject_dataset(
        s2_in_image.astype(np.float32),
        new_crs=GRID.crs,
        resampling=Resampling.average,
        fill_value=S2_CLASSES["nodata"][0],
        transform=GRID.affine,
        shape=GRID.shape,
    )

    # Compose the mask
    s2_out_image = xr.where(s2_validity_mask <= S2_CLASSES["snow_cover"][-1], s2_aggregated.astype("u1"), s2_validity_mask)
    s2_out_image.rio.write_nodata(255, inplace=True)
    s2_out_image = s2_out_image.sel(band=1).drop_vars("spatial_ref").drop_vars("band")
    s2_out_image = georef_data_array(s2_out_image, data_array_name="snow_cover", crs=GRID.crs)

    return s2_out_image


def create_s2_time_series(s2_folder: str, winter_year: WinterYear, output_folder: Path, roi_shapefile: str | None = None):
    files = get_all_s2_files_of_winter_year(s2_folder, winter_year=winter_year)
    if roi_shapefile is not None:
        roi_mask = gdf_to_binary_mask(gdf=gpd.read_file(roi_shapefile), grid=GRID)

    out_tmp_paths = []

    for day in winter_year.iterate_days():
        logger.info(f"Processing day {day}")
        empty_grid = xr.full_like(read_aggregate_s2(files[0]), fill_value=S2_CLASSES["nodata"][0])
        day_data: xr.Dataset = empty_grid.copy(deep=True)
        daily_files = [file for file in files if day.strftime("%Y%m%d") in file]

        for s2_daily_file in daily_files:
            logger.info(f"Reading file {s2_daily_file}")
            s2_resampled_image = read_aggregate_s2(s2_filename=s2_daily_file)
            # This masking approach might be inefficient
            day_data = day_data.where(
                s2_resampled_image.data_vars["snow_cover"] == S2_CLASSES["nodata"][0], s2_resampled_image
            )

        out_path = f"{str(output_folder)}/{day.strftime('%Y%m%d')}.nc"
        day_data = day_data.expand_dims(time=[day])
        if roi_shapefile is not None:
            day_data = day_data.where(roi_mask.data_vars["binary_mask"], other=S2_CLASSES["nodata"][0])
        out_tmp_paths.append(out_path)
        day_data.to_netcdf(out_path)

    all_data = xr.open_mfdataset(out_tmp_paths, combine="nested", concat_dim="time")
    all_data.to_netcdf(
        f"{output_folder}/WY_{winter_year.from_year}_{winter_year.to_year}_S2_res_{GRID.resolution}m_time_series.nc"
    )
    [os.remove(file) for file in out_tmp_paths]


if __name__ == "__main__":
    year = WinterYear(2023, 2024)
    massifs_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/vectorial/massifs/massifs.shp"
    s2_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/S2"
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/cms_workshop"

    create_s2_time_series(s2_folder=s2_folder, roi_shapefile=massifs_shapefile, winter_year=year, output_folder=output_folder)
