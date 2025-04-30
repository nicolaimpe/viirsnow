import abc
import os
from datetime import datetime
from glob import glob
from typing import Dict, List

import geopandas as gpd
import rasterio
import xarray as xr

from compression import generate_xarray_compression_encodings
from geotools import gdf_to_binary_mask
from grids import GeoGrid
from logger_setup import default_logger as logger
from products.classes import PRODUCT_CLASSES_DICT
from reductions.snow_cover_extent_cross_comparison import WinterYear


def check_input_daily_tif_files(input_tif_files: List[str]) -> List[str]:
    for day_file in input_tif_files:
        try:
            xr.open_dataset(day_file).data_vars["band_data"].values
        except (OSError, rasterio.errors.RasterioIOError, rasterio._err.CPLE_AppDefinedError):
            logger.info(f"Could not open file {day_file}. Removing it from processing")
            input_tif_files.remove(day_file)
            continue
    return input_tif_files


class HarmonisationBase:
    def __init__(self, product_name: str, output_grid: GeoGrid, data_folder: str, output_folder: str):
        self.product_name = product_name
        self.grid = output_grid
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.classes = PRODUCT_CLASSES_DICT[product_name]

    @abc.abstractmethod
    def get_all_files_of_winter_year(self, winter_year: WinterYear) -> List[str]:
        pass

    @abc.abstractmethod
    def get_daily_files(self, all_winter_year_files: List[str], day: datetime) -> List[str]:
        pass

    @abc.abstractmethod
    def check_daily_files(self, day_files: List[str]) -> List[str]:
        pass

    @abc.abstractmethod
    def create_spatial_composite(self, day_files: List[str]) -> xr.Dataset:
        pass

    def check_scf_not_empty(self, daily_composite: xr.Dataset) -> None:
        snow_cover = (
            daily_composite.data_vars["snow_cover_fraction"]
            if "snow_cover_fraction" in daily_composite.data_vars
            else daily_composite.data_vars["NDSI_Snow_Cover"]
        )
        if (
            snow_cover.where(snow_cover <= self.classes["clouds"]).count()
            == snow_cover.where(snow_cover == self.classes["clouds"]).count()
        ):
            return False
        else:
            return True

    def low_values_screen(self, daily_composite: xr.Dataset, thresholds: Dict[str, float]) -> xr.Dataset:
        for key, value in thresholds.items():
            daily_composite.data_vars[key][:] = daily_composite.data_vars[key].where(daily_composite.data_vars[key] > value, 0)
        return daily_composite

    def create_time_series(
        self,
        winter_year: WinterYear,
        roi_shapefile: str | None = None,
        low_value_thresholds: Dict[str, float] | None = None,
    ):
        files = self.get_all_files_of_winter_year(winter_year=winter_year)
        out_tmp_paths = []

        for day in winter_year.iterate_days():
            logger.info(f"Processing day {day}")

            day_files = self.get_daily_files(files, day=day)
            if day.month != 12 and day.month != 1 and day.month != 2:
                continue
            if day.day > 6:
                break

            day_files = self.check_daily_files(day_files=day_files)

            if len(day_files) == 0:
                logger.info(f"Skip day {day.date()} because 0 files were found on this date")
                continue
            daily_composite = self.create_spatial_composite(day_files=day_files)

            if roi_shapefile is not None:
                roi_mask = gdf_to_binary_mask(gdf=gpd.read_file(roi_shapefile), grid=self.grid)
                daily_composite = daily_composite.where(roi_mask, self.classes["fill"][0])
                for dv in daily_composite.data_vars.values():
                    dv.rio.write_nodata(self.classes["fill"][0], inplace=True)

            if not self.check_scf_not_empty(daily_composite):
                logger.info(f"Skip day {day.date()} because only clouds are present on this date.")
                continue
            if low_value_thresholds is not None:
                daily_composite = self.low_values_screen(daily_composite=daily_composite, thresholds=low_value_thresholds)

            out_path = f"{str(self.output_folder)}/{day.strftime('%Y%m%d')}.nc"
            out_tmp_paths.append(out_path)

            daily_composite = daily_composite.expand_dims(time=[day])
            daily_composite.to_netcdf(out_path)
        out_tmp_paths = glob(f"{str(self.output_folder)}/202[3-4]*.nc")
        time_series = xr.open_mfdataset(out_tmp_paths, mask_and_scale=False)
        encodings = generate_xarray_compression_encodings(time_series)
        encodings.update(time={"calendar": "gregorian", "units": f"days since {str(winter_year.from_year)}-10-01"})
        time_series.to_netcdf(
            f"{self.output_folder}/WY_{winter_year.from_year}_{winter_year.to_year}_{self.product_name}.nc",
            encoding=encodings,
        )
        [os.remove(file) for file in out_tmp_paths]
