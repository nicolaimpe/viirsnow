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
from products.snow_cover_product import SnowCoverProduct
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
    def __init__(self, product: SnowCoverProduct, output_grid: GeoGrid, data_folder: str, output_folder: str):
        self.product = product
        self.grid = output_grid
        self.data_folder = data_folder
        self.output_folder = output_folder

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
            snow_cover.where(snow_cover <= self.product.classes["clouds"]).count()
            == snow_cover.where(snow_cover == self.product.classes["clouds"]).count()
        ):
            return False
        else:
            return True

    def low_values_screen(self, daily_composite: xr.Dataset, thresholds: Dict[str, float]) -> xr.Dataset:
        for key, value in thresholds.items():
            daily_composite.data_vars[key][:] = daily_composite.data_vars[key].where(daily_composite.data_vars[key] > value, 0)
        return daily_composite

    def export_daily_data(self, day: datetime, daily_data: xr.Dataset):
        out_path = f"{str(self.output_folder)}/{day.strftime('%Y%m%d')}.nc"
        daily_composite = daily_data.expand_dims(time=[day])
        daily_composite.to_netcdf(out_path)

    def export_time_series(self, winter_year: WinterYear):
        out_tmp_paths = glob(f"{str(self.output_folder)}/[{winter_year.from_year}-{winter_year.to_year}]*.nc")
        time_series = xr.open_mfdataset(out_tmp_paths, mask_and_scale=False)
        encodings = generate_xarray_compression_encodings(time_series)
        encodings.update(time={"calendar": "gregorian", "units": f"days since {str(winter_year.from_year)}-10-01"})
        out_path = f"{self.output_folder}/WY_{winter_year.from_year}_{winter_year.to_year}_{self.product.name}.nc"
        logger.info(f"Exporting to {out_path}")
        time_series.to_netcdf(out_path, encoding=encodings)
        [os.remove(file) for file in out_tmp_paths]

    def create_time_series(
        self,
        winter_year: WinterYear,
        roi_shapefile: str | None = None,
        low_value_thresholds: Dict[str, float] | None = None,
    ):
        files = self.get_all_files_of_winter_year(winter_year=winter_year)

        for day in winter_year.iterate_days():
            logger.info(f"Processing day {day}")
            # if day.year == 2023 or day.month < 4:
            #     continue
            day_files = self.get_daily_files(files, day=day)

            day_files = self.check_daily_files(day_files=day_files)

            #     out_path = f"{str(self.output_folder)}/{day.strftime('%Y%m%d')}.nc"
            #     out_tmp_paths.append(out_path)
            #     continue
            if len(day_files) == 0:
                logger.info(f"Skip day {day.date()} because 0 files were found on this date")
                continue
            daily_composite = self.create_spatial_composite(day_files=day_files)

            if roi_shapefile is not None:
                roi_mask = gdf_to_binary_mask(gdf=gpd.read_file(roi_shapefile), grid=self.grid)
                daily_composite = daily_composite.where(roi_mask, self.product.classes["fill"][0])
                for dv in daily_composite.data_vars.values():
                    dv.rio.write_nodata(self.product.classes["fill"][0], inplace=True)

            if not self.check_scf_not_empty(daily_composite):
                logger.info(f"Skip day {day.date()} because only clouds are present on this date.")
                continue
            if low_value_thresholds is not None:
                daily_composite = self.low_values_screen(daily_composite=daily_composite, thresholds=low_value_thresholds)
            self.export_daily_data(day=day, daily_data=daily_composite)
        self.export_time_series(winter_year=winter_year)
