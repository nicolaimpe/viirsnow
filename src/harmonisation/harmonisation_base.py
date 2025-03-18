import abc
import os
from typing import List

import xarray as xr

from compression import generate_xarray_compression_encodings
from evaluations.snow_cover_extent_cross_comparison import WinterYear
from geotools import mask_dataarray_with_vector_file
from grids import GeoGrid
from logger_setup import default_logger as logger
from products.classes import PRODUCT_CLASSES_DICT


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
    def create_spatial_composite(self, day_files: List[str]) -> xr.Dataset:
        pass

    def create_time_series(
        self, winter_year: WinterYear, roi_shapefile: str | None = None, fsc_threshold: float | None = None
    ):
        files = self.get_all_files_of_winter_year(winter_year=winter_year)

        out_tmp_paths = []

        for day in winter_year.iterate_days():
            if day.day > 5:
                break
            logger.info(f"Processing day {day}")
            day_files = [file for file in files if day.strftime("%Y%m%d") in file]
            if len(day_files) == 0:
                logger.info(f"Skip day {day.date()} because 0 files were found on this day")
                continue
            daily_composite = self.create_spatial_composite(day_files=day_files)

            if roi_shapefile is not None:
                daily_composite = mask_dataarray_with_vector_file(
                    data_array=daily_composite.data_vars["snow_cover_fraction"],
                    roi_file=roi_shapefile,
                    output_grid=self.grid,
                    fill_value=self.classes["nodata"][0],
                )

            if fsc_threshold is not None:
                fsc_threshols_scaled = fsc_threshold * self.classes["snow_cover"][-1]
                daily_composite = daily_composite.where(daily_composite > fsc_threshols_scaled, 0)

            out_path = f"{str(self.output_folder)}/{day.strftime('%Y%m%d')}.nc"
            out_tmp_paths.append(out_path)

            daily_composite = daily_composite.expand_dims(time=[day])
            daily_composite.to_netcdf(out_path)

        time_series = xr.open_mfdataset(out_tmp_paths, mask_and_scale=False)
        encodings = generate_xarray_compression_encodings(time_series)
        encodings.update(time={"calendar": "gregorian", "units": f"days since {str(winter_year.from_year)}-10-01"})
        time_series.to_netcdf(
            f"{self.output_folder}/WY_{winter_year.from_year}_{winter_year.to_year}_{self.product_name}_res_{self.grid.resolution}m.nc"
        )
        [os.remove(file) for file in out_tmp_paths]
