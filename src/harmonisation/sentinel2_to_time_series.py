from datetime import datetime
from typing import List

import xarray as xr

from grids import GeoGrid, UTM375mGrid
from harmonisation.daily_composites import create_spatial_s2_composite
from harmonisation.harmonisation_base import HarmonisationBase
from logger_setup import default_logger as logger
from products.filenames import get_all_s2_clms_files_of_winter_year, get_all_s2_theia_files_of_winter_year
from products.plot_settings import S2_CLMS_VAR_NAME, S2_THEIA_VAR_NAME
from reductions.snow_cover_extent_cross_comparison import WinterYear


class S2CLMSHarmonisation(HarmonisationBase):
    def __init__(self, output_grid: GeoGrid, data_folder: str, output_folder: str):
        super().__init__(
            product_name=S2_CLMS_VAR_NAME, output_grid=output_grid, data_folder=data_folder, output_folder=output_folder
        )

    def get_all_files_of_winter_year(self, winter_year):
        return get_all_s2_clms_files_of_winter_year(self.data_folder, winter_year=winter_year)

    def create_spatial_composite(self, day_files):
        return create_spatial_s2_composite(day_files=day_files, output_grid=self.grid)


class S2TheiaHarmonisation(HarmonisationBase):
    def __init__(self, output_grid: GeoGrid, data_folder: str, output_folder: str):
        super().__init__(S2_THEIA_VAR_NAME, output_grid, data_folder, output_folder)

    def get_daily_files(self, all_winter_year_files: List[str], day: datetime):
        return [file for file in all_winter_year_files if day.strftime("%Y%m%d") in file]

    def get_all_files_of_winter_year(self, winter_year: WinterYear) -> List[str]:
        return get_all_s2_theia_files_of_winter_year(s2_folder=self.data_folder, winter_year=winter_year)

    def create_spatial_composite(self, day_files: List[str]) -> xr.Dataset:
        return create_spatial_s2_composite(day_files=day_files, output_grid=self.grid)


if __name__ == "__main__":
    year = WinterYear(2023, 2024)
    massifs_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/vectorial/massifs/massifs.shp"
    s2_clms_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/S2_CLMS"
    s2_theia_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/LIS_FSC_PREOP"
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_4/time_series"
    grid = UTM375mGrid()

    # logger.info('S2 CLMS processing')
    # S2CLMSHarmonisation(output_grid=grid, data_folder=s2_clms_folder, output_folder=output_folder).create_time_series(
    #     winter_year=year, roi_shapefile=massifs_shapefile
    # )

    logger.info("S2 Theia processing")
    S2TheiaHarmonisation(output_grid=grid, data_folder=s2_theia_folder, output_folder=output_folder).create_time_series(
        winter_year=year, roi_shapefile=massifs_shapefile, low_value_thresholds={"snow_cover_fraction": 10}
    )
