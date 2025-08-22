from datetime import datetime
from typing import List

import numpy as np
import xarray as xr

from fractional_snow_cover import gascoin
from grids import GeoGrid, UTM375mGrid, UTM500mGrid
from harmonisation.daily_composites import create_spatial_s2_composite, create_spatial_s2_composite_sca
from harmonisation.harmonisation_base import HarmonisationBase, check_input_daily_tif_files
from logger_setup import default_logger as logger
from products.filenames import get_all_s2_theia_files_of_winter_year
from products.plot_settings import S2_THEIA_VAR_NAME
from products.snow_cover_product import Sentinel2Theia, SnowCoverProduct
from reductions.snow_cover_extent_cross_comparison import WinterYear


class S2Harmonisation(HarmonisationBase):
    def __init__(self, product: SnowCoverProduct, output_grid: GeoGrid, data_folder: str, output_folder: str):
        super().__init__(product=product, output_grid=output_grid, data_folder=data_folder, output_folder=output_folder)

    def check_daily_files(self, day_files: List[str]) -> List[str]:
        return check_input_daily_tif_files(input_tif_files=day_files)

    def get_daily_files(self, all_winter_year_files: List[str], day: datetime):
        return [file for file in all_winter_year_files if day.strftime("%Y%m%d") in file]


class S2TheiaSCAHarmonisation(S2Harmonisation):
    def __init__(self, output_grid: GeoGrid, data_folder: str, output_folder: str, fsc_thresh: int | None = None):
        super().__init__(Sentinel2Theia(), output_grid, data_folder, output_folder)
        self.fsc_thresh = fsc_thresh

    def get_all_files_of_winter_year(self, winter_year: WinterYear) -> List[str]:
        return get_all_s2_theia_files_of_winter_year(s2_folder=self.data_folder, winter_year=winter_year)

    def create_spatial_composite(self, day_files: List[str]) -> xr.Dataset:
        return create_spatial_s2_composite_sca(day_files=day_files, output_grid=self.grid, fsc_thresh=self.fsc_thresh)


if __name__ == "__main__":
    year = WinterYear(2023, 2024)
    massifs_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/vectorial/massifs/massifs.shp"
    s2_clms_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/S2_CLMS"
    s2_theia_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/LIS_FSC_PREOP"

    grid = UTM375mGrid()
    for ndsi in [45]:
        output_folder = f"/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_8/time_series"

        fsc = int(gascoin(ndsi=ndsi / 100, f_veg=0) * 100)
        logger.info("S2 Theia processing")
        print("fsc thresh", fsc)
        S2TheiaSCAHarmonisation(
            output_grid=grid, data_folder=s2_theia_folder, output_folder=output_folder, fsc_thresh=fsc
        ).create_time_series(winter_year=year, roi_shapefile=massifs_shapefile)
