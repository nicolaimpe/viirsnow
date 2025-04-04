from datetime import datetime
from typing import List

import xarray as xr
from rasterio.enums import Resampling

from geotools import GeoGrid, reproject_using_grid
from grids import UTM375mGrid
from harmonisation.daily_composites import create_temporal_composite_meteofrance
from harmonisation.harmonisation_base import HarmonisationBase
from harmonisation.reprojections import reprojection_l3_meteofrance_to_grid_new
from logger_setup import default_logger as logger
from products.classes import METEOFRANCE_CLASSES
from products.filenames import get_all_meteofrance_sat_angle_filenames, get_all_meteofrance_synopsis_filenames
from products.plot_settings import METEOFRANCE_VAR_NAME
from reductions.snow_cover_extent_cross_comparison import WinterYear


class MeteoFranceSynopsisHarmonisation(HarmonisationBase):
    def __init__(self, output_grid: GeoGrid, data_folder: str, output_folder: str):
        super().__init__(METEOFRANCE_VAR_NAME, output_grid, data_folder, output_folder)

    def get_all_files_of_winter_year(self, winter_year: WinterYear) -> List[str]:
        snow_cover_and_sat_angle_file_list = get_all_meteofrance_synopsis_filenames(
            data_folder=self.data_folder, winter_year=winter_year
        )
        snow_cover_and_sat_angle_file_list.extend(
            get_all_meteofrance_sat_angle_filenames(data_folder=self.data_folder, winter_year=winter_year)
        )
        return snow_cover_and_sat_angle_file_list

    def get_daily_files(self, all_winter_year_files: List[str], day: datetime) -> List[str]:
        return [file for file in all_winter_year_files if day.strftime("%Y%m%d") in file]

    def create_spatial_composite(self, day_files: List[str]) -> xr.Dataset:
        # day.strftime('%Y%m%d')
        daily_temporal_composite = create_temporal_composite_meteofrance(
            daily_snow_cover_files=[f for f in day_files if "produit_synopsis" in f],
            daily_geometry_files=[f for f in day_files if "SatelliteZenithAngle" in f],
        )

        meteofrance_snow_cover = reprojection_l3_meteofrance_to_grid_new(
            meteofrance_snow_cover=daily_temporal_composite.data_vars["snow_cover_fraction"], output_grid=self.grid
        )

        meteofrance_view_angle = reproject_using_grid(
            dataset=daily_temporal_composite.data_vars["sensor_zenith_angle"],
            output_grid=grid,
            nodata=METEOFRANCE_CLASSES["nodata"][0],
            resampling_method=Resampling.bilinear,
        )
        out_dataset = xr.Dataset(
            {
                "snow_cover_fraction": meteofrance_snow_cover,
                "sensor_zenith_angle": meteofrance_view_angle.astype("u1"),
            }
        )
        return out_dataset


if __name__ == "__main__":
    year = WinterYear(2023, 2024)
    massifs_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/vectorial/massifs/massifs.shp"
    meteofrance_cms_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_rejeu"
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_5/time_series"
    grid = UTM375mGrid()

    logger.info("Méteo-France processing")
    MeteoFranceSynopsisHarmonisation(
        output_grid=grid, data_folder=meteofrance_cms_folder, output_folder=output_folder
    ).create_time_series(winter_year=year, roi_shapefile=massifs_shapefile)
