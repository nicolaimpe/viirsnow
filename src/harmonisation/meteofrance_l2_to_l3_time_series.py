from datetime import datetime
from pathlib import Path
from typing import List

import xarray as xr
from rasterio.enums import Resampling

from geotools import GeoGrid, reproject_using_grid
from grids import UTM375mGrid
from harmonisation.daily_composites import create_temporal_composite_meteofrance
from harmonisation.harmonisation_base import HarmonisationBase, check_input_daily_tif_files
from harmonisation.reprojections import reprojection_l3_meteofrance_to_grid
from logger_setup import default_logger as logger
from products.classes import METEOFRANCE_CLASSES
from products.filenames import get_all_meteofrance_sat_angle_filenames, get_all_meteofrance_type_filenames
from products.snow_cover_product import (
    SnowCoverProduct,
    VIIRSMeteoFranceArchive,
    VIIRSMeteoFranceJPSS1Prototype,
    VIIRSMeteoFranceJPSS2Prototype,
    VIIRSMeteoFranceSNPPPrototype,
)
from reductions.snow_cover_extent_cross_comparison import WinterYear


class MeteoFranceHarmonisation(HarmonisationBase):
    def __init__(self, product: SnowCoverProduct, output_grid: GeoGrid, data_folder: str, output_folder: str, suffix: str):
        super().__init__(product, output_grid, data_folder, output_folder)
        self.suffix = suffix

    def get_all_files_of_winter_year(self, winter_year: WinterYear) -> List[str]:
        snow_cover_and_sat_angle_file_list = get_all_meteofrance_type_filenames(
            data_folder=self.data_folder, winter_year=winter_year, suffix=self.suffix, platform=self.product.platform
        )
        print(snow_cover_and_sat_angle_file_list)
        snow_cover_and_sat_angle_file_list.extend(
            get_all_meteofrance_sat_angle_filenames(
                data_folder=self.data_folder, winter_year=winter_year, suffix=suffix, platform=self.product.platform
            )
        )

        return snow_cover_and_sat_angle_file_list

    def get_daily_files(self, all_winter_year_files: List[str], day: datetime) -> List[str]:
        return [file for file in all_winter_year_files if day.strftime("%Y%m%d") in file]

    def check_daily_files(self, day_files: List[str]) -> List[str]:
        return check_input_daily_tif_files(input_tif_files=day_files)

    def create_spatial_composite(self, day_files: List[str]) -> xr.Dataset:
        # day.strftime('%Y%m%d')
        daily_temporal_composite = create_temporal_composite_meteofrance(
            daily_snow_cover_files=[f for f in day_files if suffix in Path(f).name],
            daily_geometry_files=[f for f in day_files if "SatelliteZenithAngleMod" in Path(f).name],
        )

        meteofrance_snow_cover = reprojection_l3_meteofrance_to_grid(
            meteofrance_snow_cover=daily_temporal_composite.data_vars["snow_cover_fraction"], output_grid=self.grid
        )

        meteofrance_view_angle = reproject_using_grid(
            dataset=daily_temporal_composite.data_vars["sensor_zenith_angle"],
            output_grid=grid,
            nodata=METEOFRANCE_CLASSES["nodata"][0],
            resampling_method=Resampling.bilinear,
        )
        if "ndsi" in self.suffix:
            out_data_var_name = "NDSI_Snow_Cover"
        else:
            out_data_var_name = "snow_cover_fraction"
        out_dataset = xr.Dataset(
            {
                out_data_var_name: meteofrance_snow_cover,
                "sensor_zenith_angle": meteofrance_view_angle.astype("u1"),
            }
        )
        return out_dataset


if __name__ == "__main__":
    year = WinterYear(2023, 2024)

    suffixes = ["no_forest_red_band_screen"]
    massifs_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/vectorial/massifs/massifs.shp"
    meteofrance_cms_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_rejeu/JPSS2"
    grid = UTM375mGrid()
    for suffix in suffixes:
        output_folder = f"/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_7/time_series/{suffix}"

        logger.info(f"Méteo-France {suffix} processing")
        MeteoFranceHarmonisation(
            product=VIIRSMeteoFranceJPSS2Prototype(),
            output_grid=grid,
            data_folder=meteofrance_cms_folder,
            output_folder=output_folder,
            suffix=suffix,
        ).create_time_series(winter_year=year, roi_shapefile=massifs_shapefile)
