from datetime import datetime
from typing import List

import xarray as xr
from rasterio.enums import Resampling

from geotools import reproject_using_grid
from grids import GeoGrid, LatLon375mGrid, UTM375mGrid
from logger_setup import default_logger as logger
from products.classes import METEOFRANCE_COMPOSITE_CLASSES
from products.filenames import get_all_meteofrance_composite_filenames
from products.snow_cover_product import (
    MeteoFranceComposite,
    MeteoFranceEvalJPSS1,
    MeteoFranceEvalJPSS2,
    MeteoFranceEvalSNPP,
    SnowCoverProduct,
)
from regrid.regrid_base import RegridBase
from regrid.reprojections import reprojection_composite_meteofrance_to_grid
from winter_year import WinterYear


class MeteoFranceCompositeRegrid(RegridBase):
    def __init__(self, product: SnowCoverProduct, output_grid: GeoGrid, data_folder: str, output_folder: str):
        super().__init__(product, output_grid, data_folder, output_folder)

    def get_all_files_of_winter_year(self, winter_year: WinterYear) -> List[str]:
        return get_all_meteofrance_composite_filenames(
            data_folder=self.data_folder, winter_year=winter_year, platform=self.product.platform
        )

    def get_daily_files(self, all_winter_year_files: List[str], day: datetime) -> List[str]:
        return [file for file in all_winter_year_files if day.strftime("%Y%m%d") in file]

    def check_daily_files(self, day_files: List[str]) -> List[str]:
        return day_files

    def create_spatial_composite(self, day_files: List[str]) -> xr.Dataset:
        # day.strftime('%Y%m%d')

        daily_temporal_composite = xr.open_dataset(day_files[0])

        daily_temporal_composite = daily_temporal_composite.rio.write_crs(
            daily_temporal_composite.data_vars["spatial_ref"].attrs["spatial_ref"]
        )

        meteofrance_snow_cover = reprojection_composite_meteofrance_to_grid(
            meteofrance_snow_cover=daily_temporal_composite.data_vars["snow_cover_fraction"], output_grid=self.grid
        )

        meteofrance_view_angle = reproject_using_grid(
            dataset=daily_temporal_composite.data_vars["sensor_zenith_angle"],
            output_grid=grid,
            nodata=METEOFRANCE_COMPOSITE_CLASSES["nodata"][0],
            resampling_method=Resampling.nearest,
        )

        if self.product.platform == "all":
            meteofrance_platform = reproject_using_grid(
                dataset=daily_temporal_composite.data_vars["platform"],
                output_grid=grid,
                nodata=METEOFRANCE_COMPOSITE_CLASSES["nodata"][0],
                resampling_method=Resampling.nearest,
            )
            out_dataset = xr.Dataset(
                {
                    "snow_cover_fraction": meteofrance_snow_cover,
                    "sensor_zenith_angle": meteofrance_view_angle.astype("u1"),
                    "platform": meteofrance_platform.astype("u1"),
                }
            )
        else:
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
    meteofrance_cms_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_composite_multiplatform/CMS_rejeu/"
    grid = UTM375mGrid()
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_10/time_series/"

    for product in [MeteoFranceEvalSNPP()]:
        logger.info(f"{product.plot_name} processing")
        MeteoFranceCompositeRegrid(
            product=product,
            output_grid=grid,
            data_folder=meteofrance_cms_folder,
            output_folder=output_folder,
        ).create_time_series(winter_year=year, roi_shapefile=massifs_shapefile)

    # logger.info("Méteo-France multiplatform processing")
    # MeteoFranceMultiplatformRegrid(
    #     output_grid=grid,
    #     data_folder=meteofrance_cms_folder,
    #     output_folder=output_folder,
    #     suffix=suffix,
    # ).create_time_series(winter_year=year, roi_shapefile=massifs_shapefile)
