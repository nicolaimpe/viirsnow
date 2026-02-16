from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List

import rasterio
import xarray as xr
from geospatial_grid.grid_database import UTM375mGrid
from geospatial_grid.gsgrid import GSGrid
from geospatial_grid.reprojections import reproject_using_grid
from ndsi_fsc_calibration.regrid import RegridBase
from rasterio.enums import Resampling

from logger_setup import default_logger as logger
from products.classes import METEOFRANCE_ARCHIVE_CLASSES
from products.snow_cover_product import MeteoFranceArchive, MeteoFranceComposite, MeteoFrancePrototypeSNPP, SnowCoverProduct
from regrid.daily_composites import (
    create_temporal_composite_meteofrance_multiplatform,
    create_temporal_composite_meteofrance_single_platform,
)
from regrid.reprojections import reprojection_l3_meteofrance_to_grid
from winter_year import WinterYear


def get_daily_meteofrance_filenames(day: datetime, data_folder: str) -> List[str] | None:
    return glob(f"{data_folder}/VIIRS{day.year}/*EOFR62_SNPP*{day.strftime('%Y%m%d')}*.LT")


def get_all_meteofrance_archive_type_filenames(
    data_folder: str, winter_year: WinterYear, platform: str, suffix: str
) -> List[str] | None:
    # Rejeu CMS
    platform_dict = {"npp": "SNPP", "noaa20": "JPSS1", "noaa21": "JPSS2"}
    meteofrance_files = glob(
        f"{data_folder}/{platform_dict[platform]}/{suffix}/{winter_year.from_year}/1[0-2]/*{platform}*{suffix}.tif"
    )
    meteofrance_files.extend(
        glob(f"{data_folder}/{platform_dict[platform]}/{suffix}/{winter_year.to_year}/[0-9]*/*{platform}*{suffix}.tif")
    )
    return sorted(meteofrance_files)


def get_all_meteofrance_archive_sat_angle_filenames(
    data_folder: str, winter_year: WinterYear, suffix: str, platform: str
) -> List[str] | None:
    # Rejeu CMS
    platform_dict = {"npp": "SNPP", "noaa20": "JPSS1", "noaa21": "JPSS2"}
    meteofrance_files = glob(
        f"{data_folder}/{platform_dict[platform]}/{suffix}/{winter_year.from_year}/1[0-2]*/*{platform}*SatelliteZenithAngleMod.tif"
    )
    meteofrance_files.extend(
        glob(
            f"{data_folder}/{platform_dict[platform]}/{suffix}/{winter_year.to_year}/[0-9]*/*{platform}*SatelliteZenithAngleMod.tif"
        )
    )
    return sorted(meteofrance_files)


class MeteoFranceArchiveRegrid(RegridBase):
    def __init__(self, output_grid: GSGrid, data_folder: str, output_folder: str, suffix: str):
        super().__init__(
            output_grid=output_grid,
            data_folder=data_folder,
            output_folder=output_folder,
            product_classes=METEOFRANCE_ARCHIVE_CLASSES,
        )
        self.suffix = suffix

    def get_daily_files(self, all_winter_year_files: List[str], day: datetime) -> List[str]:
        return [file for file in all_winter_year_files if day.strftime("%Y%m%d") in file]

    def get_all_files_of_winter_year(self, winter_year: WinterYear) -> List[str]:
        snow_cover_and_sat_angle_file_list = get_all_meteofrance_archive_type_filenames(
            data_folder=self.data_folder, winter_year=winter_year, suffix=self.suffix, platform="npp"
        )
        snow_cover_and_sat_angle_file_list.extend(
            get_all_meteofrance_archive_sat_angle_filenames(
                data_folder=self.data_folder, winter_year=winter_year, suffix=self.suffix, platform="npp"
            )
        )

        return snow_cover_and_sat_angle_file_list

    def create_spatial_composite(self, day_files: List[str]) -> xr.Dataset:
        # day.strftime('%Y%m%d')
        daily_temporal_composite = create_temporal_composite_meteofrance_single_platform(
            daily_snow_cover_files=[f for f in day_files if self.suffix in Path(f).name],
            daily_geometry_files=[f for f in day_files if "SatelliteZenithAngleMod" in Path(f).name],
        )

        meteofrance_snow_cover = reprojection_l3_meteofrance_to_grid(
            meteofrance_snow_cover=daily_temporal_composite.data_vars["snow_cover_fraction"], output_grid=self.grid
        )

        meteofrance_view_angle = reproject_using_grid(
            dataset=daily_temporal_composite.data_vars["sensor_zenith_angle"],
            output_grid=grid,
            nodata=METEOFRANCE_ARCHIVE_CLASSES["nodata"][0],
            resampling_method=Resampling.nearest,
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


class MeteoFranceMultiplatformRegrid(RegridBase):
    def __init__(self, output_grid: GSGrid, data_folder: str, output_folder: str, suffix: str):
        super().__init__(
            product=MeteoFranceComposite(),
            output_grid=output_grid,
            data_folder=data_folder,
            output_folder=output_folder,
        )
        self.suffix = suffix

    def get_daily_files(self, all_winter_year_files: List[str], day: datetime) -> List[str]:
        return [file for file in all_winter_year_files if day.strftime("%Y%m%d") in file]

    def check_date_files(date_files: List[str]) -> List[str]:
        for day_file in date_files:
            try:
                xr.open_dataset(day_file).data_vars["band_data"].values
            except (OSError, rasterio.errors.RasterioIOError, rasterio._err.CPLE_AppDefinedError):
                logger.info(f"Could not open file {day_file}. Removing it from processing")
                date_files.remove(day_file)
                continue
        return date_files

    def get_all_files_of_winter_year(self, winter_year: WinterYear) -> List[str]:
        snow_cover_and_sat_angle_file_list = []
        platform_dict = {"npp": "SNPP", "noaa20": "JPSS1", "noaa21": "JPSS2"}
        for platform in platform_dict:
            snow_cover_and_sat_angle_file_list.extend(
                get_all_meteofrance_archive_type_filenames(
                    data_folder=f"{self.data_folder}/{platform_dict[platform]}",
                    winter_year=winter_year,
                    suffix=self.suffix,
                    platform=platform,
                )
            )

            snow_cover_and_sat_angle_file_list.extend(
                get_all_meteofrance_archive_sat_angle_filenames(
                    data_folder=f"{self.data_folder}/{platform_dict[platform]}/",
                    winter_year=winter_year,
                    suffix=self.suffix,
                    platform=platform,
                )
            )
        return snow_cover_and_sat_angle_file_list

    def create_spatial_composite(self, day_files: List[str]) -> xr.Dataset:
        daily_temporal_composite = create_temporal_composite_meteofrance_multiplatform(
            daily_snow_cover_files=[f for f in day_files if self.suffix in Path(f).name],
            daily_geometry_files=[f for f in day_files if "SatelliteZenithAngleMod" in Path(f).name],
        )

        meteofrance_snow_cover = reprojection_l3_meteofrance_to_grid(
            meteofrance_snow_cover=daily_temporal_composite.data_vars["snow_cover_fraction"], output_grid=self.grid
        )

        meteofrance_view_angle = reproject_using_grid(
            dataset=daily_temporal_composite.data_vars["sensor_zenith_angle"],
            output_grid=grid,
            nodata=METEOFRANCE_ARCHIVE_CLASSES["nodata"][0],
            resampling_method=Resampling.nearest,
        )

        meteofrance_platform = reproject_using_grid(
            dataset=daily_temporal_composite.data_vars["platform"],
            output_grid=grid,
            nodata=METEOFRANCE_ARCHIVE_CLASSES["nodata"][0],
            resampling_method=Resampling.nearest,
        )
        if "ndsi" in self.suffix:
            out_data_var_name = "NDSI_Snow_Cover"
        else:
            out_data_var_name = "snow_cover_fraction"

        out_dataset = xr.Dataset(
            {
                out_data_var_name: meteofrance_snow_cover,
                "sensor_zenith_angle": meteofrance_view_angle.astype("u1"),
                "platform": meteofrance_platform.astype("u1"),
            }
        )

        return out_dataset


if __name__ == "__main__":
    year = WinterYear(2023, 2024)

    suffixes = ["no_forest_red_band_screen"]
    massifs_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/vectorial/massifs/massifs.shp"
    meteofrance_cms_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_rejeu/"
    grid = UTM375mGrid()

    logger.info("Méteo-France multiplatform processing")
    MeteoFranceMultiplatformRegrid(
        output_grid=grid,
        data_folder=meteofrance_cms_folder,
        output_folder="./output_folder",
    ).create_time_series(winter_year=year, roi_shapefile=massifs_shapefile)
