from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List

import xarray as xr
from geospatial_grid.gsgrid import GSGrid
from geospatial_grid.reprojections import reproject_using_grid
from ndsi_fsc_calibration.regrid import RegridBase
from rasterio.enums import Resampling

from products.classes import METEOFRANCE_COMPOSITE_CLASSES
from regrid.daily_composites import create_temporal_composite_meteofrance_single_platform
from regrid.reprojections import reprojection_l3_meteofrance_to_grid

platform_dict = {"SNPP": "npp", "JPSS1": "noaa20", "JPSS2": "noaa21", "all": "all"}


def get_all_meteofrance_prototype_type_filenames(data_folder: str, platform: str, suffix: str) -> List[str] | None:
    # Rejeu CMS
    meteofrance_files = glob(f"{data_folder}/{platform}/{suffix}/*/*/*{platform_dict[platform]}*{suffix}.tif")
    return sorted(meteofrance_files)


def get_all_meteofrance_prototype_sat_angle_filenames(data_folder: str, suffix: str, platform: str) -> List[str] | None:
    # Rejeu CMS

    meteofrance_files = glob(f"{data_folder}/{platform}/{suffix}/*/*/*{platform_dict[platform]}*SatelliteZenithAngleMod.tif")
    return sorted(meteofrance_files)


class MeteoFrancePrototypeRegrid(RegridBase):
    def __init__(self, output_grid: GSGrid, data_folder: str, output_folder: str, platform: str, suffix: str):
        super().__init__(
            output_grid=output_grid,
            data_folder=data_folder,
            output_folder=output_folder,
            product_classes=METEOFRANCE_COMPOSITE_CLASSES,
        )
        self.suffix = suffix
        self.platform = platform

    def get_date_files(self, all_winter_year_files: List[str], date: datetime) -> List[str]:
        return [file for file in all_winter_year_files if date.strftime("%Y%m%d") in file]

    def get_all_files(self) -> List[str]:
        snow_cover_and_sat_angle_file_list = get_all_meteofrance_prototype_type_filenames(
            data_folder=self.data_folder, suffix=self.suffix, platform=self.platform
        )
        snow_cover_and_sat_angle_file_list.extend(
            get_all_meteofrance_prototype_sat_angle_filenames(
                data_folder=self.data_folder, suffix=self.suffix, platform=self.platform
            )
        )

        return snow_cover_and_sat_angle_file_list

    def check_date_files(self, date_files: List[str]):
        return date_files

    def create_spatial_composite(self, date_files: List[str]) -> xr.Dataset:
        # day.strftime('%Y%m%d')

        daily_temporal_composite = create_temporal_composite_meteofrance_single_platform(
            daily_snow_cover_files=[f for f in date_files if self.suffix in Path(f).name],
            daily_geometry_files=[f for f in date_files if "SatelliteZenithAngleMod" in Path(f).name],
        )

        meteofrance_snow_cover = reprojection_l3_meteofrance_to_grid(
            meteofrance_snow_cover=daily_temporal_composite.data_vars["snow_cover_fraction"], output_grid=self.grid
        )

        meteofrance_view_angle = reproject_using_grid(
            data=daily_temporal_composite.data_vars["sensor_zenith_angle"],
            output_grid=self.grid,
            nodata=METEOFRANCE_COMPOSITE_CLASSES["nodata"][0],
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
