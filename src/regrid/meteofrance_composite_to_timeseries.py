from datetime import datetime
from glob import glob
from typing import List

import xarray as xr
from geospatial_grid.gsgrid import GSGrid
from geospatial_grid.reprojections import reproject_using_grid
from ndsi_fsc_calibration.regrid import RegridBase
from rasterio.enums import Resampling

from products.classes import METEOFRANCE_COMPOSITE_CLASSES
from regrid.meteofrance_l2_to_l3_time_series import platform_dict
from regrid.reprojections import reprojection_composite_meteofrance_to_grid


def get_all_meteofrance_composite_filenames(data_folder: str, platform: str) -> List[str] | None:
    # Rejeu CMS
    meteofrance_files = glob(f"{data_folder}/*/*/*{platform_dict[platform]}*.nc")
    return sorted(meteofrance_files)


class MeteoFranceCompositeRegrid(RegridBase):
    def __init__(self, platform: str, output_grid: GSGrid, data_folder: str, output_folder: str):
        super().__init__(
            output_grid=output_grid,
            data_folder=data_folder,
            output_folder=output_folder,
            product_classes=METEOFRANCE_COMPOSITE_CLASSES,
        )
        self.platform = platform

    def get_all_files(self) -> List[str]:
        return get_all_meteofrance_composite_filenames(data_folder=self.data_folder, platform=self.platform)

    def get_date_files(self, all_winter_year_files: List[str], date: datetime) -> List[str]:
        return [file for file in all_winter_year_files if date.strftime("%Y%m%d") in file]

    def check_date_files(self, date_files: List[str]) -> List[str]:
        return date_files

    def create_spatial_composite(self, date_files: List[str]) -> xr.Dataset:
        # day.strftime('%Y%m%d')

        daily_temporal_composite = xr.open_dataset(date_files[0])

        daily_temporal_composite = daily_temporal_composite.rio.write_crs(
            daily_temporal_composite.data_vars["spatial_ref"].attrs["spatial_ref"]
        )

        meteofrance_snow_cover = reprojection_composite_meteofrance_to_grid(
            meteofrance_snow_cover=daily_temporal_composite.data_vars["snow_cover_fraction"], output_grid=self.grid
        )

        meteofrance_view_angle = reproject_using_grid(
            data=daily_temporal_composite.data_vars["sensor_zenith_angle"],
            output_grid=self.grid,
            nodata=METEOFRANCE_COMPOSITE_CLASSES["nodata"][0],
            resampling_method=Resampling.nearest,
        )

        if self.platform == "all":
            meteofrance_platform = reproject_using_grid(
                data=daily_temporal_composite.data_vars["platform"],
                output_grid=self.grid,
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
