from datetime import datetime
from glob import glob
from typing import List

import xarray as xr
from geospatial_grid.gsgrid import GSGrid
from geospatial_grid.reprojections import reproject_using_grid
from ndsi_fsc_calibration.regrid import RegridBase
from rasterio.enums import Resampling

from products.classes import METEOFRANCE_COMPOSITE_CLASSES
from regrid.reprojections import reprojection_composite_meteofrance_to_grid
from winter_year import WinterYear


def get_all_meteofrance_composite_filenames(data_folder: str, winter_year: WinterYear, platform: str) -> List[str] | None:
    # Rejeu CMS
    meteofrance_files = glob(f"{data_folder}/{winter_year.from_year}1[0-2]/1[0-2]/*{platform}*.nc")
    meteofrance_files.extend(glob(f"{data_folder}/{winter_year.to_year}0[1-9]/0[1-9]/*{platform}*.nc"))
    return sorted(meteofrance_files)


class MeteoFranceCompositeRegrid(RegridBase):
    def __init__(self, platform: str, output_grid: GSGrid, data_folder: str, output_folder: str):
        super().__init__(output_grid=output_grid, data_folder=data_folder, output_folder=output_folder)
        self.platform = platform

    def get_all_files_of_winter_year(self, winter_year: WinterYear) -> List[str]:
        return get_all_meteofrance_composite_filenames(
            data_folder=self.data_folder, winter_year=winter_year, platform=self.platform
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
            output_grid=self.grid,
            nodata=METEOFRANCE_COMPOSITE_CLASSES["nodata"][0],
            resampling_method=Resampling.nearest,
        )

        if self.platform == "all":
            meteofrance_platform = reproject_using_grid(
                dataset=daily_temporal_composite.data_vars["platform"],
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
