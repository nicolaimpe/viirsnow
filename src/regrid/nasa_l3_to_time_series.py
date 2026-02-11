import os
from datetime import datetime
from glob import glob
from typing import List

import xarray as xr

from compression import generate_xarray_compression_encodings
from fractional_snow_cover import nasa_ndsi_snow_cover_to_fraction
from grids import GeoGrid, UTM375mGrid, UTM500mGrid, SIN375mGrid
from logger_setup import default_logger as logger
from products.filenames import get_all_nasa_filenames_per_product, open_modis_ndsi_snow_cover
from products.snow_cover_product import MOD10A1, VJ110A1, VNP10A1, NASASnowCoverProduct, SnowCoverProduct, V10A1Multiplatform
from regrid.daily_composites import (
    create_spatial_l3_nasa_modis_composite,
    create_spatial_l3_nasa_viirs_composite,
    create_temporal_composite_nasa,
    create_temporal_l3_naive_composite_nasa,
    match_daily_snow_cover_and_geometry_nasa,
)
from regrid.regrid_base import RegridBase
from regrid.reprojections import reprojection_l3_nasa_to_grid
from winter_year import WinterYear


class V10Regrid(RegridBase):
    def __init__(self, product: NASASnowCoverProduct, output_grid: GeoGrid, data_folder: str, output_folder: str):
        super().__init__(product, output_grid, data_folder, output_folder)

    def get_all_files_of_winter_year(self, winter_year: WinterYear) -> List[str]:
        snow_cover_file_list = get_all_nasa_filenames_per_product(
            data_folder=self.data_folder, product_id=self.product.product_id, winter_year=winter_year
        )
        return snow_cover_file_list

    def get_daily_files(self, all_winter_year_files: List[str], day: datetime) -> List[str]:
        return [file for file in all_winter_year_files if day.strftime("A%Y%j") in file]

    def check_daily_files(self, day_files: List[str]) -> List[str]:
        for day_file in day_files:
            try:
                xr.open_dataset(day_file, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields", engine="netcdf4").data_vars[
                    "NDSI_Snow_Cover"
                ].values
            except OSError:
                logger.info(f"Could not open file {day_file}. Removing it from processing")
                day_files.remove(day_file)
                continue
        return day_files

    def create_spatial_composite(self, day_files: List[str]) -> xr.Dataset:
        daily_spatial_composite = create_spatial_l3_nasa_viirs_composite(daily_snow_cover_files=day_files)
        nasa_snow_cover = reprojection_l3_nasa_to_grid(nasa_snow_cover=daily_spatial_composite, output_grid=self.grid)
        nasa_snow_cover.attrs.pop("valid_range")
        out_dataset = xr.Dataset(
            {
                "NDSI_Snow_Cover": nasa_snow_cover,
                "snow_cover_fraction": xr.DataArray(
                    nasa_ndsi_snow_cover_to_fraction(nasa_ndsi_snow_cover_product=nasa_snow_cover.values, method="mine"),
                    coords=nasa_snow_cover.coords,
                ),
            }
        )
        return out_dataset


class MOD10Regrid(RegridBase):
    def __init__(self, product: NASASnowCoverProduct, output_grid: GeoGrid, data_folder: str, output_folder: str):
        super().__init__(product, output_grid, data_folder, output_folder)

    def get_all_files_of_winter_year(self, winter_year: WinterYear) -> List[str]:
        snow_cover_file_list = get_all_nasa_filenames_per_product(
            data_folder=self.data_folder, product_id=self.product.product_id, winter_year=winter_year
        )
        return snow_cover_file_list

    def get_daily_files(self, all_winter_year_files: List[str], day: datetime) -> List[str]:
        return [file for file in all_winter_year_files if day.strftime("A%Y%j") in file]

    def check_daily_files(self, day_files: List[str]) -> List[str]:
        for day_file in day_files:
            try:
                open_modis_ndsi_snow_cover(day_file).values
            except OSError:
                logger.info(f"Could not open file {day_file}. Removing it from processing")
                day_files.remove(day_file)
                continue
        return day_files

    def create_spatial_composite(self, day_files: List[str]) -> xr.Dataset:
        daily_spatial_composite = create_spatial_l3_nasa_modis_composite(daily_snow_cover_files=day_files)
        nasa_snow_cover = reprojection_l3_nasa_to_grid(nasa_snow_cover=daily_spatial_composite, output_grid=self.grid)
        out_dataset = xr.Dataset(
            {
                "NDSI_Snow_Cover": nasa_snow_cover,
                "snow_cover_fraction": xr.DataArray(
                    nasa_ndsi_snow_cover_to_fraction(nasa_ndsi_snow_cover_product=nasa_snow_cover.values),
                    coords=nasa_snow_cover.coords,
                ),
            }
        )
        return out_dataset


# class V10MultiplatformRegrid(RegridBase):
#     def __init__(self, data_folder: str, output_folder: str):
#         self.data_folder = data_folder
#         self.output_folder = output_folder
#         self.product = V10A1Multiplatform()

#     def create_multiplatform_composite(self, winter_year: WinterYear) -> xr.Dataset:
#         snpp_year = xr.open_dataset(
#             f"{self.data_folder}/time_series/WY_{winter_year.from_year}_{winter_year.to_year}_{VNP10A1().name}.nc",
#             mask_and_scale=False,
#         )
#         jpss1_year = xr.open_dataset(
#             f"{self.data_folder}/time_series/WY_{winter_year.from_year}_{winter_year.to_year}_{VJ110A1().name}.nc",
#             mask_and_scale=False,
#         )

#         for day in winter_year.iterate_days():
#             logger.info(f"Processing day {day}")

#             day_data_arrays = []
#             if day in snpp_year.coords["time"].values:
#                 day_data_arrays.append(snpp_year.data_vars["NDSI_Snow_Cover"].sel(time=day))
#             if day in jpss1_year.coords["time"].values:
#                 day_data_arrays.append(jpss1_year.data_vars["NDSI_Snow_Cover"].sel(time=day))

#             if len(day_data_arrays) == 0:
#                 logger.info(f"Skip day {day.date()} because 0 files were found on this date")
#                 continue
#             daily_temporal_ndsi_composite = create_temporal_l3_naive_composite_nasa(daily_data_arrays=day_data_arrays)
#             daily_composite = xr.Dataset(
#                 {
#                     "NDSI_Snow_Cover": daily_temporal_ndsi_composite,
#                     "snow_cover_fraction": xr.DataArray(
#                         nasa_ndsi_snow_cover_to_fraction(nasa_ndsi_snow_cover_product=daily_temporal_ndsi_composite.values),
#                         coords=daily_temporal_ndsi_composite.coords,
#                     ),
#                 }
#             )

#             self.export_daily_data(day=day, daily_data=daily_composite)
#         self.export_time_series(winter_year=winter_year)


if __name__ == "__main__":
    year = WinterYear(2023, 2024)
    massifs_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/vectorial/cantal_bbox.shp"
    nasa_l3_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/"
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/geospatial_grid/data"
    grid = SIN375mGrid()

    for product in [VJ110A1()]:
        logger.info(f"NASA L3 processing {product.name}")
        V10Regrid(
            product=product, output_grid=grid, data_folder=nasa_l3_folder, output_folder=output_folder
        ).create_time_series(winter_year=year, roi_shapefile=massifs_shapefile)

    # grid = UTM500mGrid()

    # product = VNP10A1()
    # logger.info(f"NASA L3 processing {product.name}")
    # V10Regrid(product=product, output_grid=grid, data_folder=nasa_l3_folder, output_folder=output_folder).create_time_series(
    #     winter_year=year, roi_shapefile=massifs_shapefile
    # )

    # product = MOD10A1()
    # logger.info(f"NASA L3 processing {product.name}")
    # MOD10Regrid(product=product, output_grid=grid, data_folder=nasa_l3_folder, output_folder=output_folder).create_time_series(
    #     winter_year=year, roi_shapefile=massifs_shapefile
    # )

    # logger.info("NASA psuedo L3 processing")
    # NASAPseudoL3Regrid(output_grid=grid, data_folder=nasa_l3_folder, output_folder=output_folder).create_time_series(
    #     winter_year=year, roi_shapefile=massifs_shapefile
    # )

    # logger.info("NASA multiplatform processing")
    # V10MultiplatformRegrid(
    #     data_folder="/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_6_lps",
    #     output_folder=output_folder,
    # ).create_multiplatform_composite(winter_year=year)

    # grid = UTM500mGrid()
    # platform = "terra"
    # product = MOD10A1()
    # logger.info(f"NASA L3 processing {product.name}")
    # MOD10Regrid(
    #     product=product,
    #     output_grid=grid,
    #     data_folder=nasa_l3_folder,
    #     output_folder=output_folder,
    # ).create_time_series(winter_year=year, roi_shapefile=massifs_shapefile)
