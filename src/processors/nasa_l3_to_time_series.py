from glob import glob
from pathlib import Path
from typing import List
import xarray as xr
import numpy as np
import geopandas as gpd
from daily_composites import create_nasa_composite
from metrics import WinterYear
from geotools import georef_data_array, gdf_to_binary_mask, reproject_dataset, dim_name, to_rioxarray
import os
from logger_setup import default_logger as logger
from grids import DefaultGrid, Grid, DefaultGrid_1km
from products.classes import NASA_CLASSES
from products.georef import modis_crs
from products.filenames import VIIRS_COLLECTION, get_daily_nasa_filenames_per_platform
from fractional_snow_cover import nasa_ndsi_snow_cover_to_fraction
from rasterio.enums import Resampling


def create_v10a1_time_series(
    winter_year: WinterYear,
    output_grid: Grid,
    viirs_data_folder: str,
    output_folder: str,
    output_name: str,
    roi_shapefile: str | None = None,
    platform: str = "SuomiNPP",
    ndsi_to_fsc_regression: str | None = None,
):
    # Treat user inputs
    viirs_data_filepaths = glob(str(Path(f"{viirs_data_folder}/V*10*{str(year.from_year)}*.00{VIIRS_COLLECTION}.*h5")))
    viirs_data_filepaths.extend(glob(str(Path(f"{viirs_data_folder}/V*10*{str(year.to_year)}*.00{VIIRS_COLLECTION}.*h5"))))

    outpaths = []
    for day in winter_year.iterate_days():
        if day.year == 2024:
            continue
        logger.info(f"Processing day {day}")
        day_files, n_day_files = get_daily_nasa_filenames_per_platform(
            platform=platform, year=day.year, day=day.day_of_year, viirs_data_filepaths=viirs_data_filepaths
        )
        if n_day_files == 0:
            logger.info(f"Skip day {day.date()} because 0 files were found on this day")
            continue
        try:
            nasa_composite = create_nasa_composite(day_files=day_files, output_grid=output_grid, roi_file=roi_shapefile)
        except OSError as e:
            logger.warning(f"Error {e} occured while reading VIIRS files. Skipping day {day.date()}.")
            continue

        if ndsi_to_fsc_regression is not None:
            snow_cover_fraction = nasa_ndsi_snow_cover_to_fraction(
                nasa_composite.data_vars["NDSI_Snow_Cover"].values, method=ndsi_to_fsc_regression
            )
            snow_cover_fraction_data_array = xr.zeros_like(nasa_composite.data_vars["NDSI_Snow_Cover"])
            snow_cover_fraction_data_array[:] = snow_cover_fraction
            nasa_composite = nasa_composite.assign({"snow_cover_fraction": snow_cover_fraction_data_array})
            nasa_composite.data_vars["snow_cover_fraction"].attrs["NDSI_to_FSC_conversion"] = ndsi_to_fsc_regression

        nasa_composite = nasa_composite.expand_dims(time=[day])
        outpath = f"{output_folder}/{day.strftime('%Y%j')}.nc"
        outpaths.append(outpath)
        nasa_composite.to_netcdf(outpath)

    time_series = xr.open_mfdataset(outpaths)
    output_name = Path(f"{output_folder}/{output_name}")
    time_series.to_netcdf(
        output_name,
        encoding={
            "time": {"calendar": "gregorian", "units": f"days since {str(year.from_year)}-10-01"},
        },
    )
    [os.remove(file) for file in outpaths]


if __name__ == "__main__":
    # User inputs
    year = WinterYear(2024, 2025)
    grid_375m = DefaultGrid()
    grid_1km = DefaultGrid_1km()
    grid = grid_375m

    platform = "SuomiNPP"
    folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/V10A1/VNP10A1"
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_3"
    output_name = f"WY_{year.from_year}_{year.to_year}_{platform}_nasa_l3_time_series_res_{grid.resolution}m.nc"
    roi_file = "/home/imperatoren/work/VIIRS_S2_comparison/data/vectorial/massifs/massifs.shp"

    create_v10a1_time_series(
        winter_year=year,
        output_grid=grid,
        viirs_data_folder=folder,
        output_folder=output_folder,
        output_name=output_name,
        roi_shapefile=roi_file,
        ndsi_to_fsc_regression="salomonson_appel",
    )
