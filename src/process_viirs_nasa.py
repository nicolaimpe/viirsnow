from glob import glob
from pathlib import Path
from typing import List
import xarray as xr
import numpy as np
from datetime import datetime
import re
import geopandas as gpd
from metrics import WinterYear
from geotools import georef_data_array, gdf_to_binary_mask, reproject_dataset, dim_name, to_rioxarray
import pyproj
import os
from logger_setup import default_logger as logger
from grids import DefaultGrid, RESAMPLING
from products import NASA_CLASSES
from fractional_snow_cover import nasa_ndsi_snow_cover_to_fraction

# Hardcode some parameters
PROJ4_MODIS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
VIIRS_COLLECTION = 2
TILE_PATTERN = "h1[7-8]v0[3-4]"
platforms_products_dict = {"SuomiNPP": "VNP", "JPSS-1": "VJ1"}

GRID = DefaultGrid()


def get_datetime_from_viirs_filepath(filepath: str) -> str:
    def timestamp_to_datetime(observation_timestamp: str) -> datetime:
        return datetime.strptime(observation_timestamp, f"%Y%j")

    return timestamp_to_datetime(Path(filepath).name.split(".")[1][1:])


def int_to_year_day(year: int, day: int) -> str:
    return str(year) + "{:03d}".format(day)


def get_daily_filenames_per_platform(platform: str, year: int, day: int, viirs_data_filepaths: List[str]) -> List[str] | None:
    platform_files = [path for path in viirs_data_filepaths if re.search(platforms_products_dict[platform], path)]
    day_files = [path for path in platform_files if re.search(f"A{int_to_year_day(year=year, day=day)}", path)]
    n_day_files = len(day_files)
    if n_day_files != 4:
        logger.info(
            f"Unexpected number of tiles corresponding to platform {platform} and day of the year {day} found. Expected 4, found: {n_day_files}"
        )
    return day_files, n_day_files


def create_nasa_composite(day_files: List[str], roi_file: str | None = None) -> xr.Dataset | None:
    day_data_arrays = []
    modis_crs = pyproj.CRS.from_proj4(PROJ4_MODIS)
    dims = dim_name(crs=modis_crs)
    for filepath in day_files:
        # try:
        logger.info(f"Processing product {Path(filepath).name}")

        dataset_grid = xr.open_dataset(filepath, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D", engine="netcdf4")
        ndsi_snow_cover = xr.open_dataset(
            filepath, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields", engine="netcdf4"
        ).data_vars["NDSI_Snow_Cover"]
        y_coords = dataset_grid.coords["YDim"].values
        x_coords = dataset_grid.coords["XDim"].values

        ndsi_snow_cover = ndsi_snow_cover.rename({"XDim": dims[1], "YDim": dims[0]}).assign_coords(
            coords={dims[0]: y_coords, dims[1]: x_coords}
        )

        day_data_arrays.append(georef_data_array(data_array=ndsi_snow_cover, data_array_name="snow_cover", crs=modis_crs))

    merged_day_dataset = xr.combine_by_coords(day_data_arrays, data_vars="minimal").astype(np.uint8)

    day_dataset_reprojected = reproject_dataset(
        dataset=to_rioxarray(merged_day_dataset),
        shape=GRID.shape,
        transform=GRID.affine,
        new_crs=pyproj.CRS(GRID.crs),
        resampling=RESAMPLING,
        fill_value=NASA_CLASSES["fill"][0],
    )

    if roi_file is not None:
        roi_mask = gdf_to_binary_mask(
            gdf=gpd.read_file(roi_file),
            grid=GRID,
        )

        masked = day_dataset_reprojected.data_vars["snow_cover"].values * roi_mask.data_vars["binary_mask"].values
        masked[roi_mask.data_vars["binary_mask"].values == 0] = NASA_CLASSES["fill"][0]
        day_dataset_reprojected.data_vars["snow_cover"][:] = masked

    # Apparently need to pop this attribute for correct encoding...not like it took me two hours to understand this :')
    # In practice, when a valid range attribute is encoded, a GDal driver reading the NetCDF will set all values outside this
    # range to NaN.
    # Since valid range in the V10A1 collection is {0,100}, i.e. the NDSI range, all other pixels (clouds, lakes etc.) are set to NaN
    # and that's not useful for the anamysis
    day_dataset_reprojected.data_vars["snow_cover"].attrs.pop("valid_range")
    # If we want not to encode the fill value like nodata
    # reprojected.data_vars["snow_cover"].attrs.pop("_FillValue")

    return day_dataset_reprojected


def create_v10a1_time_series(
    winter_year: WinterYear,
    viirs_data_folder: str,
    output_folder: str,
    roi_shapefile: str | None = None,
    platform: str = "SuomiNPP",
    convert_to_fsc: bool = False,
):
    # Treat user inputs
    viirs_data_filepaths = glob(
        str(Path(f"{viirs_data_folder}/V*10*{str(year.from_year)}*.{TILE_PATTERN}.00{VIIRS_COLLECTION}.*h5"))
    )
    viirs_data_filepaths.extend(
        glob(str(Path(f"{viirs_data_folder}/V*10*{str(year.to_year)}*.{TILE_PATTERN}.00{VIIRS_COLLECTION}.*h5")))
    )

    outpaths = []
    for day in winter_year.iterate_days():
        logger.info(f"Processing day {day}")
        day_files, n_day_files = get_daily_filenames_per_platform(
            platform=platform, year=day.year, day=day.day_of_year, viirs_data_filepaths=viirs_data_filepaths
        )
        if n_day_files == 0:
            logger.info(f"Skip day {day.date()} because 0 files were found on this day")
            continue
        try:
            nasa_composite = create_nasa_composite(day_files=day_files, roi_file=roi_shapefile)
        except OSError as e:
            logger.warning(f"Error {e} occured while reading VIIRS files. Skipping day {day.date()}.")
            continue
        if convert_to_fsc:
            nasa_composite.data_vars["snow_cover"][:] = nasa_ndsi_snow_cover_to_fraction(
                nasa_composite.data_vars["snow_cover"].values
            )

        nasa_composite = nasa_composite.expand_dims(time=[day])
        outpath = f"{output_folder}/{day.strftime('%Y%j')}.nc"
        outpaths.append(outpath)
        nasa_composite.to_netcdf(outpath)

    time_series = xr.open_mfdataset(outpaths)
    outfile_name = (
        f"WY_{year.from_year}_{year.to_year}_{platform}_nasa_time_series.nc"
        if not convert_to_fsc
        else f"WY_{year.from_year}_{year.to_year}_{platform}_nasa_fsc_time_series.nc"
    )
    output_name = Path(f"{output_folder}/{outfile_name}")
    time_series.to_netcdf(
        output_name,
        encoding={
            "time": {"calendar": "gregorian", "units": f"days since {str(year.from_year)}-10-01"},
        },
    )
    [os.remove(file) for file in outpaths]


if __name__ == "__main__":
    # User inputs
    year = WinterYear(2023, 2024)
    platform = "SuomiNPP"
    folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/V10A1/VNP10A1"
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/cms_workshop"
    roi_file = "/home/imperatoren/work/VIIRS_S2_comparison/data/vectorial/massifs/massifs.shp"

    create_v10a1_time_series(
        winter_year=year, viirs_data_folder=folder, output_folder=output_folder, roi_shapefile=roi_file, convert_to_fsc=False
    )
