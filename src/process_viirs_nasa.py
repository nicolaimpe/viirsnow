from glob import glob
from pathlib import Path
from typing import List
import xarray as xr
import numpy as np
from datetime import datetime
import re
import geopandas as gpd
from pathlib import Path
from metrics import WinterYear
from geotools import georef_data_array, gdf_to_binary_mask, reproject_dataset, dim_name, find_nearest_bounds_for_selection
import pyproj
import rasterio
import os
from logger_setup import default_logger as logger


# Hardcode some parameters
LAT = "lat"
LON = "lon"
PROJ4_MODIS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
OUT_GRID_CRS = 32631
OUT_GRID_RES = 250  # m
VIIRS_COLLECTION = 2
TILE_PATTERN = "h1[7-8]v0[3-4]"
RESAMPLING_METHOD = rasterio.enums.Resampling.nearest
FILL_VALUE = 255
platforms_products_dict = {"SuomiNPP": "VNP", "JPSS-1": "VJ1"}


def timestamp_to_datetime(observation_timestamp: str) -> datetime:
    return datetime.strptime(observation_timestamp, f"%Y%j")


def get_viirs_timestamp_from_filepath(filepath: str) -> str:
    return Path(filepath).name.split(".")[1][1:]


def get_dateteime_from_viirs_filepath(filepath: str) -> str:
    return timestamp_to_datetime(get_viirs_timestamp_from_filepath(filepath))


def int_to_year_day(year: int, day: int) -> str:
    return str(year) + "{:03d}".format(day)


def get_daily_filenames_per_platform(platform: str, year: int, day: int, viirs_data_filepaths: List[str]) -> List[str] | None:
    platform_files = [path for path in viirs_data_filepaths if re.search(platforms_products_dict[platform], path)]
    day_files = [path for path in platform_files if re.search(f"A{int_to_year_day(year=year, day=day)}", path)]
    if len(day_files) != 4:
        logger.info(
            f"Unexpected number of tiles corresponding to platform {platform} and day of the year {day} found. Expected 4, found: {len(day_files)}"
        )
        if len(day_files) == 0:
            return None
    else:
        return day_files


def create_nasa_composite(day_files: List[str], roi_file: str | None = None) -> xr.Dataset:
    day_data_arrays = []
    modis_crs = pyproj.CRS.from_proj4(PROJ4_MODIS)
    dims = dim_name(crs=modis_crs)
    for filepath in day_files:
        try:
            logger.info(f"Processing product {Path(filepath).name}")
            xr.open_dataset(filepath, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields", engine="netcdf4")
        except OSError as e:
            logger.warning(f"Error {e} occured while reading VIIRS files. Skipping day.")
            return None
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
    merged = xr.combine_by_coords(day_data_arrays, data_vars="minimal")
    if roi_file is not None:
        roi_mask = gdf_to_binary_mask(
            gdf=gpd.read_file(roi_file),
            out_resolution=merged.rio.resolution()[0],
            out_crs=pyproj.CRS.from_wkt(merged.data_vars["spatial_ref"].attrs["spatial_ref"]),
        )

        xmin, xmax, ymin, ymax = find_nearest_bounds_for_selection(data_array=merged, other=roi_mask)
        dims = dim_name(merged)
        data_to_reproject = merged.sel({dims[1]: slice(xmin, xmax), dims[0]: slice(ymin, ymax)})
        masked = data_to_reproject.data_vars["snow_cover"].values * roi_mask.data_vars["binary_mask"].values
        masked[masked == 0] = FILL_VALUE
        data_to_reproject.data_vars["snow_cover"][:] = masked
    else:
        data_to_reproject = merged

    reprojected = reproject_dataset(
        data_to_reproject,
        new_resolution=OUT_GRID_RES,
        new_crs=pyproj.CRS.from_epsg(OUT_GRID_CRS),
        resampling=RESAMPLING_METHOD,
    )

    return reprojected


def create_v10_time_series(
    winter_year: WinterYear,
    viirs_data_folder: str,
    output_folder: str,
    roi_shapefile: str | None = None,
    platform: str = "SuomiNPP",
):
    # Treat user inputs
    roi = gpd.read_file(Path(roi_shapefile))
    viirs_data_filepaths = glob(
        str(Path(f"{viirs_data_folder}/V*10*{str(year.from_year)}*.{TILE_PATTERN}.00{VIIRS_COLLECTION}.*h5"))
    )
    viirs_data_filepaths.extend(
        glob(str(Path(f"{viirs_data_folder}/V*10*{str(year.to_year)}*.{TILE_PATTERN}.00{VIIRS_COLLECTION}.*h5")))
    )

    outpaths = []
    for day in winter_year.iterate_days():
        logger.info(f"Processing day {day}")
        day_files = get_daily_filenames_per_platform(
            platform=platform, year=day.year, day=day.day_of_year, viirs_data_filepaths=viirs_data_filepaths
        )
        if day_files is None:
            continue
        nasa_composite = create_nasa_composite(day_files=day_files, roi=roi)
        if nasa_composite is None:
            continue
        nasa_composite = nasa_composite.expand_dims(time=[day])
        outpath = f"{output_folder}/{day.strftime("%Y%j")}.nc"
        outpaths.append(outpath)
        nasa_composite.to_netcdf(outpath)

    time_series = xr.open_mfdataset(outpaths)
    output_name = Path(f"{output_folder}/WY_{year.from_year}_{year.to_year}_{platform}_nasa_time_series.nc")
    time_series.to_netcdf(
        output_name, encoding={"time": {"calendar": "gregorian", "units": f"days since {str(year.from_year)}-10-01"}}
    )
    [os.remove(file) for file in outpaths]


if __name__ == "__main__":
    # User inputs
    year = WinterYear(2023, 2024)
    platform = "SuomiNPP"
    folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/V10A1/VNP10A1"
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/snow_cover_extent_analysis"
    roi_file = "/home/imperatoren/work/VIIRS_S2_comparison/data/vectorial/massifs/massifs.shp"

    create_v10_time_series(winter_year=year, viirs_data_folder=folder, output_folder=output_folder, roi_shapefile=roi_file)
