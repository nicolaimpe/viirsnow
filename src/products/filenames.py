import ast
import re
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import rasterio
import rioxarray
import xarray as xr
from pyhdf.SD import SD, SDC

from grids import georef_netcdf_rioxarray
from products.georef import MODIS_CRS
from winter_year import WinterYear

nasa_format_per_product = {
    "VNP10A1": "h5",
    "VJ110A1": "h5",
    "MOD10A1": "hdf",
    "VNP10_UTM_375m": "nc",
    "VNP03IMG_UTM_375m": "nc",
}
nasa_collection_per_product_id = {
    "VNP10A1": "V10A1",
    "VJ110A1": "V10A1",
    "MOD10A1": "M10A1",
    "VNP10_UTM_375m": "V10",
    "VNP03IMG_UTM_375m": "V03IMG",
}


def timestamp_nasa_to_datetime(observation_timestamp: str) -> datetime:
    return datetime.strptime(observation_timestamp, "A%Y%j%H%M")


def get_datetime_from_viirs_nasa_filepath(filepath: str) -> datetime:
    _, obs_year_day, obs_hour_minute, _, __, _ = Path(filepath).name.split(".")
    return timestamp_nasa_to_datetime(obs_year_day + obs_hour_minute)


def int_to_year_day(year: int, day: int) -> str:
    return str(year) + "{:03d}".format(day)


def get_all_nasa_filenames_per_product(product_id: str, data_folder: str, winter_year: WinterYear) -> List[str] | None:
    version_id = "002"
    if product_id == "MOD10A1":
        version_id = "061"
    file_list = []
    for day in winter_year.iterate_days():
        path_pattern = f"{data_folder}/{nasa_collection_per_product_id[product_id]}/{product_id}/{product_id}.A{day.strftime('%Y%j')}*.{version_id}*.{nasa_format_per_product[product_id]}"
        file_list.extend(glob(path_pattern))
    return file_list


def get_datetime_from_viirs_meteofrance_filepath(filepath: str) -> str:
    # This function is not use for the moment...to see whether to take it out
    def timestamp_to_datetime(observation_timestamp: str) -> datetime:
        return datetime.strptime(observation_timestamp, "%Y-%m-%dT%H:%M:%SZ")

    meteofrance_raster = rasterio.open(filepath)
    return timestamp_to_datetime(ast.literal_eval(meteofrance_raster.tags()["TIFFTAG_IMAGEDESCRIPTION"])["time"])


def open_modis_ndsi_snow_cover(filepath: str) -> xr.DataArray:
    DATAFIELD_NAME = "NDSI_Snow_Cover"

    hdf = SD(filepath, SDC.READ)

    # Read dataset.
    data2D = hdf.select(DATAFIELD_NAME)
    data = data2D[:, :].astype(np.float64)

    fattrs = hdf.attributes(full=1)
    ga = fattrs["StructMetadata.0"]
    gridmeta = ga[0]

    ul_regex = re.compile(
        r"""UpperLeftPointMtrs=\(
    (?P<upper_left_x>[+-]?\d+\.\d+)
    ,
    (?P<upper_left_y>[+-]?\d+\.\d+)
    \)""",
        re.VERBOSE,
    )
    match = ul_regex.search(gridmeta)
    x0 = float(match.group("upper_left_x"))
    y0 = float(match.group("upper_left_y"))

    lr_regex = re.compile(
        r"""LowerRightMtrs=\(
    (?P<lower_right_x>[+-]?\d+\.\d+)
    ,
    (?P<lower_right_y>[+-]?\d+\.\d+)
    \)""",
        re.VERBOSE,
    )
    match = lr_regex.search(gridmeta)
    x1 = float(match.group("lower_right_x"))
    y1 = float(match.group("lower_right_y"))
    ny, nx = data.shape
    xinc = (x1 - x0) / nx
    yinc = (y1 - y0) / ny

    x = np.linspace(x0, x0 + xinc * nx, nx)
    y = np.linspace(y0, y0 + yinc * ny, ny)

    out_data_array = xr.DataArray(data=data, coords={"y": y, "x": x})
    return georef_netcdf_rioxarray(xr.Dataset({DATAFIELD_NAME: out_data_array}), crs=MODIS_CRS)


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


def get_all_meteofrance_composite_filenames(data_folder: str, winter_year: WinterYear, platform: str) -> List[str] | None:
    # Rejeu CMS
    print(f"{data_folder}/{winter_year.from_year}1[0-2]/1[0-2]/*{platform}*.nc")
    meteofrance_files = glob(f"{data_folder}/{winter_year.from_year}1[0-2]/1[0-2]/*{platform}*.nc")
    meteofrance_files.extend(glob(f"{data_folder}/{winter_year.to_year}0[1-9]/0[1-9]/*{platform}*.nc"))
    return sorted(meteofrance_files)


def get_all_s2_clms_files_of_winter_year(s2_folder: str, winter_year: WinterYear) -> List[str]:
    s2_files = glob(str(Path(s2_folder).joinpath(f"FSC_*{winter_year.from_year}1[0-2]*/*FSCOG.tif")))
    s2_files.extend(glob(str(Path(s2_folder).joinpath(f"FSC_*{winter_year.to_year}0[1-9]*/*FSCOG.tif"))))
    return sorted(s2_files)


def get_all_s2_theia_files_of_winter_year(s2_folder: str, winter_year: WinterYear) -> List[str]:
    s2_files = glob(str(Path(s2_folder).joinpath(f"LIS_S2-SNOW-FSC_*{winter_year.from_year}1[0-2]*.tif")))
    s2_files.extend(glob(str(Path(s2_folder).joinpath(f"LIS_S2-SNOW-FSC_*{winter_year.to_year}0[1-9]*.tif"))))
    return sorted(s2_files)


def get_datetime_from_s2_filepath(filepath: str):
    observation_timestamp = Path(filepath).name.split("_")[1]
    observation_datetime = datetime.strptime(observation_timestamp[:8], "%Y%m%d")
    return observation_datetime
