import ast
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List

import rasterio

from winter_year import WinterYear

nasa_format_per_product = {"VNP10A1": "h5", "VNP10_UTM_375m": "nc", "VNP03IMG_UTM_375m": "nc"}
nasa_collection_per_product_id = {"VNP10A1": "V10A1", "VNP10_UTM_375m": "V10", "VNP03IMG_UTM_375m": "V03IMG"}


def timestamp_nasa_to_datetime(observation_timestamp: str) -> datetime:
    return datetime.strptime(observation_timestamp, "A%Y%j%H%M")


def get_datetime_from_viirs_nasa_filepath(filepath: str) -> datetime:
    _, obs_year_day, obs_hour_minute, _, __, _ = Path(filepath).name.split(".")
    return timestamp_nasa_to_datetime(obs_year_day + obs_hour_minute)


def int_to_year_day(year: int, day: int) -> str:
    return str(year) + "{:03d}".format(day)


def get_all_nasa_filenames_per_product(product_id: str, data_folder: str) -> List[str] | None:
    version_id = "002"
    return glob(
        f"{data_folder}/{nasa_collection_per_product_id[product_id]}/{product_id}/{product_id}*.{version_id}*.{nasa_format_per_product[product_id]}"
    )


def get_datetime_from_viirs_meteofrance_filepath(filepath: str) -> str:
    # This function is not use for the moment...to see whether to take it out
    def timestamp_to_datetime(observation_timestamp: str) -> datetime:
        return datetime.strptime(observation_timestamp, "%Y-%m-%dT%H:%M:%SZ")

    meteofrance_raster = rasterio.open(filepath)
    return timestamp_to_datetime(ast.literal_eval(meteofrance_raster.tags()["TIFFTAG_IMAGEDESCRIPTION"])["time"])


def get_daily_meteofrance_filenames(day: datetime, data_folder: str) -> List[str] | None:
    return glob(f"{data_folder}/VIIRS{day.year}/*EOFR62_SNPP*{day.strftime('%Y%m%d')}*.LT")


def get_all_meteofrance_type_filenames(data_folder: str, winter_year: WinterYear, suffix: str) -> List[str] | None:
    # Rejeu CMS
    meteofrance_files = glob(f"{data_folder}/{winter_year.from_year}1[0-2]/*npp*{suffix}.tif")
    meteofrance_files.extend(glob(f"{data_folder}/{winter_year.to_year}[0-9]*/*npp*{suffix}.tif"))
    return sorted(meteofrance_files)


def get_all_meteofrance_sat_angle_filenames(data_folder: str, winter_year: WinterYear) -> List[str] | None:
    # Rejeu CMS
    meteofrance_files = glob(f"{data_folder}/{winter_year.from_year}1[0-2]*/*npp*SatelliteZenithAngleMod.tif")
    meteofrance_files.extend(glob(f"{data_folder}/{winter_year.to_year}[0-9]*/*npp*SatelliteZenithAngleMod.tif"))
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
