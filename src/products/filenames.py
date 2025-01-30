from datetime import datetime
from glob import glob
from pathlib import Path
import re
from typing import List
from logger_setup import default_logger as logger

PLATFORMS_PRODUCT_DICT_NASA = {"SuomiNPP": "VNP10A1"}
PLATFORMS_PRODUCT_DICT_METEOFRANCE = {"SuomiNPP": "SNPP"}
VIIRS_COLLECTION = 2


def timestamp_viirs_to_datetime(observation_timestamp: str) -> datetime:
    return datetime.strptime(observation_timestamp, f"%Y%j")


def get_datetime_from_viirs_nasa_filepath(filepath: str) -> str:
    return timestamp_viirs_to_datetime(Path(filepath).name.split(".")[1][1:])


def int_to_year_day(year: int, day: int) -> str:
    return str(year) + "{:03d}".format(day)


def get_daily_nasa_filenames_per_platform(
    platform: str, year: int, day: int, viirs_data_filepaths: List[str]
) -> List[str] | None:
    platform_files = [path for path in viirs_data_filepaths if re.search(PLATFORMS_PRODUCT_DICT_NASA[platform], path)]
    day_files = [path for path in platform_files if re.search(f"A{int_to_year_day(year=year, day=day)}", path)]
    n_day_files = len(day_files)
    if n_day_files != 4:
        logger.info(
            f"Unexpected number of tiles corresponding to platform {platform} and day of the year {day} found. Expected 4, found: {n_day_files}"
        )
    return day_files, n_day_files


def get_datetime_from_viirs_meteofrance_filepath(filepath: str) -> str:
    # This function is not use for the moment...to see whether to take it out
    def timestamp_to_datetime(observation_timestamp: str) -> datetime:
        return datetime.strptime(observation_timestamp, "%Y%m%d%H%M%S")

    return timestamp_to_datetime(Path(filepath).name.split(".")[1].split("_")[0])


def get_daily_meteofrance_filenames_per_platform(platform: str, day: datetime, viirs_data_folder: str) -> List[str] | None:
    return glob(
        f"{viirs_data_folder}/VIIRS{day.year}/*_{PLATFORMS_PRODUCT_DICT_METEOFRANCE[platform]}_*{day.strftime('%Y%m%d')}*.LT"
    )
