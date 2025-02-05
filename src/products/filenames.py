import re
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List

from logger_setup import default_logger as logger

NASA_L2_SNOW_PRODUCTS_IDS = ["VNP10", "VJ110", "VNP10_NRT", "VJ110_NRT"]
NASA_L3_SNOW_PRODUCTS_IDS = ["VNP10A1", "VJ110A1"]
NASA_L2_GEOMETRY_PRODUCTS_IDS = ["VNP03IMG", "VJ103IMG", "VNP03IMG_NRT", "VJ103IMG_NRT"]
NASA_L2_SNOW_PRODUCTS = {
    "Standard": {"Suomi-NPP": NASA_L2_SNOW_PRODUCTS_IDS[0], "JPSS1": NASA_L2_SNOW_PRODUCTS_IDS[1]},
    "NRT": {"Suomi-NPP": NASA_L2_SNOW_PRODUCTS_IDS[2], "JPSS1": NASA_L2_SNOW_PRODUCTS_IDS[3]},
}
NASA_L3_SNOW_PRODUCTS = {
    "Standard": {"Suomi-NPP": NASA_L3_SNOW_PRODUCTS_IDS[0], "JPSS1": NASA_L3_SNOW_PRODUCTS_IDS[1]},
}

NASA_L2_GEOM_PRODUCTS = {
    "Standard": {"Suomi-NPP": NASA_L2_GEOMETRY_PRODUCTS_IDS[0], "JPSS1": NASA_L2_GEOMETRY_PRODUCTS_IDS[1]},
    "NRT": {"Suomi-NPP": NASA_L2_GEOMETRY_PRODUCTS_IDS[2], "JPSS1": NASA_L2_GEOMETRY_PRODUCTS_IDS[3]},
}
METEOFRANCE_L2 = {"Suomi-NPP": "EOFR62_SNPP"}
KNOWN_COLLECTIONS = {
    "V10": NASA_L2_SNOW_PRODUCTS,
    "V10A1": NASA_L3_SNOW_PRODUCTS,
    "V03IMG": NASA_L2_GEOM_PRODUCTS,
    "Meteo-France": METEOFRANCE_L2,
    "S2": "FSC",
}

VIIRS_COLLECTION = 2


def timestamp_nasa_to_datetime(observation_timestamp: str) -> datetime:
    return datetime.strptime(observation_timestamp, "A%Y%j%H%M")


def get_datetime_from_viirs_nasa_filepath(filepath: str) -> datetime:
    _, obs_year_day, obs_hour_minute, _, __, _ = Path(filepath).name.split(".")
    return timestamp_nasa_to_datetime(obs_year_day + obs_hour_minute)


def int_to_year_day(year: int, day: int) -> str:
    return str(year) + "{:03d}".format(day)


def get_daily_nasa_filenames_per_platform(
    product_id: str, year: int, day: int, viirs_data_filepaths: List[str]
) -> List[str] | None:
    platform_files = [path for path in viirs_data_filepaths if re.search(product_id, path)]
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
    return glob(f"{viirs_data_folder}/VIIRS{day.year}/*_{METEOFRANCE_L2[platform]}_*{day.strftime('%Y%m%d')}*.LT")
