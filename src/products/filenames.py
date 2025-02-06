import ast
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List

import rasterio

NASA_L2_SNOW_PRODUCTS_IDS = ["VNP10", "VJ110", "VNP10_NRT", "VJ110_NRT"]
NASA_L3_SNOW_PRODUCTS_IDS = ["VNP10A1", "VJ110A1"]
NASA_L2_GEOMETRY_PRODUCTS_IDS = ["VNP03IMG", "VJ103IMG", "VNP03IMG_NRT", "VJ103IMG_NRT"]
NASA_L2_SNOW_PRODUCTS = {
    "Standard": {"SNPP": NASA_L2_SNOW_PRODUCTS_IDS[0], "JPSS1": NASA_L2_SNOW_PRODUCTS_IDS[1]},
    "NRT": {"SNPP": NASA_L2_SNOW_PRODUCTS_IDS[2], "JPSS1": NASA_L2_SNOW_PRODUCTS_IDS[3]},
}
NASA_L3_SNOW_PRODUCTS = {
    "Standard": {"SNPP": NASA_L3_SNOW_PRODUCTS_IDS[0], "JPSS1": NASA_L3_SNOW_PRODUCTS_IDS[1]},
}

NASA_L2_GEOM_PRODUCTS = {
    "Standard": {"SNPP": NASA_L2_GEOMETRY_PRODUCTS_IDS[0], "JPSS1": NASA_L2_GEOMETRY_PRODUCTS_IDS[1]},
    "NRT": {"SNPP": NASA_L2_GEOMETRY_PRODUCTS_IDS[2], "JPSS1": NASA_L2_GEOMETRY_PRODUCTS_IDS[3]},
}
METEOFRANCE_L2 = {"SNPP": "EOFR62_SNPP"}
KNOWN_COLLECTIONS = {
    "V10": NASA_L2_SNOW_PRODUCTS,
    "V10A1": NASA_L3_SNOW_PRODUCTS,
    "V03IMG": NASA_L2_GEOM_PRODUCTS,
    "Meteo-France": METEOFRANCE_L2,
    "S2": "FSC",
}

VIIRS_NASA_VERSION = 2


def timestamp_nasa_to_datetime(observation_timestamp: str) -> datetime:
    return datetime.strptime(observation_timestamp, "A%Y%j%H%M")


def get_datetime_from_viirs_nasa_filepath(filepath: str) -> datetime:
    _, obs_year_day, obs_hour_minute, _, __, _ = Path(filepath).name.split(".")
    return timestamp_nasa_to_datetime(obs_year_day + obs_hour_minute)


def int_to_year_day(year: int, day: int) -> str:
    return str(year) + "{:03d}".format(day)


def get_daily_nasa_filenames_per_product(product_id: str, day: datetime, data_folder: str) -> List[str] | None:
    return glob(f"{data_folder}/{product_id}_*A{day.strftime('%Y%j')}*.nc")


def get_datetime_from_viirs_meteofrance_filepath(filepath: str) -> str:
    # This function is not use for the moment...to see whether to take it out
    def timestamp_to_datetime(observation_timestamp: str) -> datetime:
        return datetime.strptime(observation_timestamp, "%Y-%m-%dT%H:%M:%SZ")

    meteofrance_raster = rasterio.open(filepath)
    return timestamp_to_datetime(ast.literal_eval(meteofrance_raster.tags()["TIFFTAG_IMAGEDESCRIPTION"])["time"])


def get_daily_meteofrance_filenames(day: datetime, data_folder: str) -> List[str] | None:
    return glob(f"{data_folder}/VIIRS{day.year}/*{METEOFRANCE_L2['SNPP']}_*{day.strftime('%Y%m%d')}*.LT")
