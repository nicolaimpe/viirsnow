from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import rasterio

from products.classes import METEOFRANCE_ARCHIVE_CLASSES
from winter_year import WinterYear


def get_all_meteofrance_type_rejeu_filenames(data_folder: str, winter_year: WinterYear, suffix: str) -> List[str] | None:
    # Rejeu CMS
    meteofrance_files = glob(f"{data_folder}/produit_synopsis/{winter_year.from_year}/1[0-2]/*npp*{suffix}.tif")
    meteofrance_files.extend(glob(f"{data_folder}/produit_synopsis/{winter_year.to_year}/0[0-9]/*npp*{suffix}.tif"))
    return sorted(meteofrance_files)


def get_all_meteofrance_red_band_rejeu_filenames(data_folder: str, winter_year: WinterYear, suffix: str) -> List[str] | None:
    # Rejeu CMS
    meteofrance_files = glob(f"{data_folder}/red_band/{winter_year.from_year}1[0-2]/*npp*{suffix}.tif")
    meteofrance_files.extend(glob(f"{data_folder}/red_band/{winter_year.to_year}0[0-9]/*npp*{suffix}.tif"))
    return sorted(meteofrance_files)


# def get_all_meteofrance_red_band_filenames(data_folder: str, winter_year: WinterYear) -> List[str] | None:
#     # Rejeu CMS

#     meteofrance_files = glob(f"{data_folder}/{winter_year.from_year}1[0-2]/*I01.tif")
#     meteofrance_files.extend(glob(f"{data_folder}/{winter_year.to_year}0[0-9]/*I01.tif"))
#     return sorted(meteofrance_files)


class MeteoFranceExperience:
    def __init__(self, data_folder: str, winter_year: WinterYear, forest_mask_path: str | None = None):
        self.data_folder = data_folder
        self.cloud_mask_relaxed_product_filenames = get_all_meteofrance_type_rejeu_filenames(
            data_folder=data_folder, winter_year=winter_year, suffix="synopsis"
        )
        if forest_mask_path:
            self.forest_mask = rasterio.open(forest_mask_path).read(1)
        self.parse_filenames_fun_dict = {
            "no_forest_red_band_screen": self.get_all_fsc_and_red_band_filenames,
            "no_forest_red_band_screen_10": self.get_all_fsc_and_red_band_filenames,
        }
        self.new_product_fun_dict = {
            "no_forest_red_band_screen": self.create_no_forest_red_band_screen,
            "no_forest_red_band_screen_10": self.create_no_forest_red_band_screen,
        }

    def get_all_fsc_and_red_band_filenames(self):
        self.fsc_files = get_all_meteofrance_type_rejeu_filenames(data_folder=folder, winter_year=wy, suffix="fsc")
        self.red_band_files = get_all_meteofrance_red_band_rejeu_filenames(data_folder=folder, winter_year=wy, suffix="I01")
        # self.red_band_files = get_all_meteofrance_red_band_filenames(data_folder=folder, winter_year=wy)

    def create_no_forest_red_band_screen(self, time: datetime, old_product: np.array, red_band_screen_value: int = 10):
        self.get_all_fsc_and_red_band_filenames()
        fsc_file = [f for f in self.fsc_files if time.strftime("%Y%m%d_%H%M") in f][0]
        red_band_file = [f for f in self.red_band_files if time.strftime("%Y%m%d_%H%M") in f]
        fsc = rasterio.open(fsc_file).read(1)
        no_forest = np.where(old_product == METEOFRANCE_ARCHIVE_CLASSES["forest_with_snow"], fsc, old_product)

        if len(red_band_file) != 1:
            print("0 or more than 1 band files found for this acquisition. Red band screen not applied.")
            modified = no_forest
        else:
            red_band = rasterio.open(red_band_file[0]).read(1)
            low_refl_mask = red_band <= red_band_screen_value
            modified = np.where(
                no_forest > METEOFRANCE_ARCHIVE_CLASSES["snow_cover"][-1],
                no_forest,
                np.where(1 - low_refl_mask, no_forest, METEOFRANCE_ARCHIVE_CLASSES["no_snow"][0]),
            )
        return modified

    def create_new_meteofrance_product(self, which: str):
        for product_filename in self.cloud_mask_relaxed_product_filenames:
            time = datetime.strptime(Path(product_filename).name[:13], "%Y%m%d_%H%M")
            print(f"Processing image with time stamp {time.strftime('%Y%m%d_%H%M')}")
            cloud_relaxed_product = rasterio.open(product_filename).read(1)
            profile = rasterio.open(product_filename).profile
            new_product = self.new_product_fun_dict[which](time, cloud_relaxed_product)
            with rasterio.open(
                product_filename.replace("produit_synopsis", which).replace(self.data_folder, f"{self.data_folder}/"),
                "w",
                **profile,
            ) as dst:
                dst.write(new_product, 1)


if __name__ == "__main__":
    wy = WinterYear(2023, 2024)
    folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_rejeu/SNPP"
    MeteoFranceExperience(
        data_folder=folder,
        winter_year=wy,
        forest_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2006/corine_2006_forest_mask_geo.tif",
    ).create_new_meteofrance_product(which="no_forest_red_band_screen_10")
