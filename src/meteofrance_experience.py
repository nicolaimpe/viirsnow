from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import rasterio

from products.classes import METEOFRANCE_CLASSES
from winter_year import WinterYear


def get_all_meteofrance_type_rejeu_filenames(data_folder: str, winter_year: WinterYear, suffix: str) -> List[str] | None:
    # Rejeu CMS
    meteofrance_files = glob(f"{data_folder}/{winter_year.from_year}1[0-2]/*npp*{suffix}.tif")
    meteofrance_files.extend(glob(f"{data_folder}/{winter_year.to_year}0[0-9]/*npp*{suffix}.tif"))
    return sorted(meteofrance_files)


def get_all_meteofrance_red_band_filenames(data_folder: str, winter_year: WinterYear) -> List[str] | None:
    # Rejeu CMS

    meteofrance_files = glob(f"{data_folder}/red_band/{winter_year.from_year}1[0-2]/*I01.tif")
    meteofrance_files.extend(glob(f"{data_folder}/red_band/{winter_year.to_year}0[0-9]/*I01.tif"))
    return sorted(meteofrance_files)


class MeteoFranceExperience:
    def __init__(self, data_folder: str, winter_year: WinterYear, forest_mask_path: str | None = None):
        self.data_folder = data_folder
        self.cloud_mask_relaxed_product_filenames = get_all_meteofrance_type_rejeu_filenames(
            data_folder=data_folder, winter_year=winter_year, suffix="synopsis"
        )
        if forest_mask_path:
            self.forest_mask = rasterio.open(forest_mask_path).read(1)
        self.parse_filenames_fun_dict = {"no_forest_red_band_screen": self.get_all_fsc_and_red_band_filenames}
        self.new_product_fun_dict = {"no_forest_red_band_screen": self.create_no_forest_red_band_screen}

    def get_all_fsc_and_red_band_filenames(self):
        self.fsc_files = get_all_meteofrance_type_rejeu_filenames(data_folder=folder, winter_year=wy, suffix="fsc")
        self.red_band_files = get_all_meteofrance_red_band_filenames(data_folder=folder, winter_year=wy)

    def create_no_forest_red_band_screen(self, time: datetime, old_product: np.array, red_band_screen_value: int = 7):
        self.get_all_fsc_and_red_band_filenames()

        red_band_file_list = []
        red_band_file_list.extend([f for f in self.red_band_files if time.strftime("%Y%m%d_%H%M") in f])
        red_band_file_list.extend(
            [f for f in self.red_band_files if (time - timedelta(minutes=1)).strftime("%Y%m%d_%H%M") in f]
        )
        red_band_file_list.extend(
            [f for f in self.red_band_files if (time - timedelta(minutes=-1)).strftime("%Y%m%d_%H%M") in f]
        )
        red_band_file_list.extend(
            [f for f in self.red_band_files if (time - timedelta(minutes=2)).strftime("%Y%m%d_%H%M") in f]
        )

        fsc_file = [f for f in self.fsc_files if time.strftime("%Y%m%d_%H%M") in f][0]
        fsc = rasterio.open(fsc_file).read(1)
        no_forest = np.where(old_product == METEOFRANCE_CLASSES["forest_with_snow"], fsc, old_product)

        if len(red_band_file_list) != 1:
            print("0 or more than 1 band files found for this acquisition. Red band screen not applied.")
            modified = no_forest
        else:
            red_band = rasterio.open(red_band_file_list[0]).read(1)
            low_refl_mask = red_band <= red_band_screen_value
            modified = np.where(
                no_forest > METEOFRANCE_CLASSES["snow_cover"][-1],
                no_forest,
                np.where(1 - low_refl_mask, no_forest, METEOFRANCE_CLASSES["no_snow"][0]),
            )
        return modified

    def create_new_meteofrance_product(self, which: str):
        for product_filename in self.cloud_mask_relaxed_product_filenames:
            time = datetime.strptime(Path(product_filename).name[:13], "%Y%m%d_%H%M")
            if time.month >= 10:
                continue
            if time.month < 7:
                continue
            print(f"Processing image with time stamp {time.strftime('%Y%m%d_%H%M')}")
            cloud_relaxed_product = rasterio.open(product_filename).read(1)
            profile = rasterio.open(product_filename).profile
            new_product = self.new_product_fun_dict[which](time, cloud_relaxed_product)
            with rasterio.open(
                product_filename.replace("produit_synopsis", which).replace(self.data_folder, f"{self.data_folder}/{which}/"),
                "w",
                **profile,
            ) as dst:
                dst.write(new_product, 1)


if __name__ == "__main__":
    wy = WinterYear(2023, 2024)
    folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_rejeu"
    MeteoFranceExperience(
        data_folder=folder,
        winter_year=wy,
        forest_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2006_forest_mask_geo.tif",
    ).create_new_meteofrance_product(which="no_forest_red_band_screen")

# cloud_files = get_all_meteofrance_type_rejeu_filenames(data_folder=folder, winter_year=wy, suffix="cloud")
# fsc_files = get_all_meteofrance_type_rejeu_filenames(data_folder=folder, winter_year=wy, suffix="fsc")
# cc_mask_files = get_all_meteofrance_type_rejeu_filenames(data_folder=folder, winter_year=wy, suffix="CCNPPJSNOW_mask")
# cc_files = get_all_meteofrance_type_rejeu_filenames(data_folder=folder, winter_year=wy, suffix="CCNPPJSNOW_reproj")
# ndsi_files = get_all_meteofrance_type_rejeu_filenames(data_folder=folder, winter_year=wy, suffix="ndsi")


# count = 0
#
#     time = datetime.strptime(Path(rejeu_file).name[:13], "%Y%m%d_%H%M")
#     cloud_file = [f for f in cloud_files if time.strftime("%Y%m%d_%H%M") in f][0]
#     fsc_file = [f for f in fsc_files if time.strftime("%Y%m%d_%H%M") in f][0]
#     cc_mask_file = [f for f in cc_mask_files if time.strftime("%Y%m%d_%H%M") in f][0]
#     cc_file = [f for f in cc_files if time.strftime("%Y%m%d_%H%M") in f][0]
#     ndsi_file = [f for f in ndsi_files if time.strftime("%Y%m%d_%H%M") in f][0]
#     print("Processing ", rejeu_file)
#     print("Processing ", cloud_file)
#     print("Processing ", fsc_file)
#     print("Processing ", cc_mask_file)
#     print("Processing ", cc_file)
#     print("Processing ", ndsi_file)
#     print(" ")

#     # if count < 110:
#     #     count += 1
#     #     continue
#     # if count > 112:
#     #     break
#     profile = rasterio.open(rejeu_file).profile
#     rejeu = rasterio.open(rejeu_file).read(1)
#     cloud = rasterio.open(cloud_file).read(1)
#     fsc = rasterio.open(fsc_file).read(1)
#     cc_mask = rasterio.open(cc_mask_file).read(1)
#     cc_nir_800 = rasterio.open(cc_file).read(1)
#     ndsi = rasterio.open(ndsi_file).read(1)

#     # orig = np.where(cloud == 2, METEOFRANCE_CLASSES["clouds"][0], rejeu)

#     # distance_mask = distance_transform_edt(cc_mask)
#     # distance_mask[distance_mask < 17] = 1
#     # distance_mask[distance_mask >= 17] = 0
#     # rejeu_no_cc_mask = np.where((1 - distance_mask) * (fsc > 0) * (rejeu < METEOFRANCE_CLASSES["water"][0]), fsc, rejeu)

#     # rejeu_no_cc_mask = np.where(
#     #     forest_mask * (rejeu_no_cc_mask > 0) * (rejeu_no_cc_mask <= METEOFRANCE_CLASSES["snow_cover"][-1]),
#     #     METEOFRANCE_CLASSES["forest_with_snow"][0],
#     #     rejeu_no_cc_mask,
#     # )

#     # FSC<100 approx NDSI 0.66
#     low_refl_mask = (cc_nir_800 < 80) * (fsc <= 20) * (fsc > 0)
#     # modified = np.where(
#     #     rejeu > METEOFRANCE_CLASSES["forest_with_snow"][0],
#     #     rejeu,
#     #     np.where(1 - low_refl_mask, rejeu, METEOFRANCE_CLASSES["no_snow"][0]),
#     # )
#     # modified = np.where((modified == 0) * forest_mask, METEOFRANCE_CLASSES["forest_without_snow"][0], modified)

#     # with rasterio.open(rejeu_file.replace("produit_synopsis", "cc_dist_mask"), "w", **profile) as dst:
#     #     dst.write(distance_mask, 1)

#     # with rasterio.open(rejeu_file.replace("produit_synopsis", "produit_orig"), "w", **profile) as dst:
#     #     dst.write(orig, 1)

#     # with rasterio.open(rejeu_file.replace("produit_synopsis", "produit_no_cc_mask"), "w", **profile) as dst:
#     #     dst.write(rejeu_no_cc_mask, 1)

#     # with rasterio.open(rejeu_file.replace("produit_synopsis", "produit_modified"), "w", **profile) as dst:
#     #     dst.write(modified, 1)

#     # ndsi_snow_cover = np.where((rejeu > 0) * (rejeu <= 200), (ndsi - 100) * 2, rejeu)
#     # with rasterio.open(rejeu_file.replace("produit_synopsis", "ndsi_snow_cover"), "w", **profile) as dst:
#     #     dst.write(ndsi_snow_cover, 1)

#     no_forest = np.where(rejeu == METEOFRANCE_CLASSES["forest_with_snow"], fsc, rejeu)
#     # with rasterio.open(rejeu_file.replace("produit_synopsis", "no_forest"), "w", **profile) as dst:
#     #     dst.write(no_forest, 1)

#     # ndsi_no_forest = np.where((no_forest > 0) * (no_forest <= 200), (ndsi - 100) * 2, no_forest)
#     # with rasterio.open(rejeu_file.replace("produit_synopsis", "ndsi_no_forest"), "w", **profile) as dst:
#     #     dst.write(ndsi_no_forest, 1)

#     no_forest_modified = np.where(
#         no_forest > METEOFRANCE_CLASSES["forest_with_snow"][0],
#         no_forest,
#         np.where(1 - low_refl_mask, no_forest, METEOFRANCE_CLASSES["no_snow"][0]),
#     )
#     no_forest_modified = np.where(
#         (no_forest_modified == 0) * forest_mask, METEOFRANCE_CLASSES["forest_without_snow"][0], no_forest_modified
#     )
#     with rasterio.open(
#         rejeu_file.replace("produit_synopsis", "no_forest_modified_new").replace(
#             "CMS_rejeu", "CMS_rejeu/no_forest_modified_new"
#         ),
#         "w",
#         **profile,
#     ) as dst:
#         dst.write(no_forest_modified, 1)
