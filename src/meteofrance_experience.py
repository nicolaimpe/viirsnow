from glob import glob
from typing import List

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

from products.filenames import get_all_meteofrance_synopsis_filenames
from winter_year import WinterYear

folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_rejeu"


def get_all_meteofrance_cloud_filenames(data_folder: str, winter_year: WinterYear) -> List[str] | None:
    # Rejeu CMS
    meteofrance_files = glob(f"{data_folder}/{winter_year.from_year}1[0-2]/*npp*cloud.tif")
    meteofrance_files.extend(glob(f"{data_folder}/{winter_year.to_year}[0-9]*/*npp*cloud.tif"))
    return sorted(meteofrance_files)


def get_all_meteofrance_fsc_filenames(data_folder: str, winter_year: WinterYear) -> List[str] | None:
    # Rejeu CMS
    meteofrance_files = glob(f"{data_folder}/{winter_year.from_year}1[0-2]/*npp*fsc.tif")
    meteofrance_files.extend(glob(f"{data_folder}/{winter_year.to_year}[0-9]*/*npp*fsc.tif"))
    return sorted(meteofrance_files)


def get_all_meteofrance_cc_mask_filenames(data_folder: str, winter_year: WinterYear) -> List[str] | None:
    # Rejeu CMS
    meteofrance_files = glob(f"{data_folder}/{winter_year.from_year}1[0-2]/*npp*CCNPPJSNOW_mask.tif")
    meteofrance_files.extend(glob(f"{data_folder}/{winter_year.to_year}[0-9]*/*npp*CCNPPJSNOW_mask.tif"))
    return sorted(meteofrance_files)


def get_all_meteofrance_cc_filenames(data_folder: str, winter_year: WinterYear) -> List[str] | None:
    # Rejeu CMS
    meteofrance_files = glob(f"{data_folder}/{winter_year.from_year}1[0-2]/*npp*CCNPPJSNOW_reproj.tif")
    meteofrance_files.extend(glob(f"{data_folder}/{winter_year.to_year}[0-9]*/*npp*CCNPPJSNOW_reproj.tif"))
    return sorted(meteofrance_files)


wy = WinterYear(2023, 2024)
rejeu_files = get_all_meteofrance_synopsis_filenames(data_folder=folder, winter_year=wy)
cloud_files = get_all_meteofrance_cloud_filenames(data_folder=folder, winter_year=wy)
fsc_files = get_all_meteofrance_fsc_filenames(data_folder=folder, winter_year=wy)
cc_mask_files = get_all_meteofrance_cc_mask_filenames(data_folder=folder, winter_year=wy)
cc_files = get_all_meteofrance_cc_filenames(data_folder=folder, winter_year=wy)

forest_mask = rasterio.open(
    "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2006_forest_mask_geo.tif"
).read(1)

count = 0
for rejeu_file, cloud_file, fsc_file, cc_mask_file, cc_file in zip(
    rejeu_files, cloud_files, fsc_files, cc_mask_files, cc_files
):
    print("Processing ", rejeu_file)
    profile = rasterio.open(rejeu_file).profile
    rejeu = rasterio.open(rejeu_file).read(1)
    cloud = rasterio.open(cloud_file).read(1)
    fsc = rasterio.open(fsc_file).read(1)
    cc_mask = rasterio.open(cc_mask_file).read(1)
    cc_nir_800 = rasterio.open(cc_file).read(1)

    orig = np.where(cloud == 2, 250, rejeu)

    distance_mask = distance_transform_edt(cc_mask)
    distance_mask[distance_mask < 17] = 1
    distance_mask[distance_mask >= 17] = 0
    rejeu_no_cc_mask = np.where((1 - distance_mask) * (fsc >= 0) * (rejeu <= 215), fsc, rejeu)
    rejeu_no_cc_mask = np.where(forest_mask * (rejeu_no_cc_mask == 0), 215, rejeu_no_cc_mask)
    rejeu_no_cc_mask = np.where(forest_mask * (rejeu_no_cc_mask > 0) * (rejeu_no_cc_mask <= 200), 210, rejeu_no_cc_mask)

    low_refl_mask = (cc_nir_800 < 40) * (fsc <= 100) * (fsc > 0)
    modified = np.where(rejeu > 210, rejeu, np.where(1 - low_refl_mask, rejeu, 0))
    modified = np.where((modified == 0) * forest_mask, 215, modified)

    with rasterio.open(rejeu_file.replace("produit_synopsis", "cc_dist_mask"), "w", **profile) as dst:
        dst.write(distance_mask, 1)

    with rasterio.open(rejeu_file.replace("produit_synopsis", "produit_orig"), "w", **profile) as dst:
        dst.write(orig, 1)

    with rasterio.open(rejeu_file.replace("produit_synopsis", "produit_no_cc_mask"), "w", **profile) as dst:
        dst.write(rejeu_no_cc_mask, 1)

    with rasterio.open(rejeu_file.replace("produit_synopsis", "produit_modified"), "w", **profile) as dst:
        dst.write(modified, 1)
