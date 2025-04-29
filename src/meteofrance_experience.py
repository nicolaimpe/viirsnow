import re
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

from products.classes import METEOFRANCE_CLASSES
from products.filenames import get_all_meteofrance_type_filenames, get_daily_meteofrance_filenames
from winter_year import WinterYear

folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_rejeu"


wy = WinterYear(2023, 2024)
rejeu_files = get_all_meteofrance_type_filenames(data_folder=folder, winter_year=wy, suffix="synopsis")
cloud_files = get_all_meteofrance_type_filenames(data_folder=folder, winter_year=wy, suffix="cloud")
fsc_files = get_all_meteofrance_type_filenames(data_folder=folder, winter_year=wy, suffix="fsc")
cc_mask_files = get_all_meteofrance_type_filenames(data_folder=folder, winter_year=wy, suffix="CCNPPJSNOW_mask")
cc_files = get_all_meteofrance_type_filenames(data_folder=folder, winter_year=wy, suffix="CCNPPJSNOW_reproj")
ndsi_files = get_all_meteofrance_type_filenames(data_folder=folder, winter_year=wy, suffix="ndsi")

forest_mask = rasterio.open(
    "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2006_forest_mask_geo.tif"
).read(1)


count = 0


for rejeu_file in rejeu_files:
    time = datetime.strptime(Path(rejeu_file).name[:13], "%Y%m%d_%H%M")
    cloud_file = [f for f in cloud_files if time.strftime("%Y%m%d_%H%M") in f][0]
    fsc_file = [f for f in fsc_files if time.strftime("%Y%m%d_%H%M") in f][0]
    cc_mask_file = [f for f in cc_mask_files if time.strftime("%Y%m%d_%H%M") in f][0]
    cc_file = [f for f in cc_files if time.strftime("%Y%m%d_%H%M") in f][0]
    ndsi_file = [f for f in ndsi_files if time.strftime("%Y%m%d_%H%M") in f][0]
    print("Processing ", rejeu_file)
    print("Processing ", cloud_file)
    print("Processing ", fsc_file)
    print("Processing ", cc_mask_file)
    print("Processing ", cc_file)
    print("Processing ", ndsi_file)

    # if count < 290:
    #     count += 1
    #     continue
    profile = rasterio.open(rejeu_file).profile
    rejeu = rasterio.open(rejeu_file).read(1)
    cloud = rasterio.open(cloud_file).read(1)
    fsc = rasterio.open(fsc_file).read(1)
    cc_mask = rasterio.open(cc_mask_file).read(1)
    cc_nir_800 = rasterio.open(cc_file).read(1)
    ndsi = rasterio.open(ndsi_file).read(1)

    orig = np.where(cloud == 2, METEOFRANCE_CLASSES["clouds"][0], rejeu)

    distance_mask = distance_transform_edt(cc_mask)
    distance_mask[distance_mask < 17] = 1
    distance_mask[distance_mask >= 17] = 0
    rejeu_no_cc_mask = np.where((1 - distance_mask) * (fsc > 0) * (rejeu < METEOFRANCE_CLASSES["water"][0]), fsc, rejeu)

    rejeu_no_cc_mask = np.where(
        forest_mask * (rejeu_no_cc_mask > 0) * (rejeu_no_cc_mask <= METEOFRANCE_CLASSES["snow_cover"][-1]),
        METEOFRANCE_CLASSES["forest_with_snow"][0],
        rejeu_no_cc_mask,
    )

    # FSC<100 approx NDSI 0.66
    low_refl_mask = (cc_nir_800 < 40) * (fsc <= 100) * (fsc > 0)
    modified = np.where(
        rejeu > METEOFRANCE_CLASSES["forest_with_snow"][0],
        rejeu,
        np.where(1 - low_refl_mask, rejeu, METEOFRANCE_CLASSES["no_snow"][0]),
    )
    modified = np.where((modified == 0) * forest_mask, METEOFRANCE_CLASSES["forest_without_snow"][0], modified)

    with rasterio.open(rejeu_file.replace("produit_synopsis", "cc_dist_mask"), "w", **profile) as dst:
        dst.write(distance_mask, 1)

    with rasterio.open(rejeu_file.replace("produit_synopsis", "produit_orig"), "w", **profile) as dst:
        dst.write(orig, 1)

    with rasterio.open(rejeu_file.replace("produit_synopsis", "produit_no_cc_mask"), "w", **profile) as dst:
        dst.write(rejeu_no_cc_mask, 1)

    with rasterio.open(rejeu_file.replace("produit_synopsis", "produit_modified"), "w", **profile) as dst:
        dst.write(modified, 1)

    ndsi_snow_cover = np.where((rejeu > 0) * (rejeu <= 200), (ndsi - 100) * 2, rejeu)
    with rasterio.open(rejeu_file.replace("produit_synopsis", "ndsi_snow_cover"), "w", **profile) as dst:
        dst.write(ndsi_snow_cover, 1)
