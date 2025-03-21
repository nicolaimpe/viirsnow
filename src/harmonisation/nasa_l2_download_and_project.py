import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import earthaccess
import numpy as np
import xarray as xr

from compression import generate_xarray_compression_encodings
from fractional_snow_cover import nasa_ndsi_snow_cover_to_fraction
from grids import GeoGrid, UTM375mGrid, georef_data_array
from logger_setup import default_logger as logger
from products.classes import NASA_CLASSES
from products.filenames import (
    KNOWN_COLLECTIONS,
    NASA_L2_GEOMETRY_PRODUCTS_IDS,
    NASA_L2_SNOW_PRODUCTS_IDS,
    get_datetime_from_viirs_nasa_filepath,
)
from reprojections import reproject_l2_nasa_to_grid
from winter_year import WinterYear


def download_daily_products_from_home(
    day: datetime, product_name: str, output_folder: str, bounding_box: Tuple[float, float, float, float]
):
    logger.info(f"Process day {day.strftime('%Y-%m-%d')}")
    # 2. Search
    results = earthaccess.search_data(
        short_name=product_name,  # ATLAS/ICESat-2 L3A Land Ice Height, VNP10
        bounding_box=bounding_box,  # Only include files in area of interest...
        temporal=(day.strftime("%Y-%m-%d"), day.strftime("%Y-%m-%d")),  # ...and time period of interest
    )

    if product_name in NASA_L2_SNOW_PRODUCTS_IDS:
        # Only version 2
        results = [
            result
            for result in results
            if result["umm"]["CollectionReference"]["EntryTitle"] == "VIIRS/NPP Snow Cover 6-Min L2 Swath 375m V002"
        ]
    # Exclude night bands
    if product_name in NASA_L2_GEOMETRY_PRODUCTS_IDS:
        results = [result for result in results if result["umm"]["DataGranule"]["DayNightFlag"] != "Night"]
    if product_name in NASA_L2_SNOW_PRODUCTS_IDS:
        results = [
            result
            for result in results
            if get_datetime_from_viirs_nasa_filepath(result["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]).hour > 9
        ]
    # 3. Access
    files = earthaccess.download(results, f"{output_folder}")

    return files


def download_daily_products_from_sxcen(day: datetime, product_name: str, download_urls_list: List[str], output_folder: str):
    logger.info(f"Process day {day.strftime('%Y-%m-%d')}")
    fs = earthaccess.get_fsspec_https_session()
    daily_urls = [path for path in download_urls_list if re.search(f"A{day.strftime('%Y%j')}", path)]
    output_filepaths = []
    for url in daily_urls:
        product_filename = url.split("/")[-1]

        output_filepath = f"{output_folder}/{product_filename}"
        output_filepaths.append(output_filepath)
        with fs.open(url) as f:
            logger.info("Reading granule")
            data = f.read()
        with open(output_filepath, "wb") as f:
            logger.info(f"Exporting to {output_filepath}")
            f.write(data)
    return output_filepaths


def reproject_l2_snow_cover_product(l2_nasa_filename: str, output_path: str, output_grid: GeoGrid):
    l2_geoloc = xr.open_dataset(l2_nasa_filename, group="/GeolocationData")
    l2_dataset = xr.open_dataset(l2_nasa_filename, group="/SnowData")
    selected = l2_dataset.data_vars["NDSI_Snow_Cover"] != NASA_CLASSES["bowtie_trim"][0]
    reprojected_dataset = reproject_l2_nasa_to_grid(
        l2_geolocation_dataset=l2_geoloc,
        l2_dataset=l2_dataset,
        output_grid=output_grid,
        bowtie_trim_mask=selected,
        output_filename=None,
        fill_value=NASA_CLASSES["fill"][0],
    )
    fractional_snow_cover = nasa_ndsi_snow_cover_to_fraction(
        reprojected_dataset["NDSI_Snow_Cover"].values, method="salomonson_appel"
    )
    fsc_dataset = georef_data_array(
        xr.DataArray(
            data=fractional_snow_cover,
            coords={"y": output_grid.ycoords, "x": output_grid.xcoords},
            attrs={"NDSI_to_FSC_method": "salomonson_appel"},
        ),
        data_array_name="snow_cover_fraction",
        crs=output_grid.crs,
    )
    reprojected_dataset = reprojected_dataset.assign(fsc_dataset)
    reprojected_dataset = reprojected_dataset.astype("u1")
    reprojected_dataset.to_netcdf(output_path, encoding=generate_xarray_compression_encodings(reprojected_dataset))


def reproject_l2_geometry_product(l2_nasa_filename: str, output_path: str, output_grid: GeoGrid):
    l2_geoloc = xr.open_dataset(l2_nasa_filename, group="/geolocation_data")
    l2_dataset = xr.open_dataset(l2_nasa_filename, group="/geolocation_data").drop_vars(
        ["land_water_mask", "latitude", "longitude", "range", "quality_flag", "sensor_azimuth", "solar_azimuth"]
    )

    reproject_l2_nasa_to_grid(
        l2_geolocation_dataset=l2_geoloc,
        l2_dataset=l2_dataset,
        output_grid=output_grid,
        output_filename=output_path,
        fill_value=np.nan,
    )


def reproject_daily_products(
    daily_l2_filenames: List[str],
    output_folder: str,
    output_grid: GeoGrid,
    product_id: str,
    delete_downloaded_swath_files: bool = False,
):
    output_product_id = f"{product_id}_{output_grid.name}"
    if product_id in NASA_L2_SNOW_PRODUCTS_IDS:
        reprojection_fun = reproject_l2_snow_cover_product
    elif product_id in NASA_L2_GEOMETRY_PRODUCTS_IDS:
        reprojection_fun = reproject_l2_geometry_product
    else:
        raise NotImplementedError
    for file in daily_l2_filenames:
        logger.info(f"Process file {file}")
        output_filename = Path(file).name.replace(product_id, output_product_id)
        output_path = f"{output_folder.replace(product_id, output_product_id)}/{output_filename}"
        reprojection_fun(l2_nasa_filename=file, output_path=output_path, output_grid=output_grid)
    if delete_downloaded_swath_files:
        [os.remove(file) for file in daily_l2_filenames]


if __name__ == "__main__":
    download_from = "home"  # "office", "home"
    data_folder = (
        "/home/imperatoren/work/VIIRS_S2_comparison/data"
        if download_from == "home"
        else "/home/imperatoren/work/viirsnow/data"
    )
    product_collection = "V10"  # V10 V03IMG
    product_type = "Standard"  # Standard, NRT (Near Real Time)
    platform = "SNPP"  # SNPP, JPSS1
    product_id = KNOWN_COLLECTIONS[product_collection][product_type][platform]
    output_folder = f"{data_folder}/{product_collection}/{product_id}/"

    # If download from office
    granule_list_filepath = (
        f"/home/imperatoren/work/viirsnow/data/{product_collection}/vnp03img_wy_2023_2024_granules_list.txt"
    )

    year = WinterYear(2023, 2024)
    output_grid = UTM375mGrid()

    if download_from == "office":
        with open(granule_list_filepath) as f:
            list_product_urls = [line.strip() for line in f.readlines()]
    earthaccess.login(strategy="interactive")

    bad_days_count = []
    for day in year.iterate_days():
        if day.year == 2023:
            continue
        if day.day_of_year < 164:
            continue
        try:
            if download_from == "home":
                daily_products_filenames = download_daily_products_from_home(
                    day=day,
                    product_name=product_id,
                    output_folder=output_folder,
                    bounding_box=output_grid.bounds_projected_to_epsg(4326),
                )
            elif download_from == "office":
                daily_products_filenames = download_daily_products_from_sxcen(
                    day=day,
                    product_name=product_id,
                    download_urls_list=list_product_urls,
                    output_folder=output_folder,
                )

        except Exception as e:
            logger.warning(f"Error {e} during download. Skipping day {day}.")
            bad_days_count.append(day)
            continue
        try:
            reproject_daily_products(
                daily_l2_filenames=daily_products_filenames,
                output_folder=output_folder,
                output_grid=output_grid,
                product_id=product_id,
                delete_downloaded_swath_files=True,
            )
        except Exception as e:
            logger.warning(f"Error {e} during reprojection. Skipping day {day}.")
            bad_days_count.append(day)
            continue

    print("Unsuccessfull days", bad_days_count)
