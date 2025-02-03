import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import earthaccess
import xarray as xr

from compression import generate_xarray_compression_encodings
from fractional_snow_cover import nasa_ndsi_snow_cover_to_fraction
from geotools import georef_data_array
from grids import Grid, UTM375mGrid
from logger_setup import default_logger as logger
from products.classes import NASA_CLASSES
from products.filenames import KNOWN_COLLECTIONS, NASA_L2_GEOMETRY_PRODUCTS_IDS, NASA_L2_SNOW_PRODUCTS_IDS
from reprojections import reproject_l2_nasa_to_grid
from winter_year import WinterYear


def download_daily_products(
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
    if product_name in NASA_L2_GEOMETRY_PRODUCTS_IDS:
        results = [result for result in results if result["umm"]["DataGranule"]["DayNightFlag"] != "Night"]
    # 3. Access
    files = earthaccess.download(results, f"{output_folder}")
    return files


def reproject_l2_snow_cover_product(l2_nasa_filename: str, output_path: str, output_grid: Grid):
    l2_geoloc = xr.open_dataset(l2_nasa_filename, group="/GeolocationData")
    l2_dataset = xr.open_dataset(l2_nasa_filename, group="/SnowData")
    selected = l2_dataset.data_vars["NDSI_Snow_Cover"] != NASA_CLASSES["bowtie_trim"][0]
    reprojected_dataset = reproject_l2_nasa_to_grid(
        l2_geolocation_dataset=l2_geoloc,
        l2_dataset=l2_dataset,
        output_grid=output_grid,
        bowtie_trim_mask=selected,
        output_filename=None,
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
        data_array_name="fractional_snow_cover",
        crs=output_grid.crs,
    )
    reprojected_dataset = reprojected_dataset.assign(fsc_dataset)
    reprojected_dataset = reprojected_dataset.astype("u1")
    reprojected_dataset.to_netcdf(output_path, encoding=generate_xarray_compression_encodings(reprojected_dataset))


def reproject_l2_geometry_product(l2_nasa_filename: str, output_path: str, output_grid: Grid):
    l2_geoloc = xr.open_dataset(l2_nasa_filename, group="/geolocation_data")
    l2_dataset = xr.open_dataset(l2_nasa_filename, group="/geolocation_data").drop_vars(
        ["land_water_mask", "latitude", "longitude", "range", "quality_flag", "sensor_azimuth", "solar_azimuth"]
    )

    reproject_l2_nasa_to_grid(
        l2_geolocation_dataset=l2_geoloc,
        l2_dataset=l2_dataset,
        output_grid=output_grid,
        output_filename=output_path,
    )


def reproject_daily_products(
    daily_l2_filenames: List[str],
    output_folder: str,
    output_grid: Grid,
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
    product_collection = "V03IMG"  # V10 V03IMG
    product_type = "Standard"  # Standard, NRT (Near Real Time)
    platform = "Suomi-NPP"  # Suomi-NPP, JPSS1
    data_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data"

    product_id = KNOWN_COLLECTIONS[product_collection][product_type][platform]
    output_folder = f"{data_folder}/{product_collection}/{product_id}/"

    year = WinterYear(2023, 2024)
    output_grid = UTM375mGrid()

    earthaccess.login()

    for day in year.iterate_days():
        daily_products_filenames = download_daily_products(
            day=day,
            product_name=product_id,
            output_folder=output_folder,
            bounding_box=output_grid.bounds_projected_to_epsg(4326),
        )
        reproject_daily_products(
            daily_l2_filenames=daily_products_filenames,
            output_folder=output_folder,
            output_grid=output_grid,
            product_id=product_id,
            delete_downloaded_swath_files=True,
        )
