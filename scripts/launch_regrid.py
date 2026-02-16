from datetime import datetime

from geospatial_grid.grid_database import UTM375mGrid
from ndsi_fsc_calibration.regrid import V10A1Regrid

from fractional_snow_cover import nasa_ndsi_snow_cover_to_fraction
from logger_setup import default_logger as logger
from regrid.meteofrance_l2_to_l3_time_series import MeteoFrancePrototypeRegrid

if __name__ == "__main__":
    start, end = datetime(year=2023, month=11, day=1), datetime(year=2023, month=11, day=5)

    massifs_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/vectorial/massifs/massifs.shp"
    meteofrance_cms_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_rejeu/"
    grid = UTM375mGrid()
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_11/time_series/wy_2023_2024"

    logger.info("Météo-France prototype regridding")
    platform = "SNPP"
    MeteoFrancePrototypeRegrid(
        output_grid=grid,
        data_folder=meteofrance_cms_folder,
        output_folder=f"{output_folder}/meteofrance_snpp_{grid.name.lower()}",
        suffix="no_forest_red_band_screen",
        platform=platform,
    ).create_time_series(start_date=start, end_date=end, roi_shapefile=massifs_shapefile)

    nasa_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/V10A1"
    for prod_id in ("VNP10A1", "VJ110A1", "VJ210A1"):
        logger.info(f"{prod_id} prototype regridding")
        output_path = f"{output_folder}/{prod_id.lower()}_{grid.name.lower()}"
        ndsi_snow_cover_data = V10A1Regrid(
            output_grid=grid,
            data_folder=f"{nasa_folder}/{prod_id}",
            output_folder=output_path,
        ).create_time_series(start_date=start, end_date=end, roi_shapefile=massifs_shapefile)
        out_data = ndsi_snow_cover_data.copy()
        out_data = out_data.assign({"snow_cover_fraction": xr.Data})
