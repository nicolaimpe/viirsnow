from datetime import datetime

from geospatial_grid.grid_database import UTM375mGrid
from ndsi_fsc_calibration.regrid import S2TheiaRegrid

from logger_setup import default_logger as logger
from regrid.meteofrance_composite_to_timeseries import MeteoFranceCompositeRegrid
from regrid.meteofrance_l2_to_l3_time_series import MeteoFrancePrototypeRegrid
from regrid.modis_l3_to_time_series import MODA1FSCRegrid, UTM500mGrid
from regrid.viirs_l3_to_time_series import V10A1FSCRegrid

if __name__ == "__main__":
    start, end = datetime(year=2023, month=11, day=1), datetime(year=2023, month=11, day=5)

    massifs_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/vectorial/massifs/massifs.shp"
    meteofrance_cms_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_rejeu/"
    grid = UTM375mGrid()
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_11/time_series/wy_2023_2024"

    ### Evaluation of V[NP|J1|J2]10A1 and MF-FSC-L3-SNPP on winter year 2023/2024
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
        V10A1FSCRegrid(
            output_grid=grid,
            data_folder=f"{nasa_folder}/{prod_id}",
            output_folder=output_path,
        ).create_time_series(start_date=start, end_date=end, roi_shapefile=massifs_shapefile)

    s2_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/S2_THEIA"

    S2TheiaRegrid(
        output_grid=grid, data_folder=s2_folder, output_folder=f"{output_folder}/s2_{grid.name.lower()}"
    ).create_time_series(roi_shapefile=massifs_shapefile, start_date=start, end_date=end)
    ### Evaluation of VNP10A1 and MOD10A1 on winter year 2023/2024

    grid_500 = UTM500mGrid()

    prod_id = "VNP10A1"
    output_path = f"{output_folder}/{prod_id.lower()}_{grid_500.name.lower()}"

    V10A1FSCRegrid(
        output_grid=grid_500,
        data_folder=f"{nasa_folder}/{prod_id}",
        output_folder=output_path,
    ).create_time_series(start_date=start, end_date=end, roi_shapefile=massifs_shapefile)

    prod_id = "MOD10A1"
    output_path = f"{output_folder}/{prod_id.lower()}_{grid_500.name.lower()}"
    modis_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/M10A1"
    MODA1FSCRegrid(
        output_grid=grid_500,
        data_folder=f"{modis_folder}/{prod_id}",
        output_folder=output_path,
    ).create_time_series(start_date=start, end_date=end, roi_shapefile=massifs_shapefile)

    S2TheiaRegrid(
        output_grid=grid_500, data_folder=s2_folder, output_folder=f"{output_folder}/s2_{grid_500.name.lower()}"
    ).create_time_series(roi_shapefile=massifs_shapefile, start_date=start, end_date=end)
    ### Evaluation of MF-V[NP|J1|J2|MP]-FSC-L3 on winter year 2024/2025
    start, end = datetime(year=2024, month=11, day=1), datetime(year=2024, month=11, day=5)
    meteofrance_composite_folder = (
        "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_composite_multiplatform/rejeu_2024_2025"
    )
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_11/time_series/wy_2024_2025"
    for platform in ("SNPP", "JPSS1", "JPSS2", "all"):
        MeteoFranceCompositeRegrid(
            output_grid=grid,
            data_folder=meteofrance_composite_folder,
            output_folder=f"{output_folder}/meteofrance_{platform.lower()}_{grid.name.lower()}",
            platform=platform,
        ).create_time_series(start_date=start, end_date=end, roi_shapefile=massifs_shapefile)

    S2TheiaRegrid(
        output_grid=grid, data_folder=s2_folder, output_folder=f"{output_folder}/s2_{grid.name.lower()}"
    ).create_time_series(roi_shapefile=massifs_shapefile, start_date=start, end_date=end)
