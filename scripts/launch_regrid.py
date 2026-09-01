from datetime import datetime

from geospatial_grid.grid_database import SIN375mGrid, UTM375mGrid
from ndsi_fsc_calibration.regrid import S2TheiaRegrid

from logger_setup import default_logger as logger
from products.snow_cover_product import MeteoFranceComposite, MeteoFranceEvalJPSS1, MeteoFranceEvalJPSS2, MeteoFranceEvalSNPP
from regrid.meteofrance_composite_to_timeseries import MeteoFranceCompositeRegrid
from regrid.meteofrance_l2_to_l3_time_series import MeteoFrancePrototypeRegrid
from regrid.modis_l3_to_time_series import MOD10A1FSCRegrid, MYD10A1FSCRegrid, UTM500mGrid
from regrid.viirs_l3_to_time_series import V10A1FSCRegrid

if __name__ == "__main__":
    # start, end = datetime(year=2023, month=11, day=1), datetime(year=2024, month=6, day=30)

    massifs_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/vectorial/massifs/massifs.shp"
    # meteofrance_cms_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_rejeu/"
    # grid = UTM375mGrid()
    # output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_12_my_regression/wy_2023_2024/"

    # ### Evaluation of V[NP|J1|J2]10A1 and MF-FSC-L3-SNPP on winter year 2023/2024
    # logger.info("Météo-France prototype regridding")
    # platform = "SNPP"
    # MeteoFrancePrototypeRegrid(
    #     output_grid=grid,
    #     data_folder=meteofrance_cms_folder,
    #     output_folder=f"{output_folder}/mf-fsc-vnp-l3_{grid.name.lower()}/time_series",
    #     suffix="no_forest_red_band_screen",
    #     platform=platform,
    # ).create_time_series(start_date=start, end_date=end, roi_shapefile=massifs_shapefile)

    # nasa_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/V10A1"
    # for prod_id in ["VNP10A1"]:
    #     logger.info(f"{prod_id} prototype regridding")
    #     output_path = f"{output_folder}/{prod_id.lower()}_{grid.name.lower()}/time_series"
    #     V10A1FSCRegrid(
    #         output_grid=grid,
    #         data_folder=f"{nasa_folder}/{prod_id}",
    #         output_folder=output_path,
    #         forest_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2018/corine_2018_forest_mask_utm_375m.tif",
    #     ).create_time_series(start_date=start, end_date=end, roi_shapefile=massifs_shapefile)

    # s2_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/S2_THEIA"

    # S2TheiaRegrid(
    #     output_grid=grid, data_folder=s2_folder, output_folder=f"{output_folder}/s2_{grid.name.lower()}/time_series"
    # ).create_time_series(roi_shapefile=massifs_shapefile, start_date=start, end_date=end)

    ### Evaluation of VNP10A1 and MOD10A1 on winter year 2023/2024

    # grid_500 = UTM500mGrid()

    # prod_id = "VNP10A1"
    # output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_12/wy_2023_2024/"
    # nasa_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/V10A1"
    # s2_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/S2_THEIA"
    # output_path = f"{output_folder}/{prod_id.lower()}_{grid_500.name.lower()}/time_series"

    # V10A1FSCRegrid(
    #     output_grid=grid_500,
    #     data_folder=f"{nasa_folder}/{prod_id}",
    #     output_folder=output_path,
    # ).create_time_series(start_date=start, end_date=end, roi_shapefile=massifs_shapefile)

    # prod_id = "MOD10A1"
    # output_path = f"{output_folder}/{prod_id.lower()}_{grid_500.name.lower()}/time_series"
    # modis_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/M10A1"
    # MOD10A1FSCRegrid(
    #     output_grid=grid_500,
    #     data_folder=f"{modis_folder}/{prod_id}",
    #     output_folder=output_path,
    # ).create_time_series(start_date=start, end_date=end, roi_shapefile=massifs_shapefile)

    # prod_id = "MYD10A1"
    # output_path = f"{output_folder}/{prod_id.lower()}_{grid_500.name.lower()}/time_series"
    # modis_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/M10A1"
    # MYD10A1FSCRegrid(
    #     output_grid=grid_500,
    #     data_folder=f"{modis_folder}/{prod_id}",
    #     output_folder=output_path,
    # ).create_time_series(start_date=start, end_date=end, roi_shapefile=massifs_shapefile)

    # S2TheiaRegrid(
    #     output_grid=grid_500, data_folder=s2_folder, output_folder=f"{output_folder}/s2_{grid_500.name.lower()}/time_series"
    # ).create_time_series(roi_shapefile=massifs_shapefile, start_date=start, end_date=end)

    ### Evaluation of MF-V[NP|J1|J2|MP]-FSC-L3 on winter year 2024/2025
    start, end = datetime(year=2024, month=11, day=1), datetime(year=2025, month=6, day=30)
    grid = UTM375mGrid()
    meteofrance_composite_folder = (
        "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_composite_multiplatform/rejeu_2024_2025"
    )
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_12/wy_2024_2025/"
    platforms = ["SNPP", "JPSS1", "JPSS2", "all"]
    for product, pl in zip(
        [MeteoFranceEvalSNPP(), MeteoFranceEvalJPSS1(), MeteoFranceEvalJPSS2(), MeteoFranceComposite()], platforms
    ):
        MeteoFranceCompositeRegrid(
            output_grid=grid,
            data_folder=meteofrance_composite_folder,
            output_folder=f"{output_folder}/{product.prod_id.lower()}_{grid.name.lower()}/time_series",
            platform=pl,
        ).create_time_series(start_date=start, end_date=end, roi_shapefile=massifs_shapefile)

    s2_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/S2_THEIA"
    S2TheiaRegrid(
        output_grid=grid, data_folder=s2_folder, output_folder=f"{output_folder}/s2_{grid.name.lower()}/time_series"
    ).create_time_series(roi_shapefile=massifs_shapefile, start_date=start, end_date=end)
