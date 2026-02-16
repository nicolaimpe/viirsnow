from logger_setup import default_logger as logger

if __name__ == "__main__":
    year = WinterYear(2023, 2024)

    massifs_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/vectorial/massifs/massifs.shp"
    meteofrance_cms_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_composite_multiplatform/CMS_rejeu/"
    grid = UTM375mGrid()
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_10/time_series/"

    for platform in ("SNPP", "JPSS2", "JPSS1", "all"):
        logger.info(f"{platform} processing")
        MeteoFranceCompositeRegrid(
            product=platform,
            output_grid=grid,
            data_folder=meteofrance_cms_folder,
            output_folder=f"{output_folder}/{platform}",
        ).create_time_series(winter_year=year, roi_shapefile=massifs_shapefile)

    year = WinterYear(2023, 2024)

    suffixes = ["no_forest_red_band_screen"]
    massifs_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/vectorial/massifs/massifs.shp"
    meteofrance_cms_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_rejeu/"
    grid = UTM375mGrid()

    logger.info("Méteo-France multiplatform processing")
    MeteoFranceMultiplatformRegrid(
        output_grid=grid,
        data_folder=meteofrance_cms_folder,
        output_folder="./output_folder",
    ).create_time_series(winter_year=year, roi_shapefile=massifs_shapefile)
