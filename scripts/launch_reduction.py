import xarray as xr

from logger_setup import default_logger as logger
from products.snow_cover_product import VNP10A1
from reductions.semidistributed import MountainParams

output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_8"
config = MountainParams(
    forest_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2006_forest_mask_utm_375m.tif",
    slope_map_path=None,
    aspect_map_path=None,
    dem_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/DEM_MSF_UTM31_375m_lanczos.tif",
)
products = [VNP10A1()]
reduction_type = "completeness"
if __name__ == "__main__":
    for product in products:
        logger.info(f"Evaluating product {product.name}")

        if reduction_type == "completeness":
            logger.info("Completeness analysis starting")
            test_series = xr.open_dataset(f"{output_folder}/time_series/WY_2023_2024_{product.name}.nc")
            product.analyzer.year_temporal_analysis(
                snow_cover_product_time_series=test_series,
                netcdf_export_path=f"{output_folder}/analyses/completeness/completeness_WY_2023_2024_{product.name}.nc",
                period=None,
                config=config,
            )
