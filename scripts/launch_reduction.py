from typing import List

import xarray as xr

from logger_setup import default_logger as logger
from products.snow_cover_product import (
    VJ110A1,
    VNP10A1,
    MeteoFranceJPSS1Prototype,
    MeteoFranceJPSS2Prototype,
    MeteoFranceMultiplatformPrototype,
    MeteoFranceSNPPPrototype,
    Sentinel2Theia,
    SnowCoverProduct,
)
from reductions.confusion_table import ConfusionTable
from reductions.correlation import Scatter
from reductions.semidistributed import MountainParams
from reductions.statistics_base import EvaluationConfig, generate_evaluation_io
from reductions.uncertainty import Uncertainty
from winter_year import WinterYear

output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_8"
working_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_8"
config_mountains = MountainParams(
    forest_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2018/corine_2018_forest_mask_utm_375m.tif",
    slope_map_path=None,
    aspect_map_path=None,
    dem_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/DEM_MSF_UTM31_375m_lanczos.tif",
)

config = EvaluationConfig(ref_fsc_step=25, sensor_zenith_analysis=False, sub_roi_mask_path=None, **config_mountains.__dict__)

products: List[SnowCoverProduct] = [
    # MeteoFranceSNPPPrototype(),
    # MeteoFranceJPSS1Prototype(),
    # MeteoFranceJPSS2Prototype(),
    # MeteoFranceMultiplatformPrototype(),
    # VJ110A1(),
    # MeteoFranceSNPPPrototype(),
    VNP10A1(),
]

reduction_type = "scatter"

if __name__ == "__main__":
    for product in products:
        logger.info(f"Evaluating product {product}")

        if reduction_type == "completeness":
            logger.info("Completeness analysis starting")
            test_series = xr.open_dataset(f"{output_folder}/time_series/WY_2023_2024_{product.name}.nc")
            product.analyzer.year_temporal_analysis(
                snow_cover_product_time_series=test_series,
                netcdf_export_path=f"{output_folder}/analyses/completeness/completeness_WY_2023_2024_{product.name}.nc",
                period=None,
                config=config,
            )

        if reduction_type == "uncertainty":
            logger.info("Uncertainty analysis")

            ref_time_series, test_time_series, output_filename = generate_evaluation_io(
                analysis_type="uncertainty",
                working_folder=working_folder,
                year=WinterYear(2023, 2024),
                test_product_name=product.name,
                ref_product_name="S2_theia",
                period=slice("2023-11", "2024-06"),
            )

            metrics_calcuator = Uncertainty(reference_analyzer=Sentinel2Theia().analyzer, test_analyzer=product.analyzer)

        if reduction_type == "confusion_table":
            logger.info("Contingency analysis")

            ref_time_series, test_time_series, output_filename = generate_evaluation_io(
                analysis_type="confusion_table",
                working_folder=working_folder,
                year=WinterYear(2023, 2024),
                test_product_name=product.name,
                ref_product_name="S2_theia",
                period=slice("2023-11", "2024-06"),
            )
            metrics_calcuator = ConfusionTable(
                reference_analyzer=Sentinel2Theia().analyzer,
                test_analyzer=product.analyzer,
                test_fsc_threshold=50,
                ref_fsc_threshold=50,
            )

        if reduction_type == "scatter":
            logger.info("Scatter analysis")

            ref_time_series, test_time_series, output_filename = generate_evaluation_io(
                analysis_type="scatter",
                working_folder=working_folder,
                year=WinterYear(2023, 2024),
                test_product_name=product.name,
                ref_product_name="S2_theia",
                period=slice("2023-11", "2024-06"),
            )

            metrics_calcuator = Scatter(
                reference_analyzer=Sentinel2Theia().analyzer,
                test_analyzer=product.analyzer,
            )

            config.test_var_name = ("NDSI_Snow_Cover",)

        metrics_calcuator.launch_analysis(
            test_time_series=test_time_series,
            ref_time_series=ref_time_series,
            config=config,
            netcdf_export_path=output_filename,
        )
