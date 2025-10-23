from typing import List

import xarray as xr

from grids import LatLon375mGrid, UTM375mGrid, UTM500mGrid
from logger_setup import default_logger as logger
from products.snow_cover_product import (
    MOD10A1,
    VJ110A1,
    VNP10A1,
    MeteoFranceComposite,
    MeteoFranceEvalJPSS1,
    MeteoFranceEvalJPSS2,
    MeteoFranceEvalSNPP,
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
    forest_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2018/corine_2018_forest_mask_utm_500m.tif",
    slope_map_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/SLP_MSF_UTM31_500m_lanczos.tif",
    aspect_map_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/ASP_MSF_UTM31_500m_lanczos.tif",
    dem_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/DEM_MSF_UTM31_500m_lanczos.tif",
)


config = EvaluationConfig(ref_fsc_step=25, sensor_zenith_analysis=False, sub_roi_mask_path=None, **config_mountains.__dict__)

products: List[SnowCoverProduct] = [
    # MeteoFranceEvalSNPP(),
    # MeteoFranceEvalJPSS1(),
    # MeteoFranceEvalJPSS2(),
    # MeteoFranceComposite(),
    # VJ110A1(),
    # MeteoFranceSNPPPrototype(),
    VNP10A1(),
    MOD10A1(),
]

reduction_type = "uncertainty"
wy = WinterYear(2023, 2024)
grid = UTM500mGrid()
period = slice(f"{wy.from_year}-11-01", f"{wy.to_year}-06-30")
if __name__ == "__main__":
    for product in products:
        logger.info(f"Evaluating product {product}")

        if reduction_type == "completeness":
            logger.info("Completeness analysis starting")
            test_series = xr.open_dataset(
                f"{output_folder}/time_series/WY_{wy.from_year}_{wy.to_year}_{product.name}_{grid.name.lower()}.nc"
            )

            product.analyzer.year_temporal_analysis(
                snow_cover_product_time_series=test_series,
                netcdf_export_path=f"{output_folder}/analyses/completeness/completeness_WY_{wy.from_year}_{wy.to_year}_{product.name}_{grid.name.lower()}.nc",
                period=period,
                config=MountainParams(
                    dem_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/DEM_MSF_UTM31_500m_lanczos.tif"
                ),
            )
            continue

        if reduction_type == "uncertainty":
            logger.info("Uncertainty analysis")

            ref_time_series, test_time_series, output_filename = generate_evaluation_io(
                analysis_type="uncertainty",
                working_folder=working_folder,
                year=wy,
                test_product_name=product.name,
                ref_product_name="S2_theia",
                grid=grid,
                period=period,
            )
            metrics_calcuator = Uncertainty(reference_analyzer=Sentinel2Theia().analyzer, test_analyzer=product.analyzer)

        if reduction_type == "confusion_table":
            logger.info("Contingency analysis")

            ref_time_series, test_time_series, output_filename = generate_evaluation_io(
                analysis_type="confusion_table",
                working_folder=working_folder,
                year=wy,
                grid=grid,
                test_product_name=product.name,
                ref_product_name="S2_theia",
                period=period,
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
                year=wy,
                grid=grid,
                period=period,
                test_product_name=product.name,
                ref_product_name="S2_theia",
            )

            metrics_calcuator = Scatter(
                reference_analyzer=Sentinel2Theia().analyzer,
                test_analyzer=product.analyzer,
            )

            config.test_var_name = ("NDSI_Snow_Cover",)
            config.ref_fsc_step = 1

        metrics_calcuator.launch_analysis(
            test_time_series=test_time_series,
            ref_time_series=ref_time_series,
            config=config,
            netcdf_export_path=output_filename,
        )
