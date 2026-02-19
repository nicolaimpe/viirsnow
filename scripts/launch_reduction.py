from copy import deepcopy
from typing import Dict, List

import xarray as xr
from geospatial_grid.grid_database import UTM375mGrid

from logger_setup import default_logger as logger
from products.snow_cover_product import (
    MOD10A1,
    VJ110A1,
    VJ210A1,
    VNP10A1,
    MeteoFranceComposite,
    MeteoFranceEvalJPSS1,
    MeteoFranceEvalJPSS2,
    MeteoFranceEvalSNPP,
    Sentinel2Theia,
    SnowCoverProduct,
)
from reductions.confusion_table import ConfusionTable
from reductions.correlation import ScatterAnalysis
from reductions.statistics_base import EvaluationConfig, EvaluationVsHighResBase
from reductions.uncertainty import Uncertainty
from regrid.modis_l3_to_time_series import UTM500mGrid
from winter_year import WinterYear

config_base = EvaluationConfig(
    forest_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2018/corine_2018_forest_mask_utm_375m.tif",
    slope_map_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/SLP_MSF_UTM31_375m_lanczos.tif",
    aspect_map_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/ASP_MSF_UTM31_375m_lanczos.tif",
    dem_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/DEM_MSF_UTM31_375m_lanczos.tif",
    sensor_zenith_analysis=False,
)


if __name__ == "__main__":
    s2_analyzer = Sentinel2Theia().analyzer

    wy = WinterYear(2023, 2024)
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_11/wy_2023_2024"
    grid = UTM375mGrid()
    period = slice(f"{wy.from_year}-11-01", f"{wy.from_year}-11-05")
    products: List[SnowCoverProduct] = [MeteoFranceEvalSNPP()]
    reductions = ["confusion_table", "uncertainty"]
    config_sza = deepcopy(config_base)
    config_sza.sensor_zenith_analysis = True

    config_scatter = deepcopy(config_base)
    config_scatter.slope_map_path = None
    config_scatter.eval_var_name = "NDSI_Snow_Cover"

    reduction_dict: Dict[str, EvaluationVsHighResBase] = {
        "confusion_table": ConfusionTable,
        "uncertainty": Uncertainty,
    }
    eval_config_dict: Dict[str, Dict[str, EvaluationConfig]] = {
        "confusion_table": {
            "VNP10A1": config_base,
            "VJ110A1": config_base,
            "VJ210A1": config_base,
            "MF-FSC-VNP-L3": config_sza,
        },
        "uncertainty": {"VNP10A1": config_base, "VJ110A1": config_base, "VJ210A1": config_base, "MF-FSC-VNP-L3": config_sza},
    }
    for reduction in reductions:
        for product in products:
            logger.info(f"Evaluation {reduction} for product {product}")
            metrics_calculator = reduction_dict[reduction](
                evaluation_config=eval_config_dict[reduction][product.prod_id],
                eval_analyzer=product.analyzer,
                reference_analyzer=s2_analyzer,
            )
            metrics_calculator.launch_analysis(
                eval_time_series=xr.open_dataset(
                    f"{output_folder}/{product.prod_id.lower()}_{grid.name.lower()}/regridded.nc"
                ),
                ref_time_series=xr.open_dataset(f"{output_folder}/s2_{grid.name.lower()}/regridded.nc"),
                netcdf_export_path=f"{output_folder}/{product.prod_id.lower()}_{grid.name.lower()}/{reduction}.nc",
            )
    logger.info("Evaluation scatter for product VNP10A1")
    metrics_calculator = ScatterAnalysis(
        evaluation_config=config_scatter,
        eval_analyzer=VNP10A1().analyzer,
        reference_analyzer=s2_analyzer,
    )
    metrics_calculator.launch_analysis(
        eval_time_series=xr.open_dataset(f"{output_folder}/{VNP10A1().prod_id.lower()}_{grid.name.lower()}/regridded.nc"),
        ref_time_series=xr.open_dataset(f"{output_folder}/s2_{grid.name.lower()}/regridded.nc"),
        netcdf_export_path=f"{output_folder}/{VNP10A1().prod_id.lower()}_{grid.name.lower()}/scatter.nc",
    )
    # if reduction_type == "uncertainty":
    #     logger.info("Uncertainty analysis")

    #     ref_time_series, test_time_series, output_filename = generate_evaluation_io(
    #         analysis_type="uncertainty",
    #         working_folder=working_folder,
    #         year=wy,
    #         test_product_name=product.name,
    #         ref_product_name="S2_theia",
    #         grid=grid,
    #         period=period,
    #     )
    #     metrics_calcuator = Uncertainty(reference_analyzer=Sentinel2Theia().analyzer, test_analyzer=product.analyzer)

    # if reduction_type == "confusion_table":
    #     logger.info("Contingency analysis")

    #     ref_time_series, test_time_series, output_filename = generate_evaluation_io(
    #         analysis_type="confusion_table",
    #         working_folder=working_folder,
    #         year=wy,
    #         grid=grid,
    #         test_product_name=product.name,
    #         ref_product_name="S2_theia",
    #         period=period,
    #     )

    #     metrics_calcuator = ConfusionTable(
    #         reference_analyzer=Sentinel2Theia().analyzer,
    #         test_analyzer=product.analyzer,
    #         test_fsc_threshold=50,
    #         ref_fsc_threshold=50,
    #     )

    # if reduction_type == "scatter":
    #     logger.info("Scatter analysis")

    #     ref_time_series, test_time_series, output_filename = generate_evaluation_io(
    #         analysis_type="scatter",
    #         working_folder=working_folder,
    #         year=wy,
    #         grid=grid,
    #         period=period,
    #         test_product_name=product.name,
    #         ref_product_name="S2_theia",
    #     )

    #     metrics_calcuator = Scatter(
    #         reference_analyzer=Sentinel2Theia().analyzer,
    #         test_analyzer=product.analyzer,
    #     )

    #     config.test_var_name = ("NDSI_Snow_Cover",)
    #     config.ref_fsc_step = 1

    # metrics_calcuator.launch_analysis(
    #     test_time_series=test_time_series,
    #     ref_time_series=ref_time_series,
    #     config=config,
    #     netcdf_export_path=output_filename,
    # )
