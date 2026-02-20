from copy import deepcopy
from typing import Dict, List

import xarray as xr
from geospatial_grid.grid_database import UTM375mGrid
from mountain_data_binner.mountain_binner import MountainBinnerConfig

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
from reductions.completeness import MeteoFranceCompositeCompleteness, NASACompleteness, SnowCoverProductCompleteness
from reductions.confusion_table import ConfusionTable
from reductions.correlation import ScatterAnalysis
from reductions.statistics_base import EvaluationConfig, EvaluationVsHighResBase
from reductions.uncertainty import Uncertainty
from regrid.modis_l3_to_time_series import UTM500mGrid
from winter_year import WinterYear

if __name__ == "__main__":
    ### Detailed analysis of VIIRS products on winter year 2023/2024
    s2_analyzer = Sentinel2Theia().analyzer

    # config_base = EvaluationConfig(
    #     forest_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2018/corine_2018_forest_mask_utm_375m.tif",
    #     slope_map_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/SLP_MSF_UTM31_375m_lanczos.tif",
    #     aspect_map_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/ASP_MSF_UTM31_375m_lanczos.tif",
    #     dem_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/DEM_MSF_UTM31_375m_lanczos.tif",
    #     sensor_zenith_analysis=False,
    # )

    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_11/wy_2023_2024"
    # grid = UTM375mGrid()

    # products: List[SnowCoverProduct] = [VNP10A1(), VJ110A1(), VJ210A1(), MeteoFranceEvalSNPP()]
    # reductions = ["confusion_table", "uncertainty"]
    # config_sza = deepcopy(config_base)
    # config_sza.sensor_zenith_analysis = True

    # config_scatter = deepcopy(config_base)
    # config_scatter.slope_map_path = None
    # config_scatter.eval_var_name = ("NDSI_Snow_Cover",)
    # config_scatter.ref_fsc_step = 1

    # reduction_dict: Dict[str, EvaluationVsHighResBase] = {
    #     "confusion_table": ConfusionTable,
    #     "uncertainty": Uncertainty,
    # }
    # eval_config_dict: Dict[str, Dict[str, EvaluationConfig]] = {
    #     "confusion_table": {
    #         "VNP10A1": config_base,
    #         "VJ110A1": config_base,
    #         "VJ210A1": config_base,
    #         "MF-FSC-VNP-L3": config_sza,
    #     },
    #     "uncertainty": {"VNP10A1": config_base, "VJ110A1": config_base, "VJ210A1": config_base, "MF-FSC-VNP-L3": config_sza},
    # # }
    # for reduction in reductions:
    #     for product in products:
    #         logger.info(f"Evaluation {reduction} for product {product}")
    #         # metrics_calculator = reduction_dict[reduction](
    #     evaluation_config=eval_config_dict[reduction][product.prod_id],
    #     eval_analyzer=product.analyzer,
    #     reference_analyzer=s2_analyzer,
    # )
    # metrics_calculator.launch_analysis(
    #     eval_time_series=xr.open_dataset(
    #         f"{output_folder}/{product.prod_id.lower()}_{grid.name.lower()}/time_series/regridded.nc"
    #     ),
    #     ref_time_series=xr.open_dataset(f"{output_folder}/s2_{grid.name.lower()}/time_series/regridded.nc"),
    #     netcdf_export_path=f"{output_folder}/{product.prod_id.lower()}_{grid.name.lower()}/analyses/{reduction}.nc",
    # )
    # logger.info("Evaluation scatter for product VNP10A1")
    # metrics_calculator = ScatterAnalysis(
    #     evaluation_config=config_scatter,
    #     eval_analyzer=VNP10A1().analyzer,
    #     reference_analyzer=s2_analyzer,
    # )
    # metrics_calculator.launch_analysis(
    #     eval_time_series=xr.open_dataset(
    #         f"{output_folder}/{VNP10A1().prod_id.lower()}_{grid.name.lower()}/time_series/regridded.nc"
    #     ),
    #     ref_time_series=xr.open_dataset(f"{output_folder}/s2_{grid.name.lower()}/time_series/regridded.nc"),
    #     netcdf_export_path=f"{output_folder}/{VNP10A1().prod_id.lower()}_{grid.name.lower()}/analyses/scatter.nc",
    # )

    # ## MODIS vs VIIRS rough analysis winter year 2023/2024

    # output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_11/wy_2023_2024"
    # grid = UTM500mGrid()
    # products: List[SnowCoverProduct] = [MOD10A1(), VNP10A1()]

    # config_base = EvaluationConfig(
    #     forest_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2018/corine_2018_forest_mask_utm_500m.tif",
    #     slope_map_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/SLP_MSF_UTM31_500m_lanczos.tif",
    #     aspect_map_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/ASP_MSF_UTM31_500m_lanczos.tif",
    #     dem_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/DEM_MSF_UTM31_500m_lanczos.tif",
    #     sensor_zenith_analysis=False,
    # )

    # eval_config_dict: Dict[str, Dict[str, EvaluationConfig]] = {
    #     "uncertainty": {
    #         "VNP10A1": config_base,
    #         "MOD10A1": config_base,
    #     },
    # }
    # config_completeness = MountainBinnerConfig(
    #     forest_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2018/corine_2018_forest_mask_utm_500m.tif",
    #     dem_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/DEM_MSF_UTM31_500m_lanczos.tif",
    # )

    # for product in products:
    #     logger.info(f"Evaluation uncertainty for product {product}")
    #     metrics_calculator = Uncertainty(
    #         evaluation_config=eval_config_dict["uncertainty"][product.prod_id],
    #         eval_analyzer=product.analyzer,
    #         reference_analyzer=s2_analyzer,
    #     )
    #     prod_time_series = xr.open_dataset(
    #         f"{output_folder}/{product.prod_id.lower()}_{grid.name.lower()}/time_series/regridded.nc"
    #     )
    #     metrics_calculator.launch_analysis(
    #         eval_time_series=prod_time_series,
    #         ref_time_series=xr.open_dataset(f"{output_folder}/s2_{grid.name.lower()}/time_series/regridded.nc"),
    #         netcdf_export_path=f"{output_folder}/{product.prod_id.lower()}_{grid.name.lower()}/analyses/uncertainty.nc",
    #     )

    #     logger.info(f"Evaluation completeness for product {product}")
    #     area_calculator = NASACompleteness()
    #     area_calculator.year_temporal_analysis(
    #         snow_cover_product_time_series=prod_time_series,
    #         netcdf_export_path=f"{output_folder}/{product.prod_id.lower()}_{grid.name.lower()}/analyses/completeness.nc",
    #         config=config_completeness,
    #     )

    ### Météo-France 3 platforms + composite winter year 2024/2025

    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_11/wy_2024_2025"
    grid = UTM375mGrid()
    products: List[SnowCoverProduct] = [
        MeteoFranceEvalSNPP(),
        MeteoFranceEvalJPSS1(),
        MeteoFranceEvalJPSS2(),
        MeteoFranceComposite(),
    ]

    config_base = EvaluationConfig(
        forest_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2018/corine_2018_forest_mask_utm_375m.tif",
        slope_map_path=None,
        aspect_map_path=None,
        dem_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/DEM_MSF_UTM31_375m_lanczos.tif",
        sensor_zenith_analysis=False,
    )

    eval_config_dict: Dict[str, Dict[str, EvaluationConfig]] = {
        "uncertainty": {
            "MF-FSC-VNP-L3": config_base,
            "MF-FSC-VJ1-L3": config_base,
            "MF-FSC-VJ2-L3": config_base,
            "MF-FSC-VMP-L3": config_base,
        },
    }
    config_completeness = MountainBinnerConfig(
        forest_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2018/corine_2018_forest_mask_utm_375m.tif",
        dem_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/DEM_MSF_UTM31_375m_lanczos.tif",
    )

    for product in products:
        logger.info(f"Evaluation uncertainty for product {product}")
        metrics_calculator = Uncertainty(
            evaluation_config=eval_config_dict["uncertainty"][product.prod_id],
            eval_analyzer=product.analyzer,
            reference_analyzer=s2_analyzer,
        )
        prod_time_series = xr.open_dataset(
            f"{output_folder}/{product.prod_id.lower()}_{grid.name.lower()}/time_series/regridded.nc"
        )
        metrics_calculator.launch_analysis(
            eval_time_series=prod_time_series,
            ref_time_series=xr.open_dataset(f"{output_folder}/s2_{grid.name.lower()}/time_series/regridded.nc"),
            netcdf_export_path=f"{output_folder}/{product.prod_id.lower()}_{grid.name.lower()}/analyses/uncertainty.nc",
        )

        logger.info(f"Evaluation completeness for product {product}")
        area_calculator = MeteoFranceCompositeCompleteness()
        area_calculator.year_temporal_analysis(
            snow_cover_product_time_series=prod_time_series,
            netcdf_export_path=f"{output_folder}/{product.prod_id.lower()}_{grid.name.lower()}/analyses/completeness.nc",
            config=config_completeness,
        )
