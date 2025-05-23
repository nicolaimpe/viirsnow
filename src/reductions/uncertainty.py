from copy import deepcopy
from typing import Dict

import numpy as np
import xarray as xr
from xarray.groupers import BinGrouper

from logger_setup import default_logger as logger
from reductions.completeness import (
    MeteoFranceSnowCoverProductCompleteness,
    NASASnowCoverProductCompleteness,
    S2SnowCoverProductCompleteness,
)
from reductions.statistics_base import EvaluationConfig, EvaluationVsHighResBase, generate_evaluation_io
from winter_year import WinterYear


class Uncertainty(EvaluationVsHighResBase):
    @staticmethod
    def biais_bins():
        return BinGrouper(np.arange(-101, 101, 1), labels=np.arange(-100, 101, 1))

    def time_step_analysis(self, dataset: xr.Dataset, bins_dict: Dict[str, xr.groupers.Grouper]):
        logger.info(f"Processing time of the year {dataset.coords['time'].values[0].astype('M8[D]').astype('O')}")
        quant_mask_test = self.test_analyzer.quantitative_mask(dataset.data_vars["test"])
        valid_test = dataset.data_vars["test"].where(quant_mask_test)
        valid_test = valid_test * 100 / self.test_analyzer.max_fsc
        quant_mask_ref = self.ref_analyzer.quantitative_mask(dataset.data_vars["ref"])
        valid_ref = dataset.data_vars["ref"].where(quant_mask_ref)
        valid_ref = valid_ref * 100 / self.ref_analyzer.max_fsc
        dataset = dataset.assign(biais=valid_test - valid_ref)

        n_intersecting_pixels = (quant_mask_test & quant_mask_ref).sum()

        if n_intersecting_pixels < 2:
            logger.info("No intersection found on this day. Returning a zeros array.")
            dummy_dict = {k + "_bins": v.labels for k, v in bins_dict.items()}
            dummy_dict.update({"biais_bins": [0]})
            return xr.DataArray(name="n_occurrences", data=np.nan, coords=xr.Coordinates(dummy_dict))
        histograms = dataset.groupby(bins_dict).map(self.compute_biais_histogram)
        return histograms

    def compute_biais_histogram(self, dataset: xr.Dataset):
        if dataset.data_vars["biais"].count() == 0:
            out_dataset = xr.DataArray(name="n_occurrences", data=np.nan, coords={"biais_bins": [0]}, dims=("biais_bins",))
        else:
            out_dataset = dataset.groupby(biais=self.biais_bins()).map(self.count_biais_bin)
        return out_dataset

    def count_biais_bin(self, dataset: xr.Dataset):
        return dataset.data_vars["biais"].count().rename("n_occurrences")

    def uncertainty_analysis(
        self,
        test_time_series: xr.Dataset,
        ref_time_series: xr.Dataset,
        config: EvaluationConfig,
        netcdf_export_path: str | None = None,
    ) -> xr.Dataset:
        combined_dataset, analysis_bin_dict = self.prepare_analysis(
            test_time_series=test_time_series,
            ref_time_series=ref_time_series,
            config=config,
        )

        result = combined_dataset.groupby("time").map(self.time_step_analysis, bins_dict=analysis_bin_dict)
        if netcdf_export_path:
            logger.info(f"Exporting to {netcdf_export_path}")
            result.to_netcdf(netcdf_export_path, encoding={"n_occurrences": {"zlib": True}})
        return result


class UncertaintyMeteoFrance(Uncertainty):
    def __init__(self) -> None:
        super().__init__(
            reference_analyzer=S2SnowCoverProductCompleteness(),
            test_analyzer=MeteoFranceSnowCoverProductCompleteness(),
        )


class UncertaintyNASA(Uncertainty):
    def __init__(self) -> None:
        super().__init__(
            reference_analyzer=S2SnowCoverProductCompleteness(),
            test_analyzer=NASASnowCoverProductCompleteness(),
        )


if __name__ == "__main__":
    config = EvaluationConfig(
        ref_fsc_step=10,
        sensor_zenith_analysis=True,
        # Use of forest mask with max resampling because of Météo-France forest with snow class resampling issue.
        # See reprojection_l3_meteofrance_to_grid function
        # In resume, all fractions next to forest with snow class are imprecise because when resampling using average we set this class to 50% FSC
        forest_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2006_forest_mask_utm_max.tif",
        slope_map_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/SLP_MSF_UTM31_375m_lanczos.tif",
        aspect_map_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/ASP_MSF_UTM31_375m_lanczos.tif",
        sub_roi_mask_path=None,
        dem_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/DEM_MSF_UTM31_375m_lanczos.tif",
    )

    config_nasa_l3 = deepcopy(config)
    config_nasa_l3.sensor_zenith_analysis = False

    working_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_6/"

    evaluation_dict: Dict[str, Dict[str, Uncertainty]] = {
        # "meteofrance_orig": {"evaluator": UncertaintyMeteoFrance(), "config": config},
        # "meteofrance_synopsis": {"evaluator": UncertaintyMeteoFrance(), "config": config},
        # "meteofrance_no_cc_mask": {"evaluator": UncertaintyMeteoFrance(), "config": config},
        # "meteofrance_modified": {"evaluator": UncertaintyMeteoFrance(), "config": config},
        # "meteofrance_no_forest": {"evaluator": UncertaintyMeteoFrance(), "config": config},
        # "meteofrance_no_forest_modified": {"evaluator": UncertaintyMeteoFrance(), "config": config},
        "meteofrance_no_forest_red_band_screen": {"evaluator": UncertaintyMeteoFrance(), "config": config},
        # "nasa_pseudo_l3_snpp": {"evaluator": UncertaintyNASA(), "config": config},
        # "nasa_l3_snpp": {"evaluator": UncertaintyNASA(), "config": config_nasa_l3},
        # "nasa_l3_jpss1": {"evaluator": UncertaintyNASA(), "config": config_nasa_l3},
        # "nasa_l3_multiplatform": {"evaluator": UncertaintyNASA(), "config": config_nasa_l3},
    }

    for product, evaluator in evaluation_dict.items():
        ref_time_series, test_time_series, output_filename = generate_evaluation_io(
            analysis_type="uncertainty",
            working_folder=working_folder,
            year=WinterYear(2023, 2024),
            test_product_name=product,
            ref_product_name="s2_theia",
            period=slice("2023-11", "2024-06"),
        )
        logger.info(f"Evaluating product {product}")
        metrics_calcuator = evaluator["evaluator"]
        metrics_calcuator.uncertainty_analysis(
            test_time_series=test_time_series,
            ref_time_series=ref_time_series,
            config=evaluation_dict[product]["config"],
            netcdf_export_path=output_filename,
        )
