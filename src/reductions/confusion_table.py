from copy import deepcopy
from typing import Dict

import xarray as xr

from logger_setup import default_logger as logger
from products.plot_settings import NASA_L3_MULTIPLATFORM_VAR_NAME
from reductions.completeness import (
    MeteoFranceSnowCoverProductCompleteness,
    NASASnowCoverProductCompleteness,
    S2SnowCoverProductCompleteness,
    SnowCoverProductCompleteness,
    mask_of_pixels_in_range,
)
from reductions.statistics_base import EvaluationConfig, EvaluationVsHighResBase, generate_evaluation_io
from winter_year import WinterYear


class ConfusionTable(EvaluationVsHighResBase):
    def __init__(
        self,
        reference_analyzer: SnowCoverProductCompleteness,
        test_analyzer: SnowCoverProductCompleteness,
        ref_fsc_threshold: float | None = None,
        test_fsc_threshold: float | None = None,
    ):
        if ref_fsc_threshold == 0:
            raise ValueError("Ref FSC threshold cannot be 0. Select None or an integer between 1 and 100.")
        self.ref_fsc_threshold = ref_fsc_threshold if ref_fsc_threshold is not None else 1
        if test_fsc_threshold == 0:
            raise ValueError("Test FSC threshold cannot be 0. Select None or an integer between 1 and 100.")
        self.test_fsc_threshold = test_fsc_threshold if test_fsc_threshold is not None else 1
        super().__init__(reference_analyzer, test_analyzer)

    def compute_binary_metrics(self, dataset: xr.Dataset, bins_dict: Dict[str, xr.groupers.Grouper]):
        logger.info(f"Processing time of the year {dataset.coords['time'].values[0].astype('M8[D]').astype('O')}")
        quant_mask_ref = self.ref_analyzer.quantitative_mask(dataset.data_vars["ref"])
        quant_mask_test = self.test_analyzer.quantitative_mask(dataset.data_vars["test"])
        n_intersecting_pixels = (quant_mask_test & quant_mask_ref).sum()

        if n_intersecting_pixels < 2:
            logger.info("No intersection found on this day. Returning a zeros array.")
            return xr.Dataset(
                {
                    k: xr.DataArray(0, coords=xr.Coordinates({k + "_bins": v.labels for k, v in bins_dict.items()}))
                    for k in ["true_positive", "true_negative", "false_positive", "false_negative"]
                }
            )
        snow_test = self.test_analyzer.total_snow_mask(data_array=dataset["test"])
        no_snow_test = self.test_analyzer.total_no_snow_mask(dataset["test"])

        if self.test_fsc_threshold > 1:
            mask_snow_0_50_test = mask_of_pixels_in_range(
                range=range(1, self.test_fsc_threshold * int(self.test_analyzer.max_fsc / 100)), data_array=dataset["test"]
            )
            snow_test = snow_test ^ mask_snow_0_50_test
            no_snow_test = no_snow_test + mask_snow_0_50_test

        snow_ref = self.ref_analyzer.total_snow_mask(data_array=dataset["ref"])
        no_snow_ref = self.ref_analyzer.total_no_snow_mask(dataset["ref"])
        if self.ref_fsc_threshold > 1:
            mask_snow_0_50_ref = mask_of_pixels_in_range(
                range=range(1, self.ref_fsc_threshold * int(self.ref_analyzer.max_fsc / 100)), data_array=dataset["ref"]
            )
            snow_ref = snow_ref ^ mask_snow_0_50_ref
            no_snow_ref = no_snow_ref + mask_snow_0_50_ref

        dataset = dataset.assign({"true_positive": snow_test & snow_ref})
        dataset = dataset.assign({"true_negative": no_snow_test & no_snow_ref})
        dataset = dataset.assign({"false_positive": snow_test & no_snow_ref})
        dataset = dataset.assign({"false_negative": no_snow_test & snow_ref})

        out_dataset = dataset.groupby(bins_dict).map(self.sum_masks)
        return out_dataset

    def sum_masks(self, dataset: xr.Dataset):
        return xr.Dataset(
            {
                "true_positive": dataset.data_vars["true_positive"].sum(),
                "true_negative": dataset.data_vars["true_negative"].sum(),
                "false_positive": dataset.data_vars["false_positive"].sum(),
                "false_negative": dataset.data_vars["false_negative"].sum(),
            }
        )

    def contingency_analysis(
        self,
        test_time_series: xr.Dataset,
        ref_time_series: xr.Dataset,
        config: EvaluationConfig,
        netcdf_export_path: str | None = None,
    ) -> xr.Dataset:
        combined_dataset, analysis_bin_dict = self.prepare_analysis(
            test_time_series=test_time_series, ref_time_series=ref_time_series, config=config
        )
        result = combined_dataset.groupby("time").map(self.compute_binary_metrics, bins_dict=analysis_bin_dict)

        if netcdf_export_path:
            logger.info(f"Exporting to {netcdf_export_path}")
            result.to_netcdf(netcdf_export_path)
        return result


class ConfusionTableMeteoFrance(ConfusionTable):
    def __init__(self, ref_fsc_threshold: float | None = None, test_fsc_threshold: float | None = None) -> None:
        super().__init__(
            reference_analyzer=S2SnowCoverProductCompleteness(),
            test_analyzer=MeteoFranceSnowCoverProductCompleteness(),
            ref_fsc_threshold=ref_fsc_threshold,
            test_fsc_threshold=test_fsc_threshold,
        )


class ConfusionTableNASA(ConfusionTable):
    def __init__(self, ref_fsc_threshold: float | None = None, test_fsc_threshold: float | None = None) -> None:
        super().__init__(
            reference_analyzer=S2SnowCoverProductCompleteness(),
            test_analyzer=NASASnowCoverProductCompleteness(),
            ref_fsc_threshold=ref_fsc_threshold,
            test_fsc_threshold=test_fsc_threshold,
        )


class ConfusionTableMeteoFGranceVsNASA(ConfusionTable):
    def __init__(self, ref_fsc_threshold: float | None = None, test_fsc_threshold: float | None = None) -> None:
        super().__init__(
            reference_analyzer=MeteoFranceSnowCoverProductCompleteness(),
            test_analyzer=NASASnowCoverProductCompleteness(),
            ref_fsc_threshold=ref_fsc_threshold,
            test_fsc_threshold=test_fsc_threshold,
        )


if __name__ == "__main__":
    config = EvaluationConfig(
        ref_fsc_step=98,
        sensor_zenith_analysis=True,
        forest_mask_path=None,
        slope_map_path=None,
        aspect_map_path=None,
        sub_roi_mask_path=None,
        dem_path=None,
    )

    config_nasa_l3 = deepcopy(config)
    config_nasa_l3.sensor_zenith_analysis = False

    working_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_6/"

    ref_fsc_threshold = 15
    test_fsc_threshold = 15

    evaluation_dict: Dict[str, Dict[str, ConfusionTable]] = {
        # "meteofrance_orig": {
        #     "evaluator": ConfusionTableMeteoFrance(ref_fsc_threshold=ref_fsc_threshold, test_fsc_threshold=test_fsc_threshold),
        #     "config": config,
        # },
        # "meteofrance_synopsis": {
        #     "evaluator": ConfusionTableMeteoFrance(ref_fsc_threshold=ref_fsc_threshold, test_fsc_threshold=test_fsc_threshold),
        #     "config": config,
        # },
        # "meteofrance_no_forest": {
        #     "evaluator": ConfusionTableMeteoFrance(ref_fsc_threshold=ref_fsc_threshold, test_fsc_threshold=test_fsc_threshold),
        #     "config": config,
        # },
        # "meteofrance_no_cc_mask": {
        #     "evaluator": ConfusionTableMeteoFrance(ref_fsc_threshold=ref_fsc_threshold, test_fsc_threshold=test_fsc_threshold),
        #     "config": config,
        # },
        # "meteofrance_modified": {
        #     "evaluator": ConfusionTableMeteoFrance(ref_fsc_threshold=ref_fsc_threshold, test_fsc_threshold=test_fsc_threshold),
        #     "config": config,
        # },
        # "nasa_pseudo_l3": {
        #     "evaluator": ConfusionTableNASA(ref_fsc_threshold=ref_fsc_threshold, test_fsc_threshold=test_fsc_threshold),
        #     "config": config,
        # },
        # "nasa_l3_snpp": {
        #     "evaluator": ConfusionTableNASA(ref_fsc_threshold=ref_fsc_threshold, test_fsc_threshold=test_fsc_threshold),
        #     "config": config_nasa_l3,
        # },
        # NASA_L3_MULTIPLATFORM_VAR_NAME: {
        #     "evaluator": ConfusionTableNASA(ref_fsc_threshold=ref_fsc_threshold, test_fsc_threshold=test_fsc_threshold),
        #     "config": config_nasa_l3,
        # },
        # "nasa_l3_jpss1": {
        #     "evaluator": ConfusionTableNASA(ref_fsc_threshold=ref_fsc_threshold, test_fsc_threshold=test_fsc_threshold),
        #     "config": config_nasa_l3,
        # },
    }

    evaluation_dict: Dict[str, Dict[str, ConfusionTable]] = {
        "meteofrance_orig_vs_nasa_snpp": {
            "evaluator": ConfusionTableNASA(ref_fsc_threshold=ref_fsc_threshold, test_fsc_threshold=test_fsc_threshold),
            "config": config_nasa_l3,
        },
    }

    for product, evaluator in evaluation_dict.items():
        ref_time_series, test_time_series, output_filename = generate_evaluation_io(
            analysis_type="confusion_table",
            working_folder=working_folder,
            year=WinterYear(2023, 2024),
            ref_product_name="meteofrance_orig",
            test_product_name="nasa_l3_snpp",
            period=slice("2023-11", "2024-06"),
        )
        logger.info(f"Evaluating product {product}")
        config_to_print = config.__dict__
        config_to_print.update(ref_fsc_threshold=ref_fsc_threshold, test_fsc_threshold=test_fsc_threshold)
        logger.info(f"Config: {config_to_print}")
        metrics_calcuator = evaluator["evaluator"]
        metrics_calcuator.contingency_analysis(
            test_time_series=test_time_series,
            ref_time_series=ref_time_series,
            config=evaluation_dict[product]["config"],
            netcdf_export_path=output_filename,
        )
