from copy import deepcopy
from typing import Dict

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from scores.categorical import BasicContingencyManager
from sklearn.metrics import ConfusionMatrixDisplay

from evaluations.completeness import (
    MeteoFranceSnowCoverProductCompleteness,
    NASASnowCoverProductCompleteness,
    S2SnowCoverProductCompleteness,
    SnowCoverProductCompleteness,
)
from evaluations.statistics_base import EvaluationConfig, EvaluationVsHighResBase, generate_evaluation_io
from logger_setup import default_logger as logger
from winter_year import WinterYear

SCORES = ["accuracy", "precision", "recall", "f1_score", "commission_error", "omission_error"]


def compute_score(dataset: xr.Dataset, score_name: str):
    tp, tn, fp, fn = (
        dataset.data_vars["true_positive"].sum(),
        dataset.data_vars["true_negative"].sum(),
        dataset.data_vars["false_positive"].sum(),
        dataset.data_vars["false_negative"].sum(),
    )
    scores_manager = BasicContingencyManager(
        counts={"tp_count": tp, "tn_count": tn, "fp_count": fp, "fn_count": fn, "total_count": tp + tn + fp + fn}
    )

    return getattr(scores_manager, score_name)()


def omission_error(dataset: xr.Dataset):
    return dataset["false_negative"].sum() / (dataset["true_positive"].sum() + dataset["false_negative"].sum())


def compute_all_scores(dataset: xr.Dataset):
    out_scores_dict = {}
    for score in SCORES:
        if score == "omission_error":
            out_scores_dict.update({score: omission_error(dataset)})
        elif score == "commission_error":
            out_scores_dict.update({score: compute_score(dataset, score_name="false_alarm_rate")})
        else:
            out_scores_dict.update({score: compute_score(dataset, score_name=score)})
    return xr.Dataset(out_scores_dict)


def plot_confusion_table(dataset: xr.Dataset, axes: Axes | None = None):
    tot = np.sum(np.array([dataset[dv].sum() for dv in dataset]))

    confusion_matrix = np.array(
        [
            [
                dataset.data_vars["true_positive"].sum().values / tot * 100,
                dataset.data_vars["false_negative"].sum().values / tot * 100,
            ],
            [
                dataset.data_vars["false_positive"].sum().values / tot * 100,
                dataset.data_vars["true_negative"].sum().values / tot * 100,
            ],
        ],
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["snow", "no_snow"])
    disp.plot(ax=axes, colorbar=False)


class ConfusionTable(EvaluationVsHighResBase):
    def __init__(
        self,
        reference_analyzer: SnowCoverProductCompleteness,
        test_analyzer: SnowCoverProductCompleteness,
        fsc_threshold: float | None = None,
    ):
        self.fsc_threshold = fsc_threshold if fsc_threshold is not None else 0
        super().__init__(reference_analyzer, test_analyzer)

    def compute_binary_metrics(self, dataset: xr.Dataset, bins_dict: Dict[str, xr.groupers.Grouper]):
        logger.info(f"Processing time of the year {dataset.coords['time'].values[0].astype('M8[D]').astype('O')}")

        snow_test = self.test_analyzer.total_snow_mask(dataset["test"])
        no_snow_test = self.test_analyzer.total_no_snow_mask(dataset["test"])

        snow_ref = self.ref_analyzer.total_snow_mask(dataset["ref"])
        no_snow_ref = self.ref_analyzer.total_no_snow_mask(dataset["ref"])

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
    def __init__(self, fsc_threshold: float | None = None) -> None:
        super().__init__(
            reference_analyzer=S2SnowCoverProductCompleteness(),
            test_analyzer=MeteoFranceSnowCoverProductCompleteness(),
            fsc_threshold=fsc_threshold,
        )


class ConfusionTableNASA(ConfusionTable):
    def __init__(self, fsc_threshold: float | None = None) -> None:
        super().__init__(
            reference_analyzer=S2SnowCoverProductCompleteness(),
            test_analyzer=NASASnowCoverProductCompleteness(),
            fsc_threshold=fsc_threshold,
        )


if __name__ == "__main__":
    config = EvaluationConfig(
        ref_fsc_step=10,
        sensor_zenith_analysis=True,
        forest_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2006_forest_mask.tif",
        slope_map_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/SLP_MSF_UTM31_375m_lanczos.tif",
        aspect_map_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/ASP_MSF_UTM31_375m_lanczos.tif",
        sub_roi_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/MSF_MACRO_FRANCE_UTM31_375m.tif",
        dem_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/DEM_MSF_UTM31_375m_lanczos.tif",
    )

    config_nasa_l3 = deepcopy(config)
    config_nasa_l3.sensor_zenith_analysis = False

    working_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_4/"

    fsc_threshold = None
    evaluation_dict: Dict[str, Dict[str, ConfusionTable]] = {
        "meteofrance_l3": {"evaluator": ConfusionTableMeteoFrance(fsc_threshold=fsc_threshold), "config": config},
        "nasa_pseudo_l3": {"evaluator": ConfusionTableNASA(fsc_threshold=fsc_threshold), "config": config},
        "nasa_l3": {"evaluator": ConfusionTableNASA(fsc_threshold=fsc_threshold), "config": config_nasa_l3},
    }

    for product, evaluator in evaluation_dict.items():
        ref_time_series, test_time_series, output_filename = generate_evaluation_io(
            analysis_type="confusion_table",
            working_folder=working_folder,
            year=WinterYear(2023, 2024),
            resolution=375,
            platform="SNPP",
            product_name=product,
            period=None,
        )
        logger.info(f"Evaluating product {product}")
        metrics_calcuator = evaluator["evaluator"]
        metrics_calcuator.contingency_analysis(
            test_time_series=test_time_series,
            ref_time_series=ref_time_series,
            config=evaluation_dict[product]["config"],
            netcdf_export_path=output_filename,
        )
