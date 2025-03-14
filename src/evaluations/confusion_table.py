from typing import Dict

import numpy as np
import xarray as xr
from scores.categorical import BasicContingencyManager
from sklearn.metrics import ConfusionMatrixDisplay

from evaluations.completeness import (
    MeteoFranceSnowCoverProductCompleteness,
    NASASnowCoverProductCompleteness,
    S2SnowCoverProductCompleteness,
    SnowCoverProductCompleteness,
    mask_of_pixels_in_range,
)
from evaluations.statistics_base import EvaluationVsHighResBase
from logger_setup import default_logger as logger
from winter_year import WinterYear

SCORES = ["precision", "recall", "f1_score", "commission_error", "omission_error", "accuracy"]


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


def plot_confusion_table(dataset: xr.Dataset):
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
    disp.plot()


class ConfusionTable(EvaluationVsHighResBase):
    def __init__(
        self,
        reference_analyzer: SnowCoverProductCompleteness,
        test_analyzer: SnowCoverProductCompleteness,
        fsc_threshold: float | None = None,
    ):
        self.fsc_threshold = fsc_threshold if fsc_threshold is not None else None
        super().__init__(reference_analyzer, test_analyzer)

    def compute_binary_metrics(self, dataset: xr.Dataset, bins_dict: Dict[str, xr.groupers.Grouper]):
        logger.info(f"Processing time of the year {dataset.coords['time'].values[0].astype('M8[D]').astype('O')}")

        snow_test = mask_of_pixels_in_range(
            range(int(fsc_threshold * self.test_analyzer.max_fsc), self.test_analyzer.max_fsc), dataset["test"]
        )
        no_snow_test = self.test_analyzer.total_no_snow_mask(dataset["test"])

        snow_ref = mask_of_pixels_in_range(
            range(int(fsc_threshold * self.ref_analyzer.max_fsc), self.ref_analyzer.max_fsc), dataset["ref"]
        )
        no_snow_ref = self.ref_analyzer.total_no_snow_mask(dataset["ref"])

        # snow_test = self.test_analyzer.total_snow_mask(dataset["test"])
        # no_snow_test = self.test_analyzer.total_no_snow_mask(dataset["test"])

        # snow_ref = self.ref_analyzer.total_snow_mask(dataset["ref"])
        # no_snow_ref = self.ref_analyzer.total_no_snow_mask(dataset["ref"])

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
        sensor_zenith_analysis: bool = True,
        forest_mask_path: str | None = None,
        slope_map_path: str | None = None,
        sub_roi_mask_path: str | None = None,
        aspect_map_path: str | None = None,
        dem_path: str | None = None,
        netcdf_export_path: str | None = None,
    ) -> xr.Dataset:
        combined_dataset, analysis_bin_dict = self.prepare_analysis(
            test_time_series=test_time_series,
            ref_time_series=ref_time_series,
            ref_fsc_step=25,
            sensor_zenith_analysis=sensor_zenith_analysis,
            forest_mask_path=forest_mask_path,
            sub_roi_mask_path=sub_roi_mask_path,
            slope_map_path=slope_map_path,
            aspect_map_path=aspect_map_path,
            dem_path=dem_path,
        )
        result = combined_dataset.groupby("time").map(self.compute_binary_metrics, bins_dict=analysis_bin_dict)

        if netcdf_export_path:
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
    platform = "SNPP"
    year = WinterYear(2023, 2024)
    products_to_evaluate = ["meteofrance_l3", "nasa_pseudo_l3", "nasa_l3"]
    resolution = 375
    fsc_threshold = 0.15
    forest_mask_path = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2006_forest_mask.tif"
    slope_map_path = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/SLP_MSF_UTM31_375m_lanczos.tif"
    aspect_map_path = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/ASP_MSF_UTM31_375m_lanczos.tif"
    massifs_mask_path = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/MSF_MACRO_FRANCE_UTM31_375m.tif"
    dem_path = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/DEM_MSF_UTM31_375m_lanczos.tif"
    for product_to_evaluate in products_to_evaluate:
        working_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_3/"
        output_folder = f"{working_folder}/analyses/confusion_table_test"
        ref_time_series_name = f"WY_{year.from_year}_{year.to_year}_S2_res_{resolution}m.nc"
        test_time_series_name = f"WY_{year.from_year}_{year.to_year}_{platform}_{product_to_evaluate}_res_{resolution}m.nc"
        output_filename = f"{output_folder}/confusion_table_WY_{year.from_year}_{year.to_year}_{platform}_{product_to_evaluate}_res_{resolution}m.nc"
        test_time_series = xr.open_dataset(f"{working_folder}/{test_time_series_name}").isel(time=slice(100, 120))
        ref_time_series = xr.open_dataset(f"{working_folder}/{ref_time_series_name}").isel(time=slice(100, 120))
        logger.info(f"Evaluating product {product_to_evaluate}")

        if product_to_evaluate in "nasa_l3":
            metrics_calcuator = ConfusionTableNASA(fsc_threshold=fsc_threshold)
            metrics_calcuator.contingency_analysis(
                test_time_series=test_time_series,
                ref_time_series=ref_time_series,
                sensor_zenith_analysis=False,
                forest_mask_path=forest_mask_path,
                sub_roi_mask_path=massifs_mask_path,
                slope_map_path=slope_map_path,
                aspect_map_path=aspect_map_path,
                dem_path=dem_path,
                netcdf_export_path=output_filename,
            )

        elif product_to_evaluate == "nasa_pseudo_l3":
            metrics_calcuator = ConfusionTableNASA(fsc_threshold=fsc_threshold)
            metrics_calcuator.contingency_analysis(
                test_time_series=test_time_series,
                ref_time_series=ref_time_series,
                sensor_zenith_analysis=True,
                forest_mask_path=forest_mask_path,
                sub_roi_mask_path=massifs_mask_path,
                slope_map_path=slope_map_path,
                aspect_map_path=aspect_map_path,
                dem_path=dem_path,
                netcdf_export_path=output_filename,
            )

        elif product_to_evaluate == "meteofrance_l3":
            metrics_calcuator = ConfusionTableMeteoFrance(fsc_threshold=fsc_threshold)
            metrics_calcuator.contingency_analysis(
                test_time_series=test_time_series,
                ref_time_series=ref_time_series,
                sensor_zenith_analysis=True,
                forest_mask_path=forest_mask_path,
                sub_roi_mask_path=massifs_mask_path,
                slope_map_path=slope_map_path,
                aspect_map_path=aspect_map_path,
                dem_path=dem_path,
                netcdf_export_path=output_filename,
            )

        else:
            raise NotImplementedError(f"Unknown product: {product_to_evaluate}")
