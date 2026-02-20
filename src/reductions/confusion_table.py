from typing import Dict

import xarray as xr

from logger_setup import default_logger as logger
from reductions.completeness import SnowCoverProductCompleteness, mask_of_pixels_in_range
from reductions.statistics_base import EvaluationConfig, EvaluationVsHighResBase


class ConfusionTable(EvaluationVsHighResBase):
    def __init__(
        self,
        evaluation_config: EvaluationConfig,
        reference_analyzer: SnowCoverProductCompleteness,
        eval_analyzer: SnowCoverProductCompleteness,
        eval_fsc_threshold: int = 50,
        ref_fsc_threshold: int = 50,
    ):
        super().__init__(evaluation_config, reference_analyzer, eval_analyzer)
        self.eval_fsc_threshold = eval_fsc_threshold
        self.ref_fsc_threshold = ref_fsc_threshold

    def time_step_analysis(self, dataset: xr.Dataset, bins_dict: Dict[str, xr.groupers.Grouper]):
        logger.info(f"Processing time of the year {dataset.coords['time'].values[0].astype('M8[D]').astype('O')}")
        quant_mask_ref = self.ref_analyzer.quantitative_mask(dataset.data_vars["ref"])
        quant_mask_eval = self.eval_analyzer.quantitative_mask(dataset.data_vars["eval"])
        n_intersecting_pixels = (quant_mask_eval & quant_mask_ref).sum()

        if n_intersecting_pixels < 2:
            logger.info("No intersection found on this day. Returning a zeros array.")
            return xr.Dataset(
                {
                    k: xr.DataArray(0, coords=xr.Coordinates({k + "_bins": v.labels for k, v in bins_dict.items()}))
                    for k in ["true_positive", "true_negative", "false_positive", "false_negative"]
                }
            )
        snow_eval = self.eval_analyzer.total_snow_mask(data_array=dataset["eval"])
        no_snow_eval = self.eval_analyzer.total_no_snow_mask(dataset["eval"])

        if self.eval_fsc_threshold > 1:
            low_snow_eval = mask_of_pixels_in_range(
                range=range(1, self.eval_fsc_threshold * int(self.eval_analyzer.max_fsc / 100)), data_array=dataset["eval"]
            )
            snow_eval = snow_eval ^ low_snow_eval
            no_snow_eval = no_snow_eval + low_snow_eval

        snow_ref = self.ref_analyzer.total_snow_mask(data_array=dataset["ref"])
        no_snow_ref = self.ref_analyzer.total_no_snow_mask(dataset["ref"])
        if self.ref_fsc_threshold > 1:
            low_snow_ref = mask_of_pixels_in_range(
                range=range(1, self.ref_fsc_threshold * int(self.ref_analyzer.max_fsc / 100)), data_array=dataset["ref"]
            )
            snow_ref = snow_ref ^ low_snow_ref
            no_snow_ref = no_snow_ref + low_snow_ref

        dataset = dataset.assign({"true_positive": snow_eval & snow_ref})
        dataset = dataset.assign({"true_negative": no_snow_eval & no_snow_ref})
        dataset = dataset.assign({"false_positive": snow_eval & no_snow_ref})
        dataset = dataset.assign({"false_negative": no_snow_eval & snow_ref})

        if "spatial_ref" in dataset.coords:
            dataset = dataset.drop_vars("spatial_ref")
        if "Projection" in dataset.coords:
            dataset = dataset.drop_vars("Projection")
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
