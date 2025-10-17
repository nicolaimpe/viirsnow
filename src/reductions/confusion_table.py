from typing import Dict

import xarray as xr

from logger_setup import default_logger as logger
from reductions.completeness import SnowCoverProductCompleteness, mask_of_pixels_in_range
from reductions.statistics_base import EvaluationVsHighResBase


class ConfusionTable(EvaluationVsHighResBase):
    def __init__(
        self,
        reference_analyzer: SnowCoverProductCompleteness,
        test_analyzer: SnowCoverProductCompleteness,
        test_fsc_threshold: int = 50,
        ref_fsc_threshold: int = 50,
    ):
        super().__init__(reference_analyzer, test_analyzer)
        self.test_fsc_threshold = test_fsc_threshold
        self.ref_fsc_threshold = ref_fsc_threshold

    def time_step_analysis(self, dataset: xr.Dataset, bins_dict: Dict[str, xr.groupers.Grouper]):
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
