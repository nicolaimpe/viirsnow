from typing import Dict

import numpy as np
import xarray as xr
from xarray.groupers import BinGrouper

from logger_setup import default_logger as logger
from reductions.statistics_base import EvaluationVsHighResBase


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
