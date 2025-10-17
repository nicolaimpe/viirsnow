from typing import Dict

import numpy as np
import xarray as xr
from xarray.groupers import BinGrouper

from logger_setup import default_logger as logger
from reductions.statistics_base import EvaluationConfig, EvaluationVsHighResBase


class Scatter(EvaluationVsHighResBase):
    def test_bins(self):
        """Test data have to be normalized between 1 and 100 for snow cover."""
        return BinGrouper(
            np.array([*np.arange(-1, 101, 1), self.test_analyzer.max_value]),
            labels=np.array([*np.arange(0, 101, 1), self.test_analyzer.max_value]),
        )

    def time_step_analysis(self, dataset: xr.Dataset, bins_dict: Dict[str, xr.groupers.Grouper]):
        logger.info(f"Processing time of the year {dataset.coords['time'].values[0].astype('M8[D]').astype('O')}")
        bins_dict.update(test=self.test_bins())

        quant_mask_ref = self.ref_analyzer.quantitative_mask(dataset.data_vars["ref"])
        quant_mask_test = self.test_analyzer.quantitative_mask(dataset.data_vars["test"])
        quantitative_mask_union = quant_mask_test & quant_mask_ref
        n_intersecting_pixels = quantitative_mask_union.sum()

        if n_intersecting_pixels < 2:
            logger.info("No intersection found on this day. Returning a zeros array.")
            return xr.DataArray(0, coords=xr.Coordinates({k + "_bins": v.labels for k, v in bins_dict.items()}))

        dataset.data_vars["ref"].values = (
            dataset.data_vars["ref"].where(quantitative_mask_union) * 100 / self.test_analyzer.max_fsc
        )
        dataset.data_vars["test"].values = (
            dataset.data_vars["test"].where(quantitative_mask_union) * 100 / self.test_analyzer.max_fsc
        )
        scatter = dataset.groupby(bins_dict).map(self.compute_scatter_plot)

        return scatter

    def compute_scatter_plot(self, dataset: xr.Dataset):
        # Counting ref or test doesn't really change here
        return dataset.data_vars["ref"].count().rename("n_occurrences")

    def scatter_analysis(
        self,
        test_time_series: xr.Dataset,
        ref_time_series: xr.Dataset,
        config: EvaluationConfig,
        netcdf_export_path: str | None = None,
    ) -> xr.Dataset:
        combined_dataset, analysis_bin_dict = self.prepare_analysis(
            test_time_series=test_time_series, ref_time_series=ref_time_series, config=config
        )

        result = combined_dataset.groupby("time").map(self.time_step_analysis, bins_dict=analysis_bin_dict)
        if netcdf_export_path:
            logger.info(f"Exporting to {netcdf_export_path}")
            result.to_netcdf(netcdf_export_path, encoding={"n_occurrences": {"zlib": True}})
        return result
