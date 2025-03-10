from typing import Dict

import numpy as np
import xarray as xr
from xarray.groupers import BinGrouper

from evaluations.completeness import (
    MeteoFranceSnowCoverProductCompleteness,
    NASASnowCoverProductCompleteness,
    S2SnowCoverProductCompleteness,
)
from evaluations.statistics_base import EvaluationVsHighResBase
from logger_setup import default_logger as logger
from winter_year import WinterYear


def histograms_to_biais_rmse(metrics_dataset: xr.Dataset) -> xr.Dataset:
    # From a histogram of differences (bias), compute uncertainty figures
    tot_occurrences = metrics_dataset.data_vars["n_occurrences"].sum()
    n_occur = metrics_dataset.data_vars["n_occurrences"]
    biais_bins = metrics_dataset.coords["biais_bins"]
    biais = (n_occur * biais_bins).sum() / tot_occurrences
    rmse = np.sqrt((n_occur * biais_bins**2).sum() / tot_occurrences)
    unbiaised_rmse = np.sqrt((n_occur * (biais_bins - biais) ** 2).sum() / tot_occurrences)
    return xr.Dataset({"biais": biais, "rmse": rmse, "unbiaised_rmse": unbiaised_rmse})


class Uncertainty(EvaluationVsHighResBase):
    @staticmethod
    def biais_bins():
        return BinGrouper(np.arange(-101, 101, 1), labels=np.arange(-100, 101, 1))

    @staticmethod
    def biais_bins_dynamical(biais_min: float, biais_max: float):
        return BinGrouper(
            np.array([*np.arange(biais_min - 1, biais_max + 1, 1)]),
            labels=np.array([*np.arange(biais_min, biais_max + 1, 1)]),
        )

    def time_step_analysis(self, dataset: xr.Dataset, bins_dict: Dict[str, xr.groupers.Grouper]):
        logger.info(f"Processing time of the year {dataset.coords['time'].values[0].astype('M8[D]').astype('O')}")
        valid_test = (
            dataset.data_vars["test"].where(self.test_analyzer.quantitative_mask(dataset.data_vars["test"]))
            * 100
            / self.test_analyzer.classes["snow_cover"][-1]
        )
        valid_ref = (
            dataset.data_vars["ref"].where(self.ref_analyzer.quantitative_mask(dataset.data_vars["ref"]))
            * 100
            / self.ref_analyzer.classes["snow_cover"][-1]
        )
        dataset = dataset.assign(biais=valid_test - valid_ref)
        histograms = dataset.groupby(bins_dict).map(self.compute_biais_histogram)
        # histograms = (histograms.where(~np.isnan(histograms), 0) + 100).astype("u1")
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
        sensor_zenith_analysis: bool = True,
        forest_mask_path: str | None = None,
        netcdf_export_path: str | None = None,
    ) -> xr.Dataset:
        combined_dataset, analysis_bin_dict = self.prepare_analysis(
            test_time_series=test_time_series,
            ref_time_series=ref_time_series,
            sensor_zenith_analysis=sensor_zenith_analysis,
            forest_mask_path=forest_mask_path,
        )

        result = combined_dataset.groupby("time").map(self.time_step_analysis, bins_dict=analysis_bin_dict)
        if netcdf_export_path:
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
    platform = "SNPP"
    year = WinterYear(2023, 2024)
    products_to_evaluate = ["meteofrance_l3", "nasa_l3", "nasa_pseudo_l3"]
    resolution = 375
    forest_mask_path = "/home/imperatoren/work/VIIRS_S2_comparison/data/vectorial/corine_2006_forest_mask.tif"

    for product_to_evaluate in products_to_evaluate:
        working_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_3/"
        output_folder = f"{working_folder}/analyses/uncertainty"
        ref_time_series_name = f"WY_{year.from_year}_{year.to_year}_S2_res_{resolution}m.nc"
        test_time_series_name = f"WY_{year.from_year}_{year.to_year}_{platform}_{product_to_evaluate}_res_{resolution}m.nc"
        output_filename = f"{output_folder}/uncertainty_WY_{year.from_year}_{year.to_year}_{platform}_{product_to_evaluate}_res_{resolution}m.nc"
        test_time_series = xr.open_dataset(f"{working_folder}/{test_time_series_name}").sel(
            time=slice("2024-03-01", "2024-04-05")
        )
        ref_time_series = xr.open_dataset(f"{working_folder}/{ref_time_series_name}").sel(
            time=slice("2024-03-01", "2024-04-05")
        )
        logger.info(f"Evaluating product {product_to_evaluate}")

        if product_to_evaluate == "nasa_l3":
            metrics_calcuator = UncertaintyNASA()
            metrics_calcuator.uncertainty_analysis(
                test_time_series=test_time_series,
                ref_time_series=ref_time_series,
                sensor_zenith_analysis=False,
                forest_mask_path=forest_mask_path,
                netcdf_export_path=output_filename,
            )

        elif product_to_evaluate == "nasa_pseudo_l3":
            metrics_calcuator = UncertaintyNASA()
            metrics_calcuator.uncertainty_analysis(
                test_time_series=test_time_series,
                ref_time_series=ref_time_series,
                sensor_zenith_analysis=True,
                forest_mask_path=forest_mask_path,
                netcdf_export_path=output_filename,
            )

        elif product_to_evaluate == "meteofrance_l3":
            metrics_calcuator = UncertaintyMeteoFrance()
            metrics_calcuator.uncertainty_analysis(
                test_time_series=test_time_series,
                ref_time_series=ref_time_series,
                sensor_zenith_analysis=True,
                forest_mask_path=forest_mask_path,
                netcdf_export_path=output_filename,
            )

        else:
            raise NotImplementedError(f"Unknown product: {product_to_evaluate}")
