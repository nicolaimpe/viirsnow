import abc
from typing import Dict

import numpy as np
import rioxarray
import xarray as xr
from xarray.groupers import BinGrouper

from evaluations.completeness import (
    MeteoFranceSnowCoverProductCompleteness,
    NASASnowCoverProductCompleteness,
    S2SnowCoverProductCompleteness,
    SnowCoverProductCompleteness,
)
from logger_setup import default_logger as logger
from winter_year import WinterYear


def accuracy(dataset: xr.Dataset):
    tot = np.sum(np.array([dataset[dv].sum() for dv in dataset]))
    return (dataset.data_vars["true_positive"].sum() + dataset.data_vars["true_negative"].sum()) / tot


class ConfusionTable:
    def __init__(
        self,
        reference_analyzer: SnowCoverProductCompleteness,
        test_analyzer: SnowCoverProductCompleteness,
        fsc_threshold: float | None = None,
    ) -> None:
        if fsc_threshold is not None:
            max_fsc_ref = reference_analyzer.classes["snow_cover"][-1]
            max_fsc_test = test_analyzer.classes["snow_cover"][-1]
            reference_analyzer.classes.update(snow_cover=range(int(fsc_threshold * max_fsc_ref), max_fsc_ref + 1))
            test_analyzer.classes.update(snow_cover=range(int(fsc_threshold * max_fsc_test), max_fsc_test + 1))
        self.ref_analyzer = reference_analyzer
        self.test_analyzer = test_analyzer

    @abc.abstractmethod
    def compute_test_product_snow_mask(self, data_array: xr.DataArray):
        pass

    @abc.abstractmethod
    def compute_test_product_no_snow_mask(self, data_array: xr.DataArray):
        pass

    @staticmethod
    def sensor_zenith_bins():
        return BinGrouper(np.arange(0, 90, 15), labels=["0-15", "15-30", "30-45", "45-60", "60-75"])

    @staticmethod
    def ref_fsc_bins():
        return BinGrouper(
            np.array([-1, *np.arange(0, 99, 10), 99, 100]),
            labels=["0", "1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-99", "100"],
        )

    @staticmethod
    def forest_bins():
        return BinGrouper([-1, 0, 1], labels=["no_forest", "forest"])

    def compute_binary_metrics(self, dataset: xr.Dataset, bins_dict: Dict[str, xr.groupers.Grouper]):
        logger.info(f"Processing time of the year {dataset.coords['time'].values[0].astype('M8[D]').astype('O')}")

        snow_test = self.compute_test_product_snow_mask(dataset["test"])
        no_snow_test = self.compute_test_product_no_snow_mask(dataset["test"])

        snow_ref = self.ref_analyzer.compute_mask_of_class("snow_cover", dataset["ref"])
        no_snow_ref = self.ref_analyzer.compute_mask_of_class("no_snow", dataset["ref"])

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
        netcdf_export_path: str | None = None,
    ) -> xr.Dataset:
        common_days = np.intersect1d(test_time_series["time"], ref_time_series["time"])

        all_products_dataset = xr.Dataset(
            {
                "ref": ref_time_series.data_vars["snow_cover_fraction"].sel(time=common_days),
                "test": test_time_series.data_vars["snow_cover_fraction"].sel(time=common_days),
                "sensor_zenith": test_time_series.data_vars["sensor_zenith"].sel(time=common_days),
            },
        )

        analysis_bin_dict = dict(ref=self.ref_fsc_bins())

        if sensor_zenith_analysis:
            analysis_bin_dict.update(sensor_zenith=self.sensor_zenith_bins())

        if forest_mask_path is not None:
            forest_mask = rioxarray.open_rasterio(forest_mask_path)
            all_products_dataset = all_products_dataset.assign(forest_mask=forest_mask.sel(band=1).drop_vars("band"))
            analysis_bin_dict.update(forest_mask=self.forest_bins())

        result = all_products_dataset.groupby("time").map(self.compute_binary_metrics, bins_dict=analysis_bin_dict)

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

    def compute_test_product_snow_mask(self, data_array: xr.DataArray):
        snow_meteofrance = self.test_analyzer.compute_mask_of_class(
            "snow_cover", data_array
        ) & self.test_analyzer.compute_mask_of_class("forest_with_snow", data_array)
        return snow_meteofrance

    def compute_test_product_no_snow_mask(self, data_array: xr.DataArray):
        no_snow_meteofrance = (
            self.test_analyzer.compute_mask_of_class("no_snow", data_array)
            & self.test_analyzer.compute_mask_of_class("forest_without_snow", data_array)
            & self.test_analyzer.compute_mask_of_class("water", data_array)
        )
        return no_snow_meteofrance


class ConfusionTableNASA(ConfusionTable):
    def __init__(self, fsc_threshold: float | None = None) -> None:
        super().__init__(
            reference_analyzer=S2SnowCoverProductCompleteness(),
            test_analyzer=NASASnowCoverProductCompleteness(),
            fsc_threshold=fsc_threshold,
        )

    def compute_test_product_snow_mask(self, data_array: xr.DataArray):
        snow_nasa = self.test_analyzer.compute_mask_of_class("snow_cover", data_array)
        return snow_nasa

    def compute_test_product_no_snow_mask(self, data_array: xr.DataArray):
        no_snow_nasa = self.test_analyzer.compute_mask_of_class(
            "no_snow", data_array
        ) & self.test_analyzer.compute_mask_of_class("water", data_array)
        return no_snow_nasa


if __name__ == "__main__":
    platform = "SNPP"
    year = WinterYear(2023, 2024)
    product_to_evaluate = "nasa_pseudo_l3"
    resolution = 375
    fsc_threshold = 0.15
    forest_mask_path = "/home/imperatoren/work/VIIRS_S2_comparison/data/vectorial/corine_2006_forest_mask.tif"

    working_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_3/"
    output_folder = f"{working_folder}/analyses/confusion_table"
    ref_time_series_name = f"WY_{year.from_year}_{year.to_year}_S2_res_{resolution}m.nc"
    test_time_series_name = f"WY_{year.from_year}_{year.to_year}_{platform}_{product_to_evaluate}_res_{resolution}m.nc"
    output_filename = f"{output_folder}/confusiont_table_WY_{year.from_year}_{year.to_year}_{platform}_{product_to_evaluate}_res_{resolution}m.nc"
    test_time_series = xr.open_dataset(f"{working_folder}/{test_time_series_name}").sel(time="2023-12")
    ref_time_series = xr.open_dataset(f"{working_folder}/{ref_time_series_name}").sel(time="2023-12")

    if product_to_evaluate == "nasa_l3":
        metrics_calcuator = ConfusionTableNASA(fsc_threshold=0.15)
        metrics_calcuator.contingency_analysis(
            test_time_series=test_time_series,
            ref_time_series=ref_time_series,
            sensor_zenith_analysis=False,
            forest_mask_path=forest_mask_path,
            netcdf_export_path=output_filename,
        )
    elif product_to_evaluate == "nasa_pseudo_l3":
        metrics_calcuator = ConfusionTableNASA(fsc_threshold=0.15)
        metrics_calcuator.contingency_analysis(
            test_time_series=test_time_series,
            ref_time_series=ref_time_series,
            sensor_zenith_analysis=True,
            forest_mask_path=forest_mask_path,
            netcdf_export_path=output_filename,
        )
    elif product_to_evaluate == "meteofrance_l3":
        metrics_calcuator = ConfusionTableMeteoFrance(fsc_threshold=0.15)
        metrics_calcuator.contingency_analysis(
            test_time_series=test_time_series,
            ref_time_series=ref_time_series,
            sensor_zenith_analysis=True,
            forest_mask_path=forest_mask_path,
            netcdf_export_path=output_filename,
        )
    else:
        raise NotImplementedError(f"Unknown product: {product_to_evaluate}")
