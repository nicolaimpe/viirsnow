import abc
from typing import Dict

import numpy as np
import rioxarray
import xarray as xr
from sklearn.metrics import ConfusionMatrixDisplay
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

            reference_analyzer.classes.update(no_snow=range(0, int(fsc_threshold * max_fsc_ref)))
            test_analyzer.classes.update(no_snow=range(0, int(fsc_threshold * max_fsc_test)))

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
        return BinGrouper(np.arange(0, 90, 15), labels=np.arange(15, 90, 15))

    @staticmethod
    def ref_fsc_bins():
        return BinGrouper(
            np.array([-1, *np.arange(0, 99, 10), 99, 100]),
            labels=np.array([0, *np.arange(10, 99, 10), 99, 100]),
        )

    @staticmethod
    def forest_bins():
        return BinGrouper([-1, 0, 1], labels=[0, 1])

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
        out_dataset = out_dataset.assign_coords({"time": dataset.coords["time"].values})
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
            },
        )

        analysis_bin_dict = dict(ref=self.ref_fsc_bins())

        if sensor_zenith_analysis:
            analysis_bin_dict.update(sensor_zenith=self.sensor_zenith_bins())
            all_products_dataset = all_products_dataset.assign(
                {"sensor_zenith": test_time_series.data_vars["sensor_zenith"].sel(time=common_days)}
            )

        if forest_mask_path is not None:
            forest_mask = rioxarray.open_rasterio(forest_mask_path)
            all_products_dataset = all_products_dataset.assign(forest_mask=forest_mask.sel(band=1).drop_vars("band"))
            analysis_bin_dict.update(forest_mask=self.forest_bins())

        result = all_products_dataset.groupby("time").map(self.compute_binary_metrics, bins_dict=analysis_bin_dict)

        if netcdf_export_path:
            result.to_netcdf(
                netcdf_export_path,
                encoding=dict(time={"calendar": "gregorian", "units": f"days since {str(year.from_year)}-10-01"}),
            )
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
        ) | self.test_analyzer.compute_mask_of_class("forest_with_snow", data_array)
        return snow_meteofrance

    def compute_test_product_no_snow_mask(self, data_array: xr.DataArray):
        no_snow_meteofrance = (
            self.test_analyzer.compute_mask_of_class("no_snow", data_array)
            | self.test_analyzer.compute_mask_of_class("forest_without_snow", data_array)
            | self.test_analyzer.compute_mask_of_class("water", data_array)
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
        ) | self.test_analyzer.compute_mask_of_class("water", data_array)
        return no_snow_nasa


if __name__ == "__main__":
    platform = "SNPP"
    year = WinterYear(2023, 2024)
    products_to_evaluate = ["meteofrance_l3", "nasa_l3", "nasa_pseudo_l3"]
    resolution = 375
    fsc_threshold = 0.15
    forest_mask_path = "/home/imperatoren/work/VIIRS_S2_comparison/data/vectorial/corine_2006_forest_mask.tif"

    for product_to_evaluate in products_to_evaluate:
        working_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_3/"
        output_folder = f"{working_folder}/analyses/confusion_table"
        ref_time_series_name = f"WY_{year.from_year}_{year.to_year}_S2_res_{resolution}m.nc"
        test_time_series_name = f"WY_{year.from_year}_{year.to_year}_{platform}_{product_to_evaluate}_res_{resolution}m.nc"
        output_filename = f"{output_folder}/confusion_table_WY_{year.from_year}_{year.to_year}_{platform}_{product_to_evaluate}_res_{resolution}m.nc"
        test_time_series = xr.open_dataset(f"{working_folder}/{test_time_series_name}").isel(time=slice(100, 130))
        ref_time_series = xr.open_dataset(f"{working_folder}/{ref_time_series_name}").isel(time=slice(100, 130))

        logger.info(f"Evaluating product {products_to_evaluate}")
        if product_to_evaluate == "nasa_l3":
            metrics_calcuator = ConfusionTableNASA(fsc_threshold=fsc_threshold)
            metrics_calcuator.contingency_analysis(
                test_time_series=test_time_series,
                ref_time_series=ref_time_series,
                sensor_zenith_analysis=False,
                forest_mask_path=forest_mask_path,
                netcdf_export_path=output_filename,
            )
        elif product_to_evaluate == "nasa_pseudo_l3":
            metrics_calcuator = ConfusionTableNASA(fsc_threshold=fsc_threshold)
            metrics_calcuator.contingency_analysis(
                test_time_series=test_time_series,
                ref_time_series=ref_time_series,
                sensor_zenith_analysis=True,
                forest_mask_path=forest_mask_path,
                netcdf_export_path=output_filename,
            )
        elif product_to_evaluate == "meteofrance_l3":
            metrics_calcuator = ConfusionTableMeteoFrance(fsc_threshold=fsc_threshold)
            metrics_calcuator.contingency_analysis(
                test_time_series=test_time_series,
                ref_time_series=ref_time_series,
                sensor_zenith_analysis=True,
                forest_mask_path=forest_mask_path,
                netcdf_export_path=output_filename,
            )
        else:
            raise NotImplementedError(f"Unknown product: {product_to_evaluate}")
