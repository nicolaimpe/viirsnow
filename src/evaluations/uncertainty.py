from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from xarray.groupers import BinGrouper

from evaluations.completeness import (
    MeteoFranceSnowCoverProductCompleteness,
    NASASnowCoverProductCompleteness,
    S2SnowCoverProductCompleteness,
)
from evaluations.statistics_base import EvaluationVsHighResBase
from logger_setup import default_logger as logger
from products.plot_settings import PRODUCT_PLOT_COLORS
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


def postprocess_uncertainty_analysis(
    uncertainty_datasets: Dict[str, xr.Dataset], analysis_var: str | BinGrouper
) -> xr.Dataset:
    reduced_datasets = []
    for dataset in uncertainty_datasets.values():
        reduced_datasets.append(dataset.groupby(analysis_var).map(histograms_to_biais_rmse))
    concatenated = xr.concat(reduced_datasets, pd.Index(list(uncertainty_datasets.keys()), name="product"), coords="minimal")
    return concatenated


def barplots(postprocessed_data_array: xr.DataArray, title: str, y_lim: Tuple[int, int]):
    plot_dataframe_dict = {}
    colors = []
    for product in postprocessed_data_array.coords["product"].values:
        plot_dataframe_dict.update({product: postprocessed_data_array.sel(product=product).to_pandas()})
        colors.append(PRODUCT_PLOT_COLORS[product])
    plot_dataframe = pd.DataFrame(plot_dataframe_dict)
    if "month" in title:
        plot_dataframe.index = plot_dataframe.index.strftime("%B")
    if "aspect" in title:
        # That's because to_pandas() reorders the aspect labels alphabetically
        plot_dataframe = plot_dataframe.reindex(index=EvaluationVsHighResBase.aspect_bins().labels)
    plot_dataframe.plot.bar(figsize=(14, 4), color=colors, width=0.6, title=title)
    plt.ylim(y_lim)
    plt.grid(True, axis="y")


def biais_barplots(postprocessed_dataset: xr.Dataset, analysis_var_plot_name: str, winter_year: WinterYear):
    barplots(
        postprocessed_data_array=postprocessed_dataset.data_vars["biais"],
        title=f"Biais comparison {analysis_var_plot_name} - {str(winter_year)}",
        y_lim=(-15, 15),
    )


def rmse_barplots(postprocessed_dataset: xr.Dataset, analysis_var_plot_name: str, winter_year: WinterYear):
    barplots(
        postprocessed_data_array=postprocessed_dataset.data_vars["rmse"],
        title=f"RMSE comparison {analysis_var_plot_name} - {str(winter_year)}",
        y_lim=(0, 40),
    )


def unbiaised_rmse_barplots(postprocessed_dataset: xr.Dataset, analysis_var_plot_name: str, winter_year: WinterYear):
    barplots(
        postprocessed_data_array=postprocessed_dataset.data_vars["unbiaised_rmse"],
        title=f"Unbiaised RMSE comparison {analysis_var_plot_name} - {str(winter_year)}",
        y_lim=(0, 40),
    )


def histograms_to_distribution(metrics_ds: xr.Dataset):
    all_dims = list(metrics_ds.sizes.keys())
    all_dims.remove("biais_bins")
    metrics_squeezed = metrics_ds.sum(dim=all_dims)
    distribution = np.repeat(metrics_ds.coords["biais_bins"].values, metrics_squeezed["n_occurrences"].values.astype(np.int64))
    return distribution


class Uncertainty(EvaluationVsHighResBase):
    @staticmethod
    def biais_bins():
        return BinGrouper(np.arange(-101, 101, 1), labels=np.arange(-100, 101, 1))

    def time_step_analysis(self, dataset: xr.Dataset, bins_dict: Dict[str, xr.groupers.Grouper]):
        logger.info(f"Processing time of the year {dataset.coords['time'].values[0].astype('M8[D]').astype('O')}")
        valid_test = dataset.data_vars["test"].where(self.test_analyzer.quantitative_mask(dataset.data_vars["test"]))
        valid_test = valid_test * 100 / self.test_analyzer.classes["snow_cover"][-1]
        valid_ref = dataset.data_vars["ref"].where(self.ref_analyzer.quantitative_mask(dataset.data_vars["ref"]))
        valid_ref = valid_ref * 100 / self.ref_analyzer.classes["snow_cover"][-1]
        dataset = dataset.assign(biais=valid_test - valid_ref)
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

    def uncertainty_analysis(
        self,
        test_time_series: xr.Dataset,
        ref_time_series: xr.Dataset,
        sensor_zenith_analysis: bool = True,
        forest_mask_path: str | None = None,
        sub_roi_mask_path: str | None = None,
        slope_map_path: str | None = None,
        aspect_map_path: str | None = None,
        dem_path: str | None = None,
        netcdf_export_path: str | None = None,
    ) -> xr.Dataset:
        combined_dataset, analysis_bin_dict = self.prepare_analysis(
            test_time_series=test_time_series,
            ref_time_series=ref_time_series,
            ref_fsc_step=99,
            sensor_zenith_analysis=sensor_zenith_analysis,
            forest_mask_path=forest_mask_path,
            sub_roi_mask_path=sub_roi_mask_path,
            slope_map_path=slope_map_path,
            aspect_map_path=aspect_map_path,
            dem_path=dem_path,
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
    forest_mask_path = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2006_forest_mask.tif"
    massifs_mask_path = None
    slope_map_path = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/SLP_MSF_UTM31_375m_lanczos.tif"
    aspect_map_path = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/ASP_MSF_UTM31_375m_lanczos.tif"
    dem_path = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/DEM_MSF_UTM31_375m_lanczos.tif"
    for product_to_evaluate in products_to_evaluate:
        working_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_4/"
        output_folder = f"{working_folder}/analyses/uncertainty"
        ref_time_series_name = f"WY_{year.from_year}_{year.to_year}_s2_theia_res_{resolution}m.nc"
        test_time_series_name = f"WY_{year.from_year}_{year.to_year}_{platform}_{product_to_evaluate}_res_{resolution}m.nc"
        output_filename = f"{output_folder}/uncertainty_WY_{year.from_year}_{year.to_year}_{platform}_{product_to_evaluate}_res_{resolution}m.nc"
        test_time_series = xr.open_dataset(f"{working_folder}/time_series/{test_time_series_name}").isel(time=slice(30, 180))
        ref_time_series = xr.open_dataset(f"{working_folder}/time_series/{ref_time_series_name}").isel(time=slice(30, 180))
        logger.info(f"Evaluating product {product_to_evaluate}")

        if product_to_evaluate == "nasa_l3":
            metrics_calcuator = UncertaintyNASA()
            metrics_calcuator.uncertainty_analysis(
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
            metrics_calcuator = UncertaintyNASA()
            metrics_calcuator.uncertainty_analysis(
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

        elif product_to_evaluate == "meteofrance_l3":
            metrics_calcuator = UncertaintyMeteoFrance()
            metrics_calcuator.uncertainty_analysis(
                test_time_series=test_time_series,
                ref_time_series=ref_time_series,
                sensor_zenith_analysis=False,
                forest_mask_path=forest_mask_path,
                sub_roi_mask_path=None,
                slope_map_path=slope_map_path,
                aspect_map_path=aspect_map_path,
                dem_path=dem_path,
                netcdf_export_path=output_filename,
            )

        else:
            raise NotImplementedError(f"Unknown product: {product_to_evaluate}")
