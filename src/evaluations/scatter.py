from copy import deepcopy
from typing import Dict

import numpy as np
import xarray as xr
from matplotlib import cm, colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
from xarray.groupers import BinGrouper

from evaluations.completeness import (
    MeteoFranceSnowCoverProductCompleteness,
    NASASnowCoverProductCompleteness,
    S2SnowCoverProductCompleteness,
)
from evaluations.statistics_base import EvaluationConfig, EvaluationVsHighResBase, generate_evaluation_io
from logger_setup import default_logger as logger
from winter_year import WinterYear


def fit_regression(data_to_fit: xr.DataArray):
    test_grid, ref_grid = np.meshgrid(data_to_fit.test_bins.values, data_to_fit.ref_bins.values)
    model_x_data = ref_grid.reshape((-1, 1))
    model_y_data = test_grid.reshape((-1, 1))
    weights = data_to_fit.values.ravel()
    regression = LinearRegression().fit(X=model_x_data, y=model_y_data, sample_weight=weights)
    return (
        regression.coef_[0],
        regression.intercept_,
        regression.score(model_x_data, model_y_data, data_to_fit.values.ravel()),
    )


def fancy_scatter_plot(data_to_plt: xr.DataArray, ax: Axes, figure: Figure, gaussian_window_size: int | None = 2):
    if gaussian_window_size is not None:
        data_smooth = gaussian_filter(data_to_plt, sigma=gaussian_window_size)
    else:
        data_smooth = data_to_plt
    distr_min, distr_max = np.quantile(data_smooth, 0.05), np.quantile(data_smooth, 0.95)
    coeff_slope, intercept, score = fit_regression(data_to_plt)
    scatter_plot = ax.pcolormesh(
        data_to_plt.ref_bins.values,
        data_to_plt.test_bins.values,
        data_smooth.T,
        norm=colors.LogNorm(vmin=distr_min if distr_min > 0 else 1, vmax=distr_max),
        cmap=cm.bone,
    )
    regression_x_axis = data_to_plt.ref_bins.values
    ax.plot(regression_x_axis, regression_x_axis * coeff_slope + intercept, color="r")
    ax.plot(regression_x_axis, regression_x_axis, color="k", linewidth=0.5)
    ax.grid(True)
    cbar_ticks = np.array([1e-1, 1, 1e1, 1e2, 1e3, 1e4])
    cbar = figure.colorbar(scatter_plot, ticks=cbar_ticks)
    cbar.ax.set_yticklabels([f"{tick:n}" for tick in cbar_ticks])
    ax.legend([f"Fitted R²={score:.2f} m={float(coeff_slope):.2f} b={float(intercept):.2f}", "y=x"])
    return scatter_plot


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
        n_intersecting_pixels = (quant_mask_test & quant_mask_ref).sum()
        if n_intersecting_pixels == 0:
            logger.info("No intersection found on this day. Returning a zeros array.")
            return xr.DataArray(0, coords=xr.Coordinates({k + "_bins": v.labels for k, v in bins_dict.items()}))

        dataset.data_vars["ref"][:] = dataset.data_vars["ref"].where(quant_mask_ref) * 100 / self.ref_analyzer.max_fsc
        dataset.data_vars["test"][:] = dataset.data_vars["test"].where(quant_mask_test) * 100 / self.test_analyzer.max_fsc

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


class ScatterMeteoFrance(Scatter):
    def __init__(self) -> None:
        super().__init__(
            reference_analyzer=S2SnowCoverProductCompleteness(),
            test_analyzer=MeteoFranceSnowCoverProductCompleteness(),
        )


class ScatterNASA(Scatter):
    def __init__(self) -> None:
        super().__init__(
            reference_analyzer=S2SnowCoverProductCompleteness(),
            test_analyzer=NASASnowCoverProductCompleteness(),
        )


class ScatterMeteoFranceVsNASA(Scatter):
    def __init__(self) -> None:
        super().__init__(
            reference_analyzer=NASASnowCoverProductCompleteness(),
            test_analyzer=MeteoFranceSnowCoverProductCompleteness(),
        )


if __name__ == "__main__":
    from products.plot_settings import METEOFRANCE_VAR_NAME, NASA_L3_VAR_NAME, NASA_PSEUDO_L3_VAR_NAME

    config_eval = EvaluationConfig(
        ref_fsc_step=1,
        sensor_zenith_analysis=False,
        forest_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2006_forest_mask.tif",
        slope_map_path=None,
        aspect_map_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/ASP_MSF_UTM31_375m_lanczos.tif",
        sub_roi_mask_path=None,
        dem_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/DEM_MSF_UTM31_375m_lanczos.tif",
    )

    config_fit = EvaluationConfig(
        ref_fsc_step=1,
        sensor_zenith_analysis=False,
        forest_mask_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2006_forest_mask.tif",
        slope_map_path=None,
        aspect_map_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/ASP_MSF_UTM31_375m_lanczos.tif",
        sub_roi_mask_path=None,
        dem_path="/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/DEM_MSF_UTM31_375m_lanczos.tif",
    )

    working_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_4/"

    evaluation_dict: Dict[str, Dict[str, Scatter]] = {
        # METEOFRANCE_VAR_NAME: {"evaluator": ScatterMeteoFrance(), "config": config_eval},
        NASA_PSEUDO_L3_VAR_NAME: {"evaluator": ScatterNASA(), "config": config_eval},
        NASA_L3_VAR_NAME: {"evaluator": ScatterNASA(), "config": config_eval},
    }

    # for product, evaluator in evaluation_dict.items():
    #     ref_time_series, test_time_series, output_filename = generate_evaluation_io(
    #         analysis_type="scatter",
    #         working_folder=working_folder,
    #         year=WinterYear(2023, 2024),
    #         resolution=375,
    #         platform="SNPP",
    #         ref_product_name="s2_theia",
    #         test_product_name=product,
    #         period=None,
    #     )
    #     logger.info(f"Evaluating product {product}")
    #     metrics_calcuator = evaluator["evaluator"]
    #     metrics_calcuator.scatter_analysis(
    #         test_time_series=test_time_series,
    #         ref_time_series=ref_time_series,
    #         config=evaluation_dict[product]["config"],
    #         netcdf_export_path=output_filename,
    #     )

    # Cross evaluation MF vs NASA
    resolution = 375
    wy = WinterYear(2023, 2024)
    ref_time_series_name = f"WY_{wy.from_year}_{wy.to_year}_SNPP_nasa_l3_res_{resolution}m.nc"
    test_time_series_name = f"WY_{wy.from_year}_{wy.to_year}_SNPP_meteofrance_l3_res_{resolution}m.nc"

    output_filename = (
        f"{working_folder}/analyses/scatter/scatter_WY_{wy.from_year}_{wy.to_year}_nasa_vs_meteofrance_l3_{resolution}m.nc"
    )

    test_time_series = xr.open_dataset(f"{working_folder}/time_series/{test_time_series_name}").data_vars[
        "snow_cover_fraction"
    ]
    ref_time_series = xr.open_dataset(f"{working_folder}/time_series/{ref_time_series_name}").data_vars["snow_cover_fraction"]
    logger.info("Evaluating products Météo-France vs NASA")
    metrics_calcuator = ScatterMeteoFranceVsNASA()
    metrics_calcuator.scatter_analysis(
        test_time_series=test_time_series,
        ref_time_series=ref_time_series,
        config=config_eval,
        netcdf_export_path=output_filename,
    )

    # # Fit NASA NDSI to S2 FSC
    # ref_time_series_name = f"WY_{wy.from_year}_{wy.to_year}_s2_theia_res_{resolution}m.nc"
    # test_time_series_name = f"WY_{wy.from_year}_{wy.to_year}_SNPP_nasa_l3_res_{resolution}m.nc"

    # output_filename = (
    #     f"{working_folder}/analyses/scatter/scatter_WY_{wy.from_year}_{wy.to_year}_nasa_NDSI_vs_s2_theia_FSC_{resolution}m.nc"
    # )

    # test_time_series = xr.open_dataset(f"{working_folder}/time_series/{test_time_series_name}").data_vars["NDSI_Snow_Cover"]
    # ref_time_series = xr.open_dataset(f"{working_folder}/time_series/{ref_time_series_name}").data_vars["snow_cover_fraction"]
    # logger.info("NASA NDSI vs S2 FSC")
    # metrics_calcuator = ScatterNASA()
    # metrics_calcuator.scatter_analysis(
    #     test_time_series=test_time_series,
    #     ref_time_series=ref_time_series,
    #     config=config_fit,
    #     netcdf_export_path=output_filename,
    # )
