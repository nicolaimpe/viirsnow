from typing import Dict

import numpy as np
import xarray as xr
from matplotlib import cm, colors
from matplotlib import pyplot as plt
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
from evaluations.statistics_base import EvaluationVsHighResBase
from logger_setup import default_logger as logger


def fit_regression(data_to_fit: xr.DataArray):
    test_grid, ref_grid = np.meshgrid(data_to_fit.ref_bins.values, data_to_fit.test_bins.values)
    regression = LinearRegression().fit(ref_grid.reshape((-1, 1)), test_grid.reshape((-1, 1)), data_to_fit.values.ravel())
    return (
        regression.coef_[0],
        regression.intercept_,
        regression.score(ref_grid.reshape((-1, 1)), test_grid.reshape((-1, 1)), data_to_fit.values.ravel()),
    )


def fancy_scatter_plot(data_to_plt: xr.DataArray, ax: Axes, figure: Figure, s=2):
    data_smooth = gaussian_filter(data_to_plt, sigma=s)
    distr_min, distr_max = np.quantile(data_smooth, 0.02), np.quantile(data_smooth, 0.98)
    a, b, score = fit_regression(data_to_plt)
    scatter_plot = ax.pcolormesh(
        data_to_plt.ref_bins.values,
        data_to_plt.test_bins.values,
        data_smooth.T,
        norm=colors.LogNorm(vmax=distr_max),
        shading="gouraud",
        cmap=cm.bone,
    )
    regression_x_axis = data_to_plt.ref_bins.values
    ax.plot(regression_x_axis, regression_x_axis * a + b, color="r")
    ax.plot(regression_x_axis, regression_x_axis, color="k", linewidth=0.5)
    ax.grid(True)
    cbar_ticks = np.array([1e-1, 1, 1e1, 1e2, 1e3, 1e4])
    cbar = figure.colorbar(scatter_plot, ticks=cbar_ticks)
    cbar.ax.set_yticklabels([f"{tick:n}" for tick in cbar_ticks])
    ax.legend([f"Score r={score:.2f}"])


class Scatter(EvaluationVsHighResBase):
    @staticmethod
    def test_bins():
        return BinGrouper(np.array([*np.arange(-1, 101, 1), 255]), labels=np.array([*np.arange(0, 101, 1), 255]))

    def time_step_analysis(self, dataset: xr.Dataset, bins_dict: Dict[str, xr.groupers.Grouper]):
        logger.info(f"Processing time of the year {dataset.coords['time'].values[0].astype('M8[D]').astype('O')}")
        dataset.data_vars["test"][:] = dataset.data_vars["test"] * 100 / self.test_analyzer.classes["snow_cover"][-1]
        dataset.data_vars["ref"][:] = dataset.data_vars["ref"] * 100 / self.ref_analyzer.classes["snow_cover"][-1]
        bins_dict.update(test=self.test_bins())
        scatter = dataset.groupby(bins_dict).map(self.compute_scatter_plot)
        return scatter

    def compute_scatter_plot(self, dataset: xr.Dataset):
        return dataset.data_vars["ref"].count().rename("n_occurrences")

    def scatter_analysis(
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
            ref_fsc_step=1,
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


from products.plot_settings import METEOFRANCE_VAR_NAME, NASA_L3_VAR_NAME, NASA_PSEUDO_L3_VAR_NAME

analysis_type = "scatter"
analysis_folder = f"/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_3/analyses/{analysis_type}"
mf_metrics_ds = xr.open_dataset(
    f"{analysis_folder}/{analysis_type}_WY_2023_2024_SNPP_meteofrance_l3_res_375m.nc", decode_cf=True
)
nasa_l3_metrics_ds = xr.open_dataset(
    f"{analysis_folder}/{analysis_type}_WY_2023_2024_SNPP_nasa_l3_res_375m.nc", decode_cf=True
)
nasa_pseudo_l3_metrics_ds = xr.open_dataset(
    f"{analysis_folder}/{analysis_type}_WY_2023_2024_SNPP_nasa_pseudo_l3_res_375m_short.nc", decode_cf=True
)
analyses_dict = {
    METEOFRANCE_VAR_NAME: mf_metrics_ds,
    NASA_PSEUDO_L3_VAR_NAME: nasa_pseudo_l3_metrics_ds,
    NASA_L3_VAR_NAME: nasa_l3_metrics_ds,
}

fig, ax = plt.subplots()
reduced_alps = (
    nasa_l3_metrics_ds.sel(ref_bins=slice(1, 99), forest_mask=[0], aspect_bins=np.array(["S"]), test_bins=slice(1, 99))
    .sum(dim=("forest_mask", "sub_roi", "time", "aspect_bins"))
    .data_vars["n_occurrences"]
)

# x = np.arange(1,100)
# y=x
# yy, xx = np.meshgrid(x,y)
# fake_data_array = xr.DataArray((xx==yy)*4 + (yy==2)*3, coords={'ref_bins': x, 'test_bins': y}).sel(ref_bins=slice(1,70),test_bins=slice(1,70))

fancy_scatter_plot(reduced_alps, ax, fig)
ax.set_xlabel("REF")
ax.set_ylabel("TEST")
plt.show()
# if __name__ == "__main__":
#     platform = "SNPP"
#     year = WinterYear(2023, 2024)
#     products_to_evaluate = ["meteofrance_l3", "nasa_l3", "nasa_pseudo_l3"]
#     resolution = 375
#     forest_mask_path = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2006_forest_mask.tif"
#     massifs_mask_path = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/MSF_MACRO_FRANCE_UTM31_375m.tif"
#     slope_map_path = None
#     aspect_map_path = "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/dem/ASP_MSF_UTM31_375m_lanczos.tif"
#     dem_path = None
#     for product_to_evaluate in products_to_evaluate:
#         working_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_3/"
#         output_folder = f"{working_folder}/analyses/scatter"
#         ref_time_series_name = f"WY_{year.from_year}_{year.to_year}_S2_res_{resolution}m.nc"
#         test_time_series_name = f"WY_{year.from_year}_{year.to_year}_{platform}_{product_to_evaluate}_res_{resolution}m.nc"
#         output_filename = (
#             f"{output_folder}/scatter_WY_{year.from_year}_{year.to_year}_{platform}_{product_to_evaluate}_res_{resolution}m.nc"
#         )
#         test_time_series = xr.open_dataset(f"{working_folder}/{test_time_series_name}").isel(time=slice(60, 180))
#         ref_time_series = xr.open_dataset(f"{working_folder}/{ref_time_series_name}").isel(time=slice(60, 180))
#         logger.info(f"Evaluating product {product_to_evaluate}")

#         if product_to_evaluate == "nasa_l3":
#             metrics_calcuator = ScatterNASA()
#             metrics_calcuator.scatter_analysis(
#                 test_time_series=test_time_series,
#                 ref_time_series=ref_time_series,
#                 sensor_zenith_analysis=False,
#                 forest_mask_path=forest_mask_path,
#                 sub_roi_mask_path=massifs_mask_path,
#                 slope_map_path=slope_map_path,
#                 aspect_map_path=aspect_map_path,
#                 dem_path=dem_path,
#                 netcdf_export_path=output_filename,
#             )

#         elif product_to_evaluate == "nasa_pseudo_l3":
#             metrics_calcuator = ScatterNASA()
#             metrics_calcuator.scatter_analysis(
#                 test_time_series=test_time_series,
#                 ref_time_series=ref_time_series,
#                 sensor_zenith_analysis=False,
#                 forest_mask_path=forest_mask_path,
#                 sub_roi_mask_path=massifs_mask_path,
#                 slope_map_path=slope_map_path,
#                 aspect_map_path=aspect_map_path,
#                 dem_path=dem_path,
#                 netcdf_export_path=output_filename,
#             )

#         elif product_to_evaluate == "meteofrance_l3":
#             metrics_calcuator = ScatterMeteoFrance()
#             metrics_calcuator.scatter_analysis(
#                 test_time_series=test_time_series,
#                 ref_time_series=ref_time_series,
#                 sensor_zenith_analysis=False,
#                 forest_mask_path=forest_mask_path,
#                 sub_roi_mask_path=massifs_mask_path,
#                 slope_map_path=slope_map_path,
#                 aspect_map_path=aspect_map_path,
#                 dem_path=dem_path,
#                 netcdf_export_path=output_filename,
#             )

#         else:
#             raise NotImplementedError(f"Unknown product: {product_to_evaluate}")
