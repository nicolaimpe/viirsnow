import numpy as np
import xarray as xr
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression


def fit_regression(data_to_fit: xr.DataArray):
    xx, yy = np.meshgrid(data_to_fit.coords["x"].values, data_to_fit.coords["y"].values)
    model_x_data = xx.reshape((-1, 1))
    model_y_data = yy.reshape((-1, 1))
    weights = data_to_fit.values.ravel()
    regression = LinearRegression().fit(X=model_x_data, y=model_y_data, sample_weight=weights)
    return (
        regression.coef_[0][0],
        regression.intercept_[0],
        regression.score(model_x_data, model_y_data, weights),
    )


def fancy_scatter_plot(
    data_to_plt: xr.DataArray,
    ax: Axes,
    figure: Figure,
    low_threshold: int | None = None,
    smoothing_window_size: int | None = 2,
):
    data_to_plt = data_to_plt.transpose("y", "x")
    if low_threshold is not None:
        data_to_plt = data_to_plt.where(data_to_plt >= low_threshold, 0)
    if smoothing_window_size is not None:
        data_smooth = gaussian_filter(data_to_plt, sigma=smoothing_window_size)
    else:
        # That's ugly
        data_smooth = data_to_plt

    coeff_slope, intercept, score = fit_regression(data_to_plt)
    distr_min, distr_max = np.quantile(data_smooth, 0.20), np.quantile(data_smooth, 0.90)
    scatter_plot = ax.pcolormesh(
        data_to_plt.coords["x"].values,
        data_to_plt.coords["y"].values,
        data_smooth,
        norm=colors.LogNorm(vmin=distr_min if distr_min > 0 else 1, vmax=distr_max, clip=True),
        cmap=cm.bone,
    )
    regression_x_axis = np.arange(0, 100)
    ax.plot(
        regression_x_axis,
        regression_x_axis * coeff_slope + intercept,
        ":",
        lw=1.5,
        color="chocolate",
        label=f"Linear fit R²={score:.2f},slope={float(coeff_slope):.2f},intercept={float(intercept):.2f}",
    )
    # ax.plot(regression_x_axis, regression_x_axis, color="k", linewidth=0.5, label="y=x")
    ax.grid(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), draggable=True, fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 100)

    cbar = figure.colorbar(scatter_plot, extend="max")
    cbar_ticks = np.array([1e1, 1e2])
    cbar_labels = [f"{tick:n}" for tick in cbar_ticks]
    cbar.set_ticks(cbar_ticks, labels=cbar_labels)
    return scatter_plot

    # if __name__ == "__main__":
    #     wy = WinterYear(2023, 2024)
    #     analysis_type = "scatter"
    #     # analysis_folder = (
    #     #     f"/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_5_complete/analyses/{analysis_type}"
    #     # )

    #     # # Errors in the data correction

    #     analysis_folder = (
    #         f"/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_6_modis/analyses/{analysis_type}"
    #     )
    #     # mf_synopsis_metrics_ds = xr.open_dataset(
    #     #     f"{analysis_folder}/{analysis_type}_WY_2023_2024_meteofrance_synopsis_vs_s2_theia.nc",
    #     #     decode_cf=True,
    #     # )
    #     # mf_orig_metrics_ds = xr.open_dataset(
    #     #     f"{analysis_folder}/{analysis_type}_WY_2023_2024_meteofrance_orig_vs_s2_theia.nc",
    #     #     decode_cf=True,
    #     # )
    #     # mf_no_forest_metrics_ds = xr.open_dataset(
    #     #     f"{analysis_folder}/{analysis_type}_WY_2023_2024_meteofrance_no_forest_vs_s2_theia.nc",
    #     #     decode_cf=True,
    #     # )
    #     # mf_no_forest_red_band_screen_metrics_ds = xr.open_dataset(
    #     #     f"{analysis_folder}/{analysis_type}_WY_2023_2024_meteofrance_no_forest_red_band_screen_vs_s2_theia.nc",
    #     #     decode_cf=True,
    #     # )

    #     nasa_l3_snpp_metrics_ds = xr.open_dataset(
    #         f"{analysis_folder}/{analysis_type}_WY_2023_2024_nasa_l3_snpp_vs_s2_theia.nc", decode_cf=True
    #     )
    #     nasa_l3_snpp_metrics_ds = nasa_l3_snpp_metrics_ds.where(nasa_l3_snpp_metrics_ds > 0, drop=True)

    #     # nasa_l3_multiplatform_metrics_ds = xr.open_dataset(
    #     #     f"{analysis_folder}/{analysis_type}_WY_2023_2024_nasa_l3_multiplatform_vs_s2_theia.nc", decode_cf=True
    #     # )
    #     # nasa_l3_multiplatform_metrics_ds = nasa_l3_multiplatform_metrics_ds.where(nasa_l3_multiplatform_metrics_ds > 0, drop=True)

    #     # nasa_l3_jpss1_metrics_ds = xr.open_dataset(
    #     #     f"{analysis_folder}/{analysis_type}_WY_2023_2024_nasa_l3_jpss1_vs_s2_theia.nc", decode_cf=True
    #     # )
    #     # nasa_l3_jpss1_metrics_ds = nasa_l3_jpss1_metrics_ds.where(nasa_l3_jpss1_metrics_ds > 0, drop=True)

    #     # nasa_pseudo_l3_metrics_ds = xr.open_dataset(
    #     #     f"{analysis_folder}/{analysis_type}_WY_2023_2024_nasa_pseudo_l3_vs_s2_theia.nc", decode_cf=True
    #     # )
    #     # nasa_pseudo_l3_metrics_ds = nasa_pseudo_l3_metrics_ds.where(nasa_pseudo_l3_metrics_ds > 0, drop=True)

    #     # meteofrance_ndsi = xr.open_dataset(
    #     #     f"{analysis_folder}/{analysis_type}_WY_2023_2024_meteofrance_ndsi_no_forest_vs_s2_theia.nc",
    #     #     decode_cf=True,
    #     # )

    #     nasa_l3_terra_metrics_ds = xr.open_dataset(
    #         f"{analysis_folder}/{analysis_type}_WY_2023_2024_nasa_l3_terra_vs_s2_theia.nc", decode_cf=True
    #     )
    #     nasa_l3_terra_metrics_ds = nasa_l3_terra_metrics_ds.where(nasa_l3_terra_metrics_ds > 0, drop=True)
    #     #############

    #     analyses_dict = {
    #         # METEOFRANCE_VAR_NAME: mf_archive_metrics_ds,
    #         # MF_ORIG_VAR_NAME: mf_orig_metrics_ds,
    #         # MF_SYNOPSIS_VAR_NAME: mf_synopsis_metrics_ds,
    #         # NASA_PSEUDO_L3_VAR_NAME: nasa_pseudo_l3_metrics_ds,
    #         # MF_NO_FOREST_VAR_NAME: mf_no_forest_metrics_ds,
    #         # MF_NO_FOREST_RED_BAND_SCREEEN_VAR_NAME: mf_no_forest_red_band_screen_metrics_ds,
    #         NASA_L3_SNPP_VAR_NAME: nasa_l3_snpp_metrics_ds,
    #         # NASA_L3_MULTIPLATFORM_VAR_NAME: nasa_l3_multiplatform_metrics_ds,
    #         # NASA_L3_JPSS1_VAR_NAME: nasa_l3_jpss1_metrics_ds,
    #         # MF_SYNOPSIS_VAR_NAME: meteofrance_ndsi,
    #         NASA_L3_MODIS_TERRA_VAR_NAME: nasa_l3_terra_metrics_ds,
    #     }

    #     # title = "accumulation"
    #     selection_dict = {k: v.sel(time=slice("2023-11", "2024-06"), drop=True) for k, v in analyses_dict.items()}

    #     ####################### Launch analysis
    #     #### FSC corelation
    #     fig, ax = plt.subplots(1, len(selection_dict), figsize=(6 * len(selection_dict), 5))
    #     n_min = 10
    #     fig.suptitle("Scatter analysis")
    #     for i, (k, v) in enumerate(selection_dict.items()):
    #         reduced_v = (
    #             v.sel(ref_bins=slice(0, 95), forest_mask_bins=["forest"], test_bins=slice(0, 95))
    #             .sum(dim=("forest_mask_bins", "time", "aspect_bins"))
    #             .data_vars["n_occurrences"]
    #         )
    #         scatter_plot = fancy_scatter_plot(
    #             data_to_plt=reduced_v.rename({"ref_bins": "x", "test_bins": "y"}),
    #             ax=ax[i],
    #             figure=fig,
    #             low_threshold=n_min,
    #             smoothing_window_size=0,
    #         )
    #         ax[i].set_title(PRODUCT_PLOT_NAMES[k])
    #         ax[i].set_xlabel("S2 FSC [%]")
    #         ax[i].set_ylabel(f"{PRODUCT_PLOT_NAMES[k]} FSC [%]")

    #### NDSI-FSC regression
    # fig, ax = plt.subplots(1,
    #                        len(selection_dict), figsize=(6 * len(selection_dict), 5))
    # n_min = 20
    # fig.suptitle(f"Scatter analysis - no forest - thresh N_min = {n_min}")
    # for i, (k, v) in enumerate(selection_dict.items()):
    #     reduced_v = (
    #         v.sel(
    #             ref_bins=slice(0, 95),
    #             forest_mask_bins=["no_forest"],
    #             test_bins=slice(0, 95),
    #         )
    #         .sum(dim=("forest_mask_bins", "time", "aspect_bins", "sub_roi_bins"))
    #         .data_vars["n_occurrences"]
    #     )
    #     xax = reduced_v.test_bins.values
    #     fit_g = gascoin(xax * 0.01, f_veg=0) * 100
    #     # ax[i].plot(xax, fit_g, "g--", linewidth=1, label="gascoin at al. fveg=0")
    #     ax[i].plot(xax, salomonson_appel(xax), "k", linewidth=1, label="salomonson_appel")
    #     scatter_plot = fancy_scatter_plot(
    #         data_to_plt=reduced_v.rename({"ref_bins": "y", "test_bins": "x"}),
    #         ax=ax[i],
    #         figure=fig,
    #         low_threshold=n_min,
    #         smoothing_window_size=0,
    #     )
    #     ax[i].set_title(PRODUCT_PLOT_NAMES[k])
    #     ax[i].set_ylabel("S2 FSC [%]")
    #     ax[i].set_xlabel(f"{PRODUCT_PLOT_NAMES[k]} NDSI [%]")

    plt.show()
