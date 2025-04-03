import numpy as np
import xarray as xr
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression

from products.plot_settings import PRODUCT_PLOT_NAMES


def fit_regression(data_to_fit: xr.DataArray):
    xx, yy = np.meshgrid(data_to_fit.coords["x"].values, data_to_fit.coords["y"].values)
    model_x_data = xx.reshape((-1, 1))
    model_y_data = yy.reshape((-1, 1))
    weights = data_to_fit.values.ravel()
    regression = LinearRegression().fit(X=model_x_data, y=model_y_data, sample_weight=weights)
    return (
        regression.coef_[0][0],
        regression.intercept_[0],
        regression.score(model_x_data, model_y_data, data_to_fit.values.ravel()),
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
    distr_min, distr_max = np.quantile(data_smooth, 0.05), np.quantile(data_smooth, 0.95)
    coeff_slope, intercept, score = fit_regression(data_to_plt)
    scatter_plot = ax.pcolormesh(
        data_to_plt.coords["x"].values,
        data_to_plt.coords["y"].values,
        data_smooth,
        norm=colors.LogNorm(vmin=distr_min if distr_min > 0 else 1, vmax=distr_max),
        cmap=cm.bone,
    )
    regression_x_axis = np.arange(0, 100)
    ax.plot(
        regression_x_axis,
        regression_x_axis * coeff_slope + intercept,
        "--",
        color="gray",
        label=f"Fitted R²={score:.2f} m={float(coeff_slope):.2f} b={float(intercept):.2f}",
    )
    ax.plot(regression_x_axis, regression_x_axis, color="k", linewidth=0.5, label="y=x")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 100)
    cbar_ticks = np.array([1e-1, 1, 1e1, 1e2, 1e3, 1e4])
    cbar = figure.colorbar(scatter_plot, ticks=cbar_ticks)
    cbar.ax.set_yticklabels([f"{tick:n}" for tick in cbar_ticks])
    return scatter_plot


if __name__ == "__main__":
    import xarray as xr

    from postprocess.general_purpose import open_analysis_file, sel_evaluation_domain
    from products.plot_settings import METEOFRANCE_VAR_NAME, NASA_L3_VAR_NAME, NASA_PSEUDO_L3_VAR_NAME
    from winter_year import WinterYear

    wy = WinterYear(2023, 2024)
    analysis_type = "scatter"
    analysis_folder = f"/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_4/analyses/{analysis_type}"
    analysis_type = "scatter"
    analysis_folder = f"/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_4/analyses/{analysis_type}"

    # Errors in the data correction
    mf_metrics_ds = xr.open_dataset(
        f"{analysis_folder}/{analysis_type}_WY_2023_2024_SNPP_meteofrance_l3_res_375m.nc", decode_cf=True
    )
    mf_metrics_ds = mf_metrics_ds.assign_coords({"forest_mask": ["no_forest", "forest"]})
    nasa_l3_metrics_ds = xr.open_dataset(
        f"{analysis_folder}/{analysis_type}_WY_2023_2024_SNPP_nasa_l3_res_375m.nc", decode_cf=True
    )
    nasa_l3_metrics_ds = nasa_l3_metrics_ds.where(nasa_l3_metrics_ds > 0, drop=True)
    nasa_l3_metrics_ds = nasa_l3_metrics_ds.assign_coords({"forest_mask_bins": ["no_forest", "forest"]}).rename(
        {"forest_mask_bins": "forest_mask"}
    )
    nasa_pseudo_l3_metrics_ds = xr.open_dataset(
        f"{analysis_folder}/{analysis_type}_WY_2023_2024_SNPP_nasa_pseudo_l3_res_375m.nc", decode_cf=True
    )
    nasa_pseudo_l3_metrics_ds = nasa_pseudo_l3_metrics_ds.where(nasa_pseudo_l3_metrics_ds > 0, drop=True)
    nasa_pseudo_l3_metrics_ds = nasa_pseudo_l3_metrics_ds.assign_coords({"forest_mask_bins": ["no_forest", "forest"]}).rename(
        {"forest_mask_bins": "forest_mask"}
    )
    #############

    analyses_dict = {
        METEOFRANCE_VAR_NAME: mf_metrics_ds,
        NASA_PSEUDO_L3_VAR_NAME: nasa_pseudo_l3_metrics_ds,
        NASA_L3_VAR_NAME: nasa_l3_metrics_ds,
    }
    evaluation_domain = "ablation"
    selection_dict, title = sel_evaluation_domain(analyses_dict=analyses_dict, evaluation_domain=evaluation_domain)

    ####################### Launch analysis
    fig, ax = plt.subplots(1, len(selection_dict), figsize=(6 * len(selection_dict), 5))
    n_min = 10
    fig.suptitle(f"Scatter analysis {title} - thresh N_min = {n_min}")
    for i, (k, v) in enumerate(selection_dict.items()):
        reduced_v = (
            v.sel(ref_bins=slice(10, 95), forest_mask=["no_forest"], test_bins=slice(1, 95))
            .sum(dim=("forest_mask", "time", "aspect_bins", "altitude_bins"))
            .data_vars["n_occurrences"]
        )
        scatter_plot = fancy_scatter_plot(
            data_to_plt=reduced_v.rename({"ref_bins": "x", "test_bins": "y"}),
            ax=ax[i],
            figure=fig,
            low_threshold=n_min,
            smoothing_window_size=0,
        )
        ax[i].set_title(PRODUCT_PLOT_NAMES[k])
        ax[i].set_xlabel("S2 FSC [%]")
        ax[i].set_ylabel(f"{PRODUCT_PLOT_NAMES[k]} FSC [%]")

    plt.show()
