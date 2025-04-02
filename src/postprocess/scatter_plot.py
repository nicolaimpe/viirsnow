import numpy as np
import xarray as xr
from matplotlib import cm, colors
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
