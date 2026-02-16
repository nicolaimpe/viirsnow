from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from pandas.io.formats.style import Styler
from xarray.groupers import BinGrouper

from postprocess.general_purpose import AnalysisContainer, fancy_table, open_reduced_dataset_for_plot
from products.snow_cover_product import SnowCoverProduct


def histograms_to_biais_rmse(metrics_dataset: xr.Dataset) -> xr.Dataset:
    # From a histogram of differences (bias), compute uncertainty figures
    tot_occurrences = metrics_dataset.data_vars["n_occurrences"].sum()
    n_occur = metrics_dataset.data_vars["n_occurrences"]
    residual_bins = metrics_dataset.coords["biais_bins"]
    bias = (n_occur * residual_bins).sum() / tot_occurrences
    rmse = np.sqrt((n_occur * residual_bins**2).sum() / tot_occurrences)
    unbiaised_rmse = np.sqrt((n_occur * (residual_bins - bias) ** 2).sum() / tot_occurrences)
    return xr.Dataset({"bias": bias, "rmse": rmse, "unbiased_rmse": unbiaised_rmse})


def postprocess_uncertainty_analysis(
    snow_cover_products: List[SnowCoverProduct],
    metrics_datasets: List[xr.Dataset],
    analysis_var: str | BinGrouper,
) -> xr.Dataset:
    reduced_datasets = []
    for product, metrics in zip(snow_cover_products, metrics_datasets):
        reduced_datasets.append(metrics.groupby(analysis_var).map(histograms_to_biais_rmse))
    concatenated = xr.concat(
        reduced_datasets, pd.Index([product.name for product in snow_cover_products], name="product"), coords="minimal"
    )
    return concatenated


def histograms_to_distribution(metrics_ds) -> npt.NDArray:
    all_dims = list(metrics_ds.sizes.keys())
    all_dims.remove("biais_bins")
    metrics_squeezed = metrics_ds.sum(dim=all_dims)
    distribution = np.repeat(metrics_ds.coords["biais_bins"].values, metrics_squeezed["n_occurrences"].values.astype(np.int64))
    return distribution


def fancy_table_tot(dataframe_to_print: pd.DataFrame) -> Styler:
    # Get the built-in RdYlGn colormap
    base_cmap = plt.get_cmap("RdYlGn")

    # Sample the three anchor colors from RdYlGn:
    red = base_cmap(0.0)  # left end
    yellow = base_cmap(0.5)  # center (yellow in RdYlGn)
    green = base_cmap(1.0)  # right end
    colors_biais = [red, yellow, green, yellow, red]

    cmap_biais = LinearSegmentedColormap.from_list("green_center", colors_biais, N=256)
    color_maps = {
        "Accuracy": "RdYlGn",
        "F1-score": "RdYlGn",
        "Commission Error": "RdYlGn_r",  # Lower is better
        "Omission Error": "RdYlGn_r",
        "Bias [%]": cmap_biais,
        "RMSE [%]": "RdYlGn_r",
    }
    vmins = {
        "Accuracy": 0.6,  # Higher is better
        "F1-score": 0.6,
        "Commission Error": 0,  # Lower is better (reversed Reds)
        "Omission Error": 0,
        "Bias [%]": -5,
        "RMSE [%]": 5,
    }
    vmaxs = {
        "Accuracy": 1,  # Higher is better
        "F1-score": 1,
        "Commission Error": 0.3,  # Lower is better
        "Omission Error": 0.3,
        "Bias [%]": 5,
        "RMSE [%]": 30,
    }

    # Build the colormap

    # Apply gradient coloring
    return fancy_table(dataframe_to_print=dataframe_to_print, color_maps=color_maps, vmins=vmins, vmaxs=vmaxs)


def scatter_to_biais(dataset: xr.Dataset) -> xr.DataArray:
    data_array = dataset.data_vars["n_occurrences"].sum(dim=("forest_mask_bins", "sub_roi_bins", "aspect_bins", "time"))
    biais_bins = np.arange(-100, 101)

    occurrences_per_biais_bins = np.zeros_like(biais_bins)
    for i, b in enumerate(biais_bins):
        occurrences_per_biais_bins[i] = np.trace(data_array.values, offset=b + data_array.ref_bins.values[0])
    out_data_array = xr.DataArray(data=occurrences_per_biais_bins, coords={"biais_bins": biais_bins})
    return xr.Dataset({"n_occurrences": out_data_array})


def plot_error_bars(analysis: AnalysisContainer, analysis_var: str, ax: plt.Axes):
    percentile_min, percentile_max = 5, 95
    sample_dataset = open_reduced_dataset_for_plot(
        product=analysis.products[0],
        analysis_folder=analysis.analysis_folder,
        analysis_type="uncertainty",
        winter_year=analysis.winter_year,
        grid=analysis.grid,
    )

    x_positions = np.arange(len(sample_dataset.coords[analysis_var].values))
    x_positions = x_positions / len(x_positions)
    for product in analysis.products:
        metrics_dataset = open_reduced_dataset_for_plot(
            product=product,
            analysis_folder=analysis.analysis_folder,
            analysis_type="uncertainty",
            winter_year=analysis.winter_year,
            grid=analysis.grid,
        )
        analysis_coords = metrics_dataset.coords[analysis_var].values
        box_width_data = 0.2 / len(x_positions)
        for idx, value in enumerate(analysis_coords):
            x_pos = x_positions[idx]
            product_analysis_var_dataset = metrics_dataset.sel({analysis_var: value})  # .drop_sel(biais_bins=0)
            reduced = product_analysis_var_dataset.groupby("biais_bins").sum(list(product_analysis_var_dataset.sizes.keys()))
            biais_rmse = histograms_to_biais_rmse(reduced)
            distr = histograms_to_distribution(reduced)
            ax.scatter(x_pos, biais_rmse.data_vars["bias"], marker="o", color=product.plot_color, s=35, zorder=3)
            whiskers_min = np.percentile(distr, percentile_min)
            whiskers_max = np.percentile(distr, percentile_max)
            ax.vlines(
                x_pos, whiskers_min, whiskers_max, color=product.plot_color, linestyle="-", lw=3, label=product.plot_name
            )

            ax.hlines(whiskers_min, x_pos - box_width_data / 2, x_pos + box_width_data / 2, color=product.plot_color, lw=3)
            ax.hlines(whiskers_max, x_pos - box_width_data / 2, x_pos + box_width_data / 2, color=product.plot_color, lw=3)
        x_positions = x_positions + box_width_data

    ax.set_xticks(x_positions - box_width_data * ((len(analysis.products) + 1) // 2), labels=analysis_coords)
    # ax.set_xlim(x_positions[0] - (len(analysis.products) + 1) * box_width_data, x_positions[-1])
    ax.set_xlim(
        -1 / ((len(analysis.products) + 1) * metrics_dataset.sizes[analysis_var]),
        1 - 1 / ((len(analysis.products) + 1) * metrics_dataset.sizes[analysis_var]),
    )
    ax.set_ylim(-70, 70)
    ax.set_ylabel("Residuals [% FSC]")
    # ax.set_xlabel(analysis_var)
    (l1,) = ax.plot([0, 1], [0, 1], c="gray", lw=1e-12)
    (l2,) = ax.plot(0, 0, c="gray", markersize=1e-12)
    ax.legend(
        [l1, l2],
        [f"P{percentile_min} and P{percentile_max}", "bias"],
        handler_map={l1: HandlerSpan(), l2: HandlerPoint()},
    )
    ax.grid(axis="y")


class HandlerSpan(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        a_list = []

        a_list.append(
            Line2D(np.array([0.5, 0.5]) * width - xdescent, np.array([-0.5, 1.5]) * height - ydescent)
        )  # top vert line

        a_list.append(
            Line2D(np.array([0.38, 0.62]) * width - xdescent, np.array([1.5, 1.5]) * height - ydescent)
        )  # top whisker

        a_list.append(
            Line2D(np.array([0.38, 0.62]) * width - xdescent, np.array([-0.5, -0.5]) * height - ydescent)
        )  # bottom whisker

        # a_list.append(matplotlib.lines.Line2D(np.array([0,1])*width-xdescent,
        #                                       np.array([0.5,0.5])*height-ydescent, lw=2)) # median
        for a in a_list:
            a.set_color(orig_handle.get_color())
        return a_list


class HandlerPoint(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        a_list = []

        a_list.append(
            Line2D(
                np.array([0.5]) * width - xdescent,
                np.array([0.5]) * height - ydescent,
                color="gray",
                marker=".",
                linestyle="None",
            )
        )  # mean point
        for a in a_list:
            a.set_color(orig_handle.get_color())
        return a_list


def line_plot_rmse(analysis: AnalysisContainer, analysis_var: str, ax: Axes):
    metrics_datasets = [
        open_reduced_dataset_for_plot(
            product=prod,
            analysis_folder=analysis.analysis_folder,
            analysis_type="uncertainty",
            winter_year=analysis.winter_year,
            grid=analysis.grid,
        )
        for prod in analysis.products
    ]
    biais_rmse = postprocess_uncertainty_analysis(
        analysis.products, metrics_datasets=metrics_datasets, analysis_var=analysis_var
    )

    x_coords_unc = biais_rmse.coords[analysis_var].values

    for prod in analysis.products:
        ax.plot(
            x_coords_unc, biais_rmse.data_vars["rmse"].sel(product=prod.name), "-o", color=prod.plot_color, markersize=7, lw=3
        )

    ax.grid(True)
    ax.set_ylim(5, 30)
    ax.set_xlim(-0.5, biais_rmse.sizes[analysis_var] - 0.5)
    ax.set_ylabel("RMSE [% FSC]")
    ax.legend([Line2D([0], [0], linestyle="-", color="gray")], ["RMSE"])


def compute_uncertainty_results_df(snow_cover_products: List[SnowCoverProduct], metric_datasets: List[xr.Dataset]):
    reduced_datasets = []
    for dataset in metric_datasets:
        reduced_datasets.append(histograms_to_biais_rmse(dataset.groupby("time.month").sum()))
    concatenated = xr.concat(
        reduced_datasets, pd.Index([prod.plot_name for prod in snow_cover_products], name="product"), coords="minimal"
    )
    reduced_df = concatenated.to_dataframe().reset_index("product")
    return reduced_df
