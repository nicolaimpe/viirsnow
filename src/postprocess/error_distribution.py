from typing import Dict, List

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from pandas.io.formats.style import Styler
from xarray.groupers import BinGrouper

from postprocess.general_purpose import fancy_table, open_reduced_dataset, open_reduced_dataset_for_plot, sel_evaluation_domain
from products.snow_cover_product import SnowCoverProduct
from reductions.statistics_base import EvaluationVsHighResBase
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


def barplots(snow_cover_products: List[SnowCoverProduct], postprocessed_data_array: xr.DataArray, title: str, ax: Axes):
    plot_dataframe_dict = {}
    colors = []
    legend = []
    for product in snow_cover_products:
        plot_dataframe_dict.update({product: postprocessed_data_array.sel(product=product.name).to_pandas()})
        colors.append(product.plot_color)
        legend.append(product.plot_name)
    plot_dataframe = pd.DataFrame(plot_dataframe_dict)
    if "month" in title:
        plot_dataframe.index = plot_dataframe.index.strftime("%B")
    if "aspect" in title:
        # That's because to_pandas() reorders the aspect labels alphabetically
        plot_dataframe = plot_dataframe.reindex(index=EvaluationVsHighResBase.aspect_bins().labels)

    plot_dataframe.plot.bar(figsize=(12, 3), color=colors, width=0.6, title=title, ax=ax)
    ax.legend(legend)


def biais_barplots(
    snow_cover_products: List[SnowCoverProduct],
    postprocessed_dataset: xr.Dataset,
    analysis_var_plot_name: str,
    title_complement: str,
):
    _, ax = plt.subplots()
    ax.set_ylim(-10, 10)
    ax.set_ylabel("biais [%]")
    ax.set_xlabel(analysis_var_plot_name)
    barplots(
        snow_cover_products=snow_cover_products,
        postprocessed_data_array=postprocessed_dataset.data_vars["biais"],
        title=f"Biais vs {analysis_var_plot_name} - {title_complement}",
        ax=ax,
    )
    ax.grid(True, axis="y")


def rmse_barplots(
    snow_cover_products: List[SnowCoverProduct],
    postprocessed_dataset: xr.Dataset,
    analysis_var_plot_name: str,
    title_complement: str,
):
    _, ax = plt.subplots()
    ax.set_ylim(0, 30)
    ax.set_ylabel("rmse [%]")
    ax.set_xlabel(analysis_var_plot_name)
    barplots(
        snow_cover_products=snow_cover_products,
        postprocessed_data_array=postprocessed_dataset.data_vars["rmse"],
        title=f"RMSE vs {analysis_var_plot_name} - {title_complement}",
        ax=ax,
    )
    ax.grid(True, axis="y")
    ax.set_ylabel("RMSE [%]")


def unbiaised_rmse_barplots(
    snow_cover_products: List[SnowCoverProduct],
    postprocessed_dataset: xr.Dataset,
    analysis_var_plot_name: str,
    title_complement: str,
):
    _, ax = plt.subplots()
    ax.set_ylim(0, 30)
    ax.set_ylabel("unbiaised_rmse [%]")
    ax.set_xlabel(analysis_var_plot_name)
    barplots(
        snow_cover_products=snow_cover_products,
        postprocessed_data_array=postprocessed_dataset.data_vars["unbiaised_rmse"],
        title=f"Unbiaised RMSE vs {analysis_var_plot_name} - {title_complement}",
        ax=ax,
    )
    ax.grid(True, axis="y")


def histograms_to_distribution(metrics_ds) -> npt.NDArray:
    all_dims = list(metrics_ds.sizes.keys())
    all_dims.remove("biais_bins")
    metrics_squeezed = metrics_ds.sum(dim=all_dims)
    distribution = np.repeat(metrics_ds.coords["biais_bins"].values, metrics_squeezed["n_occurrences"].values.astype(np.int64))
    return distribution


def semidistributed_geometry_plot(
    snow_cover_products: List[SnowCoverProduct],
    analysis_folder: str,
    variable_to_plot: str,
    title_complement: str | None = None,
    altitude_ticks: npt.NDArray | None = None,
):
    slope_titles = ["< 10", "10-30", "30-50"]
    if altitude_ticks is None:
        sample_dataset = open_reduced_dataset(
            product=snow_cover_products[0], analysis_folder=analysis_folder, analysis_type="uncertainty"
        )
        altitude_ticks = np.array(
            [
                0,
                *sample_dataset.coords["altitude_bins"].values,
            ]
        )
    for product in snow_cover_products:
        dataset = open_reduced_dataset(product=product, analysis_folder=analysis_folder, analysis_type="uncertainty")
        fig, ax = plt.subplots(1, 3, figsize=(12, 5), subplot_kw=dict(projection="polar"))
        dataset_reduced = (
            dataset.groupby(["slope_bins", "aspect_bins", "altitude_bins"]).map(histograms_to_biais_rmse).reindex_like(dataset)
        )
        for i, slope in enumerate(dataset_reduced.coords["slope_bins"].values[:3]):
            ax[i].set_theta_direction(-1)
            ax[i].set_theta_zero_location("N")
            ax[i].set_xticks(np.deg2rad(EvaluationVsHighResBase.aspect_bins().bins[1:] - 22.5))
            ax[i].set_xticklabels(dataset.coords["aspect_bins"].values)
            ax[i].set_rticks(altitude_ticks)
            ax[i].set_rlim(altitude_ticks[-1], altitude_ticks[0])
            ax[i].set_title("Slope : " + slope_titles[i] + "°")
            ax[0].set_ylabel(variable_to_plot, labelpad=15, fontsize=12)

            im = ax[i].pcolor(
                np.deg2rad(EvaluationVsHighResBase.aspect_bins().bins),
                altitude_ticks,
                dataset_reduced.data_vars[variable_to_plot].sel(slope_bins=slope).transpose().values,
                cmap="coolwarm" if variable_to_plot == "biais" else "Reds",
                vmin=-15 if variable_to_plot == "biais" else 0,
                vmax=15 if variable_to_plot == "biais" else 25,
            )

            fig.colorbar(im, ax=ax[i], orientation="horizontal", label=variable_to_plot, fraction=0.05, pad=0.1)
            fig.suptitle(f"{product.name} - {title_complement}")


def raw_error_boxplots(snow_cover_products: List[SnowCoverProduct], analysis_folder: str, analysis_var: str, ax: Axes):
    sample_dataset = open_reduced_dataset(product=snow_cover_products[0])
    ticks = np.arange(len(list(sample_dataset.coords[analysis_var].values)))
    for product in snow_cover_products:
        metrics_dataset = open_reduced_dataset(product=product, analysis_folder=analysis_folder, analysis_type="uncertainty")
        error_distributions = []

        for value in metrics_dataset.coords[analysis_var].values:
            prod_selected_metrics = metrics_dataset.sel({analysis_var: value})
            error_distributions.append(histograms_to_distribution(prod_selected_metrics))

        product_boxplot = ax.boxplot(
            error_distributions,
            positions=ticks,
            widths=0.2,
            showfliers=False,
            patch_artist=True,
            label=product.plot_name,
        )

        for patch in product_boxplot["boxes"]:
            patch.set_facecolor(product.plot_color)
        ticks = ticks + 0.2

    ticks = np.arange(len(metrics_dataset.coords[analysis_var].values))
    ax.set_xticks(ticks + (len(snow_cover_products) - 1) * 0.2 / 2)
    ax.grid(True, axis="y")
    ax.legend()


def fancy_table_error_distribution(dataframe_to_print: pd.DataFrame) -> Styler:
    color_maps = {
        "biais": "RdYlGn_r",
        "rmse": "RdYlGn_r",
        "unbiaised_rmse": "RdYlGn_r",
    }
    vmins = {
        "biais": 0,
        "rmse": 8,
        "unbiaised_rmse": 8,
    }
    vmaxs = {
        "biais": 10,
        "rmse": 25,
        "unbiaised_rmse": 25,
    }

    # Apply gradient coloring
    return fancy_table(dataframe_to_print=dataframe_to_print, color_maps=color_maps, vmins=vmins, vmaxs=vmaxs)


def double_variable_barplots(
    snow_cover_products: List[SnowCoverProduct], analysis_folder: str, var1: str, var2: str, title_complement: str = ""
):
    metrics_datasets = [
        open_reduced_dataset(product=prod, analysis_folder=analysis_folder, analysis_type="uncertainty")
        for prod in snow_cover_products
    ]
    reduced_ds = postprocess_uncertainty_analysis(
        snow_cover_products=snow_cover_products, metrics_datasets=metrics_datasets, analysis_var=[var1, var2]
    )
    reduced_df = reduced_ds.to_dataframe()
    sns.set_style("whitegrid")
    plot_biais = sns.catplot(
        reduced_df,
        x="biais",
        y=var2,
        hue="product",
        col=var1,
        kind="bar",
        col_wrap=2,
        orient="h",
        palette=[product.plot_color for product in snow_cover_products],
    )
    plot_rmse = sns.catplot(
        reduced_df,
        x="rmse",
        y=var2,
        hue="product",
        col=var1,
        kind="bar",
        col_wrap=2,
        orient="h",
        palette=[product.plot_color for product in snow_cover_products],
    )
    plot_biais.figure.suptitle(f"Bias vs aspect - {title_complement}", fontsize=11, fontweight="bold")
    plot_biais.figure.subplots_adjust(top=0.85)
    plot_rmse.figure.suptitle(f"RMSE vs aspect - {title_complement}", fontsize=11, fontweight="bold")
    plot_rmse.figure.subplots_adjust(top=0.85)


def scatter_to_biais(dataset: xr.Dataset) -> xr.DataArray:
    data_array = dataset.data_vars["n_occurrences"].sum(dim=("forest_mask_bins", "sub_roi_bins", "aspect_bins", "time"))
    biais_bins = np.arange(-100, 101)

    occurrences_per_biais_bins = np.zeros_like(biais_bins)
    for i, b in enumerate(biais_bins):
        occurrences_per_biais_bins[i] = np.trace(data_array.values, offset=b + data_array.ref_bins.values[0])
    out_data_array = xr.DataArray(data=occurrences_per_biais_bins, coords={"biais_bins": biais_bins})
    return xr.Dataset({"n_occurrences": out_data_array})


def smooth_data_np_convolve(arr, span):
    return np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")


def plot_custom_spans(snow_cover_products: List[SnowCoverProduct], analysis_folder: str, analysis_var: str, ax: plt.Axes):
    percentile_min, percentile_max = 5, 95
    sample_dataset = open_reduced_dataset_for_plot(
        product=snow_cover_products[0], analysis_folder=analysis_folder, analysis_type="uncertainty"
    )

    x_positions = np.arange(len(sample_dataset.coords[analysis_var].values))
    x_positions = x_positions / len(x_positions)
    for product in snow_cover_products:
        metrics_dataset = open_reduced_dataset_for_plot(
            product=product, analysis_folder=analysis_folder, analysis_type="uncertainty"
        )
        analysis_coords = metrics_dataset.coords[analysis_var].values
        box_width_data = 0.2 / len(x_positions)
        for idx, value in enumerate(analysis_coords):
            x_pos = x_positions[idx]
            product_analysis_var_dataset = metrics_dataset.sel({analysis_var: value})  # .drop_sel(biais_bins=0)
            reduced = product_analysis_var_dataset.groupby("biais_bins").sum(list(product_analysis_var_dataset.sizes.keys()))
            biais_rmse = histograms_to_biais_rmse(reduced)
            distr = histograms_to_distribution(reduced)
            ax.scatter(x_pos, biais_rmse.data_vars["biais"], marker="o", color=product.plot_color, s=15, zorder=3)
            whiskers_min = np.percentile(distr, percentile_min)
            whiskers_max = np.percentile(distr, percentile_max)
            ax.vlines(
                x_pos, whiskers_min, whiskers_max, color=product.plot_color, linestyle="-", lw=3, label=product.plot_name
            )

            ax.hlines(whiskers_min, x_pos - box_width_data / 2, x_pos + box_width_data / 2, color=product.plot_color, lw=3)
            ax.hlines(whiskers_max, x_pos - box_width_data / 2, x_pos + box_width_data / 2, color=product.plot_color, lw=3)
        x_positions = x_positions + box_width_data

    ax.set_xticks(x_positions - box_width_data * ((len(snow_cover_products) + 1) // 2), labels=analysis_coords)
    ax.set_xlim(x_positions[0] - (len(snow_cover_products) + 1) * box_width_data, x_positions[-1])
    ax.set_ylim(-65, 65)
    ax.set_ylabel("Bias [\% FSC]")
    ax.set_xlabel(analysis_var)
    (l1,) = ax.plot([0, 1], [0, 1], c="gray", lw=1e-12)
    (l2,) = ax.plot(0, 0, c="gray", markersize=1e-12)
    ax.legend(
        [l1, l2],
        [f"{percentile_min}th and {percentile_max}th percentiles", "mean"],
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


def line_plot_rmse(snow_cover_products: List[SnowCoverProduct], analysis_folder: str, analysis_var: str, ax: Axes):
    metrics_datasets = [
        open_reduced_dataset_for_plot(product=prod, analysis_folder=analysis_folder, analysis_type="uncertainty")
        for prod in snow_cover_products
    ]
    biais_rmse = postprocess_uncertainty_analysis(
        snow_cover_products, metrics_datasets=metrics_datasets, analysis_var=analysis_var
    )

    x_coords_unc = biais_rmse.coords[analysis_var].values

    for prod in snow_cover_products:
        ax.plot(
            x_coords_unc, biais_rmse.data_vars["rmse"].sel(product=prod.name), "-o", color=prod.plot_color, markersize=5, lw=3
        )

    ax.grid(True)
    ax.set_ylim(5, 30)
    ax.set_ylabel("RMSE [\% FSC]")
    ax.set_xlabel(analysis_var)
    ax.legend([Line2D([0], [0], linestyle="-", color="gray")], ["RMSE"])


# if __name__ == "__main__":
#     wy = WinterYear(2023, 2024)
#     analysis_type = "uncertainty"
#     analysis_folder = (
#         f"/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_6_modis/analyses/{analysis_type}"
#     )
#     analyses_dict = {k: xr.open_dataset(v, decode_cf=True) for k, v in path_dict.items()}
#     evaluation_domain = "general"
#     selection_dict, title = sel_evaluation_domain(analyses_dict=analyses_dict, evaluation_domain=evaluation_domain)

#     ############## Launch analysis

#     # Temporal analysis
#     biais_barplots(
#         postprocess_uncertainty_analysis(selection_dict, analysis_var={"time": EvaluationVsHighResBase.month_bins(wy)}),
#         analysis_var_plot_name="time (month)",
#         title_complement=f"Biais temporal distribution - {title} - {str(wy)}",
#     )
# unbiaised_rmse_barplots(
#     postprocess_uncertainty_analysis(selection_dict, analysis_var={"time": EvaluationVsHighResBase.month_bins(wy)}),
#     analysis_var_plot_name="time (month)",
#     title_complement=f"Unbiaised RMSE temporal distribution - {title} - {str(wy)}",
# # )
# rmse_barplots(
#     postprocess_uncertainty_analysis(selection_dict, analysis_var={"time": EvaluationVsHighResBase.month_bins(wy)}),
#     analysis_var_plot_name="time (month)",
#     title_complement=f"RMSE temporal distribution - {title} - {str(wy)}",
# )

# # rmse_barplots(
# #     postprocess_uncertainty_analysis(selection_dict, analysis_var={"time": EvaluationVsHighResBase.month_bins(wy)}),
# #     analysis_var_plot_name="time (month)",
# #     title_complement=f"FSC 1-99% - no forest - {str(wy)}",
# # )
# # SAFRAN geometry
# semidistributed_geometry_plot(
#     selection_dict, "biais", title_complement=f"Semidistributed geometry biais distribution- {title}"
# )
# semidistributed_geometry_plot(
#     selection_dict, "unbiaised_rmse", title_complement=f"Semidistributed geometry unbiaised RMSE distribution - {title}"
# )

# # Barplots aspect
# double_variable_barplots(selection_dict, "forest_mask_bins", "aspect_bins")

# # Boxplots vza
# # fig, ax = plt.subplots(figsize=(10, 4))
# # fig.suptitle(f"Error distribution vs VZA - {title} - {str(wy)}")
# # ax.set_xticklabels(["0-15", "15-30", "30-45", "45-60", ">60"])
# # ax.set_xlabel("Sensor zenith angle [°]")
# # ax.set_ylabel("FSC [%]")
# # ax.set_ylim(-60, 60)

# # sel_vza = selection_dict.copy()
# # if "nasa_l3_snpp" in selection_dict:
# #     sel_vza.pop("nasa_l3_snpp")
# # if "nasa_l3_jpss1" in selection_dict:
# #     sel_vza.pop("nasa_l3_jpss1")
# # if "nasa_l3_multiplatform" in selection_dict:
# #     sel_vza.pop("nasa_l3_multiplatform")
# # sel_vza = {k: v.sel(sensor_zenith_bins=slice(0, 75)) for k, v in sel_vza.items()}

# # raw_error_boxplots(metrics_dict=sel_vza, analysis_var="sensor_zenith_bins", ax=ax)

# # # Boxplots slope
# fig, ax = plt.subplots(figsize=(10, 4))
# fig.suptitle(f"Error distribution vs slope - {title} - {str(wy)}")
# ax.set_xticklabels(["0-10", "10-30", "30-50"])
# ax.set_xlabel("Slope [°]")
# ax.set_ylabel("FSC [%]")
# ax.set_ylim(-60, 60)
# ax.plot()
# selection_slope_dict = {k: v.sel(slope_bins=slice(0, 60)) for k, v in selection_dict.items()}
# raw_error_boxplots(metrics_dict=selection_slope_dict, analysis_var="slope_bins", ax=ax)

# # Boxplots ref FSC

# fig, ax = plt.subplots(figsize=(10, 4))
# fig.suptitle("Error distribution vs Fractional Snow Cover")
# ax.set_xlabel("Ref FSC bin [%]")
# ax.set_ylabel("FSC error [%]")
# ax.set_ylim(-80, 80)
# ax.set_yticks(np.arange(-80, 81, 10))
# ax.set_xticklabels(
#     [
#         "[1-10]",
#         "[11-20]",
#         "[21-30]",
#         "[31-40]",
#         "[41-50]",
#         "[51-60]",
#         "[61-70]",
#         "[71-80]",
#         "[81-90]",
#         "[91-99]",
#     ]
# )
# selection_ref_fsc_dict = {k: v.sel(ref_bins=slice(1, 99)) for k, v in selection_dict.items()}
# raw_error_boxplots(metrics_dict=selection_ref_fsc_dict, analysis_var="ref_bins", ax=ax)

# # Print resume table
# reduced_datasets = []
# for dataset in selection_dict.values():
#     reduced_datasets.append(histograms_to_biais_rmse(dataset))
# concatenated = xr.concat(reduced_datasets, pd.Index(list(selection_dict.keys()), name="product"), coords="minimal")
# reduced_df = concatenated.to_dataframe().reset_index("product")
# print(reduced_df.round(decimals=2))

# plt.show()
