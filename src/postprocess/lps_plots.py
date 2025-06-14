from typing import Any, Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import font_manager, ticker
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

from postprocess.error_distribution import (
    histograms_to_biais_rmse,
    histograms_to_distribution,
    postprocess_uncertainty_analysis,
)
from postprocess.skill_scores import compute_all_scores
from products.plot_settings import (
    NASA_L3_JPSS1_VAR_NAME,
    NASA_L3_MULTIPLATFORM_VAR_NAME,
    NASA_L3_SNPP_VAR_NAME,
    PRODUCT_PLOT_COLORS,
    PRODUCT_PLOT_NAMES,
)
from reductions.statistics_base import EvaluationVsHighResBase


def sel_evaluation_domain(analyses_dict: Dict[str, xr.Dataset]) -> Tuple[Dict[str, xr.Dataset], str]:
    selection_dict = {
        k: v.sel(
            time=slice("2023-12", "2024-06"),
            altitude_bins=slice(900, None),
            ref_bins=slice(0, 101),
            slope_bins=slice(None, 60),
        )
        for k, v in analyses_dict.items()
    }

    selection_dict = {
        k: v.assign_coords(
            {
                "aspect_bins": pd.CategoricalIndex(
                    data=EvaluationVsHighResBase.aspect_bins().labels,
                    categories=EvaluationVsHighResBase.aspect_bins().labels,
                    ordered=True,
                ),
                "forest_mask_bins": ["Open", "Forest"],
                "slope_bins": np.array(["[0-10]", "[11-30]", "[31-]"]),
                "ref_bins": ["0", "1-25", "26-50", "51-75", "75-99", "100"],
            }
        )
        for k, v in selection_dict.items()
    }

    selection_dict = {
        k: v.rename(
            {"aspect_bins": "Aspect", "forest_mask_bins": "Landcover", "slope_bins": "Slope [°]", "ref_bins": "Ref FSC [%]"}
        )
        for k, v in selection_dict.items()
    }

    return selection_dict


def compute_skill_scores_for_parameter(metrics_dict: Dict[str, xr.Dataset], variable: str) -> pd.DataFrame:
    results = []
    for metrics_ds in metrics_dict.values():
        results.append(metrics_ds.groupby(variable, restore_coord_dims=True).map(compute_all_scores))
    results = xr.concat(results, pd.Index([PRODUCT_PLOT_NAMES[k] for k in metrics_dict.keys()], name="product"))
    # results = results.reset_index([variable, "product"])
    # results = results.melt(id_vars=["product", variable], var_name="score", value_name="value")
    return results


def smooth_data_np_convolve(arr, span):
    return np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")


def plot_custom_spans(metrics_dict: Dict[str, xr.Dataset], analysis_var: str, ax: plt.Axes):
    x_positions = np.arange(len(list(metrics_dict.values())[0].coords[analysis_var].values))
    x_positions = x_positions / len(x_positions)
    ax.set_ylim(-60, 60)
    for product_name, metrics_dataset in metrics_dict.items():
        color = PRODUCT_PLOT_COLORS[product_name]
        analysis_coords = metrics_dict[product_name].coords[analysis_var].values
        box_width_data = 0.2 / len(x_positions)
        for idx, value in enumerate(analysis_coords):
            x_pos = x_positions[idx]
            product_analysis_var_dataset = metrics_dataset.sel({analysis_var: value}).drop_sel(biais_bins=0)
            reduced = product_analysis_var_dataset.groupby("biais_bins").sum(list(product_analysis_var_dataset.sizes.keys()))
            smooth = smooth_data_np_convolve(reduced.data_vars["n_occurrences"], 1)
            smooth = smooth / smooth.max()

            biais_rmse = histograms_to_biais_rmse(reduced)
            distr = histograms_to_distribution(reduced)
            ax.scatter(x_pos, biais_rmse.data_vars["biais"], marker="o", color=color, s=15, zorder=3)

            whiskers_min = np.quantile(distr, 0.05)
            whiskers_max = np.quantile(distr, 0.95)
            ax.vlines(x_pos, whiskers_min, whiskers_max, color=color, linestyle="-", lw=2, label=product_name)

            ax.hlines(whiskers_min, x_pos - box_width_data / 2, x_pos + box_width_data / 2, color=color, lw=2)
            ax.hlines(whiskers_max, x_pos - box_width_data / 2, x_pos + box_width_data / 2, color=color, lw=2)
        x_positions = x_positions + box_width_data

    ax.set_xticks(x_positions - box_width_data * ((len(metrics_dict) + 1) // 2), labels=analysis_coords)
    ax.set_xlim(x_positions[0] - (len(metrics_dict) + 1) * box_width_data, x_positions[-1])
    ax.set_ylabel(f"MAE [% FSC]")
    ax.set_xlabel(analysis_var)


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


def plot_synthesis(metrics_dict_conf: Dict[str, xr.Dataset], metrics_dict_unc: Dict[str, xr.Dataset], params_list: List[str]):
    _, axs = plt.subplots(len(params_list), 3, figsize=(14, 3 * len(params_list)), layout="constrained")
    custom_leg = [
        mpatches.Patch(color=PRODUCT_PLOT_COLORS[product_name], label=PRODUCT_PLOT_NAMES[product_name])
        for product_name in metrics_dict_unc
    ]
    product_legend = axs[0, 1].legend(handles=custom_leg, loc=[0.1, 0.7])
    axs[0, 1].add_artist(product_legend)
    axs[0, 0].legend(
        [Line2D([0], [0], linestyle="-", color="gray"), Line2D([0], [0], linestyle="--", color="gray")],
        ["Accuracy", "F1 score"],
        loc=(0, 1.05),
    )
    axs[0, 1].legend([Line2D([0], [0], linestyle="-", color="gray")], ["RMSE"], loc=(0, 1.05))

    (l1,) = axs[0, 2].plot([0, 1], [0, 1], c="gray", lw=1e-12)
    (l2,) = axs[0, 2].plot(0, 0, c="gray", markersize=1e-12)
    axs[0, 2].legend(
        [l1, l2], ["5th 95th and percentile", "mean"], handler_map={l1: HandlerSpan(), l2: HandlerPoint()}, loc=(0, 1.05)
    )
    for i, var in enumerate(params_list):
        skill_scores = compute_skill_scores_for_parameter(metrics_dict_conf, variable=var)
        skill_scores = skill_scores.where(skill_scores != 0, np.nan)
        biais_rmse = postprocess_uncertainty_analysis(metrics_dict_unc, analysis_var=var)

        axs[i, 0].set_ylim(0.75, 1)
        axs[i, 0].set_ylabel("Score[-]")
        axs[i, 0].set_xlabel(var)

        axs[i, 1].set_ylim(5, 30)
        axs[i, 1].set_ylabel("RMSE [% FSC]")
        axs[i, 1].set_xlabel(var)

        plot_custom_spans(metrics_dict=metrics_dict_unc, analysis_var=var, ax=axs[i, 2])
        axs[i, 2].grid(axis="x")

        for prod in metrics_dict_conf:
            #     # print(skill_scores.loc[PRODUCT_PLOT_NAMES[prod]])
            x_coords_conf = metrics_dict_conf[list(metrics_dict_conf.keys())[0]].coords[var].values
            x_coords_unc = metrics_dict_unc[list(metrics_dict_unc.keys())[0]].coords[var].values
            if var in ("Ref FSC [%]", "Slope [°]"):
                skill_scores = skill_scores.sel({var: x_coords_conf})
                biais_rmse = biais_rmse.sel({var: x_coords_unc})

            axs[i, 0].plot(
                x_coords_conf,
                skill_scores.sel(product=PRODUCT_PLOT_NAMES[prod]).data_vars["accuracy"],
                "-o",
                color=PRODUCT_PLOT_COLORS[prod],
                markersize=2,
            )
            axs[i, 0].plot(
                x_coords_conf,
                skill_scores.sel(product=PRODUCT_PLOT_NAMES[prod]).data_vars["f1_score"],
                "--^",
                color=PRODUCT_PLOT_COLORS[prod],
                markersize=2,
            )
            axs[i, 1].plot(
                x_coords_unc,
                biais_rmse.data_vars["rmse"].sel(product=prod),
                "-o",
                color=PRODUCT_PLOT_COLORS[prod],
                markersize=2,
            )

    plt.show()


def annual_area_fancy_plot(metrics_dict_completeness: Dict[str, xr.Dataset], metrics_dict_uncertainty: Dict[str, xr.Dataset]):
    if len(metrics_dict_completeness) > 1:
        common_days = np.intersect1d(*[v.coords["time"] for v in metrics_dict_completeness.values()][:2])

    if len(metrics_dict_completeness) > 2:
        for v in metrics_dict_completeness.values():
            common_days = np.intersect1d(common_days, v.coords["time"])

    _, axs = plt.subplots(4, 1, figsize=(10, 6), sharex=True, layout="constrained")

    [ax.set_ylabel("Surface [km²]", fontproperties=font_manager.FontProperties(size=10)) for ax in axs[:3]]
    axs[3].set_ylabel("RMSE [% FSC]", fontproperties=font_manager.FontProperties(size=10))
    custom_leg = [
        mpatches.Patch(color=PRODUCT_PLOT_COLORS[product_name], label=PRODUCT_PLOT_NAMES[product_name])
        for product_name in metrics_dict_completeness
    ]
    product_legend = axs[0].legend(handles=custom_leg, loc=[0.8, 0.45])
    axs[0].add_artist(product_legend)
    old_snow_cover, old_no_snow, old_clouds = 0, 0, 0
    for product in metrics_dict_completeness:
        metrics_dict_completeness[product] = metrics_dict_completeness[product].sel(time=common_days)
        product_monthly_averages = (
            metrics_dict_completeness[product].resample({"time": "1ME"}).mean(dim="time").data_vars["surface"] * 1e-6
        )

        snow_cover = product_monthly_averages.sel(class_name="snow_cover")
        axs[0].fill_between(
            np.arange(product_monthly_averages.sizes["time"]),
            snow_cover,
            old_snow_cover,
            alpha=0.5,
            color=PRODUCT_PLOT_COLORS[product],
        )
        old_snow_cover = snow_cover
        no_snow = product_monthly_averages.sel(class_name="no_snow")
        axs[1].fill_between(
            np.arange(product_monthly_averages.sizes["time"]),
            no_snow,
            old_no_snow,
            alpha=0.5,
            color=PRODUCT_PLOT_COLORS[product],
        )
        old_no_snow = no_snow

        axs[3].plot(
            np.arange(product_monthly_averages.sizes["time"]),
            metrics_dict_uncertainty[product].rmse,
            color=PRODUCT_PLOT_COLORS[product],
            lw=2,
        )
    axs[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0e}"))
    axs[1].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0e}"))
    axs[2].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0e}"))
    axs[3].set_ylim(0, 20)
    for product in [NASA_L3_MULTIPLATFORM_VAR_NAME, NASA_L3_JPSS1_VAR_NAME, NASA_L3_SNPP_VAR_NAME]:
        product_monthly_averages = (
            metrics_dict_completeness[product].resample({"time": "1ME"}).mean(dim="time").data_vars["surface"] * 1e-6
        )
        clouds = product_monthly_averages.sel(class_name="clouds")
        axs[2].fill_between(
            np.arange(product_monthly_averages.sizes["time"]),
            clouds,
            old_clouds,
            alpha=0.5,
            color=PRODUCT_PLOT_COLORS[product],
        )
        old_clouds = clouds
    axs[2].set_xticks(np.arange(product_monthly_averages.sizes["time"]))
    axs[2].set_xticklabels(product_monthly_averages.coords["time"].to_dataframe().index.strftime("%B"))
    axs[0].set_title("Snow cover")
    axs[1].set_title("No snow")
    axs[2].set_title("Clouds")
    axs[3].set_title("RMSE")

    plt.show()
