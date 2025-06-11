from typing import Any, Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

from postprocess.error_distribution import (
    histograms_to_biais_rmse,
    histograms_to_distribution,
    postprocess_uncertainty_analysis,
)
from postprocess.skill_scores import compute_all_scores
from products.plot_settings import PRODUCT_PLOT_COLORS, PRODUCT_PLOT_NAMES
from reductions.statistics_base import EvaluationVsHighResBase


def sel_evaluation_domain(analyses_dict: Dict[str, xr.Dataset]) -> Tuple[Dict[str, xr.Dataset], str]:
    selection_dict = {
        k: v.sel(time=slice("2023-12", "2024-06"), altitude_bins=slice(900, None)) for k, v in analyses_dict.items()
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
            }
        )
        for k, v in selection_dict.items()
    }

    selection_dict = {
        k: v.rename({"aspect_bins": "Aspect", "forest_mask_bins": "Landcover"}) for k, v in selection_dict.items()
    }

    return selection_dict


def compute_skill_scores_for_parameter(metrics_dict: Dict[str, xr.Dataset], variable: str):
    results = []
    for metrics_ds in metrics_dict.values():
        results.append(metrics_ds.groupby(variable).map(compute_all_scores))
    results = xr.concat(results, pd.Index([PRODUCT_PLOT_NAMES[k] for k in metrics_dict.keys()], name="product")).to_dataframe()
    # results = results.reset_index([variable, "product"])
    # results = results.melt(id_vars=["product", variable], var_name="score", value_name="value")
    return results


def smooth_data_np_convolve(arr, span):
    return np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")


# red = metrics_dict_unc['meteofrance_synopsis'].sel(aspect_bins='S')


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
            ax.vlines(x_pos, whiskers_min, whiskers_max, color=color, linestyle="-", lw=1, label=product_name)

            ax.hlines(whiskers_min, x_pos - box_width_data / 2, x_pos + box_width_data / 2, color=color, lw=1)
            ax.hlines(whiskers_max, x_pos - box_width_data / 2, x_pos + box_width_data / 2, color=color, lw=1)
        x_positions = x_positions + box_width_data

    ax.set_xticks(x_positions - box_width_data * ((len(metrics_dict) + 1) // 2), labels=analysis_coords)
    ax.set_ylabel(f"% FSC")


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
    _, axs = plt.subplots(len(params_list), 3, figsize=(10, 3 * len(params_list)), layout="constrained")
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
        biais_rmse = postprocess_uncertainty_analysis(metrics_dict_unc, analysis_var=var)

        axs[i, 0].set_ylim(0.75, 1)
        axs[i, 0].set_ylabel("Skill score[-]")

        axs[i, 1].set_ylim(5, 25)
        axs[i, 1].set_ylabel("RMSE [% FSC]")
        axs[i, 2].set_ylim(-25, 25)
        plot_custom_spans(metrics_dict=metrics_dict_unc, analysis_var=var, ax=axs[i, 2])
        axs[i, 2].grid(axis="x")

        for j, (prod, metrics) in enumerate(metrics_dict_conf.items()):
            axs[i, 0].plot(
                skill_scores.loc[PRODUCT_PLOT_NAMES[prod]]["accuracy"], "-o", color=PRODUCT_PLOT_COLORS[prod], markersize=2
            )
            axs[i, 0].plot(
                skill_scores.loc[PRODUCT_PLOT_NAMES[prod]]["f1_score"], "--^", color=PRODUCT_PLOT_COLORS[prod], markersize=2
            )
            # axs[i,1].plot(biais_rmse.data_vars['biais'].sel(product=prod).to_pandas(), '-o',color=PRODUCT_PLOT_COLORS[prod], markersize=2)
            axs[i, 1].plot(
                biais_rmse.data_vars["rmse"].sel(product=prod).to_pandas(), "-^", color=PRODUCT_PLOT_COLORS[prod], markersize=2
            )

    plt.show()
