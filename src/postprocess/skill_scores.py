from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from pandas.io.formats.style import Styler
from scores.categorical import BasicContingencyManager
from sklearn.metrics import ConfusionMatrixDisplay

from postprocess.general_purpose import fancy_table
from products.plot_settings import MF_NO_FOREST_VAR_NAME, NASA_L3_JPSS1_VAR_NAME, PRODUCT_PLOT_NAMES

SCORES = ["accuracy", "precision", "recall", "f1_score", "commission_error", "omission_error"]


def compute_score(dataset: xr.Dataset, score_name: str):
    tp, tn, fp, fn = (
        dataset.data_vars["true_positive"].sum(),
        dataset.data_vars["true_negative"].sum(),
        dataset.data_vars["false_positive"].sum(),
        dataset.data_vars["false_negative"].sum(),
    )
    scores_manager = BasicContingencyManager(
        counts={"tp_count": tp, "tn_count": tn, "fp_count": fp, "fn_count": fn, "total_count": tp + tn + fp + fn}
    )

    return getattr(scores_manager, score_name)()


def omission_error(dataset: xr.Dataset):
    return dataset["false_negative"].sum() / (dataset["true_positive"].sum() + dataset["false_negative"].sum())


def compute_all_scores(dataset: xr.Dataset):
    out_scores_dict = {}
    for score in SCORES:
        if score == "omission_error":
            out_scores_dict.update({score: omission_error(dataset)})
        elif score == "commission_error":
            out_scores_dict.update({score: compute_score(dataset, score_name="false_alarm_rate")})
        else:
            out_scores_dict.update({score: compute_score(dataset, score_name=score)})
    return xr.Dataset(out_scores_dict)


def plot_confusion_table(dataset: xr.Dataset, axes: Axes | None = None):
    tot = np.sum(np.array([dataset[dv].sum() for dv in dataset]))

    confusion_matrix = np.array(
        [
            [
                dataset.data_vars["true_positive"].sum().values / tot * 100,
                dataset.data_vars["false_negative"].sum().values / tot * 100,
            ],
            [
                dataset.data_vars["false_positive"].sum().values / tot * 100,
                dataset.data_vars["true_negative"].sum().values / tot * 100,
            ],
        ],
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["snow", "no_snow"])
    disp.plot(ax=axes, colorbar=False)


def plot_multiple_confusion_table(metrics_dict: Dict[str, xr.Dataset], title_complement: str):
    n_prods = len(metrics_dict)
    fig, axs = plt.subplots(ncols=n_prods, figsize=(6 * n_prods, 5))
    fig.suptitle("Confusion table comparison - " + title_complement + "\n")
    for i, (product, metrics_ds) in enumerate(metrics_dict.items()):
        accuracy = compute_score(metrics_ds, "accuracy").values
        plot_confusion_table(metrics_ds, axes=axs[i])
        axs[i].set_title(f"{PRODUCT_PLOT_NAMES[product]} - accuracy = {accuracy:.2f}")


def plot_multiple_scores_sns(metrics_dict: Dict[str, xr.Dataset], variable: str, xlabel: str, title_complement: str):
    results = []
    for metrics_ds in metrics_dict.values():
        results.append(metrics_ds.groupby(variable).map(compute_all_scores))
    results = xr.concat(results, pd.Index([PRODUCT_PLOT_NAMES[k] for k in metrics_dict.keys()], name="product")).to_dataframe()
    results = results.reset_index([variable, "product"])
    results = results.melt(id_vars=["product", variable], var_name="score", value_name="value")
    sns.set_style("darkgrid")
    g = sns.relplot(
        results,
        x=variable,
        hue="score",
        y="value",
        col="product",
        kind="line",
        style="score",
        markers=True,
        col_wrap=np.min([len(metrics_dict), 4]),
    )
    g.figure.suptitle(f"Performance Metrics vs {title_complement}", fontsize=14, fontweight="bold")
    # Adjust layout to make space for the title
    g.figure.subplots_adjust(top=0.85)
    g.set_axis_labels(xlabel, "Score [-]")


def compute_results_df(metrics_dict: xr.Dataset) -> pd.DataFrame:
    results = []
    for metrics_ds in metrics_dict.values():
        results.append(compute_all_scores(metrics_ds))
    results = xr.concat(results, pd.Index(list(metrics_dict.keys()), name="product")).to_dataframe()
    results = results.reset_index(["product"])
    return pd.DataFrame(results)


def fancy_table_skill_scores(dataframe_to_print: pd.DataFrame) -> Styler:
    color_maps = {
        "accuracy": "RdYlGn",
        "precision": "RdYlGn",
        "recall": "RdYlGn",
        "f1_score": "RdYlGn",
        "commission_error": "RdYlGn_r",  # Lower is better
        "omission_error": "RdYlGn_r",
    }
    vmins = {
        "accuracy": 0.6,  # Higher is better
        "precision": 0.6,
        "recall": 0.6,
        "f1_score": 0.6,
        "commission_error": 0,  # Lower is better (reversed Reds)
        "omission_error": 0,
    }
    vmaxs = {
        "accuracy": 1,  # Higher is better
        "precision": 1,
        "recall": 1,
        "f1_score": 1,
        "commission_error": 0.3,  # Lower is better
        "omission_error": 0.3,
    }

    # Apply gradient coloring
    return fancy_table(dataframe_to_print=dataframe_to_print, color_maps=color_maps, vmins=vmins, vmaxs=vmaxs)


if __name__ == "__main__":
    import xarray as xr

    from postprocess.general_purpose import sel_evaluation_domain
    from products.plot_settings import (
        METEOFRANCE_VAR_NAME,
        MF_NO_CC_MASK_VAR_NAME,
        MF_ORIG_VAR_NAME,
        MF_REFL_SCREEN_VAR_NAME,
        MF_SYNOPSIS_VAR_NAME,
        NASA_L3_SNPP_VAR_NAME,
        NASA_PSEUDO_L3_VAR_NAME,
    )
    from reductions.statistics_base import EvaluationVsHighResBase
    from winter_year import WinterYear

    wy = WinterYear(2023, 2024)
    analysis_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_6/analyses/confusion_table"
    analysis_type = "confusion_table"

    analyses_dict = {
        # MF_ORIG_VAR_NAME: xr.open_dataset(
        #     f"{analysis_folder}/confusion_table_WY_2023_2024_meteofrance_orig_vs_s2_theia.nc", decode_cf=True
        # ),
        # MF_SYNOPSIS_VAR_NAME: xr.open_dataset(
        #     f"{analysis_folder}/confusion_table_WY_2023_2024_meteofrance_synopsis_vs_s2_theia.nc",
        #     decode_cf=True,
        # ),
        MF_NO_FOREST_VAR_NAME: xr.open_dataset(
            f"{analysis_folder}/confusion_table_WY_2023_2024_meteofrance_no_forest_vs_s2_theia.nc",
            decode_cf=True,
        ),
        # MF_NO_CC_MASK_VAR_NAME: xr.open_dataset(
        #     f"{analysis_folder}/confusion_table_WY_2023_2024_meteofrance_no_cc_mask_vs_s2_theia.nc",
        #     decode_cf=True,
        # ),
        # MF_REFL_SCREEN_VAR_NAME: xr.open_dataset(
        #     f"{analysis_folder}/confusion_table_WY_2023_2024_meteofrance_modified_vs_s2_theia.nc",
        #     decode_cf=True,
        # ),
        # NASA_PSEUDO_L3_VAR_NAME: xr.open_dataset(
        #     f"{analysis_folder}/confusion_table_WY_2023_2024_nasa_pseudo_l3_vs_s2_theia.nc",
        #     decode_cf=True,
        # ),
        NASA_L3_SNPP_VAR_NAME: xr.open_dataset(
            f"{analysis_folder}/confusion_table_WY_2023_2024_nasa_l3_snpp_vs_s2_theia.nc",
            decode_cf=True,
        ),
        # NASA_L3_JPSS1_VAR_NAME: xr.open_dataset(
        #     f"{analysis_folder}/confusion_table_WY_2023_2024_nasa_l3_jpss1_vs_s2_theia.nc",
        #     decode_cf=True,
        # ),
    }

    # analyses_dict = {
    #     MF_S2_FSC_SCREEN: xr.open_dataset(
    #         f"{analysis_folder.replace('version_5_complete', 'version_5_fsc_screen_harmo')}/confusion_table_WY_2023_2024_meteofrance_synopsis_vs_s2_theia_sca.nc",
    #         decode_cf=True,
    #     ),
    #     MF_NO_S2_FSC_SCREEN: xr.open_dataset(
    #         f"{analysis_folder}/confusion_table_WY_2023_2024_meteofrance_synopsis_vs_s2_theia.nc", decode_cf=True
    #     ),
    # }
    evaluation_domain = "accumulation"
    selection_dict, title = sel_evaluation_domain(analyses_dict=analyses_dict, evaluation_domain=evaluation_domain)
    ################# Launch analysis ###########################

    # Confusion table
    plot_multiple_confusion_table(metrics_dict=selection_dict, title_complement=f"{title} - {str(wy)}")

    df = compute_results_df(selection_dict)
    print(df.round(decimals=2))

    sel_no_vza = selection_dict.copy()
    if "nasa_l3_snpp" in sel_no_vza:
        sel_no_vza.pop("nasa_l3_snpp")
    if "nasa_l3_jpss1" in sel_no_vza:
        sel_no_vza.pop("nasa_l3_jpss1")
    # Sensor zenith
    sel_no_vza = {
        k: v.sel(sensor_zenith_bins=slice(None, 90)).assign_coords(
            {"sensor_zenith_bins": ["0-15", "15-30", "30-45", "45-60", ">60"]}
        )
        for k, v in sel_no_vza.items()
    }
    plot_multiple_scores_sns(
        metrics_dict=sel_no_vza,
        variable="sensor_zenith_bins",
        xlabel="Sensor Zenith Angle [°]",
        title_complement=f"Sensor zenith angle - {title} - {str(wy)}",
    )

    # Aspect
    selection_dict = {
        k: v.assign_coords(
            {
                "aspect_bins": pd.CategoricalIndex(
                    data=EvaluationVsHighResBase.aspect_bins().labels,
                    categories=EvaluationVsHighResBase.aspect_bins().labels,
                    ordered=True,
                )
            }
        )
        for k, v in selection_dict.items()
    }
    plot_multiple_scores_sns(
        metrics_dict=selection_dict,
        variable="aspect_bins",
        xlabel="Aspect",
        title_complement=f"Aspect - {title} - {str(wy)}",
    )

    # # Massifs
    # selection_dict = {
    #     k: v.assign_coords({"sub_roi": ["", "Alps", "Pyrenees", "Corse", "Massif Central", "Jura", "Vosges"]})
    #     for k, v in selection_dict.items()
    # }
    # plot_multiple_scores_sns(
    #     metrics_dict=selection_dict,
    #     variable="sub_roi",
    #     xlabel="Massif",
    #     title_complement=f"Massif - {title} - {str(wy)}",
    # )

    # Slope
    # selection_dict = {
    #     k: v.sel(slope_bins=slice(None, 50)).assign_coords({"slope_bins": ["0-10", "10-30", "30-50"]})
    #     for k, v in selection_dict.items()
    # }
    # plot_multiple_scores_sns(
    #     metrics_dict=selection_dict,
    #     variable="slope_bins",
    #     xlabel="Slope [°]",
    #     title_complement=f"Slope - {title} - {str(wy)}",
    # )
    # vegetation
    plot_multiple_scores_sns(
        metrics_dict=selection_dict,
        variable="forest_mask_bins",
        xlabel="Landcover",
        title_complement=f"Landcover - {title} - {str(wy)}",
    )

    # df = compute_results_df(selection_dict)
    # print(df.round(decimals=2))
    plt.show()
