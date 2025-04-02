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
from products.plot_settings import PRODUCT_PLOT_NAMES

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
    results = xr.concat(results, pd.Index(list(metrics_dict.keys()), name="product")).to_dataframe()
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
