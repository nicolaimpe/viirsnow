from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from scores.categorical import BasicContingencyManager
from sklearn.metrics import ConfusionMatrixDisplay

from postprocess.general_purpose import AnalysisContainer, open_reduced_dataset_for_plot
from products.snow_cover_product import SnowCoverProduct

SCORES = ["accuracy", "f1_score", "commission_error", "omission_error"]


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


def plot_multiple_confusion_table(snow_cover_products: List[SnowCoverProduct], analysis_folder: str, title_complement: str):
    n_prods = len(snow_cover_products)
    fig, axs = plt.subplots(ncols=n_prods, figsize=(6 * n_prods, 5))
    fig.suptitle("Confusion table comparison - " + title_complement + "\n")
    for i, product in enumerate(snow_cover_products):
        metrics_ds = open_reduced_dataset_for_plot(product, analysis_folder=analysis_folder, analysis_type="confusion_table")
        accuracy = compute_score(metrics_ds, "accuracy").values
        plot_confusion_table(metrics_ds, axes=axs[i])
        axs[i].set_title(f"{product.plot_name} - accuracy = {accuracy:.2f}")


def compute_contingency_results_df(
    snow_cover_products: List[SnowCoverProduct], metric_datasets: List[xr.Dataset]
) -> pd.DataFrame:
    results = []
    for metrics_ds in metric_datasets:
        results.append(compute_all_scores(metrics_ds))
    results = xr.concat(
        results, dim=xr.DataArray([prod.plot_name for prod in snow_cover_products], dims="product")
    ).to_dataframe()
    results = results.reset_index(["product"])
    return pd.DataFrame(results)


def compute_skill_scores_for_parameter(
    snow_cover_products: List[SnowCoverProduct], metrics_datasets: List[xr.Dataset], variable: str
) -> pd.DataFrame:
    results = []
    for product, metrics in zip(snow_cover_products, metrics_datasets):
        results.append(metrics.groupby(variable, restore_coord_dims=True).map(compute_all_scores))
    results = xr.concat(results, pd.Index([product.name for product in snow_cover_products], name="product"))
    return results


def line_plot_accuracy_f1_score(analysis: AnalysisContainer, analysis_var: str, ax: Axes):
    metrics_datasets = [
        open_reduced_dataset_for_plot(
            product=prod,
            analysis_folder=analysis.analysis_folder,
            analysis_type="confusion_table",
            winter_year=analysis.winter_year,
            grid=analysis.grid,
        )
        for prod in analysis.products
    ]

    new_metrics_datasets = []
    if "Ref FSC [%]" in metrics_datasets[0].sizes.keys():
        for md in metrics_datasets:
            fractions = (
                md.sel({"Ref FSC [%]": ["[1-25]", "[26-50]", "[51-75]", "[76-99]"]})
                .sum(dim="Ref FSC [%]")
                .assign_coords({"Ref FSC [%]": "[1-99]"})
            )
            new_metrics_datasets.append(
                xr.concat(
                    [md.sel({"Ref FSC [%]": "0"}), fractions, md.sel({"Ref FSC [%]": "100"})],
                    dim="Ref FSC [%]",
                    coords="minimal",
                    compat="override",
                )
            )

    skill_scores = compute_skill_scores_for_parameter(
        snow_cover_products=analysis.products, metrics_datasets=new_metrics_datasets, variable=analysis_var
    )
    skill_scores = skill_scores.where(skill_scores != 0, np.nan)
    x_coords_conf = new_metrics_datasets[0].coords[analysis_var].values
    skill_scores = skill_scores.sel({analysis_var: x_coords_conf})
    for prod in analysis.products:
        ax.plot(
            x_coords_conf,
            skill_scores.sel(product=prod.name).data_vars["accuracy"],
            "-o",
            color=prod.plot_color,
            markersize=5,
            lw=2,
        )
        ax.plot(
            x_coords_conf,
            skill_scores.sel(product=prod.name).data_vars["f1_score"],
            "--^",
            color=prod.plot_color,
            markersize=6,
            lw=3,
        )
    ax.legend(
        [Line2D([0], [0], linestyle="-", color="gray"), Line2D([0], [0], linestyle="--", color="gray")],
        ["Accuracy", "F1 score"],
    )
    ax.set_ylim(0.75, 1)
    ax.set_xlim(-0.5, skill_scores.sizes[analysis_var] - 0.5)
    ax.set_ylabel("Score[-]")
    # ax.set_xlabel(analysis_var)
    ax.grid(True)


def barplot_total_count(analysis: AnalysisContainer, analysis_var: str, ax: Axes):
    metrics_data_arrays = []
    for prod in analysis.products:
        metrics_ds = open_reduced_dataset_for_plot(
            product=prod,
            analysis_folder=analysis.analysis_folder,
            analysis_type="confusion_table",
            winter_year=analysis.winter_year,
            grid=analysis.grid,
        )
        metrics_ds = metrics_ds.sum(dim=[d for d in metrics_ds.sizes.keys() if d != analysis_var])
        metrics_data_arrays.append(
            metrics_ds.data_vars["true_positive"]
            + metrics_ds.data_vars["true_negative"]
            + metrics_ds.data_vars["false_positive"]
            + metrics_ds.data_vars["false_negative"]
        )

    total_count_ds = xr.concat(
        metrics_data_arrays, dim=xr.DataArray([prod.name for prod in analysis.products], dims="product")
    )
    colors = [prod.plot_color for prod in analysis.products]
    ax = sns.barplot(
        xr.Dataset({"total_count": total_count_ds}).to_dataframe(),
        x=analysis_var,
        y="total_count",
        hue="product",
        width=0.1 * total_count_ds.sizes[analysis_var],
        palette=colors,
        ax=ax,
    )

    ax.set_yscale("log")
    ax.set_ylim(bottom=1e4, top=1e7)
    ax.set_ylabel("# of match-ups [-]")
    ax.get_legend().remove()
    ax.spines[["top", "right"]].set_visible(False)


def compute_n_pixels_results_df(
    snow_cover_products: List[SnowCoverProduct], metric_datasets: List[xr.Dataset]
) -> pd.DataFrame:
    results = []
    for metrics_ds in metric_datasets:
        total_number_of_pixels = (
            metrics_ds["true_positive"].sum()
            + metrics_ds["true_negative"].sum()
            + metrics_ds["false_negative"].sum()
            + metrics_ds["false_positive"].sum()
        )
        # snow_metrics_ds = metrics_ds.sel(ref_bins=slice(25,100))
        # snow_pixels = snow_metrics_ds['true_positive'].sum() + snow_metrics_ds['false_negative'].sum()+ snow_metrics_ds['true_negative'].sum()+ snow_metrics_ds['false_positive'].sum()
        snow_pixels = metrics_ds["true_positive"].sum() + metrics_ds["false_negative"].sum()
        out_dataset = xr.Dataset({"n_tot_pixels": total_number_of_pixels, "n_snow_pixels": snow_pixels})
        results.append(out_dataset)
    results = xr.concat(
        results, dim=xr.DataArray([prod.plot_name for prod in snow_cover_products], dims="product")
    ).to_dataframe()

    results = results.reset_index(["product"])
    return pd.DataFrame(results)
