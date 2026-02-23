from typing import List

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import ticker
from matplotlib.axes import Axes

from postprocess.error_distribution import postprocess_uncertainty_analysis
from postprocess.general_purpose import AnalysisContainer, open_reduced_dataset
from products.plot_settings import MF_ORIG_VAR_NAME, MF_SYNOPSIS_VAR_NAME


def classes_bar_distribution(
    year_stats_data_array: xr.DataArray, classes_to_plot: List[str] | str = "all", ax: Axes | None = None
) -> None:
    year_data_frame = year_stats_data_array.to_pandas()
    if classes_to_plot == "all":
        classes_to_plot = year_data_frame.columns

    year_data_frame.index = year_data_frame.index.strftime("%B")
    year_data_frame[classes_to_plot].plot.bar(ax=ax)
    ax.legend(bbox_to_anchor=(1, 1))
    ax.grid(True, axis="y")


def plot_multiple_stacked(data: List[pd.Series], ax: Axes, stacked: List[bool], labels: List[str], colors: List[str]):
    if len(data) != len(stacked):
        raise ValueError
    # Bar width and x positions
    n_bars = len(data) - np.array(stacked).sum()
    bar_width = 0.75 / n_bars
    x = range(len(data[0]))

    for idx, datum in enumerate(data):
        # Plot stacked bar (sum of Column1 and Column2)

        if not stacked[idx]:
            x = x + bar_width

        if stacked[idx]:
            ax.bar(x, datum, bottom=data[idx - 1], width=bar_width, label=labels[idx], color=colors[idx])
        else:
            ax.bar(x, datum, width=bar_width, label=labels[idx], color=colors[idx])

    ax.set_xticks(np.arange(0.5, len(data[0]) + 0.5))


def plot_annual_daily_cross_sce(area_stats: xr.Dataset, ax: Axes, mode: str = "binary"):
    area_stats["meteofrance"].sel(class_name="no_forest_" + mode).plot(ax=ax, label="Météo-France no_forest")
    mf_total_snow = area_stats["meteofrance"].sel(class_name="no_forest_" + mode) + area_stats["meteofrance"].sel(
        class_name="forest_" + mode
    )
    mf_total_snow.plot(ax=ax, label="Météo-France snow cover + forest with snow")

    area_stats["nasa"].sel(class_name="no_forest_" + mode).plot(ax=ax, label="nasa no_forest")
    nasa_total_snow = area_stats["nasa"].sel(class_name="no_forest_" + mode) + area_stats["nasa"].sel(
        class_name="forest_" + mode
    )
    nasa_total_snow.plot(ax=ax, label="NASA snow cover + forest with snow")

    ax.legend()
    ax.grid(True, axis="y")
    ax.plot()


def plot_annual_area_lines(analysis: AnalysisContainer, classes: List[str], axes: List[Axes]):
    class_titles = {"snow_cover": "Snow cover", "clouds": "Clouds"}

    metrics_dataset_completeness_0 = open_reduced_dataset(
        product=analysis.products[0],
        analysis_folder=analysis.analysis_folder,
        analysis_type="completeness",
        winter_year=analysis.winter_year,
        grid=analysis.grid,
    )
    common_days = metrics_dataset_completeness_0.coords["time"]

    for prod in analysis.products[1:]:
        metrics_dataset_completeness = open_reduced_dataset(
            product=prod,
            analysis_folder=analysis.analysis_folder,
            analysis_type="completeness",
            winter_year=analysis.winter_year,
            grid=analysis.grid,
        )
        common_days = np.intersect1d(common_days, metrics_dataset_completeness.coords["time"])
    # class_ylim_top={'snow_cover': 2e4, 'clouds': 6e4}
    for product in analysis.products:
        metrics_dataset_completeness = open_reduced_dataset(
            product=product,
            analysis_folder=analysis.analysis_folder,
            analysis_type="completeness",
            winter_year=analysis.winter_year,
            grid=analysis.grid,
        )
        metrics_dataset_completeness = metrics_dataset_completeness.set_xindex("altitude_min")

        metrics_dataset_completeness.sel(time=common_days, altitude_min=slice(900, None))

        product_monthly_averages = (
            metrics_dataset_completeness.resample({"time": "1ME"}).mean(dim="time").data_vars["surface"] * 1e-6
        )
        product_monthly_averages = product_monthly_averages.sum(
            dim=[d for d in product_monthly_averages.dims if d != "time" and d != "class_name"]
        )

        time_ax = np.arange(product_monthly_averages.sizes["time"])

        for i, area_class in enumerate(classes):
            annual_surface = product_monthly_averages.sel(class_name=area_class)
            axes[i].plot(time_ax, annual_surface, ".-", color=product.plot_color, lw=2)
            axes[i].set_ylabel("Area [km²]")
            axes[i].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1e}"))
            axes[i].set_title(class_titles[area_class])

    [ax.set_ylim(bottom=0) for ax in axes]


def plot_annual_uncertainty_score_lines(analysis: AnalysisContainer, scores: List[str], axes: List[Axes]):
    score_labels = {"bias": "Bias [% FSC]", "rmse": "RMSE [% FSC]", "unbiased_rmse": "unbiased RMSE [% FSC]"}
    score_titles = {"bias": "Bias", "rmse": "RMSE", "unbiased_rmse": "unbiased RMSE [% FSC]"}
    score_ylims = {"bias": (-6, 6), "rmse": (0, 25), "unbiased_rmse": (0, 25)}

    metrics_dataset_uncertainty_0 = open_reduced_dataset(
        product=analysis.products[0],
        analysis_folder=analysis.analysis_folder,
        analysis_type="uncertainty",
        winter_year=analysis.winter_year,
        grid=analysis.grid,
    )
    common_days = metrics_dataset_uncertainty_0.coords["time"]
    for prod in analysis.products[1:]:
        metrics_dataset_uncertainty = open_reduced_dataset(
            product=prod,
            analysis_folder=analysis.analysis_folder,
            analysis_type="uncertainty",
            winter_year=analysis.winter_year,
            grid=analysis.grid,
        )
        common_days = np.intersect1d(common_days, metrics_dataset_uncertainty.coords["time"])

    for product in analysis.products:
        metrics_dataset_uncertainty = open_reduced_dataset(
            product=product,
            analysis_folder=analysis.analysis_folder,
            analysis_type="uncertainty",
            winter_year=analysis.winter_year,
            grid=analysis.grid,
        )
        metrics_dataset_uncertainty = metrics_dataset_uncertainty.set_xindex("altitude_min")
        metrics_dataset_uncertainty = (
            metrics_dataset_uncertainty.sel(time=common_days, altitude_min=slice(900, None)).resample({"time": "1ME"}).sum()
        )

        metrics_dataset_uncertainty = postprocess_uncertainty_analysis(
            snow_cover_products=[product], metrics_datasets=[metrics_dataset_uncertainty], analysis_var="time"
        )
        time_ax = np.arange(metrics_dataset_uncertainty.sizes["time"])
        for i, score in enumerate(scores):
            axes[i].plot(
                time_ax,
                metrics_dataset_uncertainty.sel(product=product.name).data_vars[score],
                "^-",
                color=product.plot_color,
                lw=2,
            )
            axes[i].set_ylabel(score_labels[score])
            axes[i].set_ylim(score_ylims[score][0], score_ylims[score][1])
            axes[i].set_title(score_titles[score])
            axes[i].set_xticks(np.arange(metrics_dataset_uncertainty.sizes["time"]))
            axes[i].set_xticklabels(metrics_dataset_uncertainty.coords["time"].to_dataframe().index.strftime("%B"))


def annual_area_fancy_plot(analysis: AnalysisContainer, classes: List[str], scores: List[str], axes: List[Axes]):
    n_classes = len(classes)
    [ax.grid() for ax in axes]

    plot_annual_area_lines(analysis=analysis, classes=classes, axes=axes[:n_classes])
    plot_annual_uncertainty_score_lines(analysis=analysis, scores=scores, axes=axes[n_classes:])


FOREST_WITH_SNOW_COLORS = {MF_ORIG_VAR_NAME: "darkgreen", MF_SYNOPSIS_VAR_NAME: "forestgreen"}
