from typing import List

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes


def print_table(year_stats_data_array: xr.DataArray, classes_to_print: List[str] | str = "all"):
    year_data_frame = year_stats_data_array.to_pandas()
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.precision = 3
    if classes_to_print == "all":
        classes_to_print = year_stats_data_array.coords["class_name"].values
    print(year_data_frame)


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
