from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes

from products.plot_settings import (
    MF_ORIG_VAR_NAME,
    MF_SYNOPSIS_VAR_NAME,
    NASA_L3_JPSS1_VAR_NAME,
    NASA_L3_MULTIPLATFORM_VAR_NAME,
    NASA_L3_SNPP_VAR_NAME,
    PRODUCT_PLOT_COLORS,
    PRODUCT_PLOT_NAMES,
)
from winter_year import WinterYear


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


FOREST_WITH_SNOW_COLORS = {MF_ORIG_VAR_NAME: "darkgreen", MF_SYNOPSIS_VAR_NAME: "forestgreen"}
if __name__ == "__main__":
    working_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_6/analyses/completeness"
    wy = WinterYear(2023, 2024)
    analysis_type = "completeness"
    analysis_folder = f"/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_6/analyses/{analysis_type}"
    analyses_dict = {
        # MF_ORIG_VAR_NAME: xr.open_dataset(f"{analysis_folder}/completeness_WY_2023_2024_meteofrance_orig.nc"),
        # MF_SYNOPSIS_VAR_NAME: xr.open_dataset(f"{analysis_folder}/completeness_WY_2023_2024_meteofrance_synopsis.nc"),
        NASA_L3_SNPP_VAR_NAME: xr.open_dataset(f"{analysis_folder}/completeness_WY_2023_2024_nasa_l3_snpp.nc"),
        NASA_L3_JPSS1_VAR_NAME: xr.open_dataset(f"{analysis_folder}/completeness_WY_2023_2024_nasa_l3_jpss1.nc"),
        NASA_L3_MULTIPLATFORM_VAR_NAME: xr.open_dataset(
            f"{analysis_folder}/completeness_WY_2023_2024_nasa_l3_multiplatform.nc"
        ),
    }

    if len(analyses_dict) > 1:
        common_days = np.intersect1d(*[v.coords["time"] for v in analyses_dict.values()][:2])

    if len(analyses_dict) > 2:
        for v in analyses_dict.values():
            common_days = np.intersect1d(common_days, v.coords["time"])

    surface_averages = []
    stacked_list = []
    label_list = []
    color_list = []
    for product in analyses_dict:
        analyses_dict[product] = analyses_dict[product].sel(time=common_days)
        product_monthly_averages = analyses_dict[product].resample({"time": "1ME"}).mean(dim="time")
        print(product)
        print(product_monthly_averages.sum(dim="time").to_dataframe())
        surface_averages.append(product_monthly_averages.sel(class_name="snow_cover").data_vars["surface"] * 1e-6)
        stacked_list.append(False)
        label_list.append(f"{PRODUCT_PLOT_NAMES[product]} snow")
        color_list.append(PRODUCT_PLOT_COLORS[product])
        if "meteofrance" in product:
            surface_averages.append(product_monthly_averages.sel(class_name="forest_with_snow").data_vars["surface"] * 1e-6)
            stacked_list.append(True)
            label_list.append(f"{PRODUCT_PLOT_NAMES[product]} forest with snow")
            color_list.append(FOREST_WITH_SNOW_COLORS[product])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.grid(axis="y", linewidth=0.4)
    plot_multiple_stacked(
        surface_averages,
        stacked=stacked_list,
        ax=ax,
        labels=label_list,
        colors=color_list,
    )
    ax.set_title(f"Average surface per observation day - {str(wy)}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Surface [km²]")
    ax.set_xticklabels(product_monthly_averages.coords["time"].to_dataframe().index.strftime("%B"))
    ax.legend()

    plt.show()

    # Invalid mask union analysis

    # nasa_vs_mf = xr.open_dataset(
    #     "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_6/analyses/confusion_table/confusion_table_WY_2023_2024_nasa_l3_snpp_vs_meteofrance_synopsis.nc"
    # )

    # # plot_confusion_table(dataset=mf_vs_nasa)

    # def confusion_table_to_surfaces(dataset: xr.Dataset) -> xr.DataArray:
    #     tp = dataset["true_positive"].sum()
    #     fp = dataset["false_positive"].sum()
    #     fn = dataset["false_negative"].sum()
    #     return xr.DataArray(
    #         [tp + fp, tp + fn],
    #         coords={"product": [PRODUCT_PLOT_NAMES[MF_ORIG_VAR_NAME], PRODUCT_PLOT_NAMES[NASA_L3_SNPP_VAR_NAME]]},
    #         dims=("product"),
    #         name="n_pixel",
    #     )

    # reduced = nasa_vs_mf.resample({"time": "1ME"}).map(confusion_table_to_surfaces)

    # sns.catplot(
    #     kind="bar",
    #     data=reduced.to_dataframe() * 375**2 * 1e-6,
    #     x="time",
    #     y="n_pixel",
    #     hue="product",
    #     palette=[PRODUCT_PLOT_COLORS[MF_ORIG_VAR_NAME], PRODUCT_PLOT_COLORS[NASA_L3_SNPP_VAR_NAME]],
    # )
    # plt.title("Surface of union of valid observations")
    # plt.ylabel("Surface [km²]")
    # plt.xlabel("Month")
    # plt.xticks(
    #     ticks=np.arange(len(reduced.coords["time"].to_dataframe().index)),
    #     labels=reduced.coords["time"].to_dataframe().index.strftime("%B"),
    # )
    # plt.grid(axis="y", linewidth=0.4)
    # plt.show()
