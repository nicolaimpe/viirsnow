from typing import Dict

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pandas.io.formats.style import Styler
from xarray.groupers import BinGrouper

from postprocess.general_purpose import fancy_table
from products.plot_settings import PRODUCT_PLOT_COLORS, PRODUCT_PLOT_NAMES
from reductions.statistics_base import EvaluationVsHighResBase


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
    uncertainty_datasets: Dict[str, xr.Dataset], analysis_var: str | BinGrouper
) -> xr.Dataset:
    reduced_datasets = []
    for dataset in uncertainty_datasets.values():
        reduced_datasets.append(dataset.groupby(analysis_var).map(histograms_to_biais_rmse))
    concatenated = xr.concat(reduced_datasets, pd.Index(list(uncertainty_datasets.keys()), name="product"), coords="minimal")
    return concatenated


def barplots(postprocessed_data_array: xr.DataArray, title: str, ax: Axes):
    plot_dataframe_dict = {}
    colors = []
    legend = []
    for product in postprocessed_data_array.coords["product"].values:
        plot_dataframe_dict.update({product: postprocessed_data_array.sel(product=product).to_pandas()})
        colors.append(PRODUCT_PLOT_COLORS[product])
        legend.append(PRODUCT_PLOT_NAMES[product])
    plot_dataframe = pd.DataFrame(plot_dataframe_dict)
    if "month" in title:
        plot_dataframe.index = plot_dataframe.index.strftime("%B")
    if "aspect" in title:
        # That's because to_pandas() reorders the aspect labels alphabetically
        plot_dataframe = plot_dataframe.reindex(index=EvaluationVsHighResBase.aspect_bins().labels)

    plot_dataframe.plot.bar(figsize=(12, 3), color=colors, width=0.6, title=title, ax=ax)
    ax.legend(legend)


def biais_barplots(postprocessed_dataset: xr.Dataset, analysis_var_plot_name: str, title_complement: str):
    _, ax = plt.subplots()
    ax.set_ylim(-10, 10)
    ax.set_ylabel("biais [%]")
    ax.set_xlabel(analysis_var_plot_name)
    barplots(
        postprocessed_data_array=postprocessed_dataset.data_vars["biais"],
        title=f"Biais vs {analysis_var_plot_name} - {title_complement}",
        ax=ax,
    )
    ax.grid(True, axis="y")


def rmse_barplots(postprocessed_dataset: xr.Dataset, analysis_var_plot_name: str, title_complement: str):
    _, ax = plt.subplots()
    ax.set_ylim(0, 30)
    ax.set_ylabel("rmse [%]")
    ax.set_xlabel(analysis_var_plot_name)
    barplots(
        postprocessed_data_array=postprocessed_dataset.data_vars["rmse"],
        title=f"RMSE vs {analysis_var_plot_name} - {title_complement}",
        ax=ax,
    )
    ax.grid(True, axis="y")
    ax.set_ylabel("RMSE [%]")


def unbiaised_rmse_barplots(postprocessed_dataset: xr.Dataset, analysis_var_plot_name: str, title_complement: str):
    _, ax = plt.subplots()
    ax.set_ylim(0, 30)
    ax.set_ylabel("unbiaised_rmse [%]")
    ax.set_xlabel(analysis_var_plot_name)
    barplots(
        postprocessed_data_array=postprocessed_dataset.data_vars["unbiaised_rmse"],
        title=f"Unbiaised RMSE vs {analysis_var_plot_name} - {title_complement}",
        ax=ax,
    )
    ax.grid(True, axis="y")


def histograms_to_distribution(metrics_ds: xr.Dataset) -> npt.NDArray:
    all_dims = list(metrics_ds.sizes.keys())
    all_dims.remove("biais_bins")
    metrics_squeezed = metrics_ds.sum(dim=all_dims)
    distribution = np.repeat(metrics_ds.coords["biais_bins"].values, metrics_squeezed["n_occurrences"].values.astype(np.int64))
    return distribution


def semidistributed_geometry_plot(
    metrics_dict: Dict[str, xr.Dataset],
    variable_to_plot: str,
    title_complement: str | None = None,
    altitude_ticks: npt.NDArray | None = None,
):
    slope_titles = ["< 10", "10-30", "30-50"]
    if altitude_ticks is None:
        altitude_ticks = np.array([0, *metrics_dict[list(metrics_dict.keys())[0]].coords["altitude_bins"].values])
    for product_name, dataset in metrics_dict.items():
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
                cmap="coolwarm",
                vmin=-30,
                vmax=30,
            )

            fig.colorbar(im, ax=ax[i], orientation="horizontal", label=variable_to_plot, fraction=0.05, pad=0.1)
            fig.suptitle(f"{PRODUCT_PLOT_NAMES[product_name]} - {title_complement}")


def raw_error_boxplots(metrics_dict: Dict[str, xr.Dataset], analysis_var: str, ax: Axes):
    ticks = np.arange(len(list(metrics_dict.values())[0].coords[analysis_var].values))
    for product_name in metrics_dict:
        error_distributions = []

        for value in metrics_dict[product_name].coords[analysis_var].values:
            prod_selected_metrics = metrics_dict[product_name].sel({analysis_var: value})
            error_distributions.append(histograms_to_distribution(prod_selected_metrics))

        product_boxplot = ax.boxplot(
            error_distributions,
            positions=ticks,
            widths=0.2,
            showfliers=False,
            patch_artist=True,
            label=PRODUCT_PLOT_NAMES[product_name],
        )

        for patch in product_boxplot["boxes"]:
            patch.set_facecolor(PRODUCT_PLOT_COLORS[product_name])
        ticks = ticks + 0.2

    ticks = np.arange(len(list(metrics_dict.values())[0].coords[analysis_var].values))
    ax.set_xticks(ticks + (len(metrics_dict) - 1) * 0.2 / 2)
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


def double_variable_barplots(analyses_dict: Dict[str, xr.Dataset], var1: str, var2: str):
    reduced_ds = postprocess_uncertainty_analysis(uncertainty_datasets=analyses_dict, analysis_var=[var1, var2])
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
        palette=list(PRODUCT_PLOT_COLORS.values()),
    )
    plot_rmse = sns.catplot(
        reduced_df,
        x="unbiaised_rmse",
        y=var2,
        hue="product",
        col=var1,
        kind="bar",
        col_wrap=2,
        orient="h",
        palette=list(PRODUCT_PLOT_COLORS.values()),
    )
    plot_biais.figure.suptitle(f"Bias vs aspect - {title} - {str(wy)}", fontsize=11, fontweight="bold")
    plot_biais.figure.subplots_adjust(top=0.85)
    plot_rmse.figure.suptitle(f"Unbiased RMSE vs aspect - {title} - {str(wy)}", fontsize=11, fontweight="bold")
    plot_rmse.figure.subplots_adjust(top=0.85)


if __name__ == "__main__":
    from postprocess.general_purpose import sel_evaluation_domain
    from products.plot_settings import MF_NO_CC_MASK_VAR_NAME, MF_ORIG_VAR_NAME, MF_REFL_SCREEN_VAR_NAME, MF_SYNOPSIS_VAR_NAME
    from reductions.statistics_base import EvaluationVsHighResBase
    from winter_year import WinterYear

    wy = WinterYear(2023, 2024)
    analysis_type = "uncertainty"
    analysis_folder = f"/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_5/analyses/{analysis_type}"
    analyses_dict = {
        # MF_ORIG_VAR_NAME: xr.open_dataset(
        #     f"{analysis_folder}/uncertainty_WY_2023_2024_meteofrance_orig_fsc_vs_s2_theia_sca_fsc_375m.nc",
        #     decode_cf=True,
        # ),
        MF_SYNOPSIS_VAR_NAME: xr.open_dataset(
            f"{analysis_folder}/uncertainty_WY_2023_2024_meteofrance_synopsis_fsc_vs_s2_theia_sca_fsc_375m.nc",
            decode_cf=True,
        ),
        MF_NO_CC_MASK_VAR_NAME: xr.open_dataset(
            f"{analysis_folder}/uncertainty_WY_2023_2024_meteofrance_no_cc_mask_fsc_vs_s2_theia_sca_fsc_375m.nc",
            decode_cf=True,
        ),
        MF_REFL_SCREEN_VAR_NAME: xr.open_dataset(
            f"{analysis_folder}/uncertainty_WY_2023_2024_meteofrance_modified_fsc_vs_s2_theia_sca_fsc_375m.nc",
            decode_cf=True,
        ),
    }

    evaluation_domain = "general"
    selection_dict, title = sel_evaluation_domain(analyses_dict=analyses_dict, evaluation_domain=evaluation_domain)

    ############## Launch analysis

    # Temporal analysis
    biais_barplots(
        postprocess_uncertainty_analysis(selection_dict, analysis_var={"time": EvaluationVsHighResBase.month_bins(wy)}),
        analysis_var_plot_name="time (month)",
        title_complement=f"Biais temporal distribution - {title} - {str(wy)}",
    )
    unbiaised_rmse_barplots(
        postprocess_uncertainty_analysis(selection_dict, analysis_var={"time": EvaluationVsHighResBase.month_bins(wy)}),
        analysis_var_plot_name="time (month)",
        title_complement=f"Unbiaised RMSE temporal distribution - {title} - {str(wy)}",
    )
    rmse_barplots(
        postprocess_uncertainty_analysis(selection_dict, analysis_var={"time": EvaluationVsHighResBase.month_bins(wy)}),
        analysis_var_plot_name="time (month)",
        title_complement=f"Unbiaised RMSE temporal distribution - {title} - {str(wy)}",
    )

    # SAFRAN geometry
    semidistributed_geometry_plot(
        selection_dict, "biais", title_complement=f"Semidistributed geometry biais distribution- {title}"
    )
    semidistributed_geometry_plot(
        selection_dict, "unbiaised_rmse", title_complement=f"Semidistributed geometry unbiaised RMSE distribution - {title}"
    )

    # Barplots aspect
    double_variable_barplots(selection_dict, "forest_mask_bins", "aspect_bins")

    # Boxplots vza
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(f"Error distribution vs VZA - {title} - {str(wy)}")
    ax.set_xticklabels(["0-15", "15-30", "30-45", "45-60", ">60"])
    ax.set_xlabel("Sensor zenith angle [°]")
    ax.set_ylabel("FSC [%]")
    ax.set_ylim(-60, 60)

    sel_vza = selection_dict.copy()
    # sel_vza.pop("nasa_l3")
    sel_vza = {k: v.sel(sensor_zenith_bins=slice(0, 75)) for k, v in sel_vza.items()}

    raw_error_boxplots(metrics_dict=sel_vza, analysis_var="sensor_zenith_bins", ax=ax)

    # Boxplots slope
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(f"Error distribution vs slope - {title} - {str(wy)}")
    ax.set_xticklabels(["0-10", "10-30", "30-50"])
    ax.set_xlabel("Slope [°]")
    ax.set_ylabel("FSC [%]")
    ax.set_ylim(-60, 60)
    ax.plot()
    selection_dict = {k: v.sel(slope_bins=slice(0, 60)) for k, v in selection_dict.items()}
    raw_error_boxplots(metrics_dict=selection_dict, analysis_var="slope_bins", ax=ax)

    reduced_datasets = []
    for dataset in selection_dict.values():
        reduced_datasets.append(histograms_to_biais_rmse(dataset))
    concatenated = xr.concat(reduced_datasets, pd.Index(list(selection_dict.keys()), name="product"), coords="minimal")
    reduced_df = concatenated.to_dataframe().reset_index("product")
    print(reduced_df.round(decimals=2))

    plt.show()
