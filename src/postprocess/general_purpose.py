from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from pandas.io.formats.style import Styler
from xarray.groupers import BinGrouper

from products.snow_cover_product import SnowCoverProduct
from reductions.statistics_base import EvaluationVsHighResBase


def fancy_table(
    dataframe_to_print: pd.DataFrame, color_maps: Dict[str, str], vmins: Dict[str, str], vmaxs: Dict[str, str]
) -> Styler:
    # Apply gradient coloring
    styled_df = dataframe_to_print.style

    def highlight_product(val):
        return "background-color: whitesmoke; color: black; font-weight: bold; text-align: center"

    styled_df = styled_df.map(highlight_product, subset=["product"])

    for col, cmap in color_maps.items():
        if col in dataframe_to_print.columns:
            styled_df = styled_df.background_gradient(subset=[col], cmap=cmap, vmin=vmins[col], vmax=vmaxs[col])

    styled_df = styled_df.set_properties(**{"text-align": "center"})  # Center-align all text
    styled_df = styled_df.set_table_styles(
        [{"selector": "th", "props": [("background-color", "lightgrey"), ("color", "black"), ("font-weight", "bold")]}]
    )
    # Compact DataFrame
    styled_df = styled_df.hide(level=None)
    # Only show 2 digits after comma
    styled_df = styled_df.format(precision=2)
    return styled_df


def sel_evaluation_domain(analyses_dict: Dict[str, xr.Dataset], evaluation_domain: str) -> Tuple[Dict[str, xr.Dataset], str]:
    if evaluation_domain == "general":
        title = "November to June > 900 m"
        selection_dict = {
            k: v.sel(time=slice("2023-11", "2024-06"), altitude_bins=slice(900, None)) for k, v in analyses_dict.items()
        }
    elif evaluation_domain == "accumulation":
        title = "Accumulation November to February > 900 m"
        selection_dict = {
            k: v.sel(time=slice("2023-11", "2024-02")).sel(altitude_bins=slice(900, None), drop=True)
            for k, v in analyses_dict.items()
        }
    elif evaluation_domain == "ablation":
        title = "Ablation March to July > 2100 m"
        selection_dict = {
            k: v.sel(time=slice("2024-03", "2024-06")).sel(altitude_bins=slice(1500, None), drop=True)
            for k, v in analyses_dict.items()
        }
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

    return selection_dict, title


def open_reduced_dataset(product: SnowCoverProduct, analysis_folder: str, analysis_type: str) -> xr.Dataset:
    return xr.open_dataset(
        f"{analysis_folder}/analyses/{analysis_type}/{analysis_type}_WY_2023_2024_{product.name}_vs_S2_theia.nc"
    )


def open_reduced_dataset_completeness(product: SnowCoverProduct, analysis_folder: str, analysis_type: str) -> xr.Dataset:
    return xr.open_dataset(f"{analysis_folder}/analyses/{analysis_type}/{analysis_type}_WY_2023_2024_{product.name}.nc")


def open_reduced_dataset_for_plot(product: SnowCoverProduct, analysis_folder: str, analysis_type: str) -> xr.Dataset:
    dataset = open_reduced_dataset(product, analysis_folder, analysis_type)
    if "sensor_zenith_bins" not in dataset.sizes:
        dataset = dataset.expand_dims({"sensor_zenith_bins": 5})
    dataset = dataset.sel(
        time=slice("2023-11", "2024-06"),
        altitude_bins=slice(900, None),
        ref_bins=slice(None, 100),
        slope_bins=slice(None, 60),
        sensor_zenith_bins=slice(None, 80),
    )
    dataset = dataset.assign_coords(
        {
            "aspect_bins": pd.CategoricalIndex(
                data=EvaluationVsHighResBase.aspect_bins().labels,
                categories=EvaluationVsHighResBase.aspect_bins().labels,
                ordered=True,
            ),
            "forest_mask_bins": ["Open", "Forest"],
            "slope_bins": np.array(["[0-10]", "[11-30]", "\>30"], dtype=str),
            "sensor_zenith_bins": np.array(["[0-15]", "[15-30]", "[30-45]", "[45-60]", "\>60"], dtype=str),
            "ref_bins": pd.CategoricalIndex(
                data=["0", "[1-25]", "[26-50]", "[51-75]", "[75-99]", "100"],
                categories=["0", "[1-25]", "[26-50]", "[51-75]", "[75-99]", "100"],
                ordered=True,
            ),
        }
    )

    dataset = dataset.rename(
        {
            "aspect_bins": "Aspect",
            "forest_mask_bins": "Landcover",
            "slope_bins": "Slope [°]",
            "sensor_zenith_bins": "View Zenith Angle [°]",
            "ref_bins": "Ref FSC [\%]",
        }
    )
    return dataset
