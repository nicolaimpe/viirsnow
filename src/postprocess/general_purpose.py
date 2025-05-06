from typing import Dict, Tuple

import pandas as pd
import xarray as xr
from pandas.io.formats.style import Styler

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
