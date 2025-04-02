from typing import Dict

import pandas as pd
from pandas.io.formats.style import Styler


def fancy_table(
    dataframe_to_print: pd.DataFrame, color_maps: Dict[str, str], vmins: Dict[str, str], vmaxs: Dict[str, str]
) -> Styler:
    # Apply gradient coloring
    styled_df = dataframe_to_print.style

    def highlight_product(val):
        return "background-color: mediumblue; color: white; font-weight: bold; text-align: center"

    styled_df = styled_df.map(highlight_product, subset=["product"])

    for col, cmap in color_maps.items():
        if col in dataframe_to_print.columns:
            styled_df = styled_df.background_gradient(subset=[col], cmap=cmap, vmin=vmins[col], vmax=vmaxs[col])

    styled_df = styled_df.set_properties(**{"text-align": "center"})  # Center-align all text
    styled_df = styled_df.set_table_styles(
        [{"selector": "th", "props": [("background-color", "darkblue"), ("color", "white"), ("font-weight", "bold")]}]
    )
    # Compact DataFrame
    styled_df = styled_df.hide(level=None)
    # Only show 2 digits after comma
    styled_df = styled_df.format(precision=2)
    return styled_df
