from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import xarray as xr
from geospatial_grid.gsgrid import GSGrid
from pandas.io.formats.style import Styler

from products.snow_cover_product import SnowCoverProduct
from reductions.statistics_base import EvaluationVsHighResBase
from winter_year import WinterYear


def fancy_table(
    dataframe_to_print: pd.DataFrame, color_maps: Dict[str, str], vmins: Dict[str, str], vmaxs: Dict[str, str]
) -> Styler:
    # Apply gradient coloring
    styled_df = dataframe_to_print.style

    for col, cmap in color_maps.items():
        if col in dataframe_to_print.columns:
            styled_df = styled_df.background_gradient(subset=[col], cmap=cmap, vmin=vmins[col], vmax=vmaxs[col])

    styled_df = styled_df.set_properties(align="center", **{"text-align": "center"})  # Center-align all text
    styled_df = styled_df.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "lightgrey"),
                    ("color", "black"),
                    ("font-weight", "bold"),
                    ("width", "75px"),
                    ("text-align", "center"),
                    ("text-usetex", True),
                ],
            }
        ]
    )
    # Compact DataFrame
    styled_df = styled_df.hide(level=None)

    def smart_format(x):
        # small or large numbers → scientific
        if type(x) is str:
            return x
        if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 0.001):
            return f"{x:.2e}"
        else:
            return f"{x:.2f}"

    styled_df.format({k: smart_format for k in dataframe_to_print.columns})
    return styled_df


@dataclass
class AnalysisContainer:
    products: List[SnowCoverProduct]
    analysis_folder: str
    winter_year: WinterYear
    grid: GSGrid


def open_reduced_dataset(
    product: SnowCoverProduct, analysis_folder: str, analysis_type: str, winter_year: WinterYear, grid: GSGrid
) -> xr.Dataset:
    return xr.open_dataset(
        f"{analysis_folder}/wy_{winter_year.from_year}_{winter_year.to_year}/{product.prod_id.lower()}_{grid.name.lower()}/analyses/{analysis_type}.nc"
    )


def open_reduced_dataset_for_plot(
    product: SnowCoverProduct, analysis_folder: str, analysis_type: str, winter_year: WinterYear, grid: GSGrid
) -> xr.Dataset:
    dataset = open_reduced_dataset(product, analysis_folder, analysis_type, winter_year, grid)
    # if "sensor_zenith_bins" not in dataset.sizes:
    #     dataset = dataset.expand_dims({"sensor_zenith_bins": 5})

    selection_dict = {}
    coord_dict = {}
    rename_dict = {}

    if "aspect" in dataset.coords:
        rename_dict.update({"aspect": "Aspect"})

    if "slope_bins" in dataset.coords:
        dataset = dataset.set_xindex("slope_max")
        selection_dict.update(slope_max=slice(None, 60))
        coord_dict.update(
            {
                "slope_bins": pd.CategoricalIndex(
                    data=["[0-10]", "[11-30]", "$>$30"], categories=["[0-10]", "[11-30]", "$>$30"], ordered=True
                ),
            },
        )
        rename_dict.update({"slope_bins": "Slope [°]"})

    if "landcover" in dataset.coords:
        rename_dict.update({"landcover": "Landcover"})

    if "sensor_zenith_bins" in dataset.coords:
        dataset = dataset.set_xindex("sensor_zenith_max")
        selection_dict.update(sensor_zenith_max=slice(None, 75))
        coord_dict.update(
            {
                "sensor_zenith_bins": pd.CategoricalIndex(
                    data=["[0-15]", "[15-30]", "[30-45]", "[45-60]", "$>$60"],
                    categories=["[0-15]", "[15-30]", "[30-45]", "[45-60]", "$>$60"],
                    ordered=True,
                ),
            },
        )
        rename_dict.update({"sensor_zenith_bins": "View Zenith Angle [°]"})

    if "ref_fsc_bins" in dataset.coords:
        dataset = dataset.set_xindex("ref_fsc_max")
        selection_dict.update(ref_fsc_max=slice(None, 101))
        if dataset.sizes["ref_fsc_bins"] == 7:
            coord_dict.update(
                {
                    "ref_fsc_bins": pd.CategoricalIndex(
                        data=["0", "[1-25]", "[26-50]", "[51-75]", "[76-99]", "100"],
                        categories=["0", "[1-25]", "[26-50]", "[51-75]", "[76-99]", "100"],
                        ordered=True,
                    ),
                },
            )
        elif dataset.sizes["ref_fsc_bins"] == 4:
            coord_dict.update(
                {
                    "ref_fsc_bins": pd.CategoricalIndex(
                        data=["0%", "[1-99]%", "100%"],
                        categories=["0%", "[1-99]%", "100%"],
                        ordered=True,
                    ),
                },
            )

        else:
            raise NotImplementedError("reference bins range not known")
        rename_dict.update({"ref_fsc_bins": "Ref FSC [%]"})

    dataset = dataset.set_xindex("altitude_min")
    dataset = dataset.sel(
        time=slice(f"{winter_year.from_year}-11", f"{winter_year.to_year}-06"),
        altitude_min=slice(900, None),
        **selection_dict,
    )

    dataset = dataset.assign_coords(coord_dict)

    dataset = dataset.rename(rename_dict)
    return dataset
