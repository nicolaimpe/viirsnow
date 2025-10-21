from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from pandas.io.formats.style import Styler
from xarray.groupers import BinGrouper

from grids import GeoGrid
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


@dataclass
class AnalysisContainer:
    products: List[SnowCoverProduct]
    analysis_folder: str
    winter_year: WinterYear
    grid: GeoGrid


def open_reduced_dataset(
    product: SnowCoverProduct, analysis_folder: str, analysis_type: str, winter_year: WinterYear
) -> xr.Dataset:
    return xr.open_dataset(
        f"{analysis_folder}/analyses/{analysis_type}/{analysis_type}_{winter_year.to_filename_format()}_{product.name}_vs_S2_theia.nc"
    )


def open_reduced_dataset_completeness(
    product: SnowCoverProduct, analysis_folder: str, winter_year: WinterYear, grid: GeoGrid
) -> xr.Dataset:
    return xr.open_dataset(
        f"{analysis_folder}/analyses/completeness/completeness_{winter_year.to_filename_format()}_{product.name}_{grid.name.lower()}.nc"
    )


def open_reduced_dataset_for_plot(
    product: SnowCoverProduct, analysis_folder: str, analysis_type: str, winter_year: WinterYear
) -> xr.Dataset:
    dataset = open_reduced_dataset(product, analysis_folder, analysis_type, winter_year)
    # if "sensor_zenith_bins" not in dataset.sizes:
    #     dataset = dataset.expand_dims({"sensor_zenith_bins": 5})

    selection_dict = {}
    coord_dict = {}
    rename_dict = {}

    if "aspect_bins" in dataset.coords:
        coord_dict.update(
            {
                "aspect_bins": pd.CategoricalIndex(
                    data=EvaluationVsHighResBase.aspect_bins().labels,
                    categories=EvaluationVsHighResBase.aspect_bins().labels,
                    ordered=True,
                )
            },
        )
        rename_dict.update({"aspect_bins": "Aspect"})

    if "slope_bins" in dataset.coords:
        selection_dict.update(slope_bins=slice(None, 60))
        coord_dict.update(
            {
                "slope_bins": pd.CategoricalIndex(
                    data=["[0-10]", "[11-30]", "$>$30"], categories=["[0-10]", "[11-30]", "$>$30"], ordered=True
                ),
            },
        )
        rename_dict.update({"slope_bins": "Slope [°]"})

    if "forest_mask_bins" in dataset.coords:
        coord_dict.update(
            {
                "forest_mask_bins": ["Open", "Forest"],
            },
        )
        rename_dict.update({"forest_mask_bins": "Landcover"})

    if "sensor_zenith_bins" in dataset.coords:
        selection_dict.update(sensor_zenith_bins=slice(None, 75))
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

    if "ref_bins" in dataset.coords:
        selection_dict.update(ref_bins=slice(None, 100))

        if dataset.sizes["ref_bins"] == 7:
            coord_dict.update(
                {
                    "ref_bins": pd.CategoricalIndex(
                        data=["0", "[1-25]", "[26-50]", "[51-75]", "[75-99]", "100"],
                        categories=["0", "[1-25]", "[26-50]", "[51-75]", "[75-99]", "100"],
                        ordered=True,
                    ),
                },
            )
        elif dataset.sizes["ref_bins"] == 4:
            coord_dict.update(
                {
                    "ref_bins": pd.CategoricalIndex(
                        data=["0", "[1-99]", "100"],
                        categories=["0", "[1-99]", "100"],
                        ordered=True,
                    ),
                },
            )

        else:
            raise NotImplementedError("reference bins range not known")
        rename_dict.update({"ref_bins": "Ref FSC [\%]"})

    dataset = dataset.sel(
        time=slice(f"{winter_year.from_year}-11", f"{winter_year.to_year}-06"),
        altitude_bins=slice(900, None),
        **selection_dict,
    )
    dataset = dataset.assign_coords(coord_dict)

    dataset = dataset.rename(rename_dict)
    return dataset
