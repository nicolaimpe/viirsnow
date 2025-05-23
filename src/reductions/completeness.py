import abc
import copy
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from logger_setup import default_logger as logger
from products.classes import METEOFRANCE_CLASSES, NASA_CLASSES, NODATA_NASA_CLASSES, S2_CLASSES


def mask_of_pixels_of(value: int, data_array: xr.DataArray) -> xr.DataArray:
    return data_array == value


def mask_of_pixels_in_range(range: range, data_array: xr.DataArray) -> xr.DataArray:
    return (data_array >= range[0]) * (data_array <= range[-1])


def compute_percentage_of_mask(mask: xr.DataArray, n_pixels_tot: int) -> float:
    return mask.sum().values / n_pixels_tot * 100


def compute_area_of_class_mask(mask: xr.DataArray) -> float:
    gsd = mask.rio.resolution()
    return mask.sum().values * np.abs(math.prod(gsd))


class SnowCoverProductCompleteness:
    def __init__(
        self,
        classes: Dict[str, Tuple[int, ...] | range],
        nodata_mapping: Tuple[str, ...] | None = None,
    ) -> None:
        self.classes = copy.deepcopy(classes)
        if nodata_mapping is not None:
            self.setup_nodata_classes(nodata_mapping=nodata_mapping)

    @property
    def max_fsc(self) -> int:
        return self.classes["snow_cover"][-1]

    @property
    def max_value(self) -> int:
        values = []
        for value in METEOFRANCE_CLASSES.values():
            if type(value) is not range:
                values.append(value[0])
            else:
                values.append(np.max(value))
        return int(np.max(values))

    def setup_nodata_classes(self, nodata_mapping: Tuple[str, ...]):
        if nodata_mapping is not None:
            new_nodata_values = ()
            if "nodata" in self.classes:
                new_nodata_values += self.classes["nodata"]

            for class_to_exclude_name in nodata_mapping:
                for value_to_exclude in self.classes[class_to_exclude_name]:
                    new_nodata_values += (value_to_exclude,)
                self.classes.pop(class_to_exclude_name)
            self.classes["nodata"] = new_nodata_values

    def mask_of_class(self, class_name: str, data_array: xr.DataArray) -> xr.DataArray:
        if type(self.classes[class_name]) is range:
            return mask_of_pixels_in_range(self.classes[class_name], data_array)
        else:
            summed_mask = xr.zeros_like(data_array, dtype=bool)
            for value in self.classes[class_name]:
                summed_mask += mask_of_pixels_of(value, data_array)
            return summed_mask

    @abc.abstractmethod
    def total_snow_mask(self, data_array: xr.DataArray) -> xr.DataArray:
        pass

    @abc.abstractmethod
    def total_no_snow_mask(self, data_array: xr.DataArray) -> xr.DataArray:
        pass

    def quantitative_mask(self, data_array: xr.DataArray):
        return mask_of_pixels_in_range(range(self.classes["no_snow"][0], self.classes["snow_cover"][-1] + 1), data_array)

    def area_of_class(self, class_name: str, data_array: xr.DataArray) -> float:
        return compute_area_of_class_mask(self.mask_of_class(class_name=class_name, data_array=data_array))

    def snow_area(self, snow_cover_data_array: xr.DataArray, consider_fraction: bool = True) -> float:
        snow_mask = self.mask_of_class("snow_cover", snow_cover_data_array)
        if consider_fraction:
            snow_cover_data_array = snow_cover_data_array / self.classes["snow_cover"][-1]
            snow_cover_extent = compute_area_of_class_mask(snow_cover_data_array.where(snow_mask))
        else:
            snow_cover_extent = compute_area_of_class_mask(snow_mask)
        return snow_cover_extent

    def count_valid_pixels(self, data_array: xr.DataArray, exclude_nodata: bool = False):
        if exclude_nodata:
            nodata_pixels = self.mask_of_class("nodata", data_array).sum().values
            n_pixels_tot = data_array.count().values - nodata_pixels

        else:
            n_pixels_tot = data_array.count().values
        return n_pixels_tot

    def _all_statistics(self, data_array: xr.DataArray, exclude_nodata: bool = False) -> Dict[str, float]:
        logger.info(f"Processing time: {data_array.coords['time'].values[0].astype('M8[D]').astype('O')}")

        results_coords = xr.Coordinates({"class_name": list(self.classes.keys())})

        results_dataset = xr.Dataset(
            {
                "n_pixels_class": xr.DataArray(np.nan, coords=results_coords, attrs={"units": "-"}),
                "percentage": xr.DataArray(np.nan, coords=results_coords, attrs={"units": "%"}),
                "surface": xr.DataArray(np.nan, coords=results_coords, attrs={"units": "m²"}),
            }
        )
        n_pixels_tot = self.count_valid_pixels(data_array, exclude_nodata=exclude_nodata)

        for class_name in self.classes:
            if class_name == "nodata" and exclude_nodata:
                continue
            class_mask = self.mask_of_class(class_name, data_array)
            results_dataset.data_vars["n_pixels_class"].loc[class_name] = class_mask.sum().values
            results_dataset.data_vars["percentage"].loc[class_name] = compute_percentage_of_mask(class_mask, n_pixels_tot)
            results_dataset.data_vars["surface"].loc[class_name] = compute_area_of_class_mask(class_mask)
        return results_dataset

    def year_temporal_analysis(
        self,
        snow_cover_product_time_series_data_array: xr.DataArray,
        exclude_nodata: bool = False,
        netcdf_export_path: str | None = None,
        period: Tuple[str | None, str | None] | None = None,
    ):
        if period is not None:
            snow_cover_product_time_series_data_array = snow_cover_product_time_series_data_array.sel(time=slice(*period))
        year_results_dataset = snow_cover_product_time_series_data_array.groupby("time").map(
            self._all_statistics, exclude_nodata=exclude_nodata
        )
        if netcdf_export_path is not None:
            year_results_dataset.to_netcdf(Path(netcdf_export_path))


class MeteoFranceSnowCoverProductCompleteness(SnowCoverProductCompleteness):
    def __init__(self) -> None:
        super().__init__(classes=METEOFRANCE_CLASSES, nodata_mapping=None)

    def total_snow_mask(self, data_array: xr.DataArray) -> xr.DataArray:
        snow_meteofrance = self.mask_of_class("snow_cover", data_array) | self.mask_of_class("forest_with_snow", data_array)
        return snow_meteofrance

    def total_no_snow_mask(self, data_array: xr.DataArray) -> xr.DataArray:
        no_snow_meteofrance = (
            self.mask_of_class("no_snow", data_array)
            | self.mask_of_class("forest_without_snow", data_array)
            | self.mask_of_class("water", data_array)
        )
        return no_snow_meteofrance


class NASASnowCoverProductCompleteness(SnowCoverProductCompleteness):
    def __init__(self) -> None:
        super().__init__(classes=NASA_CLASSES, nodata_mapping=NODATA_NASA_CLASSES)

    def total_snow_mask(self, data_array: xr.DataArray) -> xr.DataArray:
        snow_nasa = self.mask_of_class("snow_cover", data_array)
        return snow_nasa

    def total_no_snow_mask(self, data_array: xr.DataArray) -> xr.DataArray:
        no_snow_nasa = self.mask_of_class("no_snow", data_array) | self.mask_of_class("water", data_array)
        return no_snow_nasa


class S2SnowCoverProductCompleteness(SnowCoverProductCompleteness):
    def __init__(self) -> None:
        super().__init__(classes=S2_CLASSES)

    def total_snow_mask(self, data_array: xr.DataArray) -> xr.DataArray:
        snow_s2 = self.mask_of_class("snow_cover", data_array)
        return snow_s2

    def total_no_snow_mask(self, data_array: xr.DataArray) -> xr.DataArray:
        no_snow_s2 = self.mask_of_class("no_snow", data_array)
        return no_snow_s2


if __name__ == "__main__":
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_6"

    evaluation_dict: Dict[str, Dict[str, SnowCoverProductCompleteness]] = {
        # "meteofrance_orig": {"evaluator": MeteoFranceSnowCoverProductCompleteness()},
        # "meteofrance_synopsis": {"evaluator": MeteoFranceSnowCoverProductCompleteness()},
        # "meteofrance_no_cc_mask": {"evaluator": MeteoFranceSnowCoverProductCompleteness(), "config": config},
        # "meteofrance_modified": {"evaluator": MeteoFranceSnowCoverProductCompleteness(), "config": config},
        # "nasa_pseudo_l3": {"evaluator": NASASnowCoverProductCompleteness()},
        # "nasa_l3_snpp": {"evaluator": NASASnowCoverProductCompleteness()},
        # "nasa_l3_jpss1": {"evaluator": NASASnowCoverProductCompleteness()},
        "nasa_l3_multiplatform": {"evaluator": NASASnowCoverProductCompleteness()},
    }

    for product in evaluation_dict:
        logger.info(f"Evaluating product {product}")
        analyzer = evaluation_dict[product]["evaluator"]
        test_series = xr.open_dataset(f"{output_folder}/time_series/WY_2023_2024_{product}.nc").sel(
            time=slice("2023-11", "2024-06")
        )
        analyzer.year_temporal_analysis(
            snow_cover_product_time_series_data_array=test_series["snow_cover_fraction"],
            netcdf_export_path=f"{output_folder}/analyses/completeness/completeness_WY_2023_2024_{product}.nc",
        )
