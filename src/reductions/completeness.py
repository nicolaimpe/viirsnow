import abc
import copy
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from logger_setup import default_logger as logger
from products.classes import (METEOFRANCE_ARCHIVE_CLASSES,
                              METEOFRANCE_COMPOSITE_CLASSES, NASA_CLASSES,
                              NODATA_NASA_CLASSES, S2_CLASSES)
from reductions.semidistributed import MountainParametrization, MountainParams


def mask_of_pixels_of(value: int, data_array: xr.DataArray) -> xr.DataArray:
    return data_array == value


def mask_of_pixels_in_range(range: range, data_array: xr.DataArray) -> xr.DataArray:
    return (data_array >= range[0]) * (data_array <= range[-1])


def compute_percentage_of_mask(mask: xr.DataArray, n_pixels_tot: int) -> float:
    return mask.sum().values / n_pixels_tot * 100


def compute_area_of_class_mask(mask: xr.DataArray) -> float:
    gsd = mask.rio.resolution()
    return mask.sum().values * np.abs(math.prod(gsd))


def compute_area_of_class_precise(input_data_array: xr.DataArray, class_name: str='clouds'):

    class_mask = analyzer.mask_of_class(class_name, input_data_array)

    polygons = []
    for geom, val in shapes(class_mask.astype('u1').values, transform=input_data_array.rio.transform()):
        if val:                                  # only take mask pixels (val == 1)
            polygons.append(shape(geom))

    # merge into one geometry if many (optional)

    union_geom = unary_union(polygons)           # geometry in UTM coords

    
    # transform to lon/lat (EPSG:4326)
    transformer = Transformer.from_crs(input_data_array.rio.crs, "EPSG:4326", always_xy=True)
    def proj_to_lonlat(x, y, z=None):
        return transformer.transform(x, y)

    lonlat_geom = shapely_transform(proj_to_lonlat, union_geom)

    # compute ellipsoidal area (pyproj.Geod)
    geod = Geod(ellps='WGS84')
    # pyproj.Geod.geometry_area_perimeter returns (area, perimeter) for shapely geometries
    area, _ = geod.geometry_area_perimeter(lonlat_geom)
    return xr.DataArray(abs(area))


class SnowCoverProductCompleteness:
    def __init__(
        self,
        classes: Dict[str, int | range],
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
        for value in METEOFRANCE_ARCHIVE_CLASSES.values():
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

    def _all_statistics(self, dataset: xr.Dataset, exclude_nodata: bool = False) -> Dict[str, float]:
        results_coords = xr.Coordinates({"class_name": list(self.classes.keys())})

        results_dataset = xr.Dataset(
            {
                "n_pixels_class": xr.DataArray(np.nan, coords=results_coords, attrs={"units": "-"}),
                "percentage": xr.DataArray(np.nan, coords=results_coords, attrs={"units": "%"}),
                "surface": xr.DataArray(np.nan, coords=results_coords, attrs={"units": "m²"}),
            }
        )
        n_pixels_tot = self.count_valid_pixels(dataset.data_vars["snow_cover_fraction"], exclude_nodata=exclude_nodata)

        for class_name in self.classes:
            if class_name == "nodata" and exclude_nodata:
                continue
            class_mask = self.mask_of_class(class_name, dataset.data_vars["snow_cover_fraction"])
            results_dataset.data_vars["n_pixels_class"].loc[class_name] = class_mask.sum().values
            results_dataset.data_vars["percentage"].loc[class_name] = compute_percentage_of_mask(class_mask, n_pixels_tot)
            results_dataset.data_vars["surface"].loc[class_name] = compute_area_of_class_mask(class_mask)
        return results_dataset

    def day_statistics_with_params(
        self, dataset: xr.Dataset, analysis_bin_dict: Dict[str, xr.groupers.Grouper], exclude_nodata: bool = False
    ):
        logger.info(f"Processing time: {dataset.coords['time'].values[0].astype('M8[D]').astype('O')}")
        return dataset.groupby(analysis_bin_dict).map(self._all_statistics, exclude_nodata=exclude_nodata)

    def day_statistics_without_params(self, dataset: xr.Dataset, exclude_nodata: bool = False):
        logger.info(f"Processing time: {dataset.coords['time'].values[0].astype('M8[D]').astype('O')}")
        return self._all_statistics(data_array=dataset, exclude_nodata=exclude_nodata)

    def year_temporal_analysis(
        self,
        snow_cover_product_time_series: xr.Dataset,
        exclude_nodata: bool = False,
        netcdf_export_path: str | None = None,
        period: slice | None = None,
        config: MountainParams | None = None,
    ):
        if period is not None:
            snow_cover_product_time_series = snow_cover_product_time_series.sel(time=period)
        if config is not None:
            snow_cover_product_time_series, analysis_bin_dict = MountainParametrization().semidistributed_parametrization(
                dataset=snow_cover_product_time_series, config=config
            )
            year_results_dataset = snow_cover_product_time_series.groupby("time").map(
                self.day_statistics_with_params, analysis_bin_dict=analysis_bin_dict, exclude_nodata=exclude_nodata
            )
        else:
            year_results_dataset = snow_cover_product_time_series.groupby("time").map(
                self.day_statistics_without_params, exclude_nodata=exclude_nodata
            )

        if netcdf_export_path is not None:
            year_results_dataset.to_netcdf(Path(netcdf_export_path))


class MeteoFranceArchiveSnowCoverProductCompleteness(SnowCoverProductCompleteness):
    def __init__(self) -> None:
        super().__init__(classes=METEOFRANCE_ARCHIVE_CLASSES, nodata_mapping=None)

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

class MeteoFranceCompositeSnowCoverProductCompleteness(SnowCoverProductCompleteness):
    def __init__(self) -> None:
        super().__init__(classes=METEP, nodata_mapping=None)

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
