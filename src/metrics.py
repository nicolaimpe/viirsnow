from datetime import datetime
import math
from typing import Dict, List, Tuple

import numpy as np
import rasterio
import rasterio.warp
import rioxarray
import xarray as xr
from logger_setup import default_logger as logger
from pathlib import Path

import copy
from products import METEOFRANCE_CLASSES, NASA_CLASSES, NODATA_NASA_CLASSES
from geotools import to_rioxarray
from winter_year import WinterYear, Year
from pyproj import CRS
from grids import DEFAULT_CRS_PROJ


def mask_of_pixels_of(value: int, data_array: xr.DataArray) -> xr.DataArray:
    return data_array == value


def mask_of_pixels_in_range(range: range, data_array: xr.DataArray) -> xr.DataArray:
    return (data_array >= range[0]) * (data_array <= range[-1])


class SnowCoverProductCompleteness:
    def __init__(
        self,
        classes: Dict[str, Tuple[int, ...] | range],
        nodata_mapping: Tuple[str, ...] | None = None,
        mask_file: str | None = None,
        class_percentage_distribution: bool = True,
        class_cover_area: bool = True,
    ) -> None:
        self.setup_roi_mask(mask_file=mask_file)
        self.classes = copy.deepcopy(classes)
        if nodata_mapping is not None:
            self.setup_nodata_classes(nodata_mapping=nodata_mapping)
        self.class_percentage_distribution = class_percentage_distribution
        self.class_cover_area = class_cover_area
        if not self.class_percentage_distribution and not self.class_cover_area:
            raise ValueError("Specify at least one between class_percentage_distribution and class_cover_area arguments")
        self.check_projection()

    def check_projection(self):
        if self.class_cover_area:
            if self.roi_mask and not self.roi_mask.crs.is_projected:
                self.roi_mask, _ = rasterio.warp.reproject(
                    self.roi_mask.read(1),
                    src_transform=self.roi_mask.transform,
                    src_crs=self.roi_mask.crs,
                    dst_crs=CRS.from_epsg(DEFAULT_CRS_PROJ),
                )
                self.roi_mask = np.astype(self.roi_mask, np.uint8)

    def setup_roi_mask(self, mask_file: str | None = None):
        if mask_file is not None:
            self.roi_mask = rasterio.open(mask_file)
        else:
            self.roi_mask = None

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

    def compute_mask_of_class(self, class_name: str, data_array: xr.DataArray) -> xr.DataArray:
        if type(self.classes[class_name]) is range:
            return mask_of_pixels_in_range(self.classes[class_name], data_array)
        else:
            summed_mask = xr.zeros_like(data_array, dtype=bool)
            for value in self.classes[class_name]:
                summed_mask += mask_of_pixels_of(value, data_array)
            return summed_mask

    def compute_percentage_of_mask(self, mask: xr.DataArray, n_pixels_tot: int) -> float:
        return mask.sum().values / n_pixels_tot * 100

    def _compute_area_of_class_mask(self, mask: xr.Dataset) -> float:
        return mask.sum() * np.abs(math.prod(mask.rio.resolution())) / mask.sizes["time"]

    def compute_area_of_class(self, class_name: str, data_array: xr.DataArray):
        return self._compute_area_of_class_mask(self.compute_mask_of_class(class_name=class_name, data_array=data_array))

    def compute_snow_area(self, snow_cover_data_array: xr.DataArray, consider_fraction: bool = True) -> float:
        snow_mask = self.compute_mask_of_class("snow_cover", snow_cover_data_array)
        if consider_fraction:
            snow_cover_data_array = snow_cover_data_array / self.classes["snow_cover"][-1]
            snow_cover_extent = self._compute_area_of_class_mask(snow_cover_data_array.where(snow_mask, 0))
        else:
            snow_cover_extent = self._compute_area_of_class_mask(snow_mask)
        return snow_cover_extent

    def _statistics_core(
        self,
        class_name: str,
        dataset: xr.Dataset,
        n_pixels_tot: int | None = None,
    ):
        class_mask = self.compute_mask_of_class(class_name, dataset.data_vars["snow_cover"])
        if self.class_percentage_distribution:
            self.percentages_dict[class_name] = self.compute_percentage_of_mask(class_mask, n_pixels_tot)
        if self.class_cover_area:
            self.area_dict[class_name] = self._compute_area_of_class_mask(class_mask)

    def _all_statistics(self, dataset: xr.Dataset, exclude_nodata: bool = False) -> Dict[str, float]:
        self.percentages_dict: Dict[str, float] = {} if self.class_percentage_distribution else None
        self.area_dict: Dict[str, float] = {} if self.class_cover_area else None
        data_array = dataset.data_vars["snow_cover"]
        if exclude_nodata:
            nodata_pixels = self.compute_mask_of_class("nodata", data_array).sum().values
            n_pixels_tot = data_array.count().values - nodata_pixels
            for class_name in self.classes:
                if class_name == "nodata":
                    continue
                self._statistics_core(class_name, dataset, n_pixels_tot=n_pixels_tot)
        else:
            n_pixels_tot = data_array.count().values
            for class_name in self.classes:
                self._statistics_core(class_name, dataset, n_pixels_tot=n_pixels_tot)

    def monthly_statics(
        self, snow_cover_dataset: xr.Dataset, month: datetime, exclude_nodata: bool = False
    ) -> Dict[str, float | int]:
        if self.roi_mask is None:
            monthy_dataset = snow_cover_dataset.groupby("time.month")[int(month.strftime("%m"))]
        else:
            monthy_dataset = snow_cover_dataset.groupby("time.month")[int(month.strftime("%m"))].where(self.roi_mask)

        return self._all_statistics(monthy_dataset, exclude_nodata=exclude_nodata)

    def year_statistics(
        self,
        snow_cover_dataset: xr.Dataset,
        months: str | List[str] = "all",
        exclude_nodata: bool = False,
        netcdf_export_path: str | None = None,
    ) -> xr.Dataset:
        from_year = snow_cover_dataset.coords["time"][0].dt.year.values
        winter_year = WinterYear(from_year, from_year + 1)
        winter_year.select_months(months=months)
        logger.info(f"Start processing time series of year {str(winter_year)}")

        year_data_array_sample = xr.DataArray(
            data=np.empty(shape=(len(self.classes), len(months))),
            coords={
                "class_name": [*self.classes],
                "time": winter_year.to_datetime(),
            },
        )
        year_dataset = xr.Dataset(data_vars={"to_remove": year_data_array_sample})

        if self.class_percentage_distribution:
            year_data_array_percentage = year_data_array_sample.copy(deep=True)
        if self.class_cover_area:
            year_data_array_area = year_data_array_sample.copy(deep=True)
        for month_datetime in winter_year.to_datetime():
            logger.info(f"Processing month {month_datetime}")
            self.monthly_statics(snow_cover_dataset=snow_cover_dataset, month=month_datetime, exclude_nodata=exclude_nodata)

            if self.class_percentage_distribution:
                year_data_array_percentage.loc[dict(time=month_datetime, class_name=list(self.percentages_dict.keys()))] = (
                    list(self.percentages_dict.values())
                )
            if self.class_cover_area:
                year_data_array_area.loc[dict(time=month_datetime, class_name=list(self.area_dict.keys()))] = list(
                    self.area_dict.values()
                )

        if self.class_percentage_distribution:
            year_dataset = year_dataset.assign({"class_distribution_percentage": year_data_array_percentage})
        if self.class_cover_area:
            year_dataset = year_dataset.assign({"class_distribution_area": year_data_array_area})

        year_dataset = year_dataset.drop_vars("to_remove")

        # Export options
        if netcdf_export_path is not None:
            year_dataset.to_netcdf(Path(netcdf_export_path))

        return year_dataset


class MeteoFranceSnowCoverProductCompleteness(SnowCoverProductCompleteness):
    def __init__(
        self,
        mask_file: str | None = None,
        class_percentage_distribution: bool = True,
        class_cover_area: bool = True,
    ) -> None:
        super().__init__(
            classes=METEOFRANCE_CLASSES,
            nodata_mapping=None,
            mask_file=mask_file,
            class_percentage_distribution=class_percentage_distribution,
            class_cover_area=class_cover_area,
        )


class NASASnowCoverProductCompleteness(SnowCoverProductCompleteness):
    def __init__(
        self,
        mask_file: str | None = None,
        class_percentage_distribution: bool = True,
        class_cover_area: bool = True,
    ) -> None:
        super().__init__(
            classes=NASA_CLASSES,
            nodata_mapping=NODATA_NASA_CLASSES,
            mask_file=mask_file,
            class_percentage_distribution=class_percentage_distribution,
            class_cover_area=class_cover_area,
        )


class CrossComparisonSnowCoverExtent:
    def __init__(
        self,
        product_1_analyzer: SnowCoverProductCompleteness,
        product_2_analyzer: SnowCoverProductCompleteness,
        validity_mask_union: bool = True,
    ) -> None:
        self.product_1_analyzer = product_1_analyzer
        self.product_2_analyzer = product_2_analyzer
        self.product_1_analyzer.class_percentage_distribution = False
        self.product_2_analyzer.class_percentage_distribution = False
        self.product_1_analyzer.class_cover_area = True
        self.product_2_analyzer.class_cover_area = True
        self.validity_mask_union = validity_mask_union
        # self.check_resolution()

    def check_resolution(self):
        if (
            self.product_1_analyzer.snow_cover_dataset.rio.resolution()
            != self.product_2_analyzer.snow_cover_dataset.rio.resolution()
        ):
            raise NotImplementedError("The two time series to compare must have the same spatial resolution.")

    def _compute_snow_cover_area(self, dataset: xr.Dataset, consider_fraction: bool = True):
        if self.validity_mask_union:
            print("computing validity masks")
            # dataset.data_vars["product_1"].values = np.where(
            #     np.isnan(dataset.data_vars["product_1"].values), 255, dataset.data_vars["product_1"].values
            # ).astype(np.uint8)

            validity_mask_1 = self.product_1_analyzer.compute_mask_of_class("nodata", dataset.data_vars["product_1"])
            validity_mask_2 = self.product_2_analyzer.compute_mask_of_class("nodata", dataset.data_vars["product_2"])
            validity_mask_union = validity_mask_1.compute() | validity_mask_2.compute()
            dataset.data_vars["product_1"].values = np.where(
                validity_mask_union, self.product_1_analyzer.classes["clouds"], dataset.data_vars["product_1"].values
            )
            dataset.data_vars["product_2"].values = np.where(
                validity_mask_union, self.product_2_analyzer.classes["clouds"], dataset.data_vars["product_2"].values
            )

        def _separate_analyzers(data_array: xr.DataArray):
            print("computing SCE")
            if data_array.name == "product_1":
                area = self.product_1_analyzer.compute_snow_area(
                    snow_cover_data_array=data_array, consider_fraction=consider_fraction
                )
                print("1", area.values)

            elif data_array.name == "product_2":
                area = self.product_2_analyzer.compute_snow_area(
                    snow_cover_data_array=data_array, consider_fraction=consider_fraction
                )
                print("2", area.values)
            return area

        return dataset.map(lambda da: _separate_analyzers(data_array=da))

    def compare_snow_extent(self, consider_fraction: bool = True):
        both_products_dataset = xr.Dataset(
            {
                "product_1": self.product_1_analyzer.snow_cover_dataset["snow_cover"].chunk({"time": 1}),
                "product_2": self.product_2_analyzer.snow_cover_dataset["snow_cover"].chunk({"time": 1}),
            },
        )

        # both_products_dataset.groupby("time.day", "massif").map(self._compute_area("snow_cover"))
        snow_cover_extent = both_products_dataset.groupby("time.month").map(
            self._compute_snow_cover_area, consider_fraction=consider_fraction
        )
        sce = snow_cover_extent.compute()
        return sce


if __name__ == "__main__":
    time_series_folder = "../output_folder/snow_cover_extent_analysis/"
    nasa_time_series_name = "WY_2023_2024_SuomiNPP_nasa_time_series_aligned.nc"
    meteofrance_time_series_name = "WY_2023_2024_SuomiNPP_meteofrance_time_series_aligned.nc"

    meteofrance_time_series_path = Path(f"{time_series_folder}").joinpath(meteofrance_time_series_name)
    meteofrance_time_series = xr.open_dataset(meteofrance_time_series_path)  # .isel(time=slice(1, 40))
    mf_stats_calculator = MeteoFranceSnowCoverProductCompleteness(
        mask_file=None,
        class_percentage_distribution=True,
        class_cover_area=True,
    )
    stats = mf_stats_calculator.year_statistics(
        snow_cover_dataset=meteofrance_time_series,
        months=["november", "december"],
        exclude_nodata=True,
        netcdf_export_path="../output_folder/tests_area_cover/test_mf.nc",
    )

    nasa_time_series_path = Path(f"{time_series_folder}").joinpath(nasa_time_series_name)
    nasa_time_series = xr.open_dataset(nasa_time_series_path)  # .isel(time=slice(1, 40))
    nasa_stats_calculator = NASASnowCoverProductCompleteness(
        mask_file=None,
        class_percentage_distribution=True,
        class_cover_area=True,
    )
    stats = nasa_stats_calculator.year_statistics(
        snow_cover_dataset=nasa_time_series,
        months=["november", "december"],
        exclude_nodata=True,
        netcdf_export_path="../output_folder/tests_area_cover/test_nasa.nc",
    )
    # print("TRU openmf_dataset")
    # xr.open_mfdataset([meteofrance_time_series_path, nasa_time_series_path], combine="nested")
    # nasa_time_series_2 = xr.open_dataset(Path(f"{time_series_folder}").joinpath(nasa_time_series_name)).isel(time=slice(1, 40))

    # nasa_stats_calculator_2 = NASASnowCoverProductCompleteness(
    #     nasa_time_series_2,
    #     mask_file=None,
    #     class_percentage_distribution=True,
    #     class_cover_area=True,
    # )
    # stats = nasa_stats_calculator.year_statistics(
    #     months=["january"], exclude_nodata=False, netcdf_export_path="../output_folder/tests_area_cover/test_nasa.nc"
    # )
    # print(meteofrance_time_series)
    # print(nasa_time_series)
    # ds = xr.align([nasa_time_series, meteofrance_time_series])
    # ds = nasa_time_series.assign({"product_2": meteofrance_time_series})

    # sca_cross_comparator = CrossComparisonSnowCoverExtent(
    #     product_1_analyzer=mf_stats_calculator, product_2_analyzer=nasa_stats_calculator, cloud_mask_union=True
    # )
    # sce = sca_cross_comparator.compare_snow_extent()
    # sce.to_netcdf("../output_folder/snow_cover_extent_analysis/WY_2023_2024_snow_cover_area_nasa_from_ndsi.nc")
