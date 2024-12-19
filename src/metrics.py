from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
import rioxarray
import xarray as xr
from logger_setup import default_logger as logger
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import math
import time
import copy
from products import METEOFRANCE_CLASSES, NASA_CLASSES, NODATA_NASA_CLASSES
from geotools import to_rioxarray

REPROJECTION_CRS_EPSG = "32631"


class Year:
    month_dict = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    def __init__(self, year: int | None = None) -> None:
        if year is not None:
            self.year = year
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        return "Year " + str(self.year)

    def __len__(self) -> int:
        return len(self.month_dict)

    @staticmethod
    def year_month_to_datetime(year: int, month: str):
        return datetime.strptime(f"{str(year)} {month}", "%Y %B")

    def as_dict(self):
        return {f"{str(self.year)}": self.month_dict}

    def to_datetime(self):
        return [self.year_month_to_datetime(self.year, month) for month in self.month_dict]

    def to_xarray_coord(self):
        return xr.Coordinates({"time": self.to_datetime()})


class WinterYear:
    month_dict = {
        "october": 1,
        "november": 2,
        "december": 3,
        "january": 4,
        "february": 5,
        "march": 6,
        "april": 7,
        "may": 8,
        "june": 9,
        "july": 10,
        "august": 11,
        "september": 12,
    }

    def __init__(self, from_year: int, to_year: int) -> None:
        self.from_year = from_year
        self.to_year = to_year

    def iterate_days(self):
        for day in pd.date_range(start=f"{self.from_year}/10/01", end=f"{self.to_year}/09/30", freq="D"):
            yield day

    def to_datetime(self):
        a = [
            Year.year_month_to_datetime(self.from_year, month)
            for month in self.month_dict
            if month in ("october", "november", "december")
        ]
        a.extend(
            [
                Year.year_month_to_datetime(self.from_year, month)
                for month in self.month_dict
                if month in ("january", "february", "march", "april", "may", "june", "july", "august", "september")
            ]
        )
        return a

    def to_tuple(self) -> Tuple[int, int]:
        return (self.from_year, self.to_year)


class SnowCoverProductCompleteness:
    def __init__(
        self,
        snow_cover_time_series: xr.Dataset,
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

        self.snow_cover_dataset = to_rioxarray(snow_cover_time_series)
        from_year = self.snow_cover_dataset.coords["time"][0].dt.year.values
        self.year = WinterYear(from_year, from_year + 1)
        self.class_percentage_distribution = class_percentage_distribution
        self.class_cover_area = class_cover_area
        if not self.class_percentage_distribution and not self.class_cover_area:
            raise ValueError("Specify at least one between class_percentage_distribution and class_cover_area arguments")
        self.check_projection()

    def check_projection(self):
        if self.class_cover_area:
            if not self.snow_cover_dataset.rio.crs.is_projected:
                logger.info(f"Reprojecting to EPSG {REPROJECTION_CRS_EPSG} for allowing surface computations")
                self.snow_cover_dataset = self.snow_cover_dataset.rio.reproject(
                    rasterio.crs.CRS.from_epsg(REPROJECTION_CRS_EPSG).to_wkt()
                )
                self.snow_cover_dataset.to_netcdf("../output_folder/tests_area_cover/reprojected.nc")
            if self.roi_mask and not self.roi_mask.crs.is_projected:
                self.roi_mask, _ = rasterio.warp.reproject(
                    self.roi_mask.read(1),
                    src_transform=self.roi_mask.transform,
                    src_crs=self.roi_mask.crs,
                    dst_crs=rasterio.crs.CRS.from_epsg(REPROJECTION_CRS_EPSG),
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

    def count_n_pixels(self, data_array: xr.DataArray) -> int:
        if self.roi_mask is None:
            sizes = data_array.sizes
            return sizes["y"] * sizes["x"] * sizes["time"]
        else:
            return data_array.count().values

    def _mask_of_pixels_of(self, value: int, data_array: xr.DataArray) -> xr.DataArray:
        return data_array == value

    def _mask_of_pixels_in_range(self, range: range, data_array: xr.DataArray) -> xr.DataArray:
        return (data_array >= range[0]) * (data_array <= range[-1])

    def _compute_mask_of_class(self, class_name: str, data_array: xr.DataArray) -> xr.DataArray:
        if type(self.classes[class_name]) is range:
            return self._mask_of_pixels_in_range(self.classes[class_name], data_array)
        else:
            summed_mask = xr.zeros_like(data_array, dtype=bool)
            for value in self.classes[class_name]:
                summed_mask += self._mask_of_pixels_of(value, data_array)
            return summed_mask

    def _compute_number_of_pixels_of_mask(self, mask: xr.DataArray) -> float:
        return mask.sum().values

    def _compute_percentage_of_mask(self, mask: xr.DataArray, n_pixels_tot: int) -> float:
        return self._compute_number_of_pixels_of_mask(mask) / n_pixels_tot * 100

    def _compute_area_of_class_mask(self, mask: xr.Dataset) -> float:
        return mask.sum() * np.abs(math.prod(mask.rio.resolution())) / mask.sizes["time"]

    def _statistics_core(
        self,
        class_name: str,
        dataset: xr.Dataset,
        n_pixels_tot: int | None = None,
    ):
        class_mask = self._compute_mask_of_class(class_name, dataset.data_vars["snow_cover"])
        if self.class_percentage_distribution:
            self.percentages_dict[class_name] = self._compute_percentage_of_mask(class_mask, n_pixels_tot)
            print(class_name, self._compute_percentage_of_mask(class_mask, n_pixels_tot))
        if self.class_cover_area:
            class_mask = class_mask.rio.write_crs(dataset.rio.crs)
            self.area_dict[class_name] = self._compute_area_of_class_mask(class_mask)

    def _all_statistics(self, dataset: xr.Dataset, exclude_nodata: bool = False) -> Dict[str, float]:
        self.percentages_dict: Dict[str, float] = {} if self.class_percentage_distribution else None
        self.area_dict: Dict[str, float] = {} if self.class_cover_area else None
        data_array = dataset.data_vars["snow_cover"]
        if exclude_nodata:
            nodata_pixels = self._compute_number_of_pixels_of_mask(self._compute_mask_of_class("nodata", data_array))
            n_pixels_tot = self.count_n_pixels(data_array) - nodata_pixels
            for class_name in self.classes:
                if class_name == "nodata":
                    continue
                self._statistics_core(class_name, dataset, n_pixels_tot=n_pixels_tot)
        else:
            n_pixels_tot = self.count_n_pixels(data_array)
            for class_name in self.classes:
                self._statistics_core(class_name, dataset, n_pixels_tot=n_pixels_tot)

    def monthly_statics(self, month: str, exclude_nodata: bool = False) -> Dict[str, float | int]:
        if self.roi_mask is None:
            monthy_dataset = self.snow_cover_dataset.groupby("time.month")[self.year.month_dict[month]]
        else:
            monthy_dataset = self.snow_cover_dataset.groupby("time.month")[self.year.month_dict[month]].where(self.roi_mask)

        return self._all_statistics(monthy_dataset, exclude_nodata=exclude_nodata)

    def year_statistics(
        self,
        months: str | List[str] = "all",
        exclude_nodata: bool = False,
        netcdf_export_path: str | None = None,
    ) -> xr.Dataset:
        logger.info(f"Start processing time series of year {self.year.year}")

        year_data_array_sample = xr.DataArray(
            data=np.empty(shape=(len(self.classes), len(months))),
            coords={
                "class_name": [*self.classes],
                "time": [Year.year_month_to_datetime(year=self.year.year, month=month_str) for month_str in months],
            },
        )
        year_dataset = xr.Dataset(data_vars={"to_remove": year_data_array_sample})

        if months == "all":
            months = self.year.month_dict.keys()

        if self.class_percentage_distribution:
            year_data_array_percentage = year_data_array_sample.copy(deep=True)
        if self.class_cover_area:
            year_data_array_area = year_data_array_sample.copy(deep=True)
        for month_str in months:
            logger.info(f"Processing month {month_str}")
            self.monthly_statics(month=month_str, exclude_nodata=exclude_nodata)
            month_datetime = Year.year_month_to_datetime(year=self.year.year, month=month_str)

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
        snow_cover_time_series: xr.Dataset,
        mask_file: str | None = None,
        class_percentage_distribution: bool = True,
        class_cover_area: bool = True,
    ) -> None:
        super().__init__(
            snow_cover_time_series,
            classes=METEOFRANCE_CLASSES,
            nodata_mapping=None,
            mask_file=mask_file,
            class_percentage_distribution=class_percentage_distribution,
            class_cover_area=class_cover_area,
        )


class NASASnowCoverProductCompleteness(SnowCoverProductCompleteness):
    def __init__(
        self,
        snow_cover_time_series: xr.Dataset,
        mask_file: str | None = None,
        class_percentage_distribution: bool = True,
        class_cover_area: bool = True,
    ) -> None:
        super().__init__(
            snow_cover_time_series,
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
        cloud_mask_union: bool = True,
    ) -> None:
        self.product_1_analyzer = product_1_analyzer
        self.product_2_analyzer = product_2_analyzer
        self.product_1_analyzer.class_percentage_distribution = False
        self.product_2_analyzer.class_percentage_distribution = False
        self.product_1_analyzer.class_cover_area = True
        self.product_2_analyzer.class_cover_area = True
        self.cloud_mask_union = cloud_mask_union
        # self.check_resolution()

    def check_resolution(self):
        if (
            self.product_1_analyzer.snow_cover_dataset.rio.resolution()
            != self.product_2_analyzer.snow_cover_dataset.rio.resolution()
        ):
            raise NotImplementedError("The two time series to compare must have the same spatial resolution.")

    def _compute_area(self, dataset: xr.Dataset, class_name: str):
        import matplotlib.pyplot as plt

        # dataset.data_vars["product_1"].compute().isel(time=0).plot.imshow()
        # plt.show()
        # dataset.data_vars["product_2"].compute().isel(time=0).plot.imshow()
        # plt.show()
        if self.cloud_mask_union:
            print("computing cloud masks")
            dataset.data_vars["product_1"].values = np.where(
                np.isnan(dataset.data_vars["product_1"].values), 255, dataset.data_vars["product_1"].values
            ).astype(np.uint8)

            cloud_mask_1 = self.product_1_analyzer._compute_mask_of_class("clouds", dataset.data_vars["product_1"])
            cloud_mask_2 = self.product_2_analyzer._compute_mask_of_class("clouds", dataset.data_vars["product_2"])
            cloud_mask_union = cloud_mask_1.compute() | cloud_mask_2.compute()
            dataset.data_vars["product_1"].values = np.where(cloud_mask_union, 255, dataset.data_vars["product_1"].values)
            dataset.data_vars["product_2"].values = np.where(cloud_mask_union, 250, dataset.data_vars["product_2"].values)

        def _separate_analyzers(data_array: xr.DataArray):
            print("computing SCE")
            if data_array.name == "product_1":
                snow_mask = self.product_1_analyzer._compute_mask_of_class(class_name=class_name, data_array=data_array)
                snow_fraction = data_array.where(snow_mask, 0) / 200
                area = self.product_1_analyzer._compute_area_of_class_mask(snow_fraction)
                # area = self.product_1_analyzer._compute_area_of_class_mask(
                #     self.product_1_analyzer._compute_mask_of_class(class_name, data_array)
                # )
                print("1", area.values)
                return area

            elif data_array.name == "product_2":
                snow_mask = self.product_2_analyzer._compute_mask_of_class(class_name=class_name, data_array=data_array)
                data_array.values = np.where(snow_mask, np.clip(data_array.values / 100 * 1.45 - 0.01, a_min=0, a_max=1), 0)
                area = self.product_2_analyzer._compute_area_of_class_mask(
                    self.product_2_analyzer._compute_mask_of_class(class_name, data_array)
                )
                # print("2", area.values)
                return area

        return dataset.map(lambda da: _separate_analyzers(data_array=da))

    def compare_snow_extent(self):
        both_products_dataset = xr.Dataset(
            {
                "product_1": self.product_1_analyzer.snow_cover_dataset["snow_cover"].chunk({"time": 1}),
                "product_2": self.product_2_analyzer.snow_cover_dataset["snow_cover"].chunk({"time": 1}),
            },
        )

        # both_products_dataset.groupby("time.day", "massif").map(self._compute_area("snow_cover"))
        snow_cover_extent = both_products_dataset.groupby("time.month").map(self._compute_area, class_name="snow_cover")
        sce = snow_cover_extent.compute()
        return sce
        print(sce)
        print(sce.data_vars["product_1"].values)
        print(sce.data_vars["product_2"].values)


if __name__ == "__main__":
    time_series_folder = "../output_folder/snow_cover_extent_analysis/"
    nasa_time_series_name = "WY_2023_2024_SuomiNPP_nasa_time_series_aligned.nc"
    meteofrance_time_series_name = "WY_2023_2024_SuomiNPP_meteofrance_time_series_aligned.nc"

    meteofrance_time_series_path = Path(f"{time_series_folder}").joinpath(meteofrance_time_series_name)
    meteofrance_time_series = xr.open_dataset(meteofrance_time_series_path)  # .isel(time=slice(1, 40))
    mf_stats_calculator = MeteoFranceSnowCoverProductCompleteness(
        meteofrance_time_series,
        mask_file=None,
        class_percentage_distribution=True,
        class_cover_area=True,
    )
    # stats = mf_stats_calculator.year_statistics(
    #     months=["january", "february"],
    #     exclude_nodata=False,
    #     netcdf_export_path="../output_folder/tests_area_cover/test_mf.nc",
    # )

    nasa_time_series_path = Path(f"{time_series_folder}").joinpath(nasa_time_series_name)
    nasa_time_series = xr.open_dataset(nasa_time_series_path)  # .isel(time=slice(1, 40))
    nasa_stats_calculator = NASASnowCoverProductCompleteness(
        nasa_time_series,
        mask_file=None,
        class_percentage_distribution=True,
        class_cover_area=True,
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

    sca_cross_comparator = CrossComparisonSnowCoverExtent(
        product_1_analyzer=mf_stats_calculator, product_2_analyzer=nasa_stats_calculator, cloud_mask_union=True
    )
    sce = sca_cross_comparator.compare_snow_extent()
    sce.to_netcdf("../output_folder/snow_cover_extent_analysis/WY_2023_2024_snow_cover_area_from_fsc.nc")
