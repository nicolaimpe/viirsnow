from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from logger_setup import default_logger as logger
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

METEOFRANCE_CLASSES = {
    "snow_cover": range(1, 201),
    "no_snow": (0,),
    "clouds": (255,),
    "forest_without_snow": (215,),
    "forest_with_snow": (210,),
    "water": (220,),
    "nodata": (230,),
}
NASA_CLASSES = {
    "snow_cover": range(1, 101),
    "no_snow": (0,),
    "clouds": (250,),
    "water": (237, 239),
    "no_decision": (201,),
    "night": (211,),
    "missing_data": (251,),
    "L1B_unusable": (252,),
    "bowtie_trim": (253,),
    "L1B_fill": (254,),
    "fill": (255,),
}

NODATA_NASA_CLASSES = (
    "no_decision",
    "night",
    "missing_data",
    "L1B_unusable",
    "bowtie_trim",
    "L1B_fill",
    "fill",
)


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
        return "Year " + self.year

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


class SnowCoverProductCompleteness:
    def __init__(
        self,
        snow_cover_time_series: xr.Dataset,
        classes: Dict[str, Tuple[int, ...] | range],
        nodata_mapping: Tuple[str, ...] | None = None,
        mask_file: str | None = None,
        class_percentage_distribution: bool = True,
        class_cover_area: bool = False,
    ) -> None:
        self.setup_roi_mask(mask_file=mask_file)
        self.classes = classes
        if nodata_mapping is not None:
            self.setup_nodata_classes(nodata_mapping=nodata_mapping)

        self.snow_cover_data_array = snow_cover_time_series.data_vars["snow_cover"]
        self.year = Year(self.snow_cover_data_array.coords["time"][0].dt.year.values)
        if any(date.dt.year != self.year.year for date in self.snow_cover_data_array.coords["time"]):
            raise NotImplementedError
        self.class_percentage_distribution = class_percentage_distribution
        self.class_cover_area = class_cover_area
        if not self.class_percentage_distribution and not self.class_cover_area:
            raise ValueError("Specify at least one between class_percentage_distribution and class_cover_area arguments")

    def setup_roi_mask(self, mask_file: str | None = None):
        if mask_file is not None:
            self.roi_mask = rasterio.open(mask_file).read()
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
            return sizes["lon"] * sizes["lat"] * sizes["time"]
        else:
            return data_array.count().values

    # def compute_number_pixels_of_class(self, class_name: str, data_array: xr.DataArray):
    #     if type(self.classes[class_name]) is range:
    #         return self.compute_number_of_pixels_in_range(self.classes[class_name], data_array)
    #     else:
    #         summed_pixels = 0
    #         for value in self.classes[class_name]:
    #             summed_pixels += self.compute_number_of_pixels_of(value, data_array)
    #         return summed_pixels

    def _mask_of_pixels_of(self, value: int, data_array: xr.DataArray):
        return data_array == value

    def _mask_of_pixels_in_range(self, range: range, data_array: xr.DataArray):
        return (data_array >= range[0]) * (data_array <= range[-1])

    def _compute_masks_of_class(self, class_name: str, data_array: xr.DataArray):
        if type(self.classes[class_name]) is range:
            return self._mask_of_pixels_in_range(self.classes[class_name], data_array)
        else:
            summed_mask = xr.zeros_like(data_array, dtype=bool)
            for value in self.classes[class_name]:
                summed_mask += self._mask_of_pixels_of(value, data_array)
            return summed_mask

    def _compute_number_of_pixels_of_mask(self, mask: xr.DataArray):
        return mask.sum().values

    def _compute_percentage_of_mask(self, mask: xr.DataArray, n_pixels_tot: int):
        return self._compute_number_of_pixels_of_mask(mask) / n_pixels_tot * 100

    def _compute_area_of_class_mask(self, mask: xr.DataArray):
        return None

    def _statistics_core(
        self,
        class_name: str,
        data_array: xr.DataArray,
        n_pixels_tot: int | None = None,
        percentages_dict: Dict[str | float] | None = None,
        area_dict: Dict[str | float] | None = None,
    ):
        class_mask = self._compute_masks_of_class(class_name, data_array)
        if percentages_dict:
            percentages_dict[class_name] = self._compute_percentage_of_mask(class_name, class_mask, n_pixels_tot)

        if area_dict:
            area_dict[class_name] = self._compute_area_of_class_mask(class_mask)

    def _all_statistics(self, data_array: xr.DataArray, exclude_nodata: bool = False) -> Dict[str, float]:
        percentages_dict: Dict[str, float] = {} if self.class_percentage_distribution else None
        area_dict: Dict[str, float] = {} if self.class_cover_area else None

        if exclude_nodata:
            nodata_pixels = self._compute_number_of_pixels_of_mask(self._compute_masks_of_class("nodata", data_array))
            n_pixels_tot = self.count_n_pixels(data_array) - nodata_pixels
            for class_name in self.classes:
                if class_name == "nodata":
                    continue
                self._statistics_core(
                    class_name, data_array, n_pixels_tot=n_pixels_tot, percentages_dict=percentages_dict, area_dict=area_dict
                )
        else:
            n_pixels_tot = self.count_n_pixels(data_array)
            for class_name in self.classes:
                self._statistics_core(
                    class_name, data_array, n_pixels_tot=n_pixels_tot, percentages_dict=percentages_dict, area_dict=area_dict
                )
        return percentages_dict, area_dict

    def monthly_statics(self, month: str, exclude_nodata: bool = False) -> Dict[str, float | int]:
        if self.roi_mask is None:
            monthy_data_array = self.snow_cover_data_array.groupby("time.month")[self.year.month_dict[month]]
        else:
            monthy_data_array = self.snow_cover_data_array.groupby("time.month")[self.year.month_dict[month]].where(
                self.roi_mask
            )

        statistics = self._all_statistics(monthy_data_array, exclude_nodata=exclude_nodata)
        statistics["n_images"] = int(monthy_data_array.sizes["time"])
        return statistics

    def year_statistics(
        self,
        months: str | List[str] = "all",
        exclude_nodata: bool = False,
        netcdf_export_path: str | None = None,
        csv_export_path: str | None = None,
    ):
        logger.info(f"Start processing time series of year {self.year.year}")
        year_data_array = xr.DataArray(
            data=np.empty(shape=(len(self.classes) + 1, len(months))),
            coords={
                "stat_name": [*self.classes, "n_images"],
                "time": [Year.year_month_to_datetime(year=self.year.year, month=month_str) for month_str in months],
            },
        )

        if months == "all":
            months = self.year.month_dict.keys()

        for month_str in months:
            logger.info(f"Processing month {month_str}")
            month_datetime = Year.year_month_to_datetime(year=self.year.year, month=month_str)
            monthly_statistics = self.monthly_statics(month=month_str, exclude_nodata=exclude_nodata)
            year_data_array.loc[dict(time=month_datetime, stat_name=list(monthly_statistics.keys()))] = list(
                monthly_statistics.values()
            )

        # Export options
        if netcdf_export_path is not None:
            year_data_array.to_netcdf(Path(netcdf_export_path))

        if csv_export_path is not None:
            year_data_array.to_pandas().to_csv(Path(csv_export_path))

        return year_data_array

    def print_table(self, year_data_array: xr.DataArray, classes_to_print: List[str] | str = "all"):
        year_data_frame = year_data_array.to_pandas()
        pd.options.display.float_format = "{:.3f}".format
        pd.options.display.precision = 3
        if classes_to_print == "all":
            classes_to_print = year_data_array.coords["stat_name"].values
        print(year_data_frame.loc[classes_to_print])

    def classes_bar_distribution(
        self, year_data_array: xr.DataArray, classes_to_plot: List[str] | str = "all", ax: Axes | None = None
    ) -> None:
        year_data_frame = year_data_array.to_pandas()
        year_data_frame = year_data_frame.drop("n_images")
        if classes_to_plot == "all":
            classes_to_plot = year_data_frame.index

        year_data_frame = year_data_frame.transpose()
        year_data_frame.index = year_data_frame.index.strftime("%B")
        year_data_frame.loc[classes_to_plot].plot.bar(title=f"Class distribution for year {self.year.year}", ax=ax)


class MeteoFranceSnowCoverProductCompleteness(SnowCoverProductCompleteness):
    def __init__(self, meteofrance_snow_cover_time_series: xr.Dataset, mask_file: str | None = None) -> None:
        super().__init__(
            meteofrance_snow_cover_time_series, classes=METEOFRANCE_CLASSES, nodata_mapping=None, mask_file=mask_file
        )


class NASASnowCoverProductCompleteness(SnowCoverProductCompleteness):
    def __init__(self, nasa_snow_cover_time_series: xr.Dataset, mask_file: str | None = None) -> None:
        super().__init__(
            nasa_snow_cover_time_series, classes=NASA_CLASSES, nodata_mapping=NODATA_NASA_CLASSES, mask_file=mask_file
        )
