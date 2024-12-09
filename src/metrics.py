from typing import Dict, List, Tuple

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
        self.classes = classes
        if nodata_mapping is not None:
            self.setup_nodata_classes(nodata_mapping=nodata_mapping)

        self.snow_cover_dataset = self.to_rioxarray(snow_cover_time_series)
        self.year = Year(self.snow_cover_dataset.coords["time"][0].dt.year.values)
        no_valid_time_mask = self.snow_cover_dataset.coords["time"].dt.year != self.year.year
        if no_valid_time_mask.sum() > 0:
            raise NotImplementedError
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
                self.roi_mask, transform = rasterio.warp.reproject(
                    self.roi_mask.read(1),
                    src_transform=self.roi_mask.transform,
                    src_crs=self.roi_mask.crs,
                    dst_crs=rasterio.crs.CRS.from_epsg(REPROJECTION_CRS_EPSG),
                )
                self.roi_mask = np.astype(self.roi_mask, np.uint8)

                with rasterio.open(
                    "../output_folder/tests_area_cover/test_mask_reprojection.tif",
                    "w",
                    transform=transform,
                    crs=rasterio.crs.CRS.from_epsg(REPROJECTION_CRS_EPSG),
                    width=self.roi_mask.shape[2],
                    height=self.roi_mask.shape[1],
                    count=1,
                    dtype=self.roi_mask.dtype,
                ) as dst:
                    dst.write(self.roi_mask)

    def to_rioxarray(self, dataset: xr.Dataset) -> xr.Dataset:
        return dataset.rio.write_crs(dataset.data_vars["spatial_ref"].attrs["spatial_ref"])

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

    def _compute_masks_of_class(self, class_name: str, data_array: xr.DataArray) -> xr.DataArray:
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
        class_mask = self._compute_masks_of_class(class_name, dataset.data_vars["snow_cover"])
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
            nodata_pixels = self._compute_number_of_pixels_of_mask(self._compute_masks_of_class("nodata", data_array))
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


if __name__ == "__main__":
    time_series_folder = "../output_folder/completeness_analysis/"
    nasa_time_series_name = "2023_SuomiNPP_nasa_time_series_fsc.nc"
    meteofrance_time_series_name = "2023_meteofrance_time_series.nc"

    meteofrance_time_series_path = Path(f"{time_series_folder}").joinpath(meteofrance_time_series_name)
    meteofrance_time_series = xr.open_dataset(meteofrance_time_series_path).isel(time=slice(1, 30))
    mf_stats_calculator = MeteoFranceSnowCoverProductCompleteness(
        meteofrance_time_series,
        mask_file="../../data/vectorial/massifs_WGS84/massifs_WGS84/massifs_mask_eofr62.tiff",
        class_percentage_distribution=True,
        class_cover_area=True,
    )
    stats = mf_stats_calculator.year_statistics(
        months=["january"], exclude_nodata=False, netcdf_export_path="../output_folder/tests_area_cover/test_mf.nc"
    )

    # nasa_time_series_path = Path(f"{time_series_folder}").joinpath(nasa_time_series_name)
    # nasa_time_series = xr.open_dataset(nasa_time_series_path).isel(time=slice(1, 20))
    # mf_stats_calculator = NASASnowCoverProductCompleteness(
    #     nasa_time_series,
    #     mask_file="../../data/vectorial/massifs_WGS84/massifs_WGS84/massifs_mask_v10_epsg4326.tiff",
    #     class_percentage_distribution=True,
    #     class_cover_area=True,
    # )
    # stats = mf_stats_calculator.year_statistics(
    #     months=["january"], exclude_nodata=False, netcdf_export_path="../output_folder/tests_area_cover/test_nasa.nc"
    # )
