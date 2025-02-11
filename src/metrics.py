import copy
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
import rasterio.warp
import xarray as xr
from pyproj import CRS

from grids import DEFAULT_CRS_PROJ
from logger_setup import default_logger as logger
from products.classes import METEOFRANCE_CLASSES, NASA_CLASSES, NODATA_NASA_CLASSES
from winter_year import WinterYear


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

    def compute_area_of_class_mask(self, mask: xr.Dataset) -> float:
        return mask.sum() * np.abs(math.prod(mask.rio.resolution()))

    def compute_area_of_class(self, class_name: str, data_array: xr.DataArray):
        return self.compute_area_of_class_mask(self.compute_mask_of_class(class_name=class_name, data_array=data_array))

    def compute_snow_area(self, snow_cover_data_array: xr.DataArray, consider_fraction: bool = True) -> float:
        snow_mask = self.compute_mask_of_class("snow_cover", snow_cover_data_array)
        if consider_fraction:
            snow_cover_data_array = snow_cover_data_array / self.classes["snow_cover"][-1]
            snow_cover_extent = self.compute_area_of_class_mask(snow_cover_data_array.where(snow_mask, 0))
        else:
            snow_cover_extent = self.compute_area_of_class_mask(snow_mask)
        return snow_cover_extent

    def _statistics_core(
        self,
        class_name: str,
        dataset: xr.Dataset,
        n_pixels_tot: int | None = None,
    ):
        class_mask = self.compute_mask_of_class(class_name, dataset.data_vars["snow_cover_fraction"])
        if self.class_percentage_distribution:
            self.percentages_dict[class_name] = self.compute_percentage_of_mask(class_mask, n_pixels_tot)
        if self.class_cover_area:
            self.area_dict[class_name] = self.compute_area_of_class_mask(class_mask)

    def _all_statistics(self, dataset: xr.Dataset, exclude_nodata: bool = False) -> Dict[str, float]:
        self.percentages_dict: Dict[str, float] = {} if self.class_percentage_distribution else None
        self.area_dict: Dict[str, float] = {} if self.class_cover_area else None
        data_array = dataset.data_vars["snow_cover_fraction"]
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
        if months != "all":
            winter_year.select_months(months=months)
        logger.info(f"Start processing time series of year {str(winter_year)}")

        year_data_array_sample = xr.DataArray(
            data=np.empty(shape=(len(self.classes), len(winter_year))),
            coords={
                "class_name": [*self.classes],
                "time": winter_year.to_datetime(),
            },
        )
        n_days_with_observation = xr.DataArray(0, dims=("time",), coords={"time": winter_year.to_datetime()})
        year_dataset = xr.Dataset(data_vars={"to_remove": year_data_array_sample, "n_observed_days": n_days_with_observation})

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

            year_dataset.data_vars["n_observed_days"].loc[dict(time=month_datetime)] = len(
                snow_cover_dataset.groupby("time.month")[month_datetime.month].coords["time"]
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
    def __init__(self) -> None:
        INVALID_METEOFRANCE_CLASSES = (
            "clouds",
            "water",
            "nodata",
            "fill",
        )

        INVALID_NASA_CLASSES = (
            "clouds",
            "water",
            "no_decision",
            "night",
            "missing_data",
            "L1B_unusable",
            "bowtie_trim",
            "L1B_fill",
            "fill",
        )

        self.mf_analyzer = SnowCoverProductCompleteness(
            classes=METEOFRANCE_CLASSES, nodata_mapping=INVALID_METEOFRANCE_CLASSES
        )
        self.nasa_analyzer = SnowCoverProductCompleteness(classes=NASA_CLASSES, nodata_mapping=INVALID_NASA_CLASSES)

    def compute_union_valid_snow_area(
        self,
        dataset: xr.Dataset,
        analyzer_meteofrance: SnowCoverProductCompleteness,
        analyzer_nasa: SnowCoverProductCompleteness,
        consider_fraction: bool = False,
        forest_mask: str | None = None,
        fsc_threshold: float | None = None,
    ):
        logger.info(f"Processing time of the year {dataset.coords['time'].values[0].astype('M8[D]').astype('O')}")
        invalid_mask_1 = analyzer_meteofrance.compute_mask_of_class("nodata", dataset.data_vars["meteofrance"])
        invalid_mask_2 = analyzer_nasa.compute_mask_of_class("nodata", dataset.data_vars["nasa"])
        invalid_mask_union = invalid_mask_1.compute() | invalid_mask_2.compute()
        meteofrance_valid = dataset.data_vars["meteofrance"].where(1 - invalid_mask_union)
        nasa_valid = dataset.data_vars["nasa"].where(1 - invalid_mask_union)
        if fsc_threshold is not None:
            meteofrance_valid = meteofrance_valid.where(
                meteofrance_valid > fsc_threshold * METEOFRANCE_CLASSES["snow_cover"][-1]
            )
        if fsc_threshold is not None:
            nasa_valid = nasa_valid.where(nasa_valid > fsc_threshold * NASA_CLASSES["snow_cover"][-1])

        if forest_mask is not None:
            meteofrance_forest = (
                analyzer_meteofrance.compute_mask_of_class("forest_with_snow", meteofrance_valid).compute()
                | analyzer_meteofrance.compute_mask_of_class("forest_without_snow", meteofrance_valid).compute()
            )

            area_nasa = analyzer_nasa.compute_snow_area(
                nasa_valid.where(~meteofrance_forest), consider_fraction=consider_fraction
            )
            area_nasa_forest_with_snow = analyzer_nasa.compute_snow_area(
                nasa_valid.where(meteofrance_forest), consider_fraction=consider_fraction
            )
            nasa_results = xr.DataArray(
                [area_nasa, area_nasa_forest_with_snow], coords={"class_name": ["snow_cover", "forest_with_snow"]}
            )
        else:
            area_nasa = analyzer_nasa.compute_snow_area(nasa_valid, consider_fraction=consider_fraction)
            nasa_results = xr.DataArray([area_nasa], coords={"class_name": ["snow_cover"]})

        area_meteofrance = analyzer_meteofrance.compute_snow_area(meteofrance_valid, consider_fraction=consider_fraction)
        area_meteofrance_forest_with_snow = analyzer_meteofrance.compute_area_of_class("forest_with_snow", meteofrance_valid)
        return xr.Dataset(
            data_vars={
                "meteofrance": xr.DataArray(
                    [area_meteofrance, area_meteofrance_forest_with_snow],
                    coords={"class_name": ["snow_cover", "forest_with_snow"]},
                ),
                "nasa": nasa_results,
            }
        )

    def analyze_sce_valid_union(
        self,
        meteofrance_time_series: xr.Dataset,
        nasa_time_series: xr.Dataset,
        consider_fraction: bool = False,
        netcdf_export_path: str | None = None,
        forest_mask: str | None = None,
        fsc_threshold: float | None = None,
    ) -> xr.Dataset:
        common_days = np.intersect1d(meteofrance_time_series["time"], nasa_time_series["time"])
        both_products_dataset = xr.Dataset(
            {
                "meteofrance": meteofrance_time_series.data_vars["snow_cover_fraction"]
                .sel(time=common_days, drop=True)
                .chunk({"time": 1}),
                "nasa": nasa_time_series.data_vars["snow_cover_fraction"].sel(time=common_days, drop=True).chunk({"time": 1}),
            },
        )
        result = both_products_dataset.groupby("time").map(
            self.compute_union_valid_snow_area,
            [self.mf_analyzer, self.nasa_analyzer, consider_fraction, forest_mask, fsc_threshold],
        )

        if netcdf_export_path:
            result.to_netcdf(netcdf_export_path)
        return result


if __name__ == "__main__":
    platform = "SNPP"
    year = WinterYear(2023, 2024)
    resolution = 375
    working_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_3/"
    nasa_prod = "nasa_pseudo_l3"
    nasa_time_series_name = f"WY_{year.from_year}_{year.to_year}_{platform}_{nasa_prod}_res_{resolution}m.nc"
    meteofrance_time_series_name = f"WY_{year.from_year}_{year.to_year}_{platform}_meteofrance_res_{resolution}m.nc"

    mode = "snow_cover_extent_cross_comparion"  # 'class_distribution', 'snow_cover_extent_cross_comparion'
    consider_fsc = False
    fsc_threshold = 0.15
    if consider_fsc:
        outfile_name = f"WY_{year.from_year}_{year.to_year}_{platform}_meteofrance_{nasa_prod}_sce_cross_comparison_snow_cover_fraction_res_{resolution}m_fsc_thresh_{fsc_threshold}.nc"
    else:
        outfile_name = f"WY_{year.from_year}_{year.to_year}_{platform}_meteofrance_{nasa_prod}_sce_cross_comparison_binary_snow_cover_res_{resolution}m_fsc_thresh_{fsc_threshold}.nc"

    meteofrance_time_series_path = Path(f"{working_folder}").joinpath(meteofrance_time_series_name)
    meteofrance_time_series = xr.open_dataset(meteofrance_time_series_path)

    nasa_time_series_path = Path(f"{working_folder}").joinpath(nasa_time_series_name)
    nasa_time_series = xr.open_dataset(nasa_time_series_path)

    if mode == "class_distribution":
        mf_stats_calculator = MeteoFranceSnowCoverProductCompleteness(
            class_percentage_distribution=True,
            class_cover_area=True,
        )
        mf_stats_calculator.year_statistics(
            snow_cover_dataset=meteofrance_time_series,
            months="all",
            exclude_nodata=False,
            netcdf_export_path=Path(f"{working_folder}").joinpath("WY_2023_2024_SNPP_meteofrance_class_distribution.nc"),
        )

        nasa_stats_calculator = NASASnowCoverProductCompleteness(
            class_percentage_distribution=True,
            class_cover_area=True,
        )
        nasa_stats_calculator.year_statistics(
            snow_cover_dataset=nasa_time_series,
            months="all",
            exclude_nodata=False,
            netcdf_export_path=Path(f"{working_folder}").joinpath("WY_2023_2024_SNPP_nasa_class_distribution.nc"),
        )
    elif mode == "snow_cover_extent_cross_comparion":
        CrossComparisonSnowCoverExtent().analyze_sce_valid_union(
            meteofrance_time_series=meteofrance_time_series,
            nasa_time_series=nasa_time_series,
            consider_fraction=consider_fsc,
            forest_mask="yes",
            netcdf_export_path=Path(f"{working_folder}").joinpath(outfile_name),
            fsc_threshold=fsc_threshold,
        )
