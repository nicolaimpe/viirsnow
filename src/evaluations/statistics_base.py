from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from xarray.groupers import BinGrouper, UniqueGrouper

from evaluations.completeness import SnowCoverProductCompleteness
from winter_year import WinterYear


@dataclass
class EvaluationConfig:
    ref_fsc_step: int = (25,)
    sensor_zenith_analysis: bool = (True,)
    forest_mask_path: str | None = (None,)
    sub_roi_mask_path: str | None = (None,)
    slope_map_path: str | None = (None,)
    aspect_map_path: str | None = (None,)
    dem_path: str | None = (None,)


def abbreviate_data_var_name(data_var_name: str) -> str:
    if data_var_name == "snow_cover_fraction":
        abbreviated = "FSC"
    elif data_var_name == "snow_cover_fraction":
        abbreviated = "NDSI"
    else:
        abbreviated = data_var_name
    return abbreviated


def generate_evaluation_io(
    analysis_type: str,
    working_folder: str,
    year: WinterYear,
    resolution: int,
    platform: str,
    ref_product_name: str,
    test_product_name: str,
    ref_product_var: str = "snow_cover_fraction",
    test_product_var: str = "snow_cover_fraction",
    period: slice | None = None,
) -> Tuple[xr.Dataset, xr.Dataset, str]:
    output_folder = f"{working_folder}/analyses/{analysis_type}"
    ref_time_series_name = f"WY_{year.from_year}_{year.to_year}_{ref_product_name}_res_{resolution}m.nc"
    test_time_series_name = f"WY_{year.from_year}_{year.to_year}_{platform}_{test_product_name}_res_{resolution}m.nc"

    abbreviated_test, abbreviated_ref = abbreviate_data_var_name(test_product_var), abbreviate_data_var_name(ref_product_var)
    output_filename = f"{output_folder}/{analysis_type}_WY_{year.from_year}_{year.to_year}_{platform}_{test_product_name}_{abbreviated_test}_vs_{ref_product_name}_{abbreviated_ref}_{resolution}m.nc"

    test_time_series = xr.open_dataset(f"{working_folder}/time_series/{test_time_series_name}").data_vars[ref_product_var]
    ref_time_series = xr.open_dataset(f"{working_folder}/time_series/{ref_time_series_name}").data_vars[test_product_var]

    if period is not None:
        test_time_series = test_time_series.sel(time=period)
        ref_time_series = ref_time_series.sel(time=period)

    return ref_time_series, test_time_series, output_filename


class EvaluationVsHighResBase:
    def __init__(
        self,
        reference_analyzer: SnowCoverProductCompleteness,
        test_analyzer: SnowCoverProductCompleteness,
    ) -> None:
        if reference_analyzer.classes["snow_cover"] != range(1, 101):
            # ref_fsc is not coded dynamically with change of reference dataset
            raise NotImplementedError("This class supposes that the reference snow cover fraction is encoded beween 0 and 100")

        self.ref_analyzer = reference_analyzer
        self.test_analyzer = test_analyzer

    @staticmethod
    def sensor_zenith_bins(sensor_zenith_step: int = 15) -> BinGrouper:
        # degrees
        # 255 is there to account for empty bins...otherwise the code breaks
        return BinGrouper(
            np.array([*np.arange(0, 90, sensor_zenith_step), 255]),
            labels=np.array([*np.arange(sensor_zenith_step, 90, sensor_zenith_step), 255]),
            include_lowest=True,
        )

    def ref_fsc_bins(self, ref_fsc_step: int = 25) -> BinGrouper:
        # 255 is there to account for empty/invalid bins...otherwise the code breaks
        return BinGrouper(
            np.array(
                [
                    -1,
                    *np.arange(0, self.ref_analyzer.max_fsc, ref_fsc_step),
                    self.ref_analyzer.max_fsc,
                    self.ref_analyzer.max_value,
                ]
            ),
            labels=np.array(
                [
                    *np.arange(0, self.ref_analyzer.max_fsc, ref_fsc_step),
                    self.ref_analyzer.max_fsc,
                    self.ref_analyzer.max_value,
                ]
            ),
        )

    @staticmethod
    def forest_bins() -> BinGrouper:
        return BinGrouper(np.array([-1, 0, 1]), labels=["no_forest", "forest"])

    @staticmethod
    def sub_roi_bins() -> BinGrouper:
        return BinGrouper(
            np.array([-1, 0, 1, 2, 3, 4, 5, 6]), labels=["Alps", "Pyrenees", "Corse", "Massif Central", "Jura", "Vosges"]
        )

    @staticmethod
    def slope_bins() -> BinGrouper:
        return BinGrouper(
            np.array([0, *np.arange(10, 70, 20), 90]), labels=np.array([*np.arange(10, 70, 20), 90]), include_lowest=True
        )

    @staticmethod
    def aspect_bins() -> BinGrouper:
        return BinGrouper(np.arange(-22.5, 360, 45), labels=np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]))

    @staticmethod
    def altitude_bins(altitude_band: int = 600) -> BinGrouper:
        return BinGrouper(
            np.array([0, *np.arange(900, 3900, altitude_band), 4800]),
            labels=np.array([0, *np.arange(900, 3900, altitude_band)]),
        )

    @staticmethod
    def month_bins(winter_year: WinterYear) -> BinGrouper:
        wy_datetime = winter_year.to_datetime()
        wy_datetime.extend([datetime(year=wy_datetime[-1].year, month=wy_datetime[-1].month + 1, day=1)])
        return BinGrouper(wy_datetime, labels=[month_datetime for month_datetime in wy_datetime[:-1]])

    @staticmethod
    def aspect_map_transform(aspect_map: xr.DataArray) -> xr.DataArray:
        """
        Aspect map in degrees azimuth

        Transform the aspect map so that its values are monotonically incresing from N to NW,
        without dividing the North in two bins (NNW [337.5-360] and NNE [0-315])
        This is convenient for BinGrouper object

        """
        # Transform the aspect map so that its values are monotonically incresing from N to NW,
        # without dividing the North in two bins (NNW [337.5-360] and NNE [0-315])
        # This is convenient for BinGrouper object

        aspect_map = aspect_map.where(aspect_map < 337.5, aspect_map - 360)
        return aspect_map

    def prepare_analysis(
        self,
        test_time_series: xr.DataArray,
        ref_time_series: xr.DataArray,
        config: EvaluationConfig,
    ) -> Tuple[xr.Dataset, Dict[str, BinGrouper]]:
        # if ref_time_series.max() != 205:
        #     # This is supposed by limiting ref bins values to 205 just in ref_fsc_bins()
        #     # If we don' put a max value, then groupby for ref FSC bins might not work in case we have invalid data
        #     # but max value depends on the encoding so we have to put this
        #     raise NotImplementedError("This function supposes that the reference data maximum is 205")

        common_days = np.intersect1d(test_time_series["time"], ref_time_series["time"])

        combined_dataset = xr.Dataset(
            {"ref": ref_time_series.sel(time=common_days), "test": test_time_series.sel(time=common_days)},
        )

        analysis_bin_dict = dict(ref=self.ref_fsc_bins(config.ref_fsc_step))

        if config.sensor_zenith_analysis:
            analysis_bin_dict.update(sensor_zenith=self.sensor_zenith_bins())
            combined_dataset = combined_dataset.assign(
                sensor_zenith=test_time_series.data_vars["sensor_zenith"].sel(time=common_days)
            )

        if config.forest_mask_path is not None:
            forest_mask = xr.open_dataarray(config.forest_mask_path)
            combined_dataset = combined_dataset.assign(forest_mask=forest_mask.sel(band=1).drop_vars("band"))
            analysis_bin_dict.update(forest_mask=self.forest_bins())

        if config.sub_roi_mask_path is not None:
            sub_roi_mask = xr.open_dataarray(config.sub_roi_mask_path)
            combined_dataset = combined_dataset.assign(sub_roi=sub_roi_mask.sel(band=1).drop_vars("band"))
            analysis_bin_dict.update(sub_roi=self.sub_roi_bins())

        if config.slope_map_path is not None:
            slope_map = xr.open_dataarray(config.slope_map_path)
            combined_dataset = combined_dataset.assign(slope=slope_map.sel(band=1).drop_vars("band"))
            analysis_bin_dict.update(slope=self.slope_bins())

        if config.aspect_map_path is not None:
            aspect_map = xr.open_dataarray(config.aspect_map_path)
            aspect_map = self.aspect_map_transform(aspect_map.sel(band=1).drop_vars("band"))
            combined_dataset = combined_dataset.assign(aspect=aspect_map)
            analysis_bin_dict.update(aspect=self.aspect_bins())

        if config.dem_path is not None:
            dem_map = xr.open_dataarray(config.dem_path)
            combined_dataset = combined_dataset.assign(altitude=dem_map.sel(band=1).drop_vars("band"))
            analysis_bin_dict.update(altitude=self.altitude_bins())

        combined_dataset = combined_dataset.drop_vars("spatial_ref")
        return combined_dataset, analysis_bin_dict
