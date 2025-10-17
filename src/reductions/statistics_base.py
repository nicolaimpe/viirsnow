import abc
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from xarray.groupers import BinGrouper

from compression import generate_xarray_compression_encodings
from logger_setup import default_logger as logger
from reductions.completeness import SnowCoverProductCompleteness
from reductions.semidistributed import MountainParametrization, MountainParams
from winter_year import WinterYear


@dataclass
class EvaluationConfig(MountainParams):
    ref_var_name: str = ("snow_cover_fraction",)
    test_var_name: str = ("snow_cover_fraction",)
    ref_fsc_step: int = (25,)
    sensor_zenith_analysis: bool = True
    sub_roi_mask_path: str | None = None


def generate_evaluation_io(
    analysis_type: str,
    working_folder: str,
    year: WinterYear,
    ref_product_name: str,
    test_product_name: str,
    period: slice | None = None,
) -> Tuple[xr.Dataset, xr.Dataset, str]:
    output_folder = f"{working_folder}/analyses/{analysis_type}"
    ref_time_series_name = f"WY_{year.from_year}_{year.to_year}_{ref_product_name}.nc"
    test_time_series_name = f"WY_{year.from_year}_{year.to_year}_{test_product_name}.nc"

    output_filename = (
        f"{output_folder}/{analysis_type}_WY_{year.from_year}_{year.to_year}_{test_product_name}_vs_{ref_product_name}.nc"
    )

    test_time_series = xr.open_dataset(f"{working_folder}/time_series/{test_time_series_name}", mask_and_scale=False)
    ref_time_series = xr.open_dataset(f"{working_folder}/time_series/{ref_time_series_name}", mask_and_scale=False)

    if period is not None:
        test_time_series = test_time_series.sel(time=period)
        ref_time_series = ref_time_series.sel(time=period)

    return ref_time_series, test_time_series, output_filename


class EvaluationVsHighResBase(MountainParametrization):
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
                    *np.arange(0, self.ref_analyzer.max_fsc - 1, ref_fsc_step),
                    self.ref_analyzer.max_fsc - 1,
                    self.ref_analyzer.max_fsc,
                    self.ref_analyzer.max_value,
                ]
            ),
            labels=np.array(
                [
                    *np.arange(0, self.ref_analyzer.max_fsc - 1, ref_fsc_step),
                    self.ref_analyzer.max_fsc - 1,
                    self.ref_analyzer.max_fsc,
                    self.ref_analyzer.max_value,
                ]
            ),
        )

    @staticmethod
    def month_bins(winter_year: WinterYear) -> BinGrouper:
        wy_datetime = winter_year.to_datetime()
        wy_datetime.extend([datetime(year=wy_datetime[-1].year, month=wy_datetime[-1].month + 1, day=1)])
        return BinGrouper(wy_datetime[2:10], labels=[month_datetime for month_datetime in wy_datetime[2:9]])

    def prepare_analysis(
        self,
        test_time_series: xr.Dataset,
        ref_time_series: xr.Dataset,
        config: EvaluationConfig,
    ) -> Tuple[xr.Dataset, Dict[str, BinGrouper]]:
        # if ref_time_series.max() != 205:
        #     # This is supposed by limiting ref bins values to 205 just in ref_fsc_bins()
        #     # If we don' put a max value, then groupby for ref FSC bins might not work in case we have invalid data
        #     # but max value depends on the encoding so we have to put this
        #     raise NotImplementedError("This function supposes that the reference data maximum is 205")

        common_days = np.intersect1d(test_time_series["time"], ref_time_series["time"])
        combined_dataset = xr.Dataset(
            {
                "ref": ref_time_series.data_vars[config.ref_var_name[0]].sel(time=common_days),
                "test": test_time_series.data_vars[config.test_var_name[0]].sel(time=common_days),
            },
        )
        combined_dataset, analysis_bin_dict = self.semidistributed_parametrization(
            dataset=combined_dataset,
            config=MountainParams(
                slope_map_path=config.slope_map_path,
                aspect_map_path=config.aspect_map_path,
                dem_path=config.dem_path,
                forest_mask_path=config.forest_mask_path,
            ),
        )
        analysis_bin_dict.update(ref=self.ref_fsc_bins(config.ref_fsc_step))

        if config.sensor_zenith_analysis:
            analysis_bin_dict.update(sensor_zenith=self.sensor_zenith_bins())
            combined_dataset = combined_dataset.assign(
                sensor_zenith=test_time_series.data_vars["sensor_zenith_angle"].sel(time=common_days)
            )

        if config.sub_roi_mask_path is not None:
            sub_roi_mask = xr.open_dataarray(config.sub_roi_mask_path)
            combined_dataset = combined_dataset.assign(sub_roi=sub_roi_mask.sel(band=1).drop_vars("band"))
            analysis_bin_dict.update(sub_roi=self.sub_roi_bins())

        combined_dataset = combined_dataset.drop_vars("spatial_ref")
        return combined_dataset, analysis_bin_dict

    @abc.abstractmethod
    def time_step_analysis(self, bins_dict: Dict[str, BinGrouper]):
        pass

    def launch_analysis(
        self,
        test_time_series: xr.Dataset,
        ref_time_series: xr.Dataset,
        config: EvaluationConfig,
        netcdf_export_path: str | None = None,
    ) -> xr.Dataset:
        combined_dataset, analysis_bin_dict = self.prepare_analysis(
            test_time_series=test_time_series,
            ref_time_series=ref_time_series,
            config=config,
        )
        result = combined_dataset.groupby("time").map(self.time_step_analysis, bins_dict=analysis_bin_dict)
        # logger.info("Reducing time coordinate per month")
        # result = result.resample({"time": "1ME"}).sum(dim="time")
        if netcdf_export_path:
            logger.info(f"Exporting to {netcdf_export_path}")
            result.to_netcdf(netcdf_export_path, encoding=generate_xarray_compression_encodings(result))
        return result
