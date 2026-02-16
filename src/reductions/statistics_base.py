import abc
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from geospatial_grid.grid_database import UTM375mGrid
from geospatial_grid.gsgrid import GSGrid
from mountain_data_binner.mountain_binner import MountainBinner, MountainBinnerConfig
from ndsi_fsc_calibration.utils import generate_xarray_compression_encodings
from xarray.groupers import BinGrouper

from logger_setup import default_logger as logger
from reductions.completeness import SnowCoverProductCompleteness
from winter_year import WinterYear


@dataclass
class EvaluationConfig(MountainBinnerConfig):
    ref_var_name: str = ("snow_cover_fraction",)
    eval_var_name: str = ("snow_cover_fraction",)
    ref_fsc_step: int = (25,)
    sensor_zenith_analysis: bool = True


def generate_evaluation_io(
    analysis_type: str,
    working_folder: str,
    year: WinterYear,
    ref_product_name: str,
    eval_product_name: str,
    grid: GSGrid | None = UTM375mGrid(),
    period: slice | None = None,
) -> Tuple[xr.Dataset, xr.Dataset, str]:
    output_folder = f"{working_folder}/analyses/{analysis_type}"
    ref_time_series_name = f"WY_{year.from_year}_{year.to_year}_{ref_product_name}_{grid.name.lower()}.nc"
    eval_time_series_name = f"WY_{year.from_year}_{year.to_year}_{eval_product_name}_{grid.name.lower()}.nc"

    output_filename = f"{output_folder}/{analysis_type}_WY_{year.from_year}_{year.to_year}_{eval_product_name}_vs_{ref_product_name}_{grid.name.lower()}.nc"

    eval_time_series = xr.open_dataset(f"{working_folder}/time_series/{eval_time_series_name}", mask_and_scale=False)
    ref_time_series = xr.open_dataset(f"{working_folder}/time_series/{ref_time_series_name}", mask_and_scale=False)

    if period is not None:
        eval_time_series = eval_time_series.sel(time=period)
        ref_time_series = ref_time_series.sel(time=period)

    return ref_time_series, eval_time_series, output_filename


class EvaluationVsHighResBase(MountainBinner):
    def __init__(
        self,
        evaluation_config: EvaluationConfig,
        reference_analyzer: SnowCoverProductCompleteness,
        eval_analyzer: SnowCoverProductCompleteness,
    ) -> None:
        if reference_analyzer.classes["snow_cover"] != range(1, 101):
            # ref_fsc is not coded dynamically with change of reference dataset
            raise NotImplementedError("This class supposes that the reference snow cover fraction is encoded beween 0 and 100")
        self.config = evaluation_config
        self.ref_analyzer = reference_analyzer
        self.eval_analyzer = eval_analyzer

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

    @abc.abstractmethod
    def time_step_analysis(self):
        pass

    def prepare(self, distributed_data: xr.Dataset, bin_dict: Dict[str, BinGrouper]):
        """Use groupby to prepare binning.

        The resulting dataset is ready to apply a custom reduction using the map function. Example:

        mountain_binner.prepare(distributed_data, bin_dict).map(<your function>)

        Args:
            distributed_data (xr.DataArray | xr.Dataset): your geospatial dataset.
            bin_dict (Dict[str, BinGrouper]): the dictionary of bin to be used in xarray groupby

        """
        common_days = np.intersect1d(distributed_data["ref"]["time"], distributed_data["eval"]["time"])
        variable_and_auxiliary = self.stack_auxiliary_data(distributed_data=distributed_data)
        if self.config.sensor_zenith_analysis is not None:
            variable_and_auxiliary = variable_and_auxiliary.assign(
                sensor_zenith=distributed_data["eval"].data_vars["sensor_zenith_angle"].sel(time=common_days)
            )
        return variable_and_auxiliary.groupby(bin_dict)

    def index_and_rename_sza_coords(self, binned_data: xr.Dataset) -> xr.Dataset:
        bd = binned_data
        bd = bd.assign_coords(sensor_zenith_min=("sensor_zenith_bins", self.bins_min(bd, "sensor_zenith_bins")))
        bd = bd.assign_coords(sensor_zenith_max_max=("sensor_zenith_bins", self.bins_max(bd, "sensor_zenith_bins")))
        bd = bd.set_xindex("sensor_zenith_min")
        bd = bd.set_xindex("sensor_zenith_max")
        return bd

    def index_and_rename_ref_fsc_coords(self, binned_data: xr.Dataset) -> xr.Dataset:
        bd = binned_data
        bd = bd.assign_coords(ref_fsc_min=("ref_fsc_bins", self.bins_min(bd, "ref_fsc_bins")))
        bd = bd.assign_coords(ref_fsc_max_max=("ref_fsc_bins", self.bins_max(bd, "ref_fsc_bins")))
        bd = bd.set_xindex("ref_fsc_min")
        bd = bd.set_xindex("ref_fsc_max")
        return bd

    def launch_analysis(
        self, eval_time_series: xr.Dataset, ref_time_series: xr.Dataset, netcdf_export_path: str | None = None
    ):
        common_days = np.intersect1d(ref_time_series["time"], eval_time_series["time"])
        combined_dataset = xr.Dataset(
            {
                "ref": ref_time_series.data_vars[self.config.ref_var_name[0]].sel(time=common_days),
                "eval": eval_time_series.data_vars[self.config.eval_var_name[0]].sel(time=common_days),
            },
        )
        data_bins = self.create_default_bin_dict(altitude_step=300)
        if self.config.sensor_zenith_analysis:
            data_bins.update(sensor_zenith=self.sensor_zenith_bins())
        data_bins.update(ref_fsc=self.ref_fsc_bins())

        transformed = combined_dataset.groupby("time").map(
            self.transform(distributed_data=combined_dataset, bin_dict=data_bins, function=self.time_step_analysis())
        )
        transformed = self.index_and_rename_ref_fsc_coords()
        if self.config.sensor_zenith_analysis:
            transformed = self.index_and_rename_sza_coords(binned_data=transformed)
        if netcdf_export_path:
            logger.info(f"Exporting to {netcdf_export_path}")
            transformed.to_netcdf(netcdf_export_path, encoding=generate_xarray_compression_encodings(transformed))
        return transformed
