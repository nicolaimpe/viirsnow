from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import rioxarray
import xarray as xr
from xarray.groupers import BinGrouper

from evaluations.completeness import SnowCoverProductCompleteness
from winter_year import WinterYear


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
    def sensor_zenith_bins():
        return BinGrouper(
            np.array([*np.arange(0, 90, 10), 255]),
            labels=np.array([*np.arange(10, 90, 10), 255]),
        )

    @staticmethod
    def ref_fsc_bins():
        return BinGrouper(
            np.array([-1, *np.arange(0, 99, 10), 99, 100, 205]),
            labels=np.array([0, *np.arange(10, 99, 10), 99, 100, 205]),
        )

    @staticmethod
    def forest_bins():
        return BinGrouper([-1, 0, 1], labels=[0, 1])

    @staticmethod
    def month_bins(winter_year: WinterYear):
        wy_datetime = winter_year.to_datetime()
        wy_datetime.extend([datetime(year=wy_datetime[-1].year, month=wy_datetime[-1].month + 1, day=1)])
        return BinGrouper(wy_datetime, labels=[month_datetime for month_datetime in wy_datetime[:-1]])

    def prepare_analysis(
        self,
        test_time_series: xr.Dataset,
        ref_time_series: xr.Dataset,
        sensor_zenith_analysis: bool = True,
        forest_mask_path: str | None = None,
    ) -> Tuple[xr.Dataset, Dict[str, BinGrouper]]:
        if ref_time_series.data_vars["snow_cover_fraction"].max() != 205:
            # This is supposed by limiting ref bins values to 205 just in ref_fsc_bins()
            # If we don' put a max value, then groupby for ref FSC bins might not work in case we have invalid data
            # but max value depends on the encoding so we have to put this
            raise NotImplementedError("This function supposes that the reference data maximum is 205")

        common_days = np.intersect1d(test_time_series["time"], ref_time_series["time"])

        combined_dataset = xr.Dataset(
            {
                "ref": ref_time_series.data_vars["snow_cover_fraction"].sel(time=common_days),
                "test": test_time_series.data_vars["snow_cover_fraction"].sel(time=common_days),
            },
        )

        analysis_bin_dict = dict(ref=self.ref_fsc_bins())

        if sensor_zenith_analysis:
            analysis_bin_dict.update(sensor_zenith=self.sensor_zenith_bins())
            combined_dataset = combined_dataset.assign(
                {"sensor_zenith": test_time_series.data_vars["sensor_zenith"].sel(time=common_days)}
            )

        if forest_mask_path is not None:
            forest_mask = rioxarray.open_rasterio(forest_mask_path)
            combined_dataset = combined_dataset.assign(forest_mask=forest_mask.sel(band=1).drop_vars("band"))
            analysis_bin_dict.update(forest_mask=self.forest_bins())

        combined_dataset = combined_dataset.drop_vars("spatial_ref")
        return combined_dataset, analysis_bin_dict
