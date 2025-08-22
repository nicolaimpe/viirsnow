from dataclasses import dataclass

import numpy as np
import xarray as xr
from xarray.groupers import BinGrouper


@dataclass
class MountainParams:
    slope_map_path: str | None = None
    aspect_map_path: str | None = None
    dem_path: str | None = None
    forest_mask_path: str | None = None
    


class MountainParametrization:
    @staticmethod
    def forest_bins() -> BinGrouper:
        return BinGrouper(np.array([-1, 0, 1]), labels=["no_forest", "forest"], right=True)

    @staticmethod
    def sub_roi_bins() -> BinGrouper:
        return BinGrouper(
            np.array([0, 1, 2, 3, 4, 5, 6]), labels=["Alps", "Pyrenees", "Corse", "Massif Central", "Jura", "Vosges"]
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

    @staticmethod
    def altitude_bins(altitude_band: int = 600) -> BinGrouper:
        return BinGrouper(
            np.array([0, *np.arange(900, 3900, altitude_band), 4800]),
            labels=np.array([0, *np.arange(900, 3900, altitude_band)]),
        )

    def semidistributed_parametrization(self, dataset: xr.Dataset, config: MountainParams):
        analysis_bin_dict = {}

        if config.forest_mask_path is not None:
            forest_mask = xr.open_dataarray(config.forest_mask_path)
            dataset = dataset.assign(forest_mask=forest_mask.sel(band=1).drop_vars("band"))
            analysis_bin_dict.update(forest_mask=self.forest_bins())

        if config.slope_map_path is not None:
            slope_map = xr.open_dataarray(config.slope_map_path)
            dataset = dataset.assign(slope=slope_map.sel(band=1).drop_vars("band"))
            analysis_bin_dict.update(slope=self.slope_bins())

        if config.aspect_map_path is not None:
            aspect_map = xr.open_dataarray(config.aspect_map_path)
            aspect_map = self.aspect_map_transform(aspect_map.sel(band=1).drop_vars("band"))
            dataset = dataset.assign(aspect=aspect_map)
            analysis_bin_dict.update(aspect=self.aspect_bins())

        if config.dem_path is not None:
            dem_map = xr.open_dataarray(config.dem_path)
            dataset = dataset.assign(altitude=dem_map.sel(band=1).drop_vars("band"))
            analysis_bin_dict.update(altitude=self.altitude_bins())

        return dataset, analysis_bin_dict
