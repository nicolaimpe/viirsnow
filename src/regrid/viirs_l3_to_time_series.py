from typing import List

import xarray as xr
from geospatial_grid.gsgrid import GSGrid
from ndsi_fsc_calibration.regrid import V10A1Regrid, reprojection_l3_nasa_to_grid

from fractional_snow_cover import ndsi_snow_cover_to_fraction


class V10A1FSCRegrid(V10A1Regrid):
    def __init__(self, output_grid: GSGrid, data_folder: str, output_folder: str):
        super().__init__(output_grid, data_folder, output_folder)

    def create_spatial_composite(self, date_files: List[str]) -> xr.Dataset:
        """Create a reprojected daily VIIRS snow cover composite.

        The merged VIIRS mosaic is reprojected onto the target grid.

        Args:
            date_files (List[str]): List of VIIRS files for a single date.

        Returns:
            xr.Dataset: Reprojected daily dataset.
        """

        daily_spatial_composite = self.create_spatial_l3_nasa_viirs_composite(daily_snow_cover_files=date_files)
        nasa_snow_cover = reprojection_l3_nasa_to_grid(nasa_snow_cover=daily_spatial_composite, output_grid=self.grid)
        nasa_snow_cover.attrs.pop("valid_range")
        nasa_fsc = xr.DataArray(
            ndsi_snow_cover_to_fraction(nasa_snow_cover.values, snow_cover_ndsi_threshold=10, method="salomonson_appel"),
            coords=nasa_snow_cover.coords,
            dims=nasa_snow_cover.dims,
        )
        out_dataset = xr.Dataset({"NDSI_Snow_Cover": nasa_snow_cover, "snow_cover_fraction": nasa_fsc})
        return out_dataset
