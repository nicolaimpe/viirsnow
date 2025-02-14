from pathlib import Path

import numpy as np
import rasterio.warp
import xarray as xr

from logger_setup import default_logger as logger
from products.classes import METEOFRANCE_CLASSES, NASA_CLASSES
from winter_year import WinterYear


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
