from pathlib import Path

import numpy as np
import xarray as xr

from evaluations.completeness import SnowCoverProductCompleteness, compute_area_of_class_mask
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

    def compute_union_valid_snow_area_new(
        self,
        dataset: xr.Dataset,
        fsc_threshold: float | None = None,
    ):
        logger.info(f"Processing time of the year {dataset.coords['time'].values[0].astype('M8[D]').astype('O')}")

        valid_mask_meteofrance = dataset.data_vars["meteofrance"] <= METEOFRANCE_CLASSES["forest_with_snow"]
        valid_mask_nasa = dataset.data_vars["nasa"] <= NASA_CLASSES["snow_cover"][-1]
        if fsc_threshold is not None:
            if fsc_threshold <= 0 or fsc_threshold > 1:
                raise ValueError(f"Invalid fsc threshold {fsc_threshold}. It must by between 0 and 1.")
            fsc_thresh_mask_meteofrance = (
                dataset.data_vars["meteofrance"] > fsc_threshold * METEOFRANCE_CLASSES["snow_cover"][-1]
            )
            valid_mask_meteofrance = valid_mask_meteofrance & fsc_thresh_mask_meteofrance
            fsc_thresh_mask_nasa = dataset.data_vars["nasa"] > fsc_threshold * NASA_CLASSES["snow_cover"][-1]
            valid_mask_nasa = valid_mask_nasa & fsc_thresh_mask_nasa

        union_valid_mask = valid_mask_meteofrance & valid_mask_nasa

        meteofrance_valid = dataset.data_vars["meteofrance"].where(union_valid_mask)
        nasa_valid = dataset.data_vars["nasa"].where(union_valid_mask)

        area_meteofrance_snow_binary = self.mf_analyzer.compute_snow_area(meteofrance_valid, consider_fraction=False)
        area_meteofrance_snow_fraction = self.mf_analyzer.compute_snow_area(meteofrance_valid, consider_fraction=True)
        meteofrance_forest_with_snow_mask = self.mf_analyzer.compute_mask_of_class(
            "forest_with_snow", meteofrance_valid
        ).compute()
        meteofrance_forest_without_snow_mask = self.mf_analyzer.compute_mask_of_class(
            "forest_without_snow", meteofrance_valid
        ).compute()
        meteofrance_forest_mask = meteofrance_forest_with_snow_mask | meteofrance_forest_without_snow_mask
        area_meteofrance_forest_with_snow_binary = compute_area_of_class_mask(meteofrance_forest_with_snow_mask)
        area_meteofrance_forest_with_snow_fraction = area_meteofrance_forest_with_snow_binary

        nasa_valid_no_forest = nasa_valid.where(~meteofrance_forest_mask)
        nasa_valid_forest = nasa_valid.where(meteofrance_forest_with_snow_mask)
        area_nasa_snow_binary = self.nasa_analyzer.compute_snow_area(nasa_valid_no_forest, consider_fraction=False)
        area_nasa_snow_fraction = self.nasa_analyzer.compute_snow_area(nasa_valid_no_forest, consider_fraction=True)
        area_nasa_forest_with_snow_binary = self.nasa_analyzer.compute_snow_area(nasa_valid_forest, consider_fraction=False)
        area_nasa_forest_with_snow_fraction = self.nasa_analyzer.compute_snow_area(nasa_valid_forest, consider_fraction=True)
        out_dataset = xr.Dataset(
            data_vars={
                "meteofrance": xr.DataArray(
                    [
                        area_meteofrance_snow_binary,
                        area_meteofrance_snow_fraction,
                        area_meteofrance_forest_with_snow_binary,
                        area_meteofrance_forest_with_snow_fraction,
                    ],
                    coords={"class_name": ["no_forest_binary", "no_forest_fraction", "forest_binary", "forest_fraction"]},
                    attrs={"units": "m²"},
                ),
                "nasa": xr.DataArray(
                    [
                        area_nasa_snow_binary,
                        area_nasa_snow_fraction,
                        area_nasa_forest_with_snow_binary,
                        area_nasa_forest_with_snow_fraction,
                    ],
                    coords={"class_name": ["no_forest_binary", "no_forest_fraction", "forest_binary", "forest_fraction"]},
                    attrs={"units": "m²"},
                ),
            }
        )
        print(out_dataset * 1e-9)
        return out_dataset

    def analyze_sce_valid_union_new(
        self,
        meteofrance_time_series: xr.Dataset,
        nasa_time_series: xr.Dataset,
        netcdf_export_path: str | None = None,
        fsc_threshold: float | None = None,
    ) -> xr.Dataset:
        common_days = np.intersect1d(meteofrance_time_series["time"], nasa_time_series["time"])
        both_products_dataset = xr.Dataset(
            {
                "meteofrance": meteofrance_time_series.data_vars["snow_cover_fraction"]
                .sel(time=common_days)
                .chunk({"time": 1}),
                "nasa": nasa_time_series.data_vars["snow_cover_fraction"].sel(time=common_days).chunk({"time": 1}),
            },
        )
        result = both_products_dataset.groupby("time").map(self.compute_union_valid_snow_area_new, [fsc_threshold])

        if netcdf_export_path:
            result.to_netcdf(netcdf_export_path)
        return result

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
        out_dataset = xr.Dataset(
            data_vars={
                "meteofrance": xr.DataArray(
                    [area_meteofrance, area_meteofrance_forest_with_snow],
                    coords={"class_name": ["snow_cover", "forest_with_snow"]},
                ),
                "nasa": nasa_results,
            }
        )
        print(out_dataset * 1e-9)
        return out_dataset

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
                .sel(time=common_days)
                .chunk({"time": 1}),
                "nasa": nasa_time_series.data_vars["snow_cover_fraction"].sel(time=common_days).chunk({"time": 1}),
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
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_3/analyses/sce_cross_comparison"
    nasa_prod = "nasa_l3"
    nasa_time_series_name = f"WY_{year.from_year}_{year.to_year}_{platform}_{nasa_prod}_res_{resolution}m.nc"
    meteofrance_time_series_name = f"WY_{year.from_year}_{year.to_year}_{platform}_meteofrance_l3_res_{resolution}m.nc"

    fsc_threshold = None

    outfile_name = f"sce_cross_comparison_WY_{year.from_year}_{year.to_year}_{platform}_meteofrance_{nasa_prod}_res_{resolution}m_fsc_thresh_{fsc_threshold}.nc"

    meteofrance_time_series_path = Path(f"{working_folder}").joinpath(meteofrance_time_series_name)
    meteofrance_time_series = xr.open_dataset(meteofrance_time_series_path).isel(time=slice(100, 105))

    nasa_time_series_path = Path(f"{working_folder}").joinpath(nasa_time_series_name)
    nasa_time_series = xr.open_dataset(nasa_time_series_path).isel(time=slice(100, 105))

    CrossComparisonSnowCoverExtent().analyze_sce_valid_union_new(
        meteofrance_time_series=meteofrance_time_series,
        nasa_time_series=nasa_time_series,
        fsc_threshold=fsc_threshold,
        netcdf_export_path=Path(f"{output_folder}").joinpath(outfile_name),
    )

    # CrossComparisonSnowCoverExtent().analyze_sce_valid_union(
    #     meteofrance_time_series=meteofrance_time_series,
    #     nasa_time_series=nasa_time_series,
    #     consider_fraction=True,
    #     forest_mask="yes",
    #     netcdf_export_path=Path(f"{working_folder}").joinpath(outfile_name),
    #     fsc_threshold=fsc_threshold,
    # )
