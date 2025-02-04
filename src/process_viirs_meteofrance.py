import os
from pathlib import Path

import xarray as xr

from daily_composites import create_temporal_l2_composite_meteofrance
from geotools import mask_dataarray_with_vector_file
from grids import Grid
from logger_setup import default_logger as logger
from metrics import WinterYear
from products.classes import METEOFRANCE_CLASSES
from products.filenames import get_daily_meteofrance_filenames_per_platform
from reprojections import reprojection_l3_meteofrance_to_grid


def create_meteofrance_time_series(
    year: WinterYear,
    viirs_data_folder: str,
    output_folder: str,
    output_grid: Grid | None = None,
    roi_shapefile: str | None = None,
    platform: str = "Suomi-NPP",
):
    outpaths = []
    for day in year.iterate_days():
        logger.info(f"Processing day {day}")
        daily_files = get_daily_meteofrance_filenames_per_platform(
            platform=platform, day=day, viirs_data_folder=viirs_data_folder
        )
        if len(daily_files) == 0:
            logger.warning(f"No data fuond in date {day}")
            continue
        meteofrance_composite = create_temporal_l2_composite_meteofrance(daily_files=daily_files, roi_file=roi_shapefile)

        if output_grid is not None:
            meteofrance_composite = reprojection_l3_meteofrance_to_grid(
                meteofrance_dataset=meteofrance_composite, output_grid=output_grid
            )
        if roi_shapefile is not None:
            meteofrance_composite.data_vars["snow_cover"] = mask_dataarray_with_vector_file(
                data_array=meteofrance_composite.data_vars["snow_cover"],
                roi_file=roi_shapefile,
                fill_value=METEOFRANCE_CLASSES["fill"],
            )

        meteofrance_composite = meteofrance_composite.expand_dims(time=[day])
        outpath = f"{output_folder}/{day.strftime('%Y%m%d')}.nc"
        outpaths.append(outpath)
        meteofrance_composite.to_netcdf(outpath)

    time_series = xr.open_mfdataset(outpaths)
    output_name = Path(f"{output_folder}/WY_{year.from_year}_{year.to_year}_{platform}_meteofrance_time_series.nc")
    time_series.to_netcdf(
        output_name, encoding={"time": {"calendar": "gregorian", "units": f"days since {str(year.from_year)}-10-01"}}
    )
    [os.remove(file) for file in outpaths]
    return


if __name__ == "__main__":
    # User inputs
    year = WinterYear(2023, 2024)
    folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/EOFR62"
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_3"
    roi_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/vectorial/massifs/massifs.shp"

    create_meteofrance_time_series(
        year=year, viirs_data_folder=folder, output_folder=output_folder, roi_shapefile=roi_shapefile
    )
