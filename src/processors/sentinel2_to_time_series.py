import os

import xarray as xr

from daily_composites import create_spatial_s2_composite
from evaluations.metrics import WinterYear
from geotools import mask_dataarray_with_vector_file
from grids import Grid, UTM375mGrid
from logger_setup import default_logger as logger
from products.classes import S2_CLASSES
from products.filenames import get_all_s2_files_of_winter_year, get_datetime_from_s2_filepath


def add_time_dim(data_array: xr.DataArray) -> xr.DataArray:
    return data_array.expand_dims(time=[get_datetime_from_s2_filepath(s2_file=data_array.encoding["source"])])


def create_s2_time_series(
    winter_year: WinterYear, output_grid: Grid, s2_folder: str, output_folder: str, roi_shapefile: str | None = None
):
    files = get_all_s2_files_of_winter_year(s2_folder, winter_year=winter_year)

    out_tmp_paths = []

    for day in winter_year.iterate_days():
        logger.info(f"Processing day {day}")
        day_files = [file for file in files if day.strftime("%Y%m%d") in file]
        if len(day_files) == 0:
            logger.info(f"Skip day {day.date()} because 0 files were found on this day")
            continue
        daily_composite = create_spatial_s2_composite(day_files=day_files, output_grid=output_grid)

        if roi_shapefile is not None:
            daily_composite = mask_dataarray_with_vector_file(
                data_array=daily_composite.data_vars["snow_cover_fraction"],
                roi_file=roi_shapefile,
                output_grid=output_grid,
                fill_value=S2_CLASSES["nodata"][0],
            )

        out_path = f"{str(output_folder)}/{day.strftime('%Y%m%d')}.nc"
        out_tmp_paths.append(out_path)

        daily_composite = daily_composite.expand_dims(time=[day])
        daily_composite.to_netcdf(out_path)

    all_data = xr.open_mfdataset(out_tmp_paths, mask_and_scale=False)
    all_data.to_netcdf(f"{output_folder}/WY_{winter_year.from_year}_{winter_year.to_year}_S2_res_{output_grid.resolution}m.nc")
    [os.remove(file) for file in out_tmp_paths]


if __name__ == "__main__":
    year = WinterYear(2023, 2024)
    massifs_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/vectorial/massifs/massifs.shp"
    s2_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/S2"
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_3"
    grid = UTM375mGrid()
    create_s2_time_series(
        winter_year=year, output_grid=grid, s2_folder=s2_folder, output_folder=output_folder, roi_shapefile=massifs_shapefile
    )
