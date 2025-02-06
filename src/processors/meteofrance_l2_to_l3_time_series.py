import os

import numpy as np
import xarray as xr
from rasterio.enums import Resampling

from daily_composites import create_temporal_l2_composite_meteofrance
from geotools import georef_data_array, mask_dataarray_with_vector_file, reproject_using_grid, to_rioxarray
from grids import Grid, UTM375mGrid
from logger_setup import default_logger as logger
from metrics import WinterYear
from products.classes import METEOFRANCE_CLASSES
from products.filenames import KNOWN_COLLECTIONS, get_daily_meteofrance_filenames, get_daily_nasa_filenames_per_product
from reprojections import reprojection_l3_meteofrance_to_grid


def create_meteofrance_time_series(
    year: WinterYear,
    meteofrance_data_folder: str,
    nasa_geometry_reprojected_folder: str,
    output_folder: str,
    output_name: str,
    output_grid: Grid | None = None,
    roi_shapefile: str | None = None,
    platform: str = "SNPP",
):
    """
    Process Météofrance data to obtain a time series on a winter year that will be used in analysis

    The geometry NASA product is used to obtain the daily composite, sensor zenith angle criterion.
    Most manipulations with NASA geometry file are supposed to be temporary. In fact there might be some inconsistencies
    since MétéoFrance uses SDR reflectances that are not the same as NASA L1B product used for NASA L2 snow cover.
    The geometry L1B product (V03IMG) is coupled with the reflectance product (V02IMG) and not with VIIRS SDR.

    Ideally, in a future iteration we would like the Météo-France product to already contain the sensor zenith angle information.
    """
    outpaths = []
    for day in year.iterate_days():
        logger.info(f"Processing day {day}")

        # 1. Collect snow cover and geometry (for sensor zenith angle) files for a given day
        daily_snow_cover_files = get_daily_meteofrance_filenames(day=day, data_folder=meteofrance_data_folder)
        daily_geometry_files = get_daily_nasa_filenames_per_product(
            product_id=KNOWN_COLLECTIONS["V03IMG"]["Standard"][platform],
            day=day,
            data_folder=nasa_geometry_reprojected_folder,
        )
        if len(daily_snow_cover_files) == 0:
            logger.warning(f"No Météo-France data data found in date {day}. Skipping day.")
            continue
        if len(daily_geometry_files) == 0:
            logger.warning(f"No geometry data found in date {day}. Skipping day.")
            continue

        # 2. Create a composite from swath data using some criteria
        meteofrance_composite = create_temporal_l2_composite_meteofrance(
            daily_snow_cover_files=daily_snow_cover_files, daily_geometry_files=daily_geometry_files
        )

        # 3. Reprojection to output grid
        # Use a specific function for snow cover data
        # Zenith angle data are resampled by a simple nearest (don't need to be very precise on that)
        if output_grid is not None:
            meteofrance_snow_cover = meteofrance_composite.drop_vars("sensor_zenith")
            meteofrance_snow_cover = reprojection_l3_meteofrance_to_grid(
                meteofrance_dataset=meteofrance_snow_cover, output_grid=output_grid
            )
            meteofrance_view_angle = to_rioxarray(meteofrance_composite.drop_vars("fractional_snow_cover"))
            meteofrance_view_angle = reproject_using_grid(
                dataset=meteofrance_view_angle,
                output_grid=grid,
                nodata=np.nan,
                resampling_method=Resampling.nearest,
            )

        # 4. Clip on our area of interest...stilla  lot of code to optimize
        if roi_shapefile is not None:
            meteofrance_snow_cover = mask_dataarray_with_vector_file(
                data_array=meteofrance_snow_cover.data_vars["fractional_snow_cover"],
                roi_file=roi_shapefile,
                output_grid=output_grid,
                fill_value=METEOFRANCE_CLASSES["fill"][0],
            )

            meteofrance_view_angle = mask_dataarray_with_vector_file(
                data_array=meteofrance_view_angle.data_vars["sensor_zenith"],
                roi_file=roi_shapefile,
                output_grid=output_grid,
                fill_value=np.nan,
            )

        # Sorry for that it's just for georeferencing
        meteofrance_composite = xr.Dataset(
            {
                "fractional_snow_cover": georef_data_array(
                    meteofrance_snow_cover, "fractional_snow_cover", output_grid.crs
                ).data_vars["fractional_snow_cover"],
                "sensor_zenith": georef_data_array(meteofrance_view_angle, "sensor_zenith", output_grid.crs).data_vars[
                    "sensor_zenith"
                ],
            }
        ).rio.write_crs(output_grid.crs)
        # Add time dimension
        meteofrance_composite = meteofrance_composite.expand_dims(time=[day])

        # 5. Export to a temporary netcdf that will be removed in order to keep space in the RAM
        outpath = f"{output_folder}/{day.strftime('%Y%m%d')}.nc"
        outpaths.append(outpath)
        meteofrance_composite.to_netcdf(outpath)

    # 6. Reopen all the exported netcdfs for each day with mfdataset (that is able to handle the RAM and save the time series),
    # concatenate on time dimension
    time_series = xr.open_mfdataset(outpaths)
    # !!!! PUT some compression here
    time_series.to_netcdf(
        f"{output_folder}/{output_name}",
        encoding={"time": {"calendar": "gregorian", "units": f"days since {str(year.from_year)}-10-01"}},
    )
    # Clean output folder from temporary files
    [os.remove(file) for file in outpaths]
    return


if __name__ == "__main__":
    # User inputs
    year = WinterYear(2023, 2024)
    grid = UTM375mGrid()
    from grids import UTM1kmGrid

    grid = UTM1kmGrid()

    platform = "SNPP"  # SNPP, JPSS1

    data_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/"
    meteofrance_data_folder = f"{data_folder}/EOFR62"
    nasa_reprojected_geometry_folder = f"{data_folder}/V03IMG/VNP03IMG_GEO_250m"
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_3"
    output_name = f"WY_{year.from_year}_{year.to_year}_{platform}_meteofrance_time_series_res_{grid.resolution}m.nc"
    roi_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/vectorial/massifs/massifs.shp"

    create_meteofrance_time_series(
        year=year,
        meteofrance_data_folder=meteofrance_data_folder,
        nasa_geometry_reprojected_folder=nasa_reprojected_geometry_folder,
        output_folder=output_folder,
        output_name=output_name,
        output_grid=grid,
        roi_shapefile=roi_shapefile,
        platform=platform,
    )
