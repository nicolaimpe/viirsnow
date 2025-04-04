import os

import numpy as np
import xarray as xr
from rasterio.enums import Resampling

from compression import generate_xarray_compression_encodings
from fractional_snow_cover import nasa_ndsi_snow_cover_to_fraction
from geotools import mask_dataarray_with_vector_file, reproject_using_grid, to_rioxarray
from grids import GeoGrid, UTM375mGrid, georef_data_array
from harmonisation.daily_composites import create_temporal_composite_nasa
from harmonisation.reprojections import reprojection_l3_nasa_to_grid
from logger_setup import default_logger as logger
from products.classes import NASA_CLASSES
from products.filenames import get_daily_nasa_filenames_per_product
from winter_year import WinterYear


def create_nasa_pseudo_l3_time_series(
    year: WinterYear,
    nasa_snow_cover_l2_folder: str,
    nasa_geometry_reprojected_folder: str,
    output_folder: str,
    output_name: str,
    output_grid: GeoGrid | None = None,
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
        # Naming conventions are identical...sort filenames should be sufficient for correct matching of granumes
        daily_snow_cover_files = sorted(
            get_daily_nasa_filenames_per_product(product_id="VNP10_UTM_375m", day=day, data_folder=nasa_snow_cover_l2_folder)
        )
        daily_geometry_files = sorted(
            get_daily_nasa_filenames_per_product(
                product_id="VNP03IMG_UTM_375m", day=day, data_folder=nasa_geometry_reprojected_folder
            )
        )

        if len(daily_snow_cover_files) == 0:
            logger.warning(f"No files found in date {day}. Skipping day.")
            continue
        if len(daily_geometry_files) == 0:
            logger.warning(f"No geometry files found in date {day}. Skipping day.")
            continue

        # 2. Create a composite from swath data using some criteria
        nasa_composite = create_temporal_composite_nasa(
            daily_snow_cover_files=daily_snow_cover_files, daily_geometry_files=daily_geometry_files
        )

        # 3. Reprojection to output grid
        # Use a specific function for snow cover data
        # Zenith angle data are resampled by a simple nearest (don't need to be very precise on that)
        if output_grid is not None:
            nasa_ndsi_snow_cover = nasa_composite.drop_vars("sensor_zenith")
            nasa_ndsi_snow_cover = reprojection_l3_nasa_to_grid(nasa_dataset=nasa_ndsi_snow_cover, output_grid=output_grid)
            nasa_view_angle = to_rioxarray(nasa_composite.drop_vars("NDSI_Snow_Cover"))
            nasa_view_angle = reproject_using_grid(
                dataset=nasa_view_angle,
                output_grid=grid,
                nodata=np.nan,
                resampling_method=Resampling.bilinear,
            )

        # 4. Clip on our area of interest...still a  lot of code to optimize
        if roi_shapefile is not None:
            nasa_ndsi_snow_cover = mask_dataarray_with_vector_file(
                data_array=nasa_ndsi_snow_cover.data_vars["NDSI_Snow_Cover"],
                roi_file=roi_shapefile,
                output_grid=output_grid,
                fill_value=NASA_CLASSES["fill"][0],
            )

            nasa_view_angle = mask_dataarray_with_vector_file(
                data_array=nasa_view_angle.data_vars["sensor_zenith"],
                roi_file=roi_shapefile,
                output_grid=output_grid,
                fill_value=np.nan,
            )
        # Convert to snow cover fraction
        nasa_snow_cover_fraction = xr.DataArray(
            data=nasa_ndsi_snow_cover_to_fraction(nasa_ndsi_snow_cover.values), coords=nasa_ndsi_snow_cover.coords
        )
        # Sorry for that it's just for georeferencing
        nasa_pseudo_l3 = xr.Dataset(
            {
                "NDSI_Snow_Cover": georef_data_array(nasa_ndsi_snow_cover, "NDSI_Snow_Cover", output_grid.crs).data_vars[
                    "NDSI_Snow_Cover"
                ],
                "snow_cover_fraction": georef_data_array(
                    nasa_snow_cover_fraction, "snow_cover_fraction", output_grid.crs
                ).data_vars["snow_cover_fraction"],
                "sensor_zenith": georef_data_array(nasa_view_angle, "sensor_zenith", output_grid.crs).data_vars[
                    "sensor_zenith"
                ],
            }
        ).rio.write_crs(output_grid.crs)

        nasa_pseudo_l3.data_vars["snow_cover_fraction"].rio.write_nodata(NASA_CLASSES["fill"][0], inplace=True)
        # Add time dimension
        nasa_pseudo_l3 = nasa_pseudo_l3.expand_dims(time=[day])

        # 5. Export to a temporary netcdf that will be removed in order to keep space in the RAM
        outpath = f"{output_folder}/{day.strftime('%Y%m%d')}.nc"
        outpaths.append(outpath)
        nasa_pseudo_l3.to_netcdf(
            outpath, encoding={"snow_cover_fraction": {"dtype": "uint8"}, "NDSI_Snow_Cover": {"dtype": "uint8"}}
        )

    # 6. Reopen all the exported netcdfs for each day with mfdataset (that is able to handle the RAM and save the time series),
    # concatenate on time dimension
    time_series = xr.open_mfdataset(outpaths, mask_and_scale=False)
    encodings = generate_xarray_compression_encodings(time_series)
    encodings.update(time={"calendar": "gregorian", "units": f"days since {str(year.from_year)}-10-01"})
    time_series.to_netcdf(f"{output_folder}/{output_name}", encoding=encodings)
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
    nasa_snow_cover_data_folder = f"{data_folder}/V10/VNP10_UTM_375m"
    nasa_reprojected_geometry_folder = f"{data_folder}/V03IMG/VNP03IMG_UTM_375m"
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_4"
    output_name = f"WY_{year.from_year}_{year.to_year}_{platform}_nasa_pseudo_l3_res_{grid.resolution}m.nc"
    roi_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/vectorial/massifs/massifs.shp"

    create_nasa_pseudo_l3_time_series(
        year=year,
        nasa_snow_cover_l2_folder=nasa_snow_cover_data_folder,
        nasa_geometry_reprojected_folder=nasa_reprojected_geometry_folder,
        output_folder=output_folder,
        output_name=output_name,
        output_grid=grid,
        roi_shapefile=roi_shapefile,
        platform=platform,
    )
