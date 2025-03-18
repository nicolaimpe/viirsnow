import os
from pathlib import Path

import xarray as xr

from compression import generate_xarray_compression_encodings
from evaluations.snow_cover_extent_cross_comparison import WinterYear
from fractional_snow_cover import nasa_ndsi_snow_cover_to_fraction
from geotools import mask_dataarray_with_vector_file
from grids import GeoGrid, UTM1kmGrid, UTM375mGrid, georef_data_array
from harmonisation.daily_composites import create_spatial_l3_nasa_composite
from logger_setup import default_logger as logger
from products.classes import NASA_CLASSES
from products.filenames import NASA_L3_SNOW_PRODUCTS, get_daily_nasa_filenames_per_product
from reprojections import reprojection_l3_nasa_to_grid


def create_v10a1_time_series(
    winter_year: WinterYear,
    output_grid: GeoGrid,
    viirs_data_folder: str,
    output_folder: str,
    output_name: str,
    roi_shapefile: str | None = None,
    platform: str = "SNPP",
    ndsi_to_fsc_regression: str | None = None,
):
    outpaths = []
    for day in winter_year.iterate_days():
        logger.info(f"Processing day {day}")
        product_id = NASA_L3_SNOW_PRODUCTS["Standard"][platform]

        day_files = get_daily_nasa_filenames_per_product(
            product_id=product_id, day=day, data_folder=viirs_data_folder, extension=".h5"
        )

        if len(day_files) == 0:
            logger.info(f"Skip day {day.date()} because 0 files were found on this day")
            continue
        try:
            nasa_daily_composite = create_spatial_l3_nasa_composite(day_files=day_files)
            nasa_daily_composite_reprojected = reprojection_l3_nasa_to_grid(
                nasa_dataset=nasa_daily_composite, output_grid=output_grid
            )

            if roi_file is not None:
                nasa_daily_composite_reprojected = mask_dataarray_with_vector_file(
                    data_array=nasa_daily_composite_reprojected.data_vars["NDSI_Snow_Cover"],
                    roi_file=roi_shapefile,
                    output_grid=output_grid,
                    fill_value=NASA_CLASSES["fill"][0],
                )
            # Apparently need to pop this attribute for correct encoding...not like it took me two hours to understand this :')
            # In practice, when a valid range attribute is encoded, a GDal driver reading the NetCDF will set all values outside this
            # range to NaN.
            # Since valid range in the V10A1 collection is {0,100}, i.e. the NDSI range, all other pixels (clouds, lakes etc.) are set to NaN
            # and that's not useful for the anamysis
            nasa_daily_composite_reprojected.attrs.pop("valid_range")
            # If we want not to encode the fill value like nodata
            # reprojected.data_vars["snow_cover"].attrs.pop("_FillValue")
        except OSError as e:
            logger.warning(f"Error {e} occured while processing VIIRS files. Skipping day {day.date()}.")
            continue

        if ndsi_to_fsc_regression is not None:
            snow_cover_fraction = nasa_ndsi_snow_cover_to_fraction(
                nasa_daily_composite_reprojected.values, method=ndsi_to_fsc_regression
            )
            snow_cover_fraction_data_array = xr.zeros_like(nasa_daily_composite_reprojected)
            snow_cover_fraction_data_array[:] = snow_cover_fraction

            nasa_composite = georef_data_array(nasa_daily_composite_reprojected, "NDSI_Snow_Cover", output_grid.crs)
            nasa_composite = nasa_composite.assign({"snow_cover_fraction": snow_cover_fraction_data_array})
            nasa_composite.data_vars["snow_cover_fraction"].attrs["NDSI_to_FSC_conversion"] = ndsi_to_fsc_regression

        nasa_composite = nasa_composite.expand_dims(time=[day])
        outpath = f"{output_folder}/{day.strftime('%Y%j')}.nc"
        outpaths.append(outpath)
        nasa_composite.to_netcdf(outpath)

    time_series = xr.open_mfdataset(outpaths, mask_and_scale=False)
    output_name = Path(f"{output_folder}/{output_name}")
    encodings = generate_xarray_compression_encodings(time_series)
    encodings.update(time={"calendar": "gregorian", "units": f"days since {str(year.from_year)}-10-01"})
    time_series.to_netcdf(output_name, encoding=encodings)
    [os.remove(file) for file in outpaths]


if __name__ == "__main__":
    # User inputs
    year = WinterYear(2023, 2024)
    grid_375m = UTM375mGrid()
    grid_1km = UTM1kmGrid()
    grid = grid_1km
    platform = "SNPP"
    product_collection = "V10A1"

    product_id = NASA_L3_SNOW_PRODUCTS["Standard"][platform]
    folder = f"/home/imperatoren/work/VIIRS_S2_comparison/data/{product_collection}/{product_id}"
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_3"
    output_name = f"WY_{year.from_year}_{year.to_year}_{platform}_nasa_l3_res_{grid.resolution}m.nc"
    roi_file = "/home/imperatoren/work/VIIRS_S2_comparison/data/vectorial/massifs/massifs.shp"

    create_v10a1_time_series(
        winter_year=year,
        output_grid=grid,
        viirs_data_folder=folder,
        output_folder=output_folder,
        output_name=output_name,
        roi_shapefile=roi_file,
        ndsi_to_fsc_regression="salomonson_appel",
    )
