from glob import glob
from pathlib import Path
from typing import List
import xarray as xr
import numpy as np
import geopandas as gpd
from metrics import WinterYear
from geotools import georef_data_array, gdf_to_binary_mask, reproject_dataset, dim_name, to_rioxarray
import os
from logger_setup import default_logger as logger
from grids import DefaultGrid, Grid, DefaultGrid_1km
from products.classes import NASA_CLASSES
from products.georef import modis_crs
from products.filenames import VIIRS_COLLECTION, get_daily_nasa_filenames_per_platform
from fractional_snow_cover import nasa_ndsi_snow_cover_to_fraction
from rasterio.enums import Resampling


def reprojection_module(nasa_dataset: xr.Dataset, output_grid: Grid) -> xr.Dataset:
    # Validity "zombie mask": wherever there is at least one non valid pixel, the output grid pixel is set as invalid (<-> cloud)
    nasa_dataset = to_rioxarray(nasa_dataset)

    # nasa_dataset = nasa_dataset.where(nasa_dataset <= NASA_CLASSES["snow_cover"][-1], NASA_CLASSES["fill"][0])

    data_array_name = [name for name in nasa_dataset]

    # Verify that we reproject one Data Array at time
    if len(data_array_name) == 1:
        data_array_name = data_array_name[0]
    else:
        raise NotImplementedError

    nasa_lakes_mask = nasa_dataset == NASA_CLASSES["water"][0]
    nasa_oceans_mask = nasa_dataset == NASA_CLASSES["water"][1]

    nasa_water_mask = nasa_lakes_mask | nasa_oceans_mask

    nasa_without_water = xr.where(nasa_water_mask == 0, nasa_dataset, 0, keep_attrs=True)

    nasa_validity_mask = reproject_dataset(
        nasa_without_water,
        new_crs=output_grid.crs,
        resampling=Resampling.max,
        transform=output_grid.affine,
        shape=output_grid.shape,
    )

    nasa_resampled = reproject_dataset(
        nasa_without_water.astype(np.float32),
        new_crs=output_grid.crs,
        resampling=Resampling.average,
        transform=output_grid.affine,
        shape=output_grid.shape,
    )

    nasa_oceans_mask_nearest = reproject_dataset(
        nasa_oceans_mask.astype("u1"),
        new_crs=output_grid.crs,
        resampling=Resampling.nearest,
        transform=output_grid.affine,
        shape=output_grid.shape,
    )
    nasa_lakes_mask_nearest = reproject_dataset(
        nasa_lakes_mask.astype("u1"),
        new_crs=output_grid.crs,
        resampling=Resampling.nearest,
        transform=output_grid.affine,
        shape=output_grid.shape,
    )
    # Compose the mask
    nasa_out_image = xr.where(
        nasa_validity_mask <= NASA_CLASSES["snow_cover"][-1], nasa_resampled.astype("u1"), nasa_validity_mask.astype("u1")
    )
    nasa_out_image = xr.where(nasa_oceans_mask_nearest, NASA_CLASSES["water"][1], nasa_out_image)
    nasa_out_image = xr.where(nasa_lakes_mask_nearest, NASA_CLASSES["water"][0], nasa_out_image)

    nasa_out_image.data_vars[data_array_name].attrs = nasa_dataset.data_vars[data_array_name].attrs
    nasa_out_image.data_vars[data_array_name].rio.write_nodata(NASA_CLASSES["fill"][0], inplace=True)
    nasa_out_image = nasa_out_image.drop_vars("spatial_ref")
    nasa_out_image = georef_data_array(
        nasa_out_image.data_vars[data_array_name], data_array_name=data_array_name, crs=output_grid.crs
    )
    return nasa_out_image


def create_nasa_composite(
    day_files: List[str], output_grid: Grid | None = None, roi_file: str | None = None
) -> xr.Dataset | None:
    day_data_arrays = []
    dims = dim_name(crs=modis_crs)
    for filepath in day_files:
        # try:
        logger.info(f"Processing product {Path(filepath).name}")

        product_grid_data_variable = xr.open_dataset(filepath, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D", engine="netcdf4")
        bin_size = xr.open_dataset(filepath, engine="netcdf4").attrs["CharacteristicBinSize"]
        nasa_l3_grid = Grid(
            resolution=bin_size,
            x0=product_grid_data_variable.coords["XDim"][0].values,
            y0=product_grid_data_variable.coords["YDim"][0].values,
            width=len(product_grid_data_variable.coords["XDim"]),
            height=len(product_grid_data_variable.coords["YDim"]),
        )
        ndsi_snow_cover = xr.open_dataset(
            filepath, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields", engine="netcdf4"
        ).data_vars["NDSI_Snow_Cover"]

        ndsi_snow_cover = ndsi_snow_cover.rename({"XDim": dims[1], "YDim": dims[0]}).assign_coords(
            coords={dims[0]: nasa_l3_grid.ycoords, dims[1]: nasa_l3_grid.xcoords}
        )

        day_data_arrays.append(georef_data_array(data_array=ndsi_snow_cover, data_array_name="NDSI_Snow_Cover", crs=modis_crs))

    merged_day_dataset = xr.combine_by_coords(day_data_arrays, data_vars="minimal").astype(np.uint8)
    # merged_day_dataset.data_vars["NDSI_Snow_Cover"].attrs.pop("valid_range")
    # merged_day_dataset.to_netcdf("./output_folder/test_merged.nc")

    if output_grid is None:
        raise NotImplementedError(
            "Output grid must be specified or a way to extract the output grid from the NASA products has to be implemented."
        )

    day_dataset_reprojected = reprojection_module(nasa_dataset=merged_day_dataset, output_grid=output_grid)

    if roi_file is not None:
        roi_mask = gdf_to_binary_mask(
            gdf=gpd.read_file(roi_file),
            grid=output_grid,
        )

        masked = day_dataset_reprojected.data_vars["NDSI_Snow_Cover"].values * roi_mask.data_vars["binary_mask"].values
        masked[roi_mask.data_vars["binary_mask"].values == 0] = NASA_CLASSES["fill"][0]
        day_dataset_reprojected.data_vars["NDSI_Snow_Cover"][:] = masked

    # Apparently need to pop this attribute for correct encoding...not like it took me two hours to understand this :')
    # In practice, when a valid range attribute is encoded, a GDal driver reading the NetCDF will set all values outside this
    # range to NaN.
    # Since valid range in the V10A1 collection is {0,100}, i.e. the NDSI range, all other pixels (clouds, lakes etc.) are set to NaN
    # and that's not useful for the anamysis
    day_dataset_reprojected.data_vars["NDSI_Snow_Cover"].attrs.pop("valid_range")
    # If we want not to encode the fill value like nodata
    # reprojected.data_vars["snow_cover"].attrs.pop("_FillValue")

    return day_dataset_reprojected


def create_v10a1_time_series(
    winter_year: WinterYear,
    output_grid: Grid,
    viirs_data_folder: str,
    output_folder: str,
    output_name: str,
    roi_shapefile: str | None = None,
    platform: str = "SuomiNPP",
    ndsi_to_fsc_regression: str | None = None,
):
    # Treat user inputs
    viirs_data_filepaths = glob(str(Path(f"{viirs_data_folder}/V*10*{str(year.from_year)}*.00{VIIRS_COLLECTION}.*h5")))
    viirs_data_filepaths.extend(glob(str(Path(f"{viirs_data_folder}/V*10*{str(year.to_year)}*.00{VIIRS_COLLECTION}.*h5"))))

    outpaths = []
    for day in winter_year.iterate_days():
        if day.year == 2024:
            continue
        logger.info(f"Processing day {day}")
        day_files, n_day_files = get_daily_nasa_filenames_per_platform(
            platform=platform, year=day.year, day=day.day_of_year, viirs_data_filepaths=viirs_data_filepaths
        )
        if n_day_files == 0:
            logger.info(f"Skip day {day.date()} because 0 files were found on this day")
            continue
        try:
            nasa_composite = create_nasa_composite(day_files=day_files, output_grid=output_grid, roi_file=roi_shapefile)
        except OSError as e:
            logger.warning(f"Error {e} occured while reading VIIRS files. Skipping day {day.date()}.")
            continue

        if ndsi_to_fsc_regression is not None:
            snow_cover_fraction = nasa_ndsi_snow_cover_to_fraction(
                nasa_composite.data_vars["NDSI_Snow_Cover"].values, method=ndsi_to_fsc_regression
            )
            snow_cover_fraction_data_array = xr.zeros_like(nasa_composite.data_vars["NDSI_Snow_Cover"])
            snow_cover_fraction_data_array[:] = snow_cover_fraction
            nasa_composite = nasa_composite.assign({"snow_cover_fraction": snow_cover_fraction_data_array})
            nasa_composite.data_vars["snow_cover_fraction"].attrs["NDSI_to_FSC_conversion"] = ndsi_to_fsc_regression

        nasa_composite = nasa_composite.expand_dims(time=[day])
        outpath = f"{output_folder}/{day.strftime('%Y%j')}.nc"
        outpaths.append(outpath)
        nasa_composite.to_netcdf(outpath)

    time_series = xr.open_mfdataset(outpaths)
    output_name = Path(f"{output_folder}/{output_name}")
    time_series.to_netcdf(
        output_name,
        encoding={
            "time": {"calendar": "gregorian", "units": f"days since {str(year.from_year)}-10-01"},
        },
    )
    [os.remove(file) for file in outpaths]


if __name__ == "__main__":
    # User inputs
    year = WinterYear(2024, 2025)
    grid_375m = DefaultGrid()
    grid_1km = DefaultGrid_1km()
    grid = grid_375m

    platform = "SuomiNPP"
    folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/V10A1/VNP10A1"
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/version_3"
    output_name = f"WY_{year.from_year}_{year.to_year}_{platform}_nasa_l3_time_series_res_{grid.resolution}m.nc"
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
