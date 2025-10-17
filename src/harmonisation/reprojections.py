import numpy as np
import xarray as xr
from rasterio.enums import Resampling

from compression import generate_xarray_compression_encodings
from geotools import reproject_dataset, reproject_using_grid
from grids import GeoGrid, georef_netcdf_rioxarray
from products.classes import METEOFRANCE_CLASSES, NASA_CLASSES, S2_CLASSES


def reprojection_l3_nasa_to_grid(nasa_snow_cover: xr.DataArray, output_grid: GeoGrid) -> xr.DataArray:
    # Validity "zombie mask": wherever there is at least one non valid pixel, the output grid pixel is set as invalid (<-> cloud)
    # nasa_dataset = nasa_dataset.where(nasa_dataset <= NASA_CLASSES["snow_cover"][-1], NASA_CLASSES["fill"][0])

    resampled_max = reproject_using_grid(
        nasa_snow_cover,
        output_grid=output_grid,
        resampling_method=Resampling.max,
        nodata=NASA_CLASSES["fill"][0],
    )

    resampled_average = reproject_using_grid(
        nasa_snow_cover,
        output_grid=output_grid,
        resampling_method=Resampling.average,
    )

    resampled_nearest = reproject_using_grid(
        nasa_snow_cover,
        output_grid=output_grid,
        resampling_method=Resampling.nearest,
    )

    invalid_mask = resampled_max > NASA_CLASSES["snow_cover"][-1]
    water_mask = resampled_nearest == NASA_CLASSES["water"][0] | NASA_CLASSES["water"][1]
    valid_qualitative_mask = water_mask

    out_snow_cover = resampled_average.where(invalid_mask == False, resampled_max)
    # We readd water resempled with nearest
    out_snow_cover = out_snow_cover.where(valid_qualitative_mask == False, resampled_nearest)

    return out_snow_cover.astype("u1")


def reprojection_l3_meteofrance_to_grid(meteofrance_snow_cover: xr.DataArray, output_grid: GeoGrid) -> xr.DataArray:
    # Validity "zombie mask": wherever there is at least one non valid pixel, the output grid pixel is set as invalid (<-> cloud)
    # nasa_dataset = nasa_dataset.where(nasa_dataset <= NASA_CLASSES["snow_cover"][-1], NASA_CLASSES["fill"][0])

    resampled_max = reproject_using_grid(
        meteofrance_snow_cover,
        output_grid=output_grid,
        resampling_method=Resampling.max,
        nodata=METEOFRANCE_CLASSES["fill"][0],
    )

    # Tricky forest with snow when resampling using average
    # Whenever a resampled pixel includes forest with snow mask, a quantitative estimation connot be performed unless we choose a FSC value for forest with snow
    # The solution would be to resample forest with snow using max, but this is problematic when forest with snow is next to no snow because it increases the snow detections
    # Therefore we set it to 50% FSC (which means 100 in meteofrance encoding).
    # The contingency analysis will not be biased. The quantitative analysis will be more uncertain and perhaps biaised. The recommendation is to use a forest mask resampled with max for quantitative analysis
    resampled_average = reproject_using_grid(
        meteofrance_snow_cover.where(meteofrance_snow_cover <= METEOFRANCE_CLASSES["forest_with_snow"][0], 0)
        .where(meteofrance_snow_cover != METEOFRANCE_CLASSES["forest_with_snow"][0], 100)
        .astype("f4"),
        output_grid=output_grid,
        resampling_method=Resampling.average,
    )

    resampled_nearest = reproject_using_grid(
        meteofrance_snow_cover,
        output_grid=output_grid,
        resampling_method=Resampling.nearest,
    )

    water_mask = resampled_nearest == METEOFRANCE_CLASSES["water"][0]
    forest_without_snow_mask = resampled_nearest == METEOFRANCE_CLASSES["forest_without_snow"][0]
    forest_with_snow_mask = resampled_nearest == METEOFRANCE_CLASSES["forest_with_snow"][0]

    cloud_mask = resampled_max == METEOFRANCE_CLASSES["clouds"][0]
    nodata_mask = resampled_max == METEOFRANCE_CLASSES["nodata"][0]

    invalid_mask = cloud_mask | nodata_mask

    # We exclude these values from the next resampling operations
    valid_qualitative_mask = water_mask | forest_without_snow_mask | forest_with_snow_mask  # | no_snow_mask
    out_snow_cover = resampled_average.where(valid_qualitative_mask == False, resampled_nearest)
    out_snow_cover = out_snow_cover.where(invalid_mask == False, resampled_max)
    return out_snow_cover.astype("u1")


def resample_s2_to_grid(s2_dataset: xr.Dataset, output_grid: GeoGrid) -> xr.DataArray:
    # 250m resolution FSC from FSCOG S2 product with a "zombie" nodata mask

    # Validity "zombie mask": wherever there is at least one non valid pixel, the output grid pixel is set as invalid (<-> cloud)
    s2_validity_mask = reproject_dataset(
        s2_dataset,
        new_crs=output_grid.crs,
        resampling=Resampling.max,
        nodata=S2_CLASSES["nodata"][0],
        transform=output_grid.affine,
        shape=output_grid.shape,
    )

    # Aggregate the dataset at 250 m
    s2_aggregated = reproject_dataset(
        s2_dataset.astype(np.float32),
        new_crs=output_grid.crs,
        resampling=Resampling.average,
        nodata=S2_CLASSES["nodata"][0],
        transform=output_grid.affine,
        shape=output_grid.shape,
    )

    # Compose the mask
    s2_out_image = xr.where(s2_validity_mask <= S2_CLASSES["snow_cover"][-1], s2_aggregated.astype("u1"), s2_validity_mask)
    s2_out_image.rio.write_nodata(S2_CLASSES["fill"][0], inplace=True)

    return s2_out_image
