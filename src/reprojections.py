import numpy as np
import xarray as xr
from pyresample import kd_tree
from pyresample.geometry import AreaDefinition, SwathDefinition
from rasterio.enums import Resampling

from compression import generate_xarray_compression_encodings
from geotools import georef_data_array, reproject_dataset, to_rioxarray
from grids import Grid
from products.classes import NASA_CLASSES


def extract_swath_lon_lats(
    l2_geolocation_data_group: xr.Dataset, bowtie_trim_mask: xr.DataArray | None = None
) -> SwathDefinition:
    if bowtie_trim_mask is not None:
        lons_modif = np.ma.masked_array(l2_geolocation_data_group.data_vars["longitude"], ~bowtie_trim_mask)
        lats_modif = np.ma.masked_array(l2_geolocation_data_group.data_vars["latitude"], ~bowtie_trim_mask)
        swath_def = SwathDefinition(lons=lons_modif, lats=lats_modif)
    else:
        lons = np.ma.masked_array(l2_geolocation_data_group.data_vars["longitude"])
        lats = np.ma.masked_array(l2_geolocation_data_group.data_vars["latitude"])
        swath_def = SwathDefinition(lons=lons, lats=lats)
    return swath_def


def reproject_l2_nasa_to_grid(
    output_grid: Grid,
    l2_geolocation_dataset: xr.Dataset,
    l2_dataset: xr.Dataset,
    bowtie_trim_mask: xr.DataArray | None = None,
    output_filename: str | None = None,
    area_name: str = "France",
    area_description: str = "France CMS bounding box",
    radius_of_influence: float = 1000,
):
    swath_def = extract_swath_lon_lats(l2_geolocation_data_group=l2_geolocation_dataset, bowtie_trim_mask=bowtie_trim_mask)
    area_def = AreaDefinition(
        area_id=area_name,
        description=area_description,
        proj_id=area_name,
        projection=output_grid.crs,
        width=output_grid.width,
        height=output_grid.height,
        area_extent=output_grid.extent_llx_lly_urx_ury,
    )

    reprojected_data_vars = {}
    for data_var_name, data_var in l2_dataset.items():
        reprojected_l2_data = kd_tree.resample_nearest(
            source_geo_def=swath_def,
            data=data_var.values,
            target_geo_def=area_def,
            radius_of_influence=radius_of_influence,
            fill_value=NASA_CLASSES["fill"][0],
            nprocs=8,
        )
        # That's ugly sorry me of the future. Basically we need to force the data array georeferecing with its attributes
        # but then we extract it to be able to use xr.assign later
        reprojected_data_vars.update(
            {
                data_var_name: georef_data_array(
                    xr.DataArray(
                        data=reprojected_l2_data,
                        coords={"y": output_grid.ycoords, "x": output_grid.xcoords},
                    ),
                    data_array_name=data_var_name,
                    crs=output_grid.crs,
                ).data_vars[data_var_name]
            }
        )

    output_dataset = xr.Dataset(reprojected_data_vars)

    if output_filename is not None:
        output_dataset.to_netcdf(
            output_filename,
            encoding=generate_xarray_compression_encodings(output_dataset),
        )
    return output_dataset


def reprojection_l3_nasa_to_grid(nasa_dataset: xr.Dataset, output_grid: Grid) -> xr.Dataset:
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
        nasa_validity_mask <= NASA_CLASSES["snow_cover"][-1],
        nasa_resampled.astype("u1"),
        nasa_validity_mask.astype("u1"),
    )
    nasa_out_image = xr.where(nasa_oceans_mask_nearest, NASA_CLASSES["water"][1], nasa_out_image)
    nasa_out_image = xr.where(nasa_lakes_mask_nearest, NASA_CLASSES["water"][0], nasa_out_image)

    nasa_out_image.data_vars[data_array_name].attrs = nasa_dataset.data_vars[data_array_name].attrs
    nasa_out_image.data_vars[data_array_name].rio.write_nodata(NASA_CLASSES["fill"][0], inplace=True)
    nasa_out_image = nasa_out_image.drop_vars("spatial_ref")
    nasa_out_image = georef_data_array(
        nasa_out_image.data_vars[data_array_name],
        data_array_name=data_array_name,
        crs=output_grid.crs,
    )
    return nasa_out_image
