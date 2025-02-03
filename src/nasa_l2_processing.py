import xarray as xr
from pyresample import geometry, AreaDefinition, kd_tree
from products.classes import NASA_CLASSES
from geotools import georef_data_array
from grids import Grid
from fractional_snow_cover import nasa_ndsi_snow_cover_to_fraction
import numpy as np
from grids import DefaultGrid


def reproject_l2_nasa_to_grid(
    l2_nasa_filename: str,
    output_grid: Grid,
    output_filename: str | None = None,
    area_name: str = "France",
    area_description: str = "France CMS bounding box",
):
    l2_geoloc = xr.open_dataset(l2_nasa_filename, group="/GeolocationData")
    l2_data = xr.open_dataset(l2_nasa_filename, group="/SnowData").data_vars["NDSI_Snow_Cover"]

    selected = l2_data != NASA_CLASSES["bowtie_trim"][0]

    lons_modif = np.ma.masked_array(l2_geoloc.data_vars["longitude"], ~selected)
    lats_modif = np.ma.masked_array(l2_geoloc.data_vars["latitude"], ~selected)
    swath_def = geometry.SwathDefinition(lons=lons_modif, lats=lats_modif)

    area_def = AreaDefinition(
        area_id=area_name,
        description=area_description,
        proj_id=area_name,
        projection=output_grid.crs,
        width=output_grid.width,
        height=output_grid.height,
        area_extent=output_grid.extent_llx_lly_urx_ury,
    )
    reprojected_l2_data = kd_tree.resample_nearest(
        source_geo_def=swath_def,
        data=l2_data.values,
        target_geo_def=area_def,
        radius_of_influence=1000,
        fill_value=NASA_CLASSES["fill"][0],
        nprocs=1,
    )

    reprojected_l2_data = nasa_ndsi_snow_cover_to_fraction(reprojected_l2_data)
    reprojected_l2_data_array = xr.DataArray(
        data=reprojected_l2_data, coords={"y": output_grid.ycoords, "x": output_grid.xcoords}
    )

    reprojected_l2_dataset = georef_data_array(reprojected_l2_data_array, data_array_name="snow_cover", crs=output_grid.crs)
    if output_filename is not None:
        reprojected_l2_dataset.to_netcdf(output_filename)
    return reprojected_l2_dataset


if __name__ == "__main__":
    test_file = "/home/imperatoren/work/VIIRS_S2_comparison/data/V10/VNP10_NRT/VNP10_NRT.A2025029.1224.002.2025029164918.nc"
    output_filename = (
        "/home/imperatoren/work/VIIRS_S2_comparison/data/V10/VNP10_NRT_UTM/VNP10_NRT_UTM.A2025029.1224.002.2025029164918.nc"
    )
    output_grid = DefaultGrid()
    result = reproject_l2_nasa_to_grid(l2_nasa_filename=test_file, output_grid=output_grid, output_filename=output_filename)
    print(result)
