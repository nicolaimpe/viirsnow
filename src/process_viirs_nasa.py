from glob import glob
from pathlib import Path
from typing import Dict, List
import xarray as xr
import numpy as np
from datetime import datetime
import numpy.typing as npt
import logging
import re
import geopandas as gpd
from pathlib import Path
import pandas as pd

# Module configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# User inputs
year = 2018
platform = "SuomiNPP"
folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/V10A1"
output_folder = "./output_folder/completeness_analysis"
roi_file = "./data/vectorial/france_bbox.shp"


# Treat user inputs
france_footprint = gpd.read_file(Path(roi_file))
viirs_data_filepaths = glob(str(Path(f"{folder}/V*10*{str(year)}*h1[7-8]v0[3-4]*.002.*h5")))
france_geometry = france_footprint["geometry"].bounds
output_name = Path(f"{year}_{platform}_nasa_time_series_fsc.nc")

# Hardcode some parameters
LAT = "lat"
LON = "lon"
PROJ4_MODIS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
platforms_products_dict = {"SuomiNPP": "VNP", "JPSS-1": "VJ1"}


def georef_data_array(data_array: xr.DataArray, data_array_name: str, crs_wkt: str) -> xr.Dataset:
    """
    Turn a DataArray into a Dataset  for which the GDAL driver (GDAL and QGIS) is able to read the georeferencing
    https://github.com/pydata/xarray/issues/2288
    https://gis.stackexchange.com/questions/230093/set-projection-for-netcdf4-in-python
    """

    data_array.coords[LAT].attrs["axis"] = "Y"
    data_array.coords[LON].attrs["axis"] = "X"
    data_array.attrs["grid_mapping"] = "spatial_ref"

    crs_variable = xr.DataArray(0)
    crs_variable.attrs["spatial_ref"] = crs_wkt

    georeferenced_dataset = xr.Dataset({data_array_name: data_array, "spatial_ref": crs_variable})
    return georeferenced_dataset


def timestamp_to_datetime(observation_timestamp: str) -> datetime:
    return datetime.strptime(observation_timestamp, f"%Y%j")


def get_viirs_timestamp_from_filepath(filepath: str) -> str:
    return Path(filepath).name.split(".")[1][1:]


def get_dateteime_from_viirs_filepath(filepath: str) -> str:
    return timestamp_to_datetime(get_viirs_timestamp_from_filepath(filepath))


def int_to_year_day(year: int, day: int) -> str:
    return str(year) + "{:03d}".format(day)


# Get all the dates of the series
# observation_times = []
# for filepath in viirs_data_filepaths:
#     observation_times.append(get_dateteime_from_viirs_filepath(filepath))
# observation_times = sorted(observation_times)


def extract_netcdf_latlon_from_viirs_nasa(dataset_object: xr.Dataset) -> Dict[str, npt.NDArray]:
    latitudes = dataset_object.coords["YDim"].values
    longitudes = dataset_object.coords["XDim"].values
    return {LAT: latitudes, LON: longitudes}


def reproject_dataset(dataset: xr.Dataset, new_crs: str) -> xr.Dataset:
    # Rioxarray reproject nearest by default
    return (
        dataset.rio.write_crs(dataset.data_vars["spatial_ref"].attrs["spatial_ref"])
        .rio.reproject(new_crs)
        .rename({"x": LON, "y": LAT})
    )


# viirs_projection_grid_sample = xr.open_dataset(viirs_data_filepaths[0], group='HDFEOS/GRIDS/VIIRS_Grid_IMG_2D', engine='netcdf4')
# latitudes_longitudes_dict = extract_netcdf_latlon_from_viirs_nasa(viirs_projection_grid_sample)


def get_daily_filenames_per_platform(platform: str, day: int) -> List[str]:
    platform_files = [path for path in viirs_data_filepaths if re.search(platforms_products_dict[platform], path)]
    day_files = [path for path in platform_files if re.search(f"A{int_to_year_day(year=year, day=day)}", path)]
    if len(day_files) != 4:
        logging.info(
            f"Wrong number of tiles corresponding to platform {platform} and day of the year {day} found. Expected 4, found: {len(day_files)}"
        )
        return None
    else:
        return day_files


def merge_crop_nasa_day_products(day: int, day_files: List[str], roi_bounds: pd.DataFrame) -> xr.Dataset:
    day_data_arrays = []
    for filepath in day_files:
        try:
            logger.info(f"Processing day {str(day)}/{str(366)}, name of the product {Path(filepath).name}")
            xr.open_dataset(filepath, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields", engine="netcdf4")
        except OSError:
            continue
        dataset_grid = xr.open_dataset(filepath, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D", engine="netcdf4")
        ndsi_snow_cover = xr.open_dataset(
            filepath, group="HDFEOS/GRIDS/VIIRS_Grid_IMG_2D/Data Fields", engine="netcdf4"
        ).data_vars["NDSI_Snow_Cover"]
        ndsi_snow_cover = ndsi_snow_cover.rename({"XDim": LON, "YDim": LAT}).assign_coords(
            coords=extract_netcdf_latlon_from_viirs_nasa(dataset_grid)
        )
        day_data_arrays.append(
            georef_data_array(data_array=ndsi_snow_cover, data_array_name="snow_cover", crs_wkt=PROJ4_MODIS)
        )
    merged = xr.combine_by_coords(day_data_arrays)
    reprojected = reproject_data_array(merged, roi_bounds.crs)
    cropped = reprojected.sel(
        lon=slice(france_geometry.minx[0], france_geometry.maxx[0]),
        lat=slice(france_geometry.maxy[0], france_geometry.miny[0]),
    )
    return cropped


# Get all available observation times
observation_times = []
for day in range(1, 367):
    day_files = get_daily_filenames_per_platform(platform=platform, day=day)
    if day_files is not None:
        observation_times.append(timestamp_to_datetime(int_to_year_day(year=year, day=day)))

# Get an array on one day to estimate size
sample_day_files = get_daily_filenames_per_platform(platform=platform, day=1)
sample_data_array = merge_crop_nasa_day_products(day=1, day_files=sample_day_files, roi_bounds=france_footprint)

time_series = xr.DataArray(
    np.zeros((len(observation_times), sample_data_array.dims[LAT], sample_data_array.dims[LON]), dtype="uint8"),
    dims=["time", LAT, LON],
    coords={
        "time": observation_times,  # The new time instant as a coordinate
        LAT: sample_data_array.coords[LAT],
        LON: sample_data_array.coords[LON],
    },
)

# Loop over all the files to populate the time series
data_arrays = []
observation_times = []
for day in range(1, 366):
    day_files = get_daily_filenames_per_platform(platform=platform, day=day)
    # Don't fail if no file is found for a day
    if day_files is None:
        continue
    processed_product = merge_crop_nasa_day_products(day=day, day_files=day_files, roi_bounds=france_footprint)
    observation_time = timestamp_to_datetime(int_to_year_day(year=year, day=day))
    # snow_mask = (ndsi_snow_cover>=10) & (ndsi_snow_cover <=100)
    # ndsi_snow_cover[snow_mask] = ndsi_snow_cover[snow_mask] * 0.01
    # ndsi_snow_cover[snow_mask] = -0.01 + 1.45*ndsi_snow_cover[snow_mask]

    # ndsi_snow_cover = np.where((ndsi_snow_cover>1) & (ndsi_snow_cover<2), 1, ndsi_snow_cover)

    # ndsi_snow_cover[snow_mask] = (ndsi_snow_cover[snow_mask] * 200).astype(np.uint8)
    # # Clouds
    # ndsi_snow_cover[ndsi_snow_cover==250] = 255
    # # Ocean
    # ndsi_snow_cover[ndsi_snow_cover==239] = 220

    time_series.loc[dict(time=observation_time)] = processed_product.data_vars["snow_cover"]


# Export to netCDF. Compression is specified here.
georef_time_series = georef_data_array(
    data_array=time_series, data_array_name="snow_cover", crs_wkt=france_footprint.crs.to_wkt()
)
time_series.coords["time"].attrs["long_name"] = "hydrological day"
outpath = f"{output_folder}/{output_name}"
logger.info(f"Save to {outpath} and compress (it might take a while)")
georef_time_series.to_netcdf(
    outpath,
    encoding={"snow_cover": {"zlib": True}, "time": {"calendar": "gregorian", "units": f"days since {str(year-1)}-10-01"}},
)
