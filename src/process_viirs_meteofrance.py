from osgeo import gdal
from glob import glob
import xarray as xr
import numpy as np
from pathlib import Path


import numpy as np
import netCDF4 as nc
import netCDF4 as nc
import numpy.typing as npt
from typing import Tuple, Dict
from dataclasses import dataclass
from datetime import datetime
import os
import pytz 
import logging


# Module configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LAT='lat'
LON='lon'

# User inputs
year = 2018
month = '*'    # regex
folder = '/home/imperatoren/work/VIIRS_S2_comparison/data/EOFR62'
viirs_data_filepaths = glob(f'{folder}/VIIRS{str(year)}/EOFR62*{str(year)}{month}*.LT')
viirs_data_filepaths = sorted(viirs_data_filepaths, key=lambda x: x.split('.')[1])
output_folder = './output_folder/completeness_analysis'


def georef_data_array(data_array: xr.DataArray, data_array_name: str, crs_wkt: str) -> xr.Dataset:

    """
    Turn a DataArray into a Dataset  for which the GDAL driver (GDAL and QGIS) is able to read the georeferencing
    https://github.com/pydata/xarray/issues/2288
    https://gis.stackexchange.com/questions/230093/set-projection-for-netcdf4-in-python
    """

    data_array.coords[LAT].attrs["axis"]="Y"
    data_array.coords[LON].attrs["axis"]="X"
    data_array.attrs['grid_mapping'] = 'spatial_ref'

    crs_variable = xr.DataArray(0)
    crs_variable.attrs['spatial_ref'] = crs_wkt

    georeferenced_dataset = xr.Dataset({data_array_name: data_array, 'spatial_ref':crs_variable})
    return georeferenced_dataset


def extract_netcdf_latlon_from_gdal_raster(gdal_dataset_object: gdal.Dataset)->Dict[str, npt.NDArray]:
    transform = gdal_dataset_object.GetGeoTransform()
    x_off, x_scale, _, y_off, _, y_scale = transform
    n_cols, n_rows = gdal_raster.RasterXSize, gdal_dataset_object.RasterYSize
    # for GDAL the UL corener is the UL corner of the image while for xarray is the center of the upper left pixel
    # Compensate for it
    longitudes = np.arange(n_cols) * x_scale + x_off + x_scale/2    
    latitudes = np.arange(n_rows) * y_scale + y_off + y_scale/2
    return {LAT: latitudes, LON: longitudes}

def timestamp_to_datetime(observation_timestamp: str)-> datetime:
    return datetime.strptime(observation_timestamp, f"%Y%m%d%H%M%S")

def get_viirs_timestamp_from_filepath(filepath: str) -> str:
    return Path(filepath).name.split('.')[1].split('_')[0]

def get_dateteime_from_viirs_filepath(filepath: str)-> str:
    return timestamp_to_datetime(get_viirs_timestamp_from_filepath(filepath))

# Get all the dates of the series
observation_times = []
for filepath in viirs_data_filepaths:
    observation_times.append(get_dateteime_from_viirs_filepath(filepath))
observation_times = sorted(observation_times)

# Get the geographic transform
gdal_raster = gdal.Open(viirs_data_filepaths[0])
latitudes_longitudes_dict=extract_netcdf_latlon_from_gdal_raster(gdal_raster)


# Initilize a time series on the whole dataset
time_series = xr.DataArray(
                    np.zeros((len(observation_times), gdal_raster.RasterYSize, gdal_raster.RasterXSize), dtype='uint8'),
                    dims=['time', LAT, LON],
                    coords={
                        'time': observation_times,  # The new time instant as a coordinate
                        LAT: latitudes_longitudes_dict[LAT],
                        LON: latitudes_longitudes_dict[LON],
                    },
                )

   
# Loop over all the files to populate the time series
for count, filepath in enumerate(viirs_data_filepaths):
    logger.info(f'Processing image {str(count+1)}/{str(len(viirs_data_filepaths))}, file name {os.path.basename(filepath)}')
    gdal_raster = gdal.Open(filepath)
    observation_time = get_dateteime_from_viirs_filepath(filepath)
    time_series.loc[dict(time=observation_time)] = gdal_raster.GetRasterBand(1).ReadAsArray()

    
    # print('number of pixels with 210 value: ', np.sum(gdal_raster.GetRasterBand(1).ReadAsArray() == 210))
    # print('number of pixels with 215 value: ', np.sum(gdal_raster.GetRasterBand(1).ReadAsArray() == 215))

# Export to netCDF. Compression is specified here.
georef_time_series = georef_data_array(data_array=time_series, data_array_name='snow_cover', crs_wkt=gdal.Open(viirs_data_filepaths[0]).GetProjection())
georef_time_series.coords['time'].attrs['long_name']='hydrological day'
outpath = f'{output_folder}/{str(year)}_{month}_meteofrance_time_series.nc'
logger.info(f'Save to {outpath} and compress (it might take a while)')
georef_time_series.to_netcdf(outpath, encoding={'snow_cover':{'zlib': True}, 'time': {'calendar': 'gregorian', 'units': 'days since 2016-10-01'}})


"""
def georef_data_variable_netcdf(latitudes_longitudes_dict: Dict[str, npt.NDArray], data_values: npt.NDArray, variable_name: str, crs_wkt: str)->xr.Dataset:
    latitudes, longitudes = latitudes_longitudes_dict[LAT], latitudes_longitudes_dict[LON]
    
    dsout = nc.Dataset(TMP_OUTFILE_PATH.replace('.nc', str(np.random.randint(1e12))+'.nc'), 'w', clobber=True)
    #dsout = nc.Dataset(TMP_OUTFILE_PATH, 'w', clobber=True)
    dsout.createDimension(LAT, len(latitudes))
    lat = dsout.createVariable(LAT, 'f4', (LAT,))
    lat.axis = "Y"
    lat[:] = latitudes

    dsout.createDimension(LON, len(longitudes))
    lon = dsout.createVariable(LON, 'f4', (LON,))
    lon.axis = "X"
    lon[:] = longitudes

    dsout.createDimension('time', 0)

    data_variable = dsout.createVariable(
        variable_name,
        data_values.dtype,
        ('time', LAT, LON),
    )
    data_variable[0,:] = data_values
    data_variable.setncattr('grid_mapping', 'spatial_ref')

    crs = dsout.createVariable('spatial_ref', 'i4')
    crs.spatial_ref=crs_wkt
    dataset = xr.open_dataset(xr.backends.NetCDF4DataStore(dsout))
    # dsout.close()
    return dataset
"""


