from typing import List, Tuple
import rasterio
import xarray as xr
import glob
import geopandas as gpd
from metrics import WinterYear
from pathlib import Path
import rioxarray
from datetime import datetime
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from rasterio.crs import CRS
from rasterio.enums import Resampling
from xarray.groupers import BinGrouper
from logger_setup import default_logger as logging
from tqdm import tqdm
from operator import itemgetter
import time
from geotools import create_empty_grid_from_roi

DEFAULT_CRS = 32631
S2_RESOLUTION = 20  # m
OUTPUT_GRID_RES = 250  # m
OUTPUT_GRID_X0, OUTPUT_GRID_Y0 = 124300, 4635200
RESAMPLING = Resampling.nearest

S2_CLASSES = {"snow_cover": range(1, 101), "no_snow": 0, "clouds": 205, "no_data": 255}


def get_all_s2_files_of_winter_year(s2_folder: str, winter_year: WinterYear) -> List[str]:
    s2_files = glob.glob(str(Path(s2_folder).joinpath(f"FSC_*{winter_year.from_year}1[0-2]*T31*/*FSCOG.tif")))
    s2_files.extend(glob.glob(str(Path(s2_folder).joinpath(f"FSC_*{winter_year.to_year}0[1-9]*T31*/*FSCOG.tif"))))
    return sorted(s2_files)


def s2_filename_to_datetime(s2_file: str):
    observation_timestamp = Path(s2_file).name.split("_")[1]
    observation_datetime = datetime.strptime(observation_timestamp[:8], "%Y%m%d")
    return observation_datetime


def out_grid_origin(minx: float, miny: float) -> Tuple[float, float]:
    x0 = np.floor(minx / OUTPUT_GRID_RES) * OUTPUT_GRID_RES if OUTPUT_GRID_X0 is not None else OUTPUT_GRID_X0
    y0 = np.floor(miny / OUTPUT_GRID_RES) * OUTPUT_GRID_RES if OUTPUT_GRID_Y0 is not None else OUTPUT_GRID_Y0
    return x0, y0


def add_time_dim(data_array: xr.DataArray) -> xr.DataArray:
    return data_array.expand_dims(time=[s2_filename_to_datetime(s2_file=data_array.encoding["source"])])


def read_s2_image(s2_filename: str) -> xr.DataArray:
    s2_image = rioxarray.open_rasterio(s2_filename)
    print(s2_image)
    if s2_image.rio.crs.to_epsg() != DEFAULT_CRS:
        print("mi riproiettanooo")
        s2_image = s2_image.rio.reproject(
            dst_crs=CRS.from_epsg(DEFAULT_CRS).to_wkt(), resolution=s2_image.rio.resolution(), resampling=RESAMPLING
        )
        print("reprojected", s2_image)
    return s2_image


def aggregate_s2_spatially(data: xr.DataArray, new_lon: np.array, new_lat: np.array) -> xr.DataArray:
    # def s2_aggregation_rule(data: xr.DataArray) -> xr.DataArray:
    #     if data.sum() == S2_CLASSES["no_snow"]:
    #         # print("EII")
    #         return xr.DataArray(S2_CLASSES["no_snow"])
    #     if (data == S2_CLASSES["snow_cover"][-1]).all():
    #         return xr.DataArray(S2_CLASSES["snow_cover"][-1])
    #     median = data.median()
    #     if median.values in itemgetter("clouds", "no_data")(S2_CLASSES):
    #         out_data = median
    #         return out_data
    #     else:
    #         cloud_mask = data == S2_CLASSES["clouds"]
    #         no_data_mask = data == S2_CLASSES["no_data"]
    #         mask = cloud_mask | no_data_mask
    #         valid_data = data * (1 - mask)
    #         out_data = valid_data.mean() * valid_data.count() / mask.count()
    #         out_data = out_data.astype(np.uint8)
    #         # print("group time ", time.time() - t0)
    #         return out_data
    def s2_aggregation_rule(data: xr.DataArray) -> xr.DataArray:
        return data.mean(skipna=True)

    x_bins = new_lon - OUTPUT_GRID_RES / 2
    x_bins = np.append(x_bins, x_bins[-1] + OUTPUT_GRID_RES)
    y_bins = new_lat - OUTPUT_GRID_RES / 2
    y_bins = np.append(y_bins, y_bins[-1] + OUTPUT_GRID_RES)
    no_snow_mask = data == S2_CLASSES["no_snow"]
    valid_data = data.where(no_snow_mask == False, drop=True)
    aggregated_data = valid_data.groupby(x=BinGrouper(x_bins), y=BinGrouper(y_bins)).map(s2_aggregation_rule)

    return aggregated_data.rename({"x_bins": "lon", "y_bins": "lat"}).assign_coords({"lon": new_lon, "lat": new_lat})


def create_s2_time_series(s2_folder: str, roi_shapefile: str, winter_year: WinterYear, output_folder: Path):
    files = get_all_s2_files_of_winter_year(s2_folder, year)
    roi = gpd.read_file(roi_shapefile).to_crs(DEFAULT_CRS)
    sample_grid = create_empty_grid_from_roi(roi)
    lon_out, lat_out = sample_grid.coords["lon"], sample_grid.coords["lat"]
    out_tmp_paths = []
    for idx, day in tqdm(enumerate(winter_year.iterate_days()), desc="Processing day: ", leave=True):
        if idx > 90:
            continue
        daily_files = [file for file in files if day.strftime("%Y%m%d") in file]
        day_data = create_empty_grid_from_roi(roi)

        for idx_image, s2_daily_file in enumerate(daily_files):
            s2_image = read_s2_image(s2_filename=s2_daily_file)

            lon_min, lon_max, lat_min, lat_max = (
                s2_image.coords["x"].values[0],
                s2_image.coords["x"].values[-1],
                s2_image.coords["y"].values[-1],
                s2_image.coords["y"].values[0],
            )

            new_lons = lon_out.sel(lon=slice(lon_min, lon_max)).values
            new_lats = lat_out.sel(lat=slice(lat_min, lat_max)).values
            aggregated = aggregate_s2_spatially(data=s2_image, new_lon=new_lons, new_lat=new_lats)
            day_data.data_vars["snow_cover"].loc[dict(lon=new_lons, lat=new_lats)] = aggregated
        out_path = f'{str(output_folder)}/{day.strftime("%Y%m%d")}.nc'
        day_data = day_data.expand_dims(time=[day])
        out_tmp_paths.append(out_path)
        day_data.to_netcdf(out_path)

    all_data = xr.open_mfdataset(out_tmp_paths, combine="nested", concat_dim="time")
    all_data.to_netcdf(
        f"{output_folder}/WY_{winter_year.from_year}_{winter_year.to_year}_S2_res_{OUTPUT_GRID_RES}_time_series.nc"
    )


if __name__ == "__main__":
    year = WinterYear(2023, 2024)
    massifs_shapefile = "/home/imperatoren/work/VIIRS_S2_comparison/data/vectorial/massifs/massifs.shp"
    s2_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/S2"
    output_folder = "/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder/test_s2"

    create_s2_time_series(s2_folder=s2_folder, roi_shapefile=massifs_shapefile, winter_year=year, output_folder=output_folder)
    # print(xr.open_mfdataset(files[:100], engine="rasterio", preprocess=add_time_dim))

    # print(create_empty_grid_from_roi(roi))
    # print("a", empty_grid)
    # print("b", sample_data_array)
    # grid_aligned, _ = xr.align(empty_grid, sample_data_array, join="outer")
    # print("a+", grid_aligned)
