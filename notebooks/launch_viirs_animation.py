import glob

import geopandas as gpd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import xarray as xr
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
from geospatial_grid.grid_database import LatLon375mGrid
from geospatial_grid.gsgrid import GSGrid
from matplotlib.colors import LinearSegmentedColormap
from ndsi_fsc_calibration.utils import gdf_to_binary_mask
from pyproj.crs import CRS

mf_grid = LatLon375mGrid()
alpes_mask = gpd.read_file("/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/vectorial/massifs/alpes.gpkg")
pyrenees_mask = gpd.read_file("/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/vectorial/massifs/pyrenees.gpkg")

# Pretreat data to select days with low cloud coverage
folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_composite_multiplatform/"
cms_composite_oper = glob.glob(f"{folder}/samples_oper/*202[5-6]*.nc")
all_year_data = xr.open_mfdataset(cms_composite_oper, concat_dim="time", combine="nested")
all_year_data = all_year_data.data_vars["snow_cover_fraction"].sortby("time")


def extract_low_cloud_cover(year_time_series: xr.Dataset, mask: xr.DataArray, grid: GSGrid, output_file: str):
    masked_data = year_time_series.where(gdf_to_binary_mask(mask, grid=grid).values)
    low_cloud_cover_dates_mask = ((masked_data >= 230).sum(dim=("x", "y")) / masked_data.count(dim=("x", "y"))) < 0.5
    low_cloud_cover_dates = [
        low_cloud_cover_dates_mask_el.time.values
        for low_cloud_cover_dates_mask_el in low_cloud_cover_dates_mask
        if low_cloud_cover_dates_mask_el
    ]
    low_cloud_cover = masked_data.sel(time=low_cloud_cover_dates)
    georef_netcdf_rioxarray(low_cloud_cover, crs=CRS.from_epsg(4326)).to_netcdf(output_file)
    return low_cloud_cover


extract_low_cloud_cover(
    year_time_series=all_year_data,
    mask=alpes_mask,
    grid=mf_grid,
    output_file=f"{folder}/time_series/alpes_20252026_clouds50.nc",
)

# extract_low_cloud_cover(
#     year_time_series=all_year_data,
#     mask=pyrenees_mask,
#     grid=mf_grid,
#     output_file=f"{folder}/time_series/pyrenees_20252026_lowclouds.nc",
# )


# Visualization of animation

# Define color stops at specific values
fsc_color_def = [
    (0.0, (0, 0, 0)),  # 0 -> black
    (1 / 255, (8 / 255, 51 / 255, 112 / 255)),  # 1 -> light blue
    (200 / 255, (1, 1, 1)),  # 200 -> white
    (220 / 255, (0.1, 0.1, 1)),  # 220 -> blue
    (230 / 255, (0.5, 0.5, 0.5)),  # 230 -> gray
    (1.0, (0.5, 0.5, 0.5)),  # 255 -> gray
]

fsc_cmap = LinearSegmentedColormap.from_list("custom_cmap", fsc_color_def, N=256)


low_cloud_cover_alps = (
    xr.open_dataset(
        "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_composite_multiplatform/time_series/alpes_20252026_clouds50.nc",
        mask_and_scale=True,
    )
    .data_vars["snow_cover_fraction"]
    .sel(x=slice(4.8, 7.8))
    .sel(y=slice(46.5, 43))
)


def plot_snow_animation(scf_time_series: xr.Dataset, fig, ax, export_to: str):
    # Extract data and format time strings for titles
    data_list = []
    titles = []
    for day_scf in scf_time_series:
        data_list.append(day_scf.values)
        titles.append(f"Time: {str(day_scf.time.values)[:10]}")
    # Initialize plot elements
    im = ax.imshow(data_list[0], cmap=fsc_cmap)
    title = ax.set_title(titles[0])
    ax.axis("off")
    # ax.set_xticks([])
    # ax.set_yticks([])

    def update(frame):
        im.set_array(data_list[frame])
        title.set_text(titles[frame])
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=range(len(data_list)), interval=300, repeat_delay=1000, blit=False)
    ani.save(export_to)
    return ani


fig, ax = plt.subplots(figsize=(6, 8))
animation_alps = plot_snow_animation(
    scf_time_series=low_cloud_cover_alps, fig=fig, ax=ax, export_to=f"{folder}/time_series/alps_wy2526_more_clouds.gif"
)

fig, ax = plt.subplots(figsize=(8, 6))
low_cloud_cover_pyrenees = (
    xr.open_dataset(
        "/home/imperatoren/work/VIIRS_S2_comparison/data/CMS_composite_multiplatform/time_series/pyrenees_20252026_lowclouds.nc",
        mask_and_scale=True,
    )
    .data_vars["snow_cover_fraction"]
    .sel(x=slice(-1.7, 3))
    .sel(y=slice(44, 41))
)

# animation_pyr = plot_snow_animation(
#     scf_time_series=low_cloud_cover_pyrenees, fig=fig, ax=ax, export_to=f"{folder}/time_series/pyrenees_wy2526.gif"
# )
