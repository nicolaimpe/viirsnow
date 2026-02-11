import earthaccess

from regrid.nasa_l2_download_and_project import download_daily_products_from_sxcen
from winter_year import WinterYear

product_name = "VNP10A1"
data_folder = "/home/imperatoren/work/edelweiss_assimilation/observation_operator/data/viirs_nasa_radiance/downloaded"

# 1. Login
earthaccess.login(strategy="interactive")

from datetime import datetime, timedelta

dates = [
    # datetime(year=2018, month=1, day=23),
    # datetime(year=2018, month=3, day=16),
    # datetime(year=2019, month=5, day=13),
    # datetime(year=2020, month=5, day=4),
    # datetime(year=2022, month=2, day=26),
    datetime(year=2022, month=5, day=1),
]

for d in dates:
    # 2. Search
    results = earthaccess.search_data(
        short_name=product_name,  # ATLAS/ICESat-2 L3A Land Ice Height, VNP10?
        bounding_box=(6, 45, 6.5, 45.3),  # Only include files in area of interest...
        temporal=(d, d + timedelta(days=1)),  # ...and time period of interest
        day_night_flag="day",
    )

    # 3. Access
    files = earthaccess.download(results, f"{data_folder}/")


# granule_list_filepath = "/home/imperatoren/work/VIIRS_S2_comparison/data/M10A1/mod10a1_wy_2023_2024_granule_test.txt"
# year = WinterYear(2023, 2024)
# with open(granule_list_filepath) as f:
#     list_product_urls = [line.strip() for line in f.readlines()]


# for day in year.iterate_days():
#     try:
#         daily_products_filenames = download_daily_products_from_sxcen(
#             day=day,
#             download_urls_list=list_product_urls,
#             output_folder="/home/imperatoren/work/VIIRS_S2_comparison/data/M10A1/MOD10A1",
#         )

#     except Exception as e:
#         earthaccess.logger.warning(f"Error {e} during download. Skipping day {day}.")
#         continue
