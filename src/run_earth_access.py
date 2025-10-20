import earthaccess

from regrid.nasa_l2_download_and_project import download_daily_products_from_sxcen
from winter_year import WinterYear

# product_name = "VJ110A1"
# data_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/V10A1"
# # 1. Login
earthaccess.login()

# # 2. Search
# results = earthaccess.search_data(
#     short_name=product_name,  # ATLAS/ICESat-2 L3A Land Ice Height, VNP10?
#     bounding_box=(-2, 41, 10, 49),  # Only include files in area of interest...
#     temporal=("2023-10-01", "2024-09-30"),  # ...and time period of interest
# )

# # 3. Access
# files = earthaccess.download(results, f"{data_folder}/{product_name}")


granule_list_filepath = "/home/imperatoren/work/VIIRS_S2_comparison/data/M10A1/mod10a1_wy_2023_2024_granule_test.txt"
year = WinterYear(2023, 2024)
with open(granule_list_filepath) as f:
    list_product_urls = [line.strip() for line in f.readlines()]


for day in year.iterate_days():
    try:
        daily_products_filenames = download_daily_products_from_sxcen(
            day=day,
            download_urls_list=list_product_urls,
            output_folder="/home/imperatoren/work/VIIRS_S2_comparison/data/M10A1/MOD10A1",
        )

    except Exception as e:
        earthaccess.logger.warning(f"Error {e} during download. Skipping day {day}.")
        continue
