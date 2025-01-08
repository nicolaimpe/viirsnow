import earthaccess

product_name = "VNP10A1"
data_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/V10A1"
# 1. Login
earthaccess.login()

# 2. Search
results = earthaccess.search_data(
    short_name=product_name,  # ATLAS/ICESat-2 L3A Land Ice Height, VNP10?
    bounding_box=(-1, 41, 12, 51),  # Only include files in area of interest...
    temporal=("2024-01-23", "2024-02-05"),  # ...and time period of interest
)

# 3. Access
files = earthaccess.download(results, f"{data_folder}/{product_name}")
