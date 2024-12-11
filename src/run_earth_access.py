import earthaccess

product_name = "VNP10A1"
# 1. Login
earthaccess.login()

# 2. Search
results = earthaccess.search_data(
    short_name=product_name,  # ATLAS/ICESat-2 L3A Land Ice Height, VNP10?
    bounding_box=(2, 43, 3, 44),  # Only include files in area of interest...
    temporal=("2024-12-01", "2024-12-10"),  # ...and time period of interest
)

# 3. Access
files = earthaccess.download(results, f"./data/{product_name}")
