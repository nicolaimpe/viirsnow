import xarray as xr
from typing import Dict, Any


def generate_xarray_compression_encodings(dataset: xr.Dataset, compression_level: int = 3) -> Dict[str, Any]:
    output_dict = {}
    compression_encoding_dict = {"zlib": True, "complevel": compression_level}
    for data_var_name in dataset.data_vars:
        output_dict.update({data_var_name: compression_encoding_dict})

    return output_dict
