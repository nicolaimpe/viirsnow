from typing import Any, Dict

import xarray as xr


def generate_xarray_compression_encodings(data: xr.Dataset | xr.DataArray, compression_level: int = 3) -> Dict[str, Any]:
    output_dict = {}
    compression_encoding_dict = {"zlib": True, "complevel": compression_level}
    if type(data) is xr.Dataset:
        for data_var_name in data.data_vars:
            output_dict.update({data_var_name: compression_encoding_dict})
    elif type(data) is xr.DataArray:
        output_dict.update({data.name: compression_encoding_dict})
    return output_dict
