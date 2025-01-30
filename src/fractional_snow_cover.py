import numpy as np
import numpy.typing as npt
import numpy.ma as ma


def salomonson_appel_regression(ndsi: npt.NDArray) -> npt.NDArray:
    snow_cover_fraction = -0.01 + 1.45 * ndsi
    snow_cover_fraction = np.clip(snow_cover_fraction, a_max=1, a_min=0)
    return snow_cover_fraction


def nasa_ndsi_snow_cover_to_fraction(
    ndsi_snow_cover_product: npt.NDArray, snow_cover_ndsi_threshold: int = 10, method: str = "salomonson_appel"
) -> npt.NDArray:
    snow_mask = (ndsi_snow_cover_product >= snow_cover_ndsi_threshold) & (ndsi_snow_cover_product <= 100)
    masked_ndsi_snow_cover = ma.masked_array(ndsi_snow_cover_product, mask=(1 - snow_mask)) / 100
    if method == "salomonson_appel":
        snow_cover_fraction = salomonson_appel_regression(masked_ndsi_snow_cover)
    else:
        raise NotImplementedError(f"Fractional snow cover method {method} not known.")
    out_fractional_snow_cover = (snow_cover_fraction.data * 100).astype(np.uint8)
    out_fractional_snow_cover = np.where(snow_mask == 1, out_fractional_snow_cover, ndsi_snow_cover_product)

    return out_fractional_snow_cover
