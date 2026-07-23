import numpy as np
import numpy.ma as ma
import numpy.typing as npt
from ndsi_fsc_calibration.visualization import salomonson_appel


def gascoin(ndsi, f_veg):
    f_snow_toc = 0.5 * np.tanh(2.65 * ndsi - 1.42) + 0.5
    return np.minimum(1, f_snow_toc / (1 - f_veg))


def salomonson_appel_regression(masked_ndsi: npt.NDArray) -> npt.NDArray:
    snow_cover_fraction = 1.45 * masked_ndsi - 0.01
    snow_cover_fraction = np.clip(snow_cover_fraction, a_max=1, a_min=0)
    return snow_cover_fraction


def my_regression(masked_ndsi: npt.NDArray, forest_mask: npt.NDArray) -> npt.NDArray:
    snow_cover_fraction = np.where((forest_mask) * (~np.isnan(masked_ndsi)), 2.17 * masked_ndsi - 0.13, masked_ndsi)
    snow_cover_fraction = np.where(
        (1 - forest_mask) * (~np.isnan(masked_ndsi)), 1.43 * masked_ndsi - 0.04, snow_cover_fraction
    )
    snow_cover_fraction = np.clip(snow_cover_fraction, a_max=1, a_min=0)
    return snow_cover_fraction


def ndsi_snow_cover_to_fraction(
    ndsi_snow_cover_product: npt.NDArray,
    snow_cover_ndsi_threshold: int = 0,
    max_ndsi: int = 100,
    method: str = "salomonson_appel",
    forest_mask: npt.NDArray | None = None,
) -> npt.NDArray:
    snow_mask = (ndsi_snow_cover_product >= snow_cover_ndsi_threshold) & (ndsi_snow_cover_product <= max_ndsi)
    masked_ndsi_snow_cover = ma.masked_array(ndsi_snow_cover_product, mask=(1 - snow_mask)) / max_ndsi
    if method == "salomonson_appel":
        snow_cover_fraction = salomonson_appel_regression(masked_ndsi_snow_cover)
    elif method == "gascoin":
        snow_cover_fraction = gascoin(masked_ndsi_snow_cover)
    elif method == "mine":
        if forest_mask is None:
            raise ValueError("Forest mask needed if method == 'mine' ")
        snow_cover_fraction = my_regression(masked_ndsi=masked_ndsi_snow_cover, forest_mask=forest_mask)
    else:
        raise NotImplementedError(f"Fractional snow cover method {method} not known.")
    out_fractional_snow_cover = (snow_cover_fraction * max_ndsi).astype(np.uint8)
    out_fractional_snow_cover = np.where(snow_mask == 1, out_fractional_snow_cover, ndsi_snow_cover_product)

    return out_fractional_snow_cover
