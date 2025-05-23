import numpy as np
import numpy.ma as ma
import numpy.typing as npt


def gascoin(ndsi, f_veg):
    f_snow_toc = 0.5 * np.tanh(2.65 * ndsi - 1.42) + 0.5
    return np.minimum(1, f_snow_toc / (1 - f_veg))


def salomonson_appel(ndsi):
    return 1.45 * ndsi - 0.01


def salomonson_appel_regression(ndsi: npt.NDArray) -> npt.NDArray:
    snow_cover_fraction = salomonson_appel(ndsi=ndsi)
    snow_cover_fraction = np.clip(snow_cover_fraction, a_max=1, a_min=0)
    return snow_cover_fraction


def ndsi_snow_cover_to_fraction(
    ndsi_snow_cover_product: npt.NDArray,
    snow_cover_ndsi_threshold: int = 10,
    max_ndsi: int = 100,
    method: str = "salomonson_appel",
) -> npt.NDArray:
    snow_mask = (ndsi_snow_cover_product >= snow_cover_ndsi_threshold) & (ndsi_snow_cover_product <= max_ndsi)
    masked_ndsi_snow_cover = ma.masked_array(ndsi_snow_cover_product, mask=(1 - snow_mask)) / max_ndsi
    if method == "salomonson_appel":
        snow_cover_fraction = salomonson_appel_regression(masked_ndsi_snow_cover)
    else:
        raise NotImplementedError(f"Fractional snow cover method {method} not known.")
    out_fractional_snow_cover = (snow_cover_fraction.data * max_ndsi).astype(np.uint8)
    out_fractional_snow_cover = np.where(snow_mask == 1, out_fractional_snow_cover, ndsi_snow_cover_product)

    return out_fractional_snow_cover


def nasa_ndsi_snow_cover_to_fraction(nasa_ndsi_snow_cover_product: npt.NDArray) -> npt.NDArray:
    return ndsi_snow_cover_to_fraction(nasa_ndsi_snow_cover_product, snow_cover_ndsi_threshold=10, max_ndsi=100)


def meteofrance_ndsi_snow_cover_to_fraction(meteofrance_ndsi_snow_cover_product: npt.NDArray) -> npt.NDArray:
    return ndsi_snow_cover_to_fraction(meteofrance_ndsi_snow_cover_product, snow_cover_ndsi_threshold=0, max_ndsi=200)
