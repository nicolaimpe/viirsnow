from products.plot_settings import (METEOFRANCE_VAR_NAME,
                                    NASA_L3_JPSS1_VAR_NAME,
                                    NASA_L3_MODIS_TERRA_VAR_NAME,
                                    NASA_L3_MULTIPLATFORM_VAR_NAME,
                                    NASA_L3_SNPP_VAR_NAME,
                                    NASA_PSEUDO_L3_VAR_NAME, S2_THEIA_VAR_NAME)

METEOFRANCE_ARCHIVE_CLASSES = {
    "snow_cover": range(1, 201),
    "no_snow": (0,),
    "clouds": (255,),
    "forest_without_snow": (215,),
    "forest_with_snow": (210,),
    "water": (220,),
    "nodata": (230,),
    "fill": (254,),
}


METEOFRANCE_COMPOSITE_CLASSES = {
    "snow_cover": range(1, 201),
    "no_snow": (0,),
    "clouds": (255,),
    "water": (220,),
    "nodata": (230,),
    "fill": (254,),
}

NASA_CLASSES = {
    "snow_cover": range(1, 101),
    "no_snow": (0,),
    "clouds": (250,),
    "water": (237, 239),  # inland and ocean
    "no_decision": (201,),
    "night": (211,),
    "missing_data": (251,),
    "L1B_unusable": (252,),
    "bowtie_trim": (253,),
    "L1B_fill": (254,),
    "fill": (255,),
}

NODATA_NASA_CLASSES = (
    "no_decision",
    "night",
    "missing_data",
    "L1B_unusable",
    "bowtie_trim",
    "L1B_fill",
    "fill",
)

S2_CLASSES = {"snow_cover": range(1, 101), "no_snow": (0,), "clouds": (205,), "nodata": (255,), "fill": (255,)}

PRODUCT_CLASSES_DICT = {
    METEOFRANCE_VAR_NAME: METEOFRANCE_ARCHIVE_CLASSES,
    NASA_L3_SNPP_VAR_NAME: NASA_CLASSES,
    NASA_L3_JPSS1_VAR_NAME: NASA_CLASSES,
    NASA_L3_MULTIPLATFORM_VAR_NAME: NASA_CLASSES,
    NASA_PSEUDO_L3_VAR_NAME: NASA_CLASSES,
    S2_THEIA_VAR_NAME: S2_CLASSES,
    NASA_L3_MODIS_TERRA_VAR_NAME: NASA_CLASSES,
}
