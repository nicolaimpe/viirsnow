(
    METEOFRANCE_VAR_NAME,
    NASA_PSEUDO_L3_VAR_NAME,
    NASA_L3_SNPP_VAR_NAME,
    NASA_L3_JPSS1_VAR_NAME,
    NASA_L3_MULTIPLATFORM_VAR_NAME,
    S2_THEIA_VAR_NAME,
) = ("meteofrance_l3", "nasa_pseudo_l3_snpp", "nasa_l3_snpp", "nasa_l3_jpss1", "nasa_l3_multiplatform", "s2_theia")

MF_SYNOPSIS_VAR_NAME = "meteofrance_synopsis"
MF_ORIG_VAR_NAME = "meteofrance_orig"
MF_NO_CC_MASK_VAR_NAME = "meteofrance_no_cc_mask"
MF_REFL_SCREEN_VAR_NAME = "meteofrance_modified"
MF_NO_FOREST_VAR_NAME = "meteofrance_no_forest"
MF_NO_FOREST_MODIFIED_VAR_NAME = "meteofrance_no_forest_modified"
MF_NO_FOREST_RED_BAND_SCREEEN_VAR_NAME = "meteofrance_no_forest_red_band_screen"


PRODUCT_PLOT_NAMES = {
    METEOFRANCE_VAR_NAME: "Météo-France",
    NASA_PSEUDO_L3_VAR_NAME: "NASA pseudo L3 SNPP",
    NASA_L3_SNPP_VAR_NAME: "NASA SNPP",
    NASA_L3_JPSS1_VAR_NAME: "NASA JPSS1",
    NASA_L3_MULTIPLATFORM_VAR_NAME: "NASA multiplatform",
    MF_SYNOPSIS_VAR_NAME: "MF cloud mask relaxed",
    MF_ORIG_VAR_NAME: "MF archive",
    MF_NO_CC_MASK_VAR_NAME: "MF No CC mask",
    MF_REFL_SCREEN_VAR_NAME: "MF CC band R screen",
    MF_NO_FOREST_VAR_NAME: "MF no forest",
    MF_NO_FOREST_MODIFIED_VAR_NAME: "MF No forest mask CC band R screen",
    MF_NO_FOREST_RED_BAND_SCREEEN_VAR_NAME: "MF no forest mask band R screen",
}
PRODUCT_PLOT_COLORS = {
    METEOFRANCE_VAR_NAME: "tab:blue",
    NASA_PSEUDO_L3_VAR_NAME: "tab:green",
    NASA_L3_SNPP_VAR_NAME: "darkturquoise",
    NASA_L3_JPSS1_VAR_NAME: "darkcyan",
    NASA_L3_MULTIPLATFORM_VAR_NAME: "steelblue",
    MF_SYNOPSIS_VAR_NAME: "tan",
    MF_ORIG_VAR_NAME: "tab:orange",
    MF_NO_CC_MASK_VAR_NAME: "darkorange",
    MF_REFL_SCREEN_VAR_NAME: "gold",
    MF_NO_FOREST_VAR_NAME: "darkkhaki",
    MF_NO_FOREST_MODIFIED_VAR_NAME: "khaki",
    MF_NO_FOREST_RED_BAND_SCREEEN_VAR_NAME: "goldenrod",
}
