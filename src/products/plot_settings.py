(
    METEOFRANCE_VAR_NAME,
    NASA_PSEUDO_L3_VAR_NAME,
    NASA_L3_SNPP_VAR_NAME,
    NASA_L3_JPSS1_VAR_NAME,
    S2_THEIA_VAR_NAME,
) = (
    "meteofrance_l3",
    "nasa_pseudo_l3_snpp",
    "nasa_l3_snpp",
    "nasa_l3_jpss1",
    "s2_theia",
)

MF_SYNOPSIS_VAR_NAME = "meteofrance_synopsis"
MF_ORIG_VAR_NAME = "meteofrance_orig"
MF_NO_CC_MASK_VAR_NAME = "meteofrance_no_cc_mask"
MF_REFL_SCREEN_VAR_NAME = "meteofrance_modified"
MF_NO_FOREST_VAR_NAME = "meteofrance_no_forest"

PRODUCT_PLOT_NAMES = {
    METEOFRANCE_VAR_NAME: "Météo-France",
    NASA_PSEUDO_L3_VAR_NAME: "NASA pseudo L3 SNPP",
    NASA_L3_SNPP_VAR_NAME: "NASA SNPP",
    NASA_L3_JPSS1_VAR_NAME: "NASA JPSS1",
    MF_SYNOPSIS_VAR_NAME: "MF Cloud mask relaxed",
    MF_ORIG_VAR_NAME: "MF archive",
    MF_NO_CC_MASK_VAR_NAME: "No CC mask",
    MF_REFL_SCREEN_VAR_NAME: "CC band R screen",
    MF_NO_FOREST_VAR_NAME: "No forest mask",
}
PRODUCT_PLOT_COLORS = {
    METEOFRANCE_VAR_NAME: "tab:blue",
    NASA_PSEUDO_L3_VAR_NAME: "tab:green",
    NASA_L3_SNPP_VAR_NAME: "paleturquoise",
    NASA_L3_JPSS1_VAR_NAME: "darkcyan",
    MF_SYNOPSIS_VAR_NAME: "tan",
    MF_ORIG_VAR_NAME: "linen",
    MF_NO_CC_MASK_VAR_NAME: "darkorange",
    MF_REFL_SCREEN_VAR_NAME: "gold",
    MF_NO_FOREST_VAR_NAME: "darkkhaki",
}
