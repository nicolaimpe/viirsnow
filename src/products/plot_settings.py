METEOFRANCE_VAR_NAME, NASA_PSEUDO_L3_VAR_NAME, NASA_L3_VAR_NAME, S2_CLMS_VAR_NAME, S2_THEIA_VAR_NAME = (
    "meteofrance_l3",
    "nasa_pseudo_l3",
    "nasa_l3",
    "s2_clms",
    "s2_theia",
)

MF_SYNOPSIS_VAR_NAME = "meteofrance_synopsis"
MF_ORIG_VAR_NAME = "meteofrance_orig"
MF_NO_CC_MASK_VAR_NAME = "meteofrance_no_cc_mask"
MF_REFL_SCREEN_VAR_NAME = "meteofrance_modified"

PRODUCT_PLOT_NAMES = {
    METEOFRANCE_VAR_NAME: "Météo-France",
    NASA_PSEUDO_L3_VAR_NAME: "NASA pseudo L3",
    NASA_L3_VAR_NAME: "NASA",
    MF_SYNOPSIS_VAR_NAME: "MF Cloud mask relaxed",
    MF_ORIG_VAR_NAME: "MF archive",
    MF_NO_CC_MASK_VAR_NAME: "No CC mask",
    MF_REFL_SCREEN_VAR_NAME: "CC band R screen",
}
PRODUCT_PLOT_COLORS = {
    METEOFRANCE_VAR_NAME: "tab:blue",
    NASA_PSEUDO_L3_VAR_NAME: "tab:green",
    NASA_L3_VAR_NAME: "paleturquoise",
}
