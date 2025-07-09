from dataclasses import dataclass
from typing import Dict

from products.classes import METEOFRANCE_CLASSES, NASA_CLASSES, S2_CLASSES


class SnowCoverProduct:
    def __init__(
        self, name: str, classes: Dict[str, int | range], plot_color: str, plot_name: str, platform: str | None = None
    ):
        self.name = name
        self.classes = classes
        self.plot_color = plot_color
        self.plot_name = plot_name
        self.platform = platform

    def __repr__(self):
        return self.plot_name


class VIIRSMeteoFranceArchive(SnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="meteofrance_orig",
            classes=METEOFRANCE_CLASSES,
            plot_color="tab:blue",
            plot_name="Météo-France archive",
            platform="snpp",
        )


class VIIRSMeteoFranceSNPPPrototype(SnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="meteofrance_no_forest_red_band_screen",
            classes=METEOFRANCE_CLASSES,
            plot_color="goldenrod",
            plot_name="Météo-France SNPP",
            platform="snpp",
        )


class VIIRSMeteoFranceJPSS1Prototype(SnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="meteofrance_jpss1",
            classes=METEOFRANCE_CLASSES,
            plot_color="darkgoldenrod",
            plot_name="Météo-France JPSS1",
            platform="noaa20",
        )


class VIIRSMeteoFranceJPSS2Prototype(SnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="meteofrance_jpss2",
            classes=METEOFRANCE_CLASSES,
            plot_color="fiordilatte",
            plot_name="Météo-France JPSS2",
            platform="noaa21",
        )


class VNP10A1(SnowCoverProduct):
    def __init__(self):
        self.product_id = "VNP10A1"
        super().__init__(name="nasa_l3_snpp", classes=NASA_CLASSES, plot_color="darkturquoise", plot_name="NASA VIIRS SNPP")


class VJ110A1(SnowCoverProduct):
    def __init__(self):
        self.product_id = "VNP10A1"
        super().__init__(name="nasa_l3_jpss1", classes=NASA_CLASSES, plot_color="darkcyan", plot_name="NASA VIIRS JPSS1")


class V10A1Multiplatform(SnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="nasa_l3_multiplatform", classes=NASA_CLASSES, plot_color="steelblue", plot_name="NASA SNPP+JPSS1"
        )


class MOD10A1(SnowCoverProduct):
    def __init__(self):
        self.product_id = "MOD10A1"
        super().__init__(name="nasa_l3_terra", classes=NASA_CLASSES, plot_color="lightcoral", plot_name="NASA MODIS Terra")


class Sentinel2Theia(SnowCoverProduct):
    def __init__(self):
        super().__init__(name="S2_theia", classes=S2_CLASSES, plot_color="black", plot_name="Sentinel-2 Theia")
