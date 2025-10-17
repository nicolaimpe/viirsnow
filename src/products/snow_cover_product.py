from dataclasses import dataclass
from typing import Dict

from products.classes import (METEOFRANCE_ARCHIVE_CLASSES,
                              METEOFRANCE_COMPOSITE_CLASSES, NASA_CLASSES,
                              S2_CLASSES)
from reductions.completeness import (
    MeteoFranceArchiveSnowCoverProductCompleteness,
    MeteoFranceCompositeSnowCoverProductCompleteness,
    NASASnowCoverProductCompleteness, S2SnowCoverProductCompleteness)


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



class MeteoFranceArchive(SnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="meteofrance_archive",
            classes=METEOFRANCE_ARCHIVE_CLASSES,
            plot_color="tab:blue",
            plot_name="Météo-France archive",
            platform="snpp",
        )
        self.analyzer = MeteoFranceArchiveSnowCoverProductCompleteness()


class MeteoFranceEvalSNPP(SnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="meteofrance_prototype_snpp",
            classes=METEOFRANCE_COMPOSITE_CLASSES,
            plot_color="orange",
            plot_name="Météo-France SNPP",
            platform="snpp",
        )
        self.analyzer = MeteoFranceCompositeSnowCoverProductCompleteness()


class MeteoFranceEvalJPSS1(SnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="meteofrance_eval_jpss1",
            classes=METEOFRANCE_COMPOSITE_CLASSES,
            plot_color="darkgoldenrod",
            plot_name="Météo-France JPSS1",
            platform="noaa20",
        )
        self.analyzer = MeteoFranceCompositeSnowCoverProductCompleteness()


class MeteoFranceEvalJPSS2(SnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="meteofrance_eval_jpss2",
            classes=METEOFRANCE_COMPOSITE_CLASSES,
            plot_color="sienna",
            plot_name="Météo-France JPSS2",
            platform="noaa21",
        )
        self.analyzer = MeteoFranceCompositeSnowCoverProductCompleteness()


class MeteoFranceComposite(SnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="meteofrance_eval_multiplatform",
            classes=METEOFRANCE_COMPOSITE_CLASSES,
            plot_color="lightcoral",
            plot_name="Météo-France multi-platform",
            platform="all",
        )
        self.analyzer = MeteoFranceCompositeSnowCoverProductCompleteness()


class NASASnowCoverProduct(SnowCoverProduct):
    def __init__(self, name, classes, plot_color, plot_name, platform=None):
        super().__init__(name, classes, plot_color, plot_name, platform)
        self.analyzer = NASASnowCoverProductCompleteness()


class VNP10A1(NASASnowCoverProduct):
    def __init__(self):
        self.product_id = "VNP10A1"
        super().__init__(name="nasa_l3_snpp", classes=NASA_CLASSES, plot_color="darkturquoise", plot_name="NASA VIIRS SNPP")


class VJ110A1(NASASnowCoverProduct):
    def __init__(self):
        self.product_id = "VNP10A1"
        super().__init__(name="nasa_l3_jpss1", classes=NASA_CLASSES, plot_color="midnightblue", plot_name="NASA VIIRS JPSS1")


class V10A1Multiplatform(NASASnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="nasa_l3_multiplatform", classes=NASA_CLASSES, plot_color="lightskyblue", plot_name="NASA SNPP+JPSS1"
        )


class MOD10A1(NASASnowCoverProduct):
    def __init__(self):
        self.product_id = "MOD10A1"
        super().__init__(name="nasa_l3_terra", classes=NASA_CLASSES, plot_color="lightcoral", plot_name="NASA MODIS Terra")


class Sentinel2Theia(NASASnowCoverProduct):
    def __init__(self):
        super().__init__(name="S2_theia", classes=S2_CLASSES, plot_color="black", plot_name="Sentinel-2 Theia")
        self.analyzer = S2SnowCoverProductCompleteness()
