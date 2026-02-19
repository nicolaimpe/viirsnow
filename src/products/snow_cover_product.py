from typing import Dict

from products.classes import METEOFRANCE_ARCHIVE_CLASSES, METEOFRANCE_COMPOSITE_CLASSES, NASA_CLASSES, S2_CLASSES
from reductions.completeness import (
    MeteoFranceArchiveSnowCoverProductCompleteness,
    MeteoFranceCompositeSnowCoverProductCompleteness,
    NASASnowCoverProductCompleteness,
    S2SnowCoverProductCompleteness,
)


class SnowCoverProduct:
    def __init__(
        self,
        name: str,
        prod_id: str,
        classes: Dict[str, int | range],
        plot_color: str,
        analyzer: S2SnowCoverProductCompleteness,
        platform: str | None = None,
    ):
        self.name = name
        self.classes = classes
        self.plot_color = plot_color
        self.prod_id = prod_id
        self.platform = platform
        self.analyzer = analyzer

    def __repr__(self):
        return self.prod_id


class MeteoFranceArchive(SnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="meteofrance_archive",
            classes=METEOFRANCE_ARCHIVE_CLASSES,
            plot_color="tab:blue",
            platform="snpp",
            prod_id="MF-ARCHIVE",
            analyzer=MeteoFranceArchiveSnowCoverProductCompleteness(),
        )


class MeteoFranceEvalSNPP(SnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="meteofrance_snpp_l3",
            classes=METEOFRANCE_COMPOSITE_CLASSES,
            plot_color="orange",
            prod_id="MF-FSC-VNP-L3",
            platform="npp",
            analyzer=MeteoFranceCompositeSnowCoverProductCompleteness(),
        )


class MeteoFranceEvalJPSS1(SnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="meteofrance_jpss1_l3",
            classes=METEOFRANCE_COMPOSITE_CLASSES,
            plot_color="darkgoldenrod",
            prod_id="MF-FSC-VJ1-L3",
            platform="noaa20",
            analyzer=MeteoFranceCompositeSnowCoverProductCompleteness(),
        )


class MeteoFranceEvalJPSS2(SnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="meteofrance_jpss2_l3",
            classes=METEOFRANCE_COMPOSITE_CLASSES,
            plot_color="sienna",
            prod_id="MF-FSC-VJ2-L3",
            platform="noaa21",
            analyzer=MeteoFranceCompositeSnowCoverProductCompleteness(),
        )


class MeteoFranceComposite(SnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="meteofrance_multiplatform_l3",
            classes=METEOFRANCE_COMPOSITE_CLASSES,
            plot_color="lightcoral",
            prod_id="MF-FSC-VMP-L3",
            platform="all",
            analyzer=MeteoFranceCompositeSnowCoverProductCompleteness(),
        )


class NASASnowCoverProduct(SnowCoverProduct):
    def __init__(self, name, plot_color, prod_id):
        super().__init__(
            name,
            classes=NASA_CLASSES,
            plot_color=plot_color,
            prod_id=prod_id,
            analyzer=NASASnowCoverProductCompleteness(),
            platform=None,
        )


class VNP10A1(NASASnowCoverProduct):
    def __init__(self):
        super().__init__(name="nasa_l3_snpp", plot_color="darkturquoise", prod_id="VNP10A1")


class VJ110A1(NASASnowCoverProduct):
    def __init__(self):
        super().__init__(name="nasa_l3_jpss1", plot_color="midnightblue", prod_id="VJ110A1")


class VJ210A1(NASASnowCoverProduct):
    def __init__(self):
        super().__init__(name="nasa_l3_jpss2", plot_color="lightsteelblue", prod_id="VJ210A1")


class MOD10A1(NASASnowCoverProduct):
    def __init__(self):
        self.product_id = "MOD10A1"
        super().__init__(name="nasa_l3_terra", plot_color="lightcoral", prod_id="MOD10A1")


class Sentinel2Theia(SnowCoverProduct):
    def __init__(self):
        super().__init__(
            name="S2_theia",
            classes=S2_CLASSES,
            plot_color="black",
            prod_id="Sentinel-2 Theia",
            analyzer=S2SnowCoverProductCompleteness(),
        )
