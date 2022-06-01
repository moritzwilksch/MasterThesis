import colorsys
from enum import Enum

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


class Colors(Enum):
    UPBLUE = "#00305e"


def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    r, g, b = colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def set_style():
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Palatino"]
    import matplotlib_inline.backend_inline

    matplotlib_inline.backend_inline.set_matplotlib_formats("svg")
    plt.rcParams["font.family"] = "Bitstream Vera Sans"
    plt.rcParams["font.size"] = 16


def when_then_else(when: pl.Expr, then: pl.Expr, otherwise: pl.Expr) -> pl.Expr:
    return pl.when(when).then(then).otherwise(otherwise)
