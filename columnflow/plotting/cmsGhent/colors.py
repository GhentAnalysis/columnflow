
import order as od
import numpy as np
import colorsys

def change_sat(color, saturation_factor):
    clr = od.ColorMixin(color=color)
    hls = colorsys.rgb_to_hls(*clr.color)
    
    # Convert back to RGB
    clr = colorsys.hls_to_rgb(hls[0], hls[1], hls[2]*saturation_factor)

    return clr

def change_shade(color, factor):
    clr = od.ColorMixin(color=color).color
    scale = lambda x: x + (1-x)*factor
    clr = ( scale(clr[0]), scale(clr[1]), scale(clr[2]) )
    return clr
    

class DefaultColors:
    def __init__(self):
        self.white = "#fefefe"
        self.grey_pure = "#999999"
        self.grey = "#94a4a2"
        self.grey2 = "#717581"
        self.grey3 = "#5D5F66"
        self.grey4 = "#44464A"
        self.grey5 = "#3D3E3F"
        self.black = "#111111"

    def pastel(self):
        return PastelColors()

    def rainbow(self):
        return RainbowColors()

    def light_rainbow(self):
        return LightRainbow()

    def cat_six(self):
        return CATColorsSix()

    def cat_ten(self):
        return CATColorsTen()

    def btv(self):
        return BTVColors()

    def __getitem__(self, i):
        return self.colors[i%len(self.colors)]

class BTVColors(DefaultColors):
    def __init__(self):
        super().__init__()
        self.red = "#ed1e02"
        self.blue = "#15a3ef"
        self.orange = "#fcb302"
        self.purple = "#7a21dd"
        self.green = "#209a24"
        self.colors = [
            self.red,
            self.orange,
            self.blue,
            self.green,
            self.purple,
            self.grey2
        ]

class CATColorsSix(DefaultColors):
    def __init__(self):
        super().__init__()
        self.red = "#e42536"
        self.orange = "#f89c20"
        self.blue = "#5790fc"
        self.purple = "#7a21dd"
        self.violet = "#964a8b"
        self.grey = "#9c9ca1"
        self.colors = [
            self.red,
            self.orange,
            self.blue,
            self.purple,
            self.violet,
            self.grey,
        ]


class CATColorsTen(DefaultColors):
    def __init__(self):
        super().__init__()
        self.blue = "#3f90da"
        self.blue_light = "#92dadd"
        self.orange = "#ffa90e"
        self.orange_dark = "#e76300"
        self.red = "#bd1f01"
        self.purple = "#832db6"
        self.brown = "#a96b59"
        self.ochre = "#b9ac70"
        self.grey = "#94a4a2"
        self.grey2 = "#717581"
        self.colors = [
            self.blue,
            self.blue_light,
            self.orange,
            self.orange_dark,
            self.red,
            self.purple,
            self.brown,
            self.ochre,
            self.grey,
            self.grey2,
        ]


class RainbowColors(DefaultColors):
    def __init__(self):
        super().__init__()
        self.purple = "#d23be7"
        self.blue = "#4355db"
        self.blue_light = "#34bbe6"
        self.green = "#49da9a"
        self.lime = "#a3e048"
        self.yellow = "#f7d038"
        self.orange = "#eb7532"
        self.red = "#e6261f"
        self.colors = [
            self.purple,
            self.blue,
            self.blue_light,
            self.green,
            self.lime,
            self.yellow,
            self.orange,
            self.red,
        ]


class PastelColors(DefaultColors):
    def __init__(self):
        super().__init__()
        self.yellow = "#ffa600"
        self.orange = "#ff843c"
        self.orange_dark = "#ff6562"
        self.red = "#fd5385"
        self.violet_light = "#da52a2"
        self.violet = "#ab59b5"
        self.purple = "#6f5fba"
        self.blue = "#1761b0"
        self.colors = [
            self.yellow,
            self.orange,
            self.orange_dark,
            self.red,
            self.violet_light,
            self.violet,
            self.purple,
            self.blue,
        ]

class LightRainbow(DefaultColors):
    def __init__(self):
        super().__init__()
        self.yellow = "#fbf8cc"
        self.orange = "#fde4cf"
        self.salmon = "#ffcfd2"
        self.pink = "#f1c0e8"
        self.violet = "#cfbaf0"
        self.blue = "#a3c4f3"
        self.blue_light = "#90dbf4"
        self.blue_light2 = "#8eecf5"
        self.teal = "#98f5e1"
        self.green = "#89fbc0"
        self.colors = [
            self.yellow,
            self.orange,
            self.salmon,
            self.pink,
            self.violet,
            self.blue,
            self.blue_light,
            self.blue_light2,
            self.teal,
            self.green,
        ]
