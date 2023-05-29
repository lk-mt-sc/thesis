from enum import Enum


class PlottableTypes(Enum):
    FEATURE = 0
    METRIC = 1


class Plottable():
    def __init__(
            self,
            name,
            steps,
            values,
            linestyle,
            marker,
            legend,
            type_,
            linewidth=None,
            markersize=None):
        self.name = name
        self.steps = steps
        self.values = values
        self.linewidth = linewidth or 1.5
        self.linestyle = linestyle
        self.marker = marker
        self.markersize = markersize or 10
        self.legend = legend
        self.type = type_
