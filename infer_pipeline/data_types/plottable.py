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
            legend,
            type_):
        self.name = name
        self.steps = steps
        self.values = values
        self.linestyle = linestyle
        self.legend = legend
        self.type = type_
