from enum import Enum


class PlottableTypes(Enum):
    FEATURE = 0
    CONTINUOUS_METRIC = 1
    DISCRETE_METRIC = 2


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
            linewidth=1.5,
            markersize=10,
            markerfacecolor='None',
            step_plot=False,
            box_plot=False,
            log_x_axis=False,
            log_y_axis=False):
        self.name = name
        self.steps = steps
        self.values = values
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.marker = marker
        self.markersize = markersize
        self.markerfacecolor = markerfacecolor
        self.legend = legend
        self.type = type_
        self.step_plot = step_plot
        self.box_plot = box_plot
        self.log_x_axis = log_x_axis
        self.log_y_axis = log_y_axis
