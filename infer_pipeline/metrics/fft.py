import numpy as np
from scipy.fft import fft, fftfreq, fftshift

from manager.metric_manager import AllMetrics
from data_types.plottable import Plottable, PlottableTypes


class FFT():
    parameter_names = []

    def __init__(
            self,
            name=None,
            steps=None,
            values=None,
            feature=None,
            list_name=None):
        self.name = name or AllMetrics.FFT.value
        self.steps = steps
        self.values = values
        self.feature = feature
        self.list_name = list_name
        self.display_name = self.name
        self.display_modes = []
        self.display_values = []
        self.parameters = None
        self.type = AllMetrics.FFT

    def calculate(self, feature, calculate_on=None, parameters=None):
        if calculate_on is None:
            calculate_on = feature

        if self.parameters:
            parameters = self.parameters

        n = len(calculate_on.steps.copy())
        d = 1 / 25
        steps = fftfreq(n, d)
        if hasattr(calculate_on, 'values_interp'):
            values = fft(calculate_on.values_interp.copy())
        else:
            values = fft(calculate_on.values.copy())

        values = 2.0/n * np.abs(values)

        steps = fftshift(steps).tolist()
        values = fftshift(values).tolist()

        list_name = self.name

        return FFT(
            name=self.name,
            steps=steps,
            values=values,
            feature=feature,
            list_name=list_name,
        )

    def plottables(self, name=None, legend=None):
        if self.steps:
            return [
                Plottable(
                    name=name or self.name,
                    steps=self.steps,
                    values=self.values,
                    linestyle='solid',
                    marker='None',
                    legend=legend or self.name,
                    type_=PlottableTypes.METRIC
                )
            ]
        else:
            return None

    @classmethod
    def tracker_plot(cls, image, slider_value, metric_x, metric_y):
        return

    def __str__(self):
        return self.name
