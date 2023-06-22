import numpy as np
from scipy.signal import hilbert

from manager.metric_manager import AllMetrics
from data_types.plottable import Plottable, PlottableTypes


class InstantaneousFrequency():
    parameter_names = []

    def __init__(
            self,
            name=None,
            steps=None,
            values=None,
            feature=None,
            calculate_on=None,
            list_name=None,
            display_values=None):
        self.name = name or AllMetrics.INSTANTANEOUS_FREQUENCY.value
        self.steps = steps
        self.values = values
        self.feature = feature
        self.calculate_on = calculate_on
        self.list_name = list_name
        self.display_name = self.name
        self.display_modes = []
        self.display_values = display_values
        self.parameters = None
        self.type = AllMetrics.INSTANTANEOUS_FREQUENCY

    def calculate(self, feature, calculate_on=None, parameters=None):
        if calculate_on is None:
            calculate_on = feature

        if self.parameters:
            parameters = self.parameters

        steps = calculate_on.steps.copy()[1:]
        if hasattr(calculate_on, 'values_interp') and calculate_on.values_interp:
            analytic_signal = hilbert(calculate_on.values_interp.copy())
        else:
            analytic_signal = hilbert(calculate_on.values.copy())

        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * feature.fps)
        values = instantaneous_frequency

        list_name = self.name

        return InstantaneousFrequency(
            name=self.name,
            steps=steps,
            values=values,
            feature=feature,
            calculate_on=calculate_on,
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
                    type_=PlottableTypes.CONTINUOUS_METRIC
                )
            ]
        else:
            return None

    @classmethod
    def tracker_plot(cls, image, slider_value, metric_x, metric_y):
        return

    def __str__(self):
        return self.name
