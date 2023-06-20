import numpy as np

from manager.metric_manager import AllMetrics
from data_types.plottable import Plottable, PlottableTypes


class Deltas():
    parameter_names = []

    def __init__(
            self,
            name=None,
            steps=None,
            values=None,
            mean=None,
            stdd=None,
            feature=None,
            calculate_on=None,
            list_name=None,
            display_values=None):
        self.name = name or AllMetrics.DELTAS.value
        self.steps = steps
        self.values = values
        self.mean = mean
        self.stdd = stdd
        self.feature = feature
        self.calculate_on = calculate_on
        self.list_name = list_name
        self.display_name = self.name + ' (abs. sum/mean/stdd.)'
        self.display_modes = ['mean', 'mean', 'mean']
        self.display_values = display_values
        self.parameters = None
        self.type = AllMetrics.DELTAS

    def calculate(self, feature, calculate_on=None, parameters=None):
        if calculate_on is None:
            calculate_on = feature

        if self.parameters:
            parameters = self.parameters

        steps = calculate_on.steps.copy()
        if hasattr(calculate_on, 'values_interp') and calculate_on.values_interp:
            values = np.diff(calculate_on.values_interp.copy()).tolist()
        else:
            values = np.diff(calculate_on.values.copy()).tolist()
        sum_ = np.sum(np.abs(values))
        mean = np.mean(values)
        stdd = np.std(values)
        values.insert(0, 0)
        list_name = self.name + f' ({round(sum_, 3)}/{round(mean, 3)}/{round(stdd, 3)})'
        display_values = [sum_, mean, stdd]

        return Deltas(
            name=self.name,
            steps=steps,
            values=values,
            mean=mean,
            stdd=stdd,
            feature=feature,
            calculate_on=calculate_on,
            list_name=list_name,
            display_values=display_values,
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
                    type_=PlottableTypes.CONTINUOUS_METRIC,
                    step_plot=True
                )
            ]
        else:
            return None

    @classmethod
    def tracker_plot(cls, image, slider_value, metric_x, metric_y):
        return

    def __str__(self):
        return self.name
