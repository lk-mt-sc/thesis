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
            list_name=None,
            display_values=None):
        self.name = name or AllMetrics.DELTAS.value
        self.steps = steps
        self.values = values
        self.mean = mean
        self.stdd = stdd
        self.feature = feature
        self.list_name = list_name
        self.display_name = self.name + ' (mean/std. deviation)'
        self.display_modes = ['mean', 'mean']
        self.display_values = display_values
        self.parameters = None
        self.type = AllMetrics.DELTAS

    def calculate(self, feature, calculate_on=None, parameters=None):
        if calculate_on is None:
            calculate_on = feature

        steps = calculate_on.steps.copy()
        if hasattr(calculate_on, 'values_interp'):
            values = np.diff(calculate_on.values_interp.copy()).tolist()
        else:
            values = np.diff(calculate_on.values.copy()).tolist()
        mean = np.mean(values)
        stdd = np.std(values)
        values.insert(0, 0)
        list_name = self.name + f' ({round(mean, 3)}/{round(stdd, 3)})'
        display_values = [mean, stdd]

        return Deltas(
            name=self.name,
            steps=steps,
            values=values,
            mean=mean,
            stdd=stdd,
            feature=feature,
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
                    type_=PlottableTypes.METRIC,
                    step_plot=True
                )
            ]
        else:
            return None

    def __str__(self):
        return self.name