import numpy as np

from data_types.plottable import Plottable, PlottableTypes


class Delta():
    def __init__(self, feature, name=None):
        self.name = name or 'Delta'
        self.steps = []
        self.values = []
        self.mean = 0
        self.stdd = 0
        self.feature = feature
        self.list_name = None

        self._calculate()

    def _calculate(self):
        self.steps = self.feature.steps.copy()
        self.values = np.diff(self.feature.values_interp.copy()).tolist()
        self.mean = np.mean(self.values)
        self.stdd = np.std(self.values)
        self.values.insert(0, 0)

        self.list_name = self.name + f' ({round(self.mean, 3)}/{round(self.stdd, 3)})'

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
