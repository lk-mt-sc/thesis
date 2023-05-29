import numpy as np

from data_types.plottable import Plottable, PlottableTypes


class MissingPoseEstimations():
    def __init__(self, feature, name=None):
        self.name = name or 'Missing Pose Estimations'
        self.steps = []
        self.values = []
        self.count = 0
        self.feature = feature

        self._calculate(self.feature)

    def _calculate(self, feature):
        feature_steps = feature.steps.copy()
        feature_values = feature.values.copy()

        self.steps = [i for i, v in enumerate(feature_values) if v == -1]
        self.count = len(self.steps)

        if self.steps:
            interpolation_steps = [i for j, i in enumerate(feature_steps) if j not in self.steps]
            interpolation_values = [i for j, i in enumerate(feature_values) if j not in self.steps]

            self.values = np.interp(self.steps, interpolation_steps, interpolation_values)

    def plottables(self, name=None, legend=None):
        if self.steps:
            return [Plottable(
                name=name or self.name,
                steps=self.steps,
                values=self.values,
                linestyle='None',
                marker='x',
                legend=legend or self.name,
                type_=PlottableTypes.METRIC
            )]
        else:
            return None

    def __str__(self):
        return self.name
