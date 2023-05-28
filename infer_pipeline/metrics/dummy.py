from data_types.plottable import Plottable, PlottableTypes


class Dummy():
    def __init__(self, feature, name=None):
        self.name = name or 'Dummy'
        self.steps = []
        self.values = []
        self.count = 0
        self.feature = feature

        self._calculate(self.feature)

    def _calculate(self, feature):
        feature_steps = feature.steps.copy()

        self.steps = feature_steps
        self.values = [100] * len(self.steps)

    def plottables(self, name=None, legend=None):
        return [Plottable(
            name=name or self.name,
            steps=self.steps,
            values=self.values,
            linestyle=None,
            legend=legend or self.name,
            type_=PlottableTypes.METRIC
        )]

    def __str__(self):
        return self.name
