from scipy.signal import find_peaks

from data_types.plottable import Plottable, PlottableTypes


class PeakDetection():
    def __init__(self, feature, delta, name=None):
        self.name = name or 'Peaks'
        self.steps = []
        self.values = []
        self.count = 0
        self.feature = feature
        self.list_name = None
        self.delta = delta

        self._calculate()

    def _calculate(self):
        feature_values_positive = self.feature.values_interp.copy()
        feature_values_negative = [-v for v in feature_values_positive]
        self.steps_positive, self.positive_stats = find_peaks(feature_values_positive, threshold=6)
        self.steps_negative, self.negative_stats = find_peaks(feature_values_negative, threshold=6)
        self.steps_positive = self.steps_positive.tolist()
        self.steps_negative = self.steps_negative.tolist()
        self.steps = sorted(set(self.steps_positive + self.steps_negative))
        self.values = [i for j, i in enumerate(feature_values_positive) if j in self.steps]
        self.count = len(self.steps)

        self.list_name = self.name + f' ({self.count})'

    def plottables(self, name=None, legend=None):
        if self.steps:
            return [Plottable(
                name=name or self.name,
                steps=self.steps,
                values=self.values,
                linestyle='None',
                marker='o',
                markersize=6,
                legend=legend or self.name,
                type_=PlottableTypes.METRIC
            )]
        else:
            return None

    def __str__(self):
        return self.name
