from scipy.signal import find_peaks

from data_types.plottable import Plottable, PlottableTypes


class Peaks():
    def __init__(
            self,
            name=None,
            steps=None,
            values=None,
            count=None,
            feature=None,
            list_name=None,
            display_values=None,):
        self.name = name or 'Peaks'
        self.steps = steps
        self.values = values
        self.count = count
        self.feature = feature
        self.list_name = list_name
        self.display_name = self.name
        self.display_modes = ['sum']
        self.display_values = display_values

    def calculate(self, feature):
        feature_values_positive = feature.values_interp.copy()
        feature_values_negative = [-v for v in feature_values_positive]
        steps_positive, positive_stats = find_peaks(feature_values_positive)
        steps_negative, negative_stats = find_peaks(feature_values_negative)
        steps = sorted(set(steps_positive.tolist() + steps_negative.tolist()))
        values = [i for j, i in enumerate(feature_values_positive) if j in steps]
        count = len(steps)
        list_name = self.name + f' ({count})'
        display_values = [count]

        return Peaks(
            name=self.name,
            steps=steps,
            values=values,
            count=count,
            feature=feature,
            list_name=list_name,
            display_values=display_values,
        )

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
