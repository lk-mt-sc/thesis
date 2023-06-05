from manager.metric_manager import AllMetrics
from data_types.plottable import Plottable, PlottableTypes


class MissingPoseEstimations():
    parameter_names = []

    def __init__(
            self,
            name=None,
            steps=None,
            values=None,
            count=None,
            feature=None,
            list_name=None,
            display_values=None):
        self.name = name or AllMetrics.MISSING_POSE_ESTIMATIONS.value
        self.steps = steps
        self.values = values
        self.count = count
        self.feature = feature
        self.list_name = list_name
        self.display_name = self.name
        self.display_modes = ['single_sum']
        self.display_values = display_values
        self.parameters = None
        self.type = AllMetrics.MISSING_POSE_ESTIMATIONS

    def calculate(self, feature, calculate_on=None, parameters=None):
        feature_values = feature.values.copy()
        feature_values_interp = feature.values_interp.copy()
        steps = [i for i, v in enumerate(feature_values) if v == -1]
        count = len(steps)
        if steps:
            values = [i for j, i in enumerate(feature_values_interp) if j in steps]
        else:
            values = []
        list_name = self.name + f' ({count})'
        display_values = [count]

        return MissingPoseEstimations(
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
                marker='x',
                legend=legend or self.name,
                type_=PlottableTypes.METRIC
            )]
        else:
            return None

    @classmethod
    def tracker_plot(cls, image, slider_value, metric_x, metric_y):
        return

    def __str__(self):
        return self.name
