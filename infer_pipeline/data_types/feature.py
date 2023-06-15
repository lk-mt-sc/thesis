import numpy as np

from data_types.plottable import Plottable, PlottableTypes


class Feature():
    def __init__(self, name, steps=None, scores=None, values=None):
        self.name = name
        self.steps = steps or []
        self.values = values or []
        self.scores = scores or []
        self.values_interp = []

    def add(self, step, value, score):
        self.steps.append(step)
        self.values.append(value)
        self.scores.append(score)

    def interpolate_values(self):
        self.values_interp = self.values.copy()
        steps_without_pose_estimations = [i for i, v in enumerate(self.values) if v == -1]
        if steps_without_pose_estimations:
            interpolation_steps = [i for j, i in enumerate(self.steps) if j not in steps_without_pose_estimations]
            interpolation_values = [i for j, i in enumerate(self.values) if j not in steps_without_pose_estimations]

            interpolated_values = np.interp(steps_without_pose_estimations, interpolation_steps, interpolation_values)

            for i, step in enumerate(steps_without_pose_estimations):
                self.values_interp[step] = interpolated_values[i]

    def plottables(self, name=None, legend=None):
        return [Plottable(
            name=name or self.name,
            steps=self.steps,
            values=self.values_interp,
            linestyle='solid',
            marker='None',
            legend=legend or self.name,
            type_=PlottableTypes.FEATURE
        )]

    def __str__(self):
        return self.name
