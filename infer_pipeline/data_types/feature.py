import numpy as np

from data_types.plottable import Plottable, PlottableTypes


class Feature():
    def __init__(self, name, steps=None, values=None):
        self.name = name
        self.steps = steps if steps is not None else []
        self.values = values if values is not None else []

    def add(self, step, value):
        self.steps.append(step)
        self.values.append(value)

    def clean_steps(self):
        steps_without_pose_estimations = [i for i, v in enumerate(self.values) if v == -1]
        return [i for j, i in enumerate(self.steps) if j not in steps_without_pose_estimations]

    def clean_values(self):
        steps_without_pose_estimations = [i for i, v in enumerate(self.values) if v == -1]
        return [i for j, i in enumerate(self.values) if j not in steps_without_pose_estimations]

    def plottables(self, name=None, legend=None):
        return [Plottable(
            name=name or self.name,
            steps=self.clean_steps(),
            values=self.clean_values(),
            linestyle='solid',
            marker='None',
            legend=legend or self.name,
            type_=PlottableTypes.FEATURE
        )]

    def __str__(self):
        return self.name()
