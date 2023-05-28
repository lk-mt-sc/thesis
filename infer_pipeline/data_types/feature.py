from data_types.plottable import Plottable, PlottableTypes


class Feature():
    def __init__(self, name, steps=None, values=None):
        self.name = name
        self.steps = steps if steps is not None else []
        self.values = values if values is not None else []

    def add(self, step, value):
        self.steps.append(step)
        self.values.append(value)

    def plottables(self, name=None, legend=None):
        return [Plottable(
            name=name or self.name,
            steps=self.steps,
            values=self.values,
            linestyle=None,
            legend=legend or self.name,
            type_=PlottableTypes.FEATURE
        )]

    def __str__(self):
        return self.name()
