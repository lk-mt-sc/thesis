from scipy.signal import find_peaks

from manager.metric_manager import AllMetrics
from data_types.plottable import Plottable, PlottableTypes


class Peaks():
    parameter_names = [
        'Height',
        'Threshold',
        'Distance',
        'Prominence',
        'Width',
        'Window Len.',
        'Rel. Height',
        'Plateau Size'
    ]

    def __init__(
            self,
            name=None,
            steps=None,
            values=None,
            count=None,
            feature=None,
            calculate_on=None,
            list_name=None,
            display_values=None,
            parameters=None,
            func_params=None):
        self.name = name or AllMetrics.PEAKS.value
        self.steps = steps
        self.values = values
        self.count = count
        self.feature = feature
        self.calculate_on = calculate_on
        self.list_name = list_name
        self.display_name = self.name
        self.display_modes = ['sum']
        self.display_values = display_values
        self.parameters = parameters
        self.func_params = func_params
        self.type = AllMetrics.PEAKS

    def calculate(self, feature, calculate_on=None, parameters=None):
        if calculate_on is None:
            calculate_on = feature

        if self.parameters:
            parameters = self.parameters

        func_params = []
        if parameters is not None:
            func_params.append(
                self.process_parameter(parameters['Height'], float) if parameters['Height'] else None)
            func_params.append(
                self.process_parameter(parameters['Threshold'], float) if parameters['Threshold'] else None)
            func_params.append(
                self.process_parameter(parameters['Distance'], int) if parameters['Distance'] else None)
            func_params.append(
                self.process_parameter(parameters['Prominence'], float) if parameters['Prominence'] else None)
            func_params.append(
                self.process_parameter(parameters['Width'], float) if parameters['Width'] else None)
            func_params.append(
                self.process_parameter(parameters['Window Len.'], float) if parameters['Window Len.'] else None)
            func_params.append(
                self.process_parameter(parameters['Rel. Height'], float) if parameters['Rel. Height'] else None)
            func_params.append(
                self.process_parameter(parameters['Plateau Size'], int) if parameters['Plateau Size'] else None)
        else:
            func_params = [None for _ in range(0, len(Peaks.parameter_names))]

        if hasattr(calculate_on, 'values_interp') and calculate_on.values_interp:
            values_positive = calculate_on.values_interp.copy()
        else:
            values_positive = calculate_on.copy()

        values_negative = [-v for v in values_positive]
        steps_positive, positive_stats = find_peaks(values_positive, *func_params)
        steps_negative, negative_stats = find_peaks(values_negative, *func_params)
        steps = sorted(set(steps_positive.tolist() + steps_negative.tolist()))
        values = [i for j, i in enumerate(values_positive) if j in steps]
        count = len(steps)
        list_name = self.name + f' ({count})'
        display_values = [count]

        # process stats
        # ...

        return Peaks(
            name=self.name,
            steps=steps,
            values=values,
            count=count,
            feature=feature,
            calculate_on=calculate_on,
            list_name=list_name,
            display_values=display_values,
            parameters=parameters,
            func_params=func_params
        )

    def process_parameter(self, parameter, dtype):
        parameter.replace(' ', '')
        if ',' in parameter:
            parameters = parameter.split(',')
            return (dtype(parameters[0]), dtype(parameters[1]))
        else:
            return dtype(parameter)

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
                type_=PlottableTypes.DISCRETE_METRIC
            )]
        else:
            return None

    @classmethod
    def tracker_plot(cls, image, slider_value, metric_x, metric_y):
        return

    def __str__(self):
        return self.name
