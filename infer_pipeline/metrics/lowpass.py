from scipy.signal import butter, filtfilt

from metrics.all_metrics import AllMetrics
from metrics.deltas import Deltas
from data_types.plottable import Plottable, PlottableTypes


class Lowpass():
    parameter_names = [
        'Order',
        'Cutoff Freq.',
        'Sample Freq.'
    ]

    def __init__(
            self,
            name=None,
            steps=None,
            values=None,
            feature=None,
            list_name=None,
            parameters=None,
            func_params=None):
        self.name = name or AllMetrics.LOWPASS.value
        self.steps = steps
        self.values = values
        self.feature = feature
        self.list_name = list_name
        self.display_name = self.name
        self.display_modes = []
        self.display_values = []
        self.parameters = parameters
        self.type = AllMetrics.LOWPASS
        self.func_params = func_params

    def calculate(self, feature, calculate_on=None, parameters=None):
        func_params = []
        if parameters is not None:
            func_params.append(
                self.process_parameter(parameters['Order'], float) if parameters['Order'] else 4)
            func_params.append(
                self.process_parameter(parameters['Cutoff Freq.'], float) if parameters['Cutoff Freq.'] else 20)
            func_params.append('lowpass'),
            func_params.append(False),
            func_params.append('ba')
            func_params.append(
                self.process_parameter(parameters['Sample Freq.'], float) if parameters['Sample Freq.'] else None)
        else:
            func_params.append(4)
            func_params.append(20)
            func_params.append('lowpass')
            func_params.append(False)
            func_params.append('ba')
            func_params.append(None)

        b, a = butter(*func_params)
        steps = feature.steps.copy()
        values = filtfilt(b, a, feature.values_interp.copy())

        list_name = self.name

        return Lowpass(
            name=self.name,
            steps=steps,
            values=values,
            feature=feature,
            list_name=list_name,
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
                    linestyle='solid',
                    marker='None',
                    legend=legend or self.name,
                    type_=PlottableTypes.METRIC
                    )]
        else:
            return None

    def __str__(self):
        return self.name
