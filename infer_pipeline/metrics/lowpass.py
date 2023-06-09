from scipy.signal import butter, filtfilt

from metrics.all_metrics import AllMetrics
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
            calculate_on=None,
            list_name=None,
            parameters=None,
            func_params=None):
        self.name = name or AllMetrics.LOWPASS.value
        self.steps = steps
        self.values = values
        self.feature = feature
        self.calculate_on = calculate_on
        self.list_name = list_name
        self.display_name = self.name
        self.display_modes = []
        self.display_values = []
        self.parameters = parameters
        self.func_params = func_params
        self.type = AllMetrics.LOWPASS

    def calculate(self, feature, calculate_on=None, parameters=None):
        if calculate_on is None:
            calculate_on = feature

        if self.parameters:
            parameters = self.parameters

        func_params = []
        if parameters is not None:
            func_params.append(
                self.process_parameter(parameters['Order'], float) if parameters['Order'] else 4)
            func_params.append(
                self.process_parameter(parameters['Cutoff Freq.'], float) if parameters['Cutoff Freq.'] else 0.5)
            func_params.append('lowpass'),
            func_params.append(False),
            func_params.append('ba')
            func_params.append(
                self.process_parameter(parameters['Sample Freq.'], float) if parameters['Sample Freq.'] else None)
        else:
            func_params.append(4)
            func_params.append(0.5)
            func_params.append('lowpass')
            func_params.append(False)
            func_params.append('ba')
            func_params.append(None)

        b, a = butter(*func_params)
        steps = calculate_on.steps.copy()
        if hasattr(calculate_on, 'values_interp') and calculate_on.values_interp:
            values = filtfilt(b, a, calculate_on.values_interp.copy())
        else:
            values = filtfilt(b, a, calculate_on.values.copy())

        list_name = self.name

        return Lowpass(
            name=self.name,
            steps=steps,
            values=values,
            feature=feature,
            calculate_on=calculate_on,
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
                    type_=PlottableTypes.CONTINUOUS_METRIC
                    )]
        else:
            return None

    @classmethod
    def tracker_plot(cls, image, slider_value, metric_x, metric_y):
        return

    def __str__(self):
        return self.name
