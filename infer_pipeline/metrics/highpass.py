import cv2 as cv
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

from metrics.all_metrics import AllMetrics
from data_types.plottable import Plottable, PlottableTypes


class Highpass():
    parameter_names = [
        'Order',
        'Cutoff Freq.',
        'Sample Freq.',
        'Zeroing Thr.'
    ]

    def __init__(
            self,
            name=None,
            steps=None,
            values=None,
            values_abs=None,
            values_zeroed=None,
            values_non_zero_interp=None,
            values_smoothed=None,
            feature=None,
            calculate_on=None,
            list_name=None,
            parameters=None,
            func_params=None):
        self.name = name or AllMetrics.HIGHPASS.value
        self.steps = steps
        self.values = values
        self.values_abs = values_abs
        self.values_zeroed = values_zeroed
        self.values_non_zero_interp = values_non_zero_interp
        self.values_smoothed = values_smoothed
        self.feature = feature
        self.calculate_on = calculate_on
        self.list_name = list_name
        self.display_name = self.name
        self.display_modes = []
        self.display_values = []
        self.parameters = parameters
        self.func_params = func_params
        self.type = AllMetrics.HIGHPASS

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
            func_params.append('highpass'),
            func_params.append(False),
            func_params.append('ba')
            func_params.append(
                self.process_parameter(parameters['Sample Freq.'], float) if parameters['Sample Freq.'] else None)
            zeroing_threshold = self.process_parameter(
                parameters['Zeroing Thr.'], float) if parameters['Zeroing Thr.'] else 5.0
        else:
            func_params.append(4)
            func_params.append(0.5)
            func_params.append('highpass')
            func_params.append(False)
            func_params.append('ba')
            func_params.append(None)
            zeroing_threshold = 5.0

        b, a = butter(*func_params)
        steps = calculate_on.steps.copy()
        if hasattr(calculate_on, 'values_interp') and calculate_on.values_interp:
            values = filtfilt(b, a, calculate_on.values_interp.copy())
        else:
            values = filtfilt(b, a, calculate_on.values.copy())

        values_abs = list(abs(values))

        values_zeroed = []
        for value in values_abs:
            if value >= zeroing_threshold:
                values_zeroed.append(value)
            else:
                values_zeroed.append(0)

        nonzero_indices = np.nonzero(values_zeroed)[0]
        if nonzero_indices.size != 0:
            non_zero_areas = []
            non_zero_area = [nonzero_indices[0]]

            for i in range(1, len(nonzero_indices)):
                if nonzero_indices[i] == nonzero_indices[i-1] + 1:
                    non_zero_area.append(nonzero_indices[i])
                else:
                    non_zero_areas.append(non_zero_area)
                    non_zero_area = [nonzero_indices[i]]

            non_zero_areas.append(non_zero_area)

            for non_zero_area in non_zero_areas:
                if non_zero_area[0] > 0:
                    non_zero_area.insert(0, non_zero_area[0] - 1)
                else:
                    non_zero_area.insert(0, 0)
                if non_zero_area[-1] == len(steps) - 1:
                    non_zero_area.append(non_zero_area[-1])
                else:
                    non_zero_area.append(non_zero_area[-1] + 1)

            steps_to_interpolate = []
            for non_zero_area in non_zero_areas:
                steps_to_interpolate += non_zero_area

            steps_ = steps.copy()

            if hasattr(calculate_on, 'values_interp') and calculate_on.values_interp:
                values_ = calculate_on.values_interp.copy()
            else:
                values_ = calculate_on.values.copy()

            steps_ = [i for j, i in enumerate(steps_) if j not in steps_to_interpolate]
            values_ = [i for j, i in enumerate(values_) if j not in steps_to_interpolate]

            interpolated_values = np.interp(steps_to_interpolate, steps_, values_)

            if hasattr(calculate_on, 'values_interp') and calculate_on.values_interp:
                values_non_zero_interp = calculate_on.values_interp.copy()
            else:
                values_non_zero_interp = calculate_on.values.copy()

            for i, index in enumerate(steps_to_interpolate):
                values_non_zero_interp[index] = interpolated_values[i]

            values_smoothed = savgol_filter(values_non_zero_interp, 25, 5).tolist()

        else:
            values_non_zero_interp = []
            values_smoothed = []

        list_name = self.name

        return Highpass(
            name=self.name,
            steps=steps,
            values=values,
            values_abs=values_abs,
            values_zeroed=values_zeroed,
            values_non_zero_interp=values_non_zero_interp,
            values_smoothed=values_smoothed,
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
            name = name or self.name
            legend = legend or self.name
            plottables = [
                Plottable(
                    name=name,
                    steps=self.steps,
                    values=self.values,
                    linestyle='solid',
                    marker='None',
                    legend=legend,
                    type_=PlottableTypes.METRIC
                )
            ]
            if self.values_abs:
                plottables.append(
                    Plottable(
                        name=name + '_ABS',
                        steps=self.steps,
                        values=self.values_abs,
                        linestyle='solid',
                        marker='None',
                        legend=legend + '_ABS',
                        type_=PlottableTypes.METRIC
                    )
                )
            if self.values_zeroed:
                plottables.append(
                    Plottable(
                        name=name + '_ZEROED',
                        steps=self.steps,
                        values=self.values_zeroed,
                        linestyle='solid',
                        marker='None',
                        legend=legend + '_ZEROED',
                        type_=PlottableTypes.METRIC
                    )
                )
            if self.values_non_zero_interp:
                plottables.append(
                    Plottable(
                        name=name + '_INTERPOLATED',
                        steps=self.steps,
                        values=self.values_non_zero_interp,
                        linestyle='solid',
                        marker='None',
                        legend=legend + '_INTERPOLATED',
                        type_=PlottableTypes.METRIC
                    )
                )
            if self.values_smoothed:
                plottables.append(
                    Plottable(
                        name=name + '_SMOOTHED',
                        steps=self.steps,
                        values=self.values_smoothed,
                        linestyle='solid',
                        marker='None',
                        legend=legend + '_SMOOTHED',
                        type_=PlottableTypes.METRIC
                    )
                )
            return plottables
        else:
            return None

    @classmethod
    def tracker_plot(cls, image, slider_value, metric_x, metric_y):
        if not None in (metric_x, metric_y):
            if metric_x.values_non_zero_interp and metric_y.values_non_zero_interp:
                x = int(metric_x.values_non_zero_interp[slider_value])
                y = int(metric_y.values_non_zero_interp[slider_value])
                color = [255 / 256, 20 / 256, 147 / 256]
                image = cv.circle(image, center=(x, y), radius=5, thickness=-1, color=color)

    def __str__(self):
        return self.name
