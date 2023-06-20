import tkinter as tk
from tkinter import messagebox
from enum import Enum
from threading import Thread

import numpy as np
import matplotlib.pyplot as plt

from manager.status_manager import Status
from gui.gui_metric import GUIMetric
from manager.dataset_manager import KeypointsNoMetric
from metrics.all_metrics import AllMetrics
from metrics.missing_pose_estimations import MissingPoseEstimations
from metrics.deltas import Deltas
from metrics.peaks import Peaks
from metrics.lowpass import Lowpass
from metrics.highpass import Highpass
from metrics.fft import FFT


class CalculableMetrics(Enum):
    DELTAS = AllMetrics.DELTAS.value
    PEAKS = AllMetrics.PEAKS.value
    LOWPASS = AllMetrics.LOWPASS.value
    HIGHPASS = AllMetrics.HIGHPASS.value
    FFT = AllMetrics.FFT.value


class InferenceMetrics():
    def __init__(self):
        self.features = []

    def add_feature(self, feature):
        self.features.append(feature)

    def calculate(self):
        self.features = [f for f in self.features if not KeypointsNoMetric.has_value(f.name[:-2])]
        highpass_total = []
        for feature in self.features:
            highpass = Highpass(parameters={'Order': '4', 'Cutoff Freq.': '10',
                                            'Sample Freq.': '25', 'Zeroing Thr.': ''}).calculate(feature)
            # concatenation is valid, since standard deviation is a point-based metric
            highpass_total += highpass.values_abs
        StandardMetrics.highpass_zeroing_threshold = np.std(highpass_total) / 2


class StandardMetrics():
    highpass_zeroing_threshold = 5.0

    def __init__(self):
        self.metrics = [
            MissingPoseEstimations(),
            Deltas(),
            Highpass(parameters={'Order': '4', 'Cutoff Freq.': '10',
                     'Sample Freq.': '25', 'Zeroing Thr.': str(StandardMetrics.highpass_zeroing_threshold)}),
            Lowpass(parameters={'Order': '4', 'Cutoff Freq.': '2', 'Sample Freq.': '25'}),
            FFT()
        ]


class RunMetrics():
    def __init__(self, features):
        self.features = features
        self.features = [f for f in self.features if not KeypointsNoMetric.has_value(f.name[:-2])]
        self.metrics = {}

    def calculate(self):
        standard_metrics = StandardMetrics()
        for metric in standard_metrics.metrics:
            self.metrics[metric.name] = []
            for feature in self.features:
                self.metrics[metric.name].append(metric.calculate(feature))

        return self.metrics


class MetricManager():
    def __init__(self, root, status_manager):
        self.gui_metric = GUIMetric(
            root,
            calculable_metrics=CalculableMetrics,
            visualize_inference_overall_metrics_callback=self.visualize_inference_overall_metrics,
            listbox_metrics_select_callback=self.metric_selected,
            listbox_metrics_drag_callback=self.on_drag,
            listbox_metrics_drop_callback=self.on_drop,
            radiobutton_metrics_select_callback=self.calculable_metric_selected,
            button_calculate_callback=self.calculate_metric,
        )
        self.status_manager = status_manager
        self.plot_manager = None
        self.compared_inferences = [None for _ in range(0, 3)]
        self.selected_metric = None
        self.selected_inference = None
        self.selected_inference_metrics = {}
        self.selected_data = None
        self.selected_data_metrics = {}
        self.selected_feature = None
        self.selected_features_metrics = []
        self.session_metrics = []
        self.visualization_metrics = {}

        # initialize parameter inputs
        self.calculable_metric_selected()
        self._gui_disable_button_calculate()
        self._gui_disable_button_show_inference_overall_metrics()

    def set_inference(self, inference):
        self.clear_all_metrics()
        self.selected_inference = inference
        self.selected_inference_metrics = self.calculate_inference_metrics()
        self._gui_set_inference_metrics(position=-1)
        self._gui_enable_button_show_inference_overall_metrics()

    def calculate_inference_metrics(self):
        runs = self.selected_inference.runs
        inference_metrics = {}
        for metric_name, metrics in runs[0].metrics.items():
            inference_metrics[metrics[0].display_name] = [0 for _ in range(0, len(metrics[0].display_values))]

        for i, data in enumerate(runs):
            data_metrics, data_modes = self.calculate_data_metrics(data)

            for metric_name, metric_values in data_metrics.items():
                modes = data_modes[metric_name]
                for j, mode in enumerate(modes):
                    if mode in ('single_sum', 'mean', 'sum'):
                        inference_metrics[metric_name][j] += metric_values[j]

                        if mode == 'mean' and i == len(runs) - 1:
                            inference_metrics[metric_name][j] /= len(runs)

        metrics_to_delete = []
        for metric_name, metric_values in inference_metrics.items():
            if not metric_values:
                metrics_to_delete.append(metric_name)

        for metric_name in metrics_to_delete:
            del inference_metrics[metric_name]

        return inference_metrics

    def visualize_inference_overall_metrics(self):
        if not self.status_manager.has_status(Status.LOADING_VISUALISATION):
            self.status_manager.add_status(Status.LOADING_VISUALISATION)
            loading_thread = Thread(target=self._calculate_inference_overall_metrics)
            loading_thread.start()
            self.monitor_loading_thread(loading_thread)

    def _calculate_inference_overall_metrics(self):
        features = [feature for run in self.selected_inference.runs for feature in run.features]
        features = [feature for feature in features if not KeypointsNoMetric.has_value(feature.name[:-2])]
        # features = [feature for feature in features if 'wrist' in feature.name[:-2]]

        deltas = []
        ffts_frequencies = []
        ffts_values = []
        total_fft_frequencies = []
        for feature in features:
            fft = FFT().calculate(feature)
            total_fft_frequencies += fft.steps.copy()
            ffts_frequencies.append(fft.steps.copy())
            ffts_values.append(fft.values.copy())

            delta = Deltas().calculate(feature)
            deltas += delta.values

        f_vector = np.arange(-12.5, 12.6, 0.1)
        cumulative_fft = np.zeros_like(f_vector)

        for (steps, values) in zip(ffts_frequencies, ffts_values):
            steps = np.array(steps)
            values = np.array(values)
            nearest_indices = np.abs(f_vector - steps[:, np.newaxis]).argmin(axis=1)
            cumulative_fft[nearest_indices] += values

        average_fft = cumulative_fft / len(ffts_frequencies)

        self.visualization_metrics['deltas'] = deltas
        self.visualization_metrics['total_fft_frequencies'] = total_fft_frequencies
        self.visualization_metrics['total_fft'] = {
            'f_vector': f_vector,
            'average_fft': average_fft
        }

    def monitor_loading_thread(self, loading_thread):
        if loading_thread.is_alive():
            self.gui_metric.root.after(50, lambda: self.monitor_loading_thread(loading_thread))
        else:
            figure, axes = plt.subplots(2, 3)

            axes[0, 0].boxplot(self.visualization_metrics['deltas'])
            axes[0, 0].set_title('Deltas')
            axes[0, 0].set_ylabel('Delta in Pixel')
            axes[0, 0].set_xticks([])

            counts, bins = np.histogram(self.visualization_metrics['deltas'], bins=251)
            axes[1, 0].stairs(counts, bins)
            axes[1, 0].set_title('Deltas')
            axes[1, 0].set_xlabel('Delta in Pixel')
            axes[1, 0].set_ylabel('Prevalence')

            axes[0, 1].boxplot(self.visualization_metrics['total_fft_frequencies'])
            axes[0, 1].set_title('FFT Frequencies')
            axes[0, 1].set_ylabel('Frequency in Hz')
            axes[0, 1].set_xticks([])

            counts, bins = np.histogram(self.visualization_metrics['total_fft_frequencies'], bins=251)
            axes[1, 1].stairs(counts, bins)
            axes[1, 1].set_title('FFT Frequencies')
            axes[1, 1].set_xlabel('Frequency in Hz')
            axes[1, 1].set_ylabel('Prevalence')

            axes[0, 2].plot(self.visualization_metrics['total_fft']
                            ['f_vector'], self.visualization_metrics['total_fft']['average_fft'])
            axes[0, 2].set_title('Fast Fourier Transform')
            axes[0, 2].set_xlabel('Frequency in Hz')
            axes[0, 2].set_ylabel('Magnitude')

            axes[1, 2].plot(self.visualization_metrics['total_fft']
                            ['f_vector'], self.visualization_metrics['total_fft']['average_fft'])
            axes[1, 2].set_title('Fast Fourier Transform - logarithmic y-axis')
            axes[1, 2].set_yscale('log')
            axes[1, 2].set_xlabel('Frequency in Hz')
            axes[1, 2].set_ylabel('Magnitude')

            figure.canvas.manager.set_window_title(f'Overall Metrics on Inference "{self.selected_inference.name}"')
            plt.show(block=False)

            self.status_manager.remove_status(Status.LOADING_VISUALISATION)

    def add_to_compared_inferences(self, inference, position):
        set_names = all(i is None for i in self.compared_inferences)
        self.compared_inferences[position] = inference
        self._gui_set_inference_metrics(position, title=inference.name, set_names=set_names)

    def remove_from_compared_inferences(self, position):
        self.compared_inferences[position] = None
        clear_names = all(i is None for i in self.compared_inferences)
        self._gui_clear_inference_metrics(position, clear_names=clear_names)

    def clear_compared_inferences(self):
        for i in range(0, 3):
            self.remove_from_compared_inferences(i)

    def set_data(self, data):
        self.clear_feature_metrics()
        self.clear_data_metrics()
        self.selected_data = data
        self.selected_data_metrics, _ = self.calculate_data_metrics()
        self._gui_set_data_metrics()

    def calculate_data_metrics(self, data=None):
        data = data or self.selected_data
        data_modes = {}
        data_metrics = {}
        for metric_name, metrics in data.metrics.items():
            name = metrics[0].display_name
            modes = metrics[0].display_modes
            values = [0 for i in range(0, len(modes))]
            for i, mode in enumerate(modes):
                if mode == 'single_sum':
                    values[i] = metrics[0].display_values[i]
                if mode == 'sum':
                    for metric in metrics:
                        values[i] += metric.display_values[i]
                if mode == 'mean':
                    for metric in metrics:
                        values[i] += metric.display_values[i]
                    values[i] /= len(metrics)
            data_metrics[name] = values
            data_modes[name] = modes
        return data_metrics, data_modes

    def set_feature(self, feature):
        self.clear_feature_metrics()
        self.selected_feature = feature
        self.selected_features_metrics = self.find_feature_metrics()
        self._gui_set_feature_metrics()

    def find_feature_metrics(self):
        feature_metrics = []
        for _, metrics in self.selected_data.metrics.items():
            for metric in metrics:
                if self.selected_feature == metric.feature:
                    feature_metrics.append(metric)

        for metric in self.session_metrics:
            if self.selected_feature == metric.feature:
                feature_metrics.append(metric)

        return feature_metrics

    def clear_all_metrics(self):
        self.clear_inference_metrics()
        self.clear_data_metrics()
        self.clear_feature_metrics()

    def clear_inference_metrics(self):
        self.selected_inference_metrics = {}
        self._gui_clear_inference_metrics(position=-1)
        self._gui_clear_parameters(disable_entries=True)
        self._gui_set_calculated_on_name('')
        self._gui_set_calculable_metric(None)
        self._gui_disable_button_show_inference_overall_metrics()

    def clear_data_metrics(self):
        self.selected_data_metrics = {}
        self._gui_set_data_metrics()
        self._gui_clear_parameters(disable_entries=True)
        self._gui_set_calculated_on_name('')
        self._gui_set_calculable_metric(None)

    def clear_feature_metrics(self):
        self.selected_metric = None
        self.selected_features_metrics.clear()
        self._gui_set_feature_metrics()
        self._gui_disable_button_calculate()
        self._gui_clear_parameters(disable_entries=True)
        self._gui_set_calculated_on_name('')
        self._gui_set_calculable_metric(None)

    def metric_selected(self, event=None):
        self._gui_clear_parameters(disable_entries=True)
        selection = self.gui_metric.listbox_metrics.curselection()
        if not selection:
            return

        selection_str = self.gui_metric.listbox_metrics.get(selection)

        for metric in self.selected_features_metrics:
            if metric.list_name == selection_str:
                self.selected_metric = metric

        parameters = self.selected_metric.parameters
        if parameters is not None:
            self._gui_set_parameter_names(self.selected_metric.parameter_names, self.selected_metric.name)
            self._gui_set_parameter_values(parameters)

        self._gui_disable_button_calculate()
        self._gui_set_calculable_metric(None)
        self._gui_set_calculated_on_name(self.selected_metric.calculate_on.name)

    def calculable_metric_selected(self, event=None):
        self._gui_clear_parameters()

        if self.selected_feature is None:
            return

        selected_calculable_metric = self.metric_selection_to_class(self.gui_metric.add_metric_var.get())
        if selected_calculable_metric is None:
            return

        self._gui_set_parameter_names(selected_calculable_metric.parameter_names, selected_calculable_metric.name)

        calculate_on = self.selected_metric.name if self.selected_metric is not None else self.selected_feature.name
        self._gui_set_calculated_on_name(calculate_on)

        self._gui_enable_button_calculate()

    def calculate_metric(self, event=None):
        metric_type = self.gui_metric.add_metric_var.get()
        metric_name = self.gui_metric.metric_name_var.get()
        selected_calculable_metric = self.metric_selection_to_class(metric_type, metric_name)

        if selected_calculable_metric is None:
            return

        gui_parameter_name_vars = self.gui_metric.parameter_name_vars
        gui_parameter_value_vars = self.gui_metric.parameter_value_vars

        parameters = {}
        for name, value in zip(gui_parameter_name_vars, gui_parameter_value_vars):
            parameters[name.get()[:-1]] = value.get()

        if selected_calculable_metric.type in (AllMetrics.DELTAS, AllMetrics.FFT, AllMetrics.HIGHPASS, AllMetrics.LOWPASS) \
                and self.selected_metric is not None:
            calculate_on = self.selected_metric
        else:
            calculate_on = self.selected_feature

        new_metric = selected_calculable_metric.calculate(
            feature=self.selected_feature,
            calculate_on=calculate_on,
            parameters=parameters)

        if self._gui_is_metric_name_taken(new_metric.list_name):
            messagebox.showerror(title='', message='Metric name already used.')
            return

        self.session_metrics.append(new_metric)
        self.set_feature(self.selected_feature)
        self._gui_enable_button_calculate()

    def metric_selection_to_class(self, metric_selection, metric_name=None):
        if metric_name is None:
            metric_name = metric_selection

        match metric_selection:
            case AllMetrics.MISSING_POSE_ESTIMATIONS.value:
                return MissingPoseEstimations(name=metric_name)
            case AllMetrics.DELTAS.value:
                return Deltas(name=metric_name)
            case AllMetrics.PEAKS.value:
                return Peaks(name=metric_name)
            case AllMetrics.LOWPASS.value:
                return Lowpass(name=metric_name)
            case AllMetrics.HIGHPASS.value:
                return Highpass(name=metric_name)
            case AllMetrics.FFT.value:
                return FFT(name=metric_name)
            case other:
                return None

    def on_drag(self, event=None):
        if self.selected_metric is None:
            return

        self.gui_metric.root.configure(cursor='plus')

    def on_drop(self, event=None):
        self.gui_metric.root.configure(cursor='')
        if self.selected_metric is None:
            return

        name = self.selected_inference.name + '_' + str(self.selected_data.id).zfill(2) + '_' + \
            self.selected_feature.name + '_' + self.selected_metric.name
        plottables = self.selected_metric.plottables(name=name, legend=name)

        if plottables is not None:
            x, y = event.widget.winfo_pointerxy()
            self.plot_manager.add_to_plot(x, y, plottables=plottables)

    def _gui_set_inference_metrics(self, position, title=None, set_names=False):
        if position == -1:
            for m in enumerate(self.selected_inference_metrics.items()):
                if m[1][1]:
                    self.gui_metric.inference_metrics_names[m[0]].set(m[1][0])
                    self.gui_metric.inference_metrics_values[m[0]].set(', '.join(str(round(x, 3)) for x in m[1][1]))
        else:
            for m in enumerate(self.selected_inference_metrics.items()):
                if m[1][1]:
                    if set_names:
                        self.gui_metric.inferences_metrics_names[m[0]].set(m[1][0])

                    self.gui_metric.inferences_metrics_values[position][m[0]].set(
                        ', '.join(str(round(x, 3)) for x in m[1][1]))

            if len(title) > 20:
                title = title[: 18] + '...'
            self.gui_metric.inference_metrics_titles[position].set(title)

    def _gui_clear_inference_metrics(self, position, clear_names=False):
        if position == -1:
            for name in self.gui_metric.inference_metrics_names:
                name.set('...')
            for values in self.gui_metric.inference_metrics_values:
                values.set('')
            return
        else:
            if clear_names:
                for name in self.gui_metric.inferences_metrics_names:
                    name.set('...')

            for values in self.gui_metric.inferences_metrics_values[position]:
                values.set('')

            self.gui_metric.inference_metrics_titles[position].set('No Inference Selected')

    def _gui_set_data_metrics(self):
        if not self.selected_data_metrics:
            for name in self.gui_metric.data_metrics_names:
                name.set('...')
            for values in self.gui_metric.data_metrics_values:
                values.set('')
            return

        for m in enumerate(self.selected_data_metrics.items()):
            if m[1][1]:
                self.gui_metric.data_metrics_names[m[0]].set(m[1][0])
                self.gui_metric.data_metrics_values[m[0]].set(', '.join(str(round(x, 3)) for x in m[1][1]))

    def _gui_set_feature_metrics(self):
        self.gui_metric.listbox_metrics.delete(0, tk.END)
        for metric in self.selected_features_metrics:
            self.gui_metric.listbox_metrics.insert(tk.END, metric.list_name)

    def _gui_is_metric_name_taken(self, metric_name):
        for name in self.gui_metric.listbox_metrics.get(0, tk.END):
            if name == metric_name:
                return True
        return False

    def _gui_enable_button_calculate(self):
        self.gui_metric.button_calculate['state'] = 'normal'

    def _gui_disable_button_calculate(self):
        self.gui_metric.button_calculate['state'] = 'disabled'

    def _gui_enable_button_show_inference_overall_metrics(self):
        self.gui_metric.button_show_inference_overall_metrics['state'] = 'normal'

    def _gui_disable_button_show_inference_overall_metrics(self):
        self.gui_metric.button_show_inference_overall_metrics['state'] = 'disabled'

    def _gui_clear_parameters(self, disable_entries=True):
        gui_new_metric_name_var = self.gui_metric.metric_name_var
        gui_new_metric_name_entry = self.gui_metric.metric_name_entry
        gui_new_metric_name_var.set('')
        gui_new_metric_name_entry['state'] = 'disabled'

        gui_parameter_name_vars = self.gui_metric.parameter_name_vars
        gui_parameter_entries = self.gui_metric.parameter_entries
        for i, _ in enumerate(gui_parameter_name_vars):
            gui_parameter_name_vars[i].set(f'Parameter {i + 1}:')
            if disable_entries:
                gui_parameter_entries[i].delete(0, tk.END)
                gui_parameter_entries[i]['state'] = 'disabled'

    def _gui_set_parameter_names(self, parameter_names, metric_name):
        gui_new_metric_name_var = self.gui_metric.metric_name_var
        gui_new_metric_name_entry = self.gui_metric.metric_name_entry
        gui_new_metric_name_var.set(metric_name)
        gui_new_metric_name_entry['state'] = 'normal'

        gui_parameter_name_vars = self.gui_metric.parameter_name_vars
        gui_parameter_entries = self.gui_metric.parameter_entries
        for i, parameter_name in enumerate(parameter_names):
            gui_parameter_name_vars[i].set(parameter_name + ':')
            gui_parameter_entries[i]['state'] = 'normal'

    def _gui_set_parameter_values(self, parameters):
        for i, (_, parameter_value) in enumerate(parameters.items()):
            if parameter_value is None:
                parameter_value = ''
            self.gui_metric.parameter_value_vars[i].set(parameter_value)

    def _gui_set_calculated_on_name(self, calculated_on_name):
        gui_new_metric_calculated_on_name_var = self.gui_metric.metric_calculated_on_name_var
        gui_new_metric_calculated_on_name_var.set(calculated_on_name)

    def _gui_set_calculable_metric(self, calculable_metric):
        if calculable_metric is not None:
            calculable_metric = calculable_metric.value
        self.gui_metric.add_metric_var.set(calculable_metric)
