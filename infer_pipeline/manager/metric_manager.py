import tkinter as tk
from enum import Enum

from gui.gui_metric import GUIMetric
from manager.dataset_manager import KeypointsNoMetric
from metrics.peaks import Peaks
from metrics.deltas import Deltas
from metrics.missing_pose_estimations import MissingPoseEstimations


class AllMetrics(Enum):
    MISSING_POSE_ESTIMATIONS = 'Missing Pose Estimations'
    DELTAS = 'Deltas'
    PEAKS = 'Peaks'
    LOWPASS = 'Low-Pass Filter'
    HIGHPASS = 'High-Pass Filter'


class CalculableMetrics(Enum):
    PEAKS = AllMetrics.PEAKS.value
    LOWPASS = AllMetrics.LOWPASS.value
    HIGHPASS = AllMetrics.HIGHPASS.value


class StandardMetrics():
    def __init__(self):
        self.metrics = [
            MissingPoseEstimations(),
            Deltas(),
            Peaks()
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
            listbox_metrics_select_callback=self.metric_selected,
            listbox_metrics_drag_callback=self.on_drag,
            listbox_metrics_drop_callback=self.on_drop,
            button_calculate_callback=None,
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

    def set_inference(self, inference):
        self.clear_all_metrics()
        self.selected_inference = inference
        self.selected_inference_metrics = self.calculate_inference_metrics()

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

        return inference_metrics

    def add_to_compared_inferences(self, inference, position):
        set_names = all(i is None for i in self.compared_inferences)
        self.compared_inferences[position] = inference
        self._gui_set_inference_metrics(position, title=inference.name, set_names=set_names)

    def remove_from_compared_inferences(self, position):
        self.compared_inferences[position] = None
        clear_names = all(i is None for i in self.compared_inferences)
        self._gui_clear_inference_metrics(position, clear_names=clear_names)

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
        return feature_metrics

    def clear_all_metrics(self):
        self.clear_inference_metrics()
        self.clear_data_metrics()
        self.clear_feature_metrics()

    def clear_inference_metrics(self):
        self.selected_inference_metrics = {}

    def clear_data_metrics(self):
        self.selected_data_metrics = {}
        self._gui_set_data_metrics()

    def clear_feature_metrics(self):
        self.selected_metric = None
        self.selected_features_metrics.clear()
        self._gui_set_feature_metrics()

    def metric_selected(self, event=None):
        selection = self.gui_metric.listbox_metrics.curselection()
        if not selection:
            return

        selection_str = self.gui_metric.listbox_metrics.get(selection)

        for metric in self.selected_features_metrics:
            if metric.list_name == selection_str:
                self.selected_metric = metric

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

    def _gui_set_inference_metrics(self, position, title, set_names=False):
        for m in enumerate(self.selected_inference_metrics.items()):
            if set_names:
                self.gui_metric.inference_metrics_names[m[0]].set(m[1][0])

            self.gui_metric.inference_metrics_values[position][m[0]].set(', '.join(str(round(x, 3)) for x in m[1][1]))

        if len(title) > 20:
            title = title[: 18] + '...'
        self.gui_metric.inference_metrics_titles[position].set(title)

    def _gui_clear_inference_metrics(self, position, clear_names=False):
        if clear_names:
            for name in self.gui_metric.inference_metrics_names:
                name.set('...')

        for values in self.gui_metric.inference_metrics_values[position]:
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
            self.gui_metric.data_metrics_names[m[0]].set(m[1][0])
            self.gui_metric.data_metrics_values[m[0]].set(', '.join(str(round(x, 3)) for x in m[1][1]))

    def _gui_set_feature_metrics(self):
        self.gui_metric.listbox_metrics.delete(0, tk.END)
        for metric in self.selected_features_metrics:
            self.gui_metric.listbox_metrics.insert(tk.END, metric.list_name)
