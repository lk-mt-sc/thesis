import tkinter as tk
from enum import Enum

from gui.gui_metric import GUIMetric
from data_types.metric import Metric
from manager.dataset_manager import KeypointsNoMetric
from metrics.missing_pose_estimations import MissingPoseEstimations
from metrics.dummy import Dummy


class AllMetrics(Enum):
    MISSING_POSE_ESTIMATIONS = 0
    OUTLIER = 1
    LOWPASS = 2
    HIGHPASS = 3


class CalculableMetrics(Enum):
    OUTLIER = 'Outlier Detection'
    LOWPASS = 'Low-Pass Filter'
    HIGHPASS = 'High-Pass Filter'


class StandardMetrics():
    def __init__(self, features):
        self.features = features
        self.features = [f for f in self.features if not KeypointsNoMetric.has_value(f.name[:-2])]
        self.metrics = []

    def calculate(self):
        missing_pose_estimations = []
        dummies = []
        for feature in self.features:
            missing_pose_estimations.append(MissingPoseEstimations(feature, name='Missing Pose Estimations'))
            dummies.append(Dummy(feature, name='Dummy'))

        self.metrics.append(Metric('Missing Pose Estimations', missing_pose_estimations))
        self.metrics.append(Metric('Dummy', dummies))

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
        self.compared_inference_1 = None
        self.compared_inference_2 = None
        self.compared_inference_3 = None
        self.selected_metric = None
        self.selected_inference = None
        self.selected_data = None
        self.selected_feature = None
        self.available_metrics = []

    def set_inference(self, inference):
        self.clear_metrics()
        self.selected_inference = inference

    def set_data(self, data):
        self.clear_metrics()
        self.selected_data = data
        # self.calculate_data_statistics()

    def set_feature(self, feature):
        self.clear_metrics()
        self.selected_feature = feature
        self.find_available_metrics()
        self._gui_set_metrics()

    def clear_metrics(self):
        self.selected_metric = None
        self.available_metrics.clear()
        self._gui_set_metrics()

    def add_to_compared_inferences(self, inference, position):
        match position:
            case 0:
                self.compared_inference_1 = inference
            case 1:
                self.compared_inference_2 = inference
            case 2:
                self.compared_inference_3 = inference

        # self.calculate_compared_inference_statistics()
        self._gui_set_compared_inferences_statistics()

    def remove_from_compared_inferences(self, position):
        match position:
            case 0:
                self.compared_inference_1 = None
            case 1:
                self.compared_inference_2 = None
            case 2:
                self.compared_inference_3 = None

        # self.calculate_compared_inference_statistics()
        self._gui_set_compared_inferences_statistics()

    def clear_compared_inferences(self):
        self.compared_inference_1 = None
        self.compared_inference_2 = None
        self.compared_inference_3 = None
        self._gui_set_compared_inferences_statistics()

    def _gui_set_compared_inferences_statistics(self):
        name_1 = self.compared_inference_1.name if self.compared_inference_1 is not None else 'No Inference Selected'
        name_2 = self.compared_inference_2.name if self.compared_inference_2 is not None else 'No Inference Selected'
        name_3 = self.compared_inference_3.name if self.compared_inference_3 is not None else 'No Inference Selected'
        self.gui_metric.inference_1_title_var.set(name_1)
        self.gui_metric.inference_2_title_var.set(name_2)
        self.gui_metric.inference_3_title_var.set(name_3)
        # statistics ...

    def find_available_metrics(self):
        for metric in self.selected_data.metrics:
            for m in metric.metrics:
                if self.selected_feature == m.feature:
                    self.available_metrics.append(m)

    def _gui_set_metrics(self):
        self.gui_metric.listbox_metrics.delete(0, tk.END)
        for metric in self.available_metrics:
            self.gui_metric.listbox_metrics.insert(tk.END, metric.name)

    def metric_selected(self, event=None):
        selection = self.gui_metric.listbox_metrics.curselection()
        if not selection:
            return

        selection_str = self.gui_metric.listbox_metrics.get(selection)

        for metric in self.available_metrics:
            if metric.name == selection_str:
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
