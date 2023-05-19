import tkinter as tk

from gui.gui_feature import GUIFeature


class FeatureManager():
    def __init__(self, root, status_manager, plot_manager):
        self.gui_feature = GUIFeature(
            root,
            listbox_feature_callback=self.feature_selected)
        self.status_manager = status_manager
        self.plot_manager = plot_manager
        self.run_id = None
        self.inference_name = None
        self.features = []

    def set_data(self, run, inference_name):
        self.run_id = run.id
        self.inference_name = inference_name
        self.features = run.features.copy()
        self._gui_set_features()

    def clear(self):
        self.run_id = None
        self.inference_name = None
        self.features.clear()
        self._gui_clear_features()

    def _gui_set_features(self):
        self.gui_feature.listbox_features.delete(0, tk.END)
        for feature in self.features:
            self.gui_feature.listbox_features.insert(tk.END, feature)

    def _gui_clear_features(self):
        self.gui_feature.listbox_features.delete(0, tk.END)

    def feature_selected(self, event=None):
        selection = self.gui_feature.listbox_features.curselection()
        if not selection:
            return
        selection_str = self.gui_feature.listbox_features.get(selection)

        for feature in self.features:
            feature_name = feature.name.upper()
            if feature_name == selection_str:
                name = self.inference_name + '_' + str(self.run_id).zfill(2) + '_' + feature_name
                self.plot_manager.toggle_feature_plot(name, feature)
