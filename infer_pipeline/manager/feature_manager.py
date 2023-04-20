import tkinter as tk

from gui.gui_feature import GUIFeature


class FeatureManager():
    def __init__(self, root, status_manager):
        self.gui_feature = GUIFeature(
            root,
            listbox_feature_callback=self.feature_selected)
        self.status_manager = status_manager
        self.features = []

    def set_features(self, run):
        self.features = run.features
        self._gui_set_features()

    def _gui_set_features(self):
        self.gui_feature.listbox_features.delete(0, tk.END)
        for feature in self.features:
            self.gui_feature.listbox_features.insert(tk.END, feature)

    def feature_selected(self):
        pass
