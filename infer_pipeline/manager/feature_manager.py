import tkinter as tk

from gui.gui_feature import GUIFeature


class FeatureManager():
    def __init__(self, root, status_manager, plot_manager):
        self.gui_feature = GUIFeature(
            root,
            listbox_feature_select_callback=self.feature_selected,
            listbox_feature_drag_callback=self.on_drag,
            listbox_feature_drop_callback=self.on_drop)
        self.status_manager = status_manager
        self.plot_manager = plot_manager
        self.run_id = None
        self.inference_name = None
        self.features = []
        self.selected_feature_name = None
        self.selected_feature_values = []

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
            self.gui_feature.listbox_features.insert(tk.END, feature.name.upper() + '_X')
            self.gui_feature.listbox_features.insert(tk.END, feature.name.upper() + '_Y')

    def _gui_clear_features(self):
        self.gui_feature.listbox_features.delete(0, tk.END)

    def feature_selected(self, event=None):
        selection = self.gui_feature.listbox_features.curselection()
        if not selection:
            return

        selection_str = self.gui_feature.listbox_features.get(selection)

        for feature in self.features:
            feature_name = feature.name.upper() + '_X'
            if feature_name == selection_str:
                self.selected_feature_name = feature_name
                self.selected_feature_values = feature.x

            feature_name = feature.name.upper() + '_Y'
            if feature_name == selection_str:
                self.selected_feature_name = feature_name
                self.selected_feature_values = feature.y

    def on_drag(self, event=None):
        self.gui_feature.root.configure(cursor='plus')

    def on_drop(self, event=None):
        self.gui_feature.root.configure(cursor='')
        if self.selected_feature_name is None:
            return

        plottable = {
            'name': self.inference_name + '_' + str(self.run_id).zfill(2) + '_' + self.selected_feature_name,
            'values': self.selected_feature_values
        }

        x, y = event.widget.winfo_pointerxy()
        self.plot_manager.add_to_plot(x, y, plottable=plottable)
