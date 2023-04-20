import tkinter as tk

from gui.gui_mmdetection_model import GUIMMDetectionModel
from manager.status_manager import Status
from data_types.mmdetection_model import MMDetectionModel


class MMDetectionModelManager():
    def __init__(self, root, status_manager):
        self.gui_mmdetection_model = GUIMMDetectionModel(root, listbox_models_callback=self.model_selected)
        self.status_manager = status_manager
        self.models = []
        self.fetch_models()

    def fetch_models(self):
        if not self.status_manager.has_status(Status.FETCHING_MMDETECTION_MODELS):
            self.status_manager.add_status(Status.FETCHING_MMDETECTION_MODELS)

            # manually selected models using the interactive model zoo at https://platform.openmmlab.com/modelzoos
            self.models.append(MMDetectionModel(
                name='-',
                key_metric='-',
                checkpoint='-',
                config='-'
            ))

            self._gui_set_models()

    def _gui_set_models(self):
        self.gui_mmdetection_model.listbox_models.delete(0, tk.END)
        for model in self.models:
            self.gui_mmdetection_model.listbox_models.insert(tk.END, model)

    def model_selected(self):
        pass
