import tkinter as tk
from threading import Thread

from gui.gui_mmpose_model import GUIMMPoseModel
from manager.status_manager import Status
from data_types.mmpose_model import MMPoseModel
from model_zoo import ModelZoo


class MMPoseModelManager():
    def __init__(self, root, status_manager):
        self.gui_mmpose_model = GUIMMPoseModel(
            root,
            button_refresh_callback=self.fetch_models,
            listbox_models_callback=self.model_selected
        )
        self.status_manager = status_manager
        self.model_zoo = ModelZoo(redownload_model_zoo=False)
        self.models = []
        self.selected_model = None
        self.fetch_models()

    def fetch_models(self):
        if not self.status_manager.has_status(Status.FETCHING_MMPOSE_MODELS):
            self.status_manager.add_status(Status.FETCHING_MMPOSE_MODELS)
            self._gui_disable_button_refresh()
            self._gui_clear_listbox_models()
            fetch_thread = Thread(target=self._fetch_models)
            fetch_thread.start()
            self.monitor_fetch_thread(fetch_thread)

    def _fetch_models(self):
        self.models = self.model_zoo.get_models(
            dataset='all',
            redownload_model_zoo=self.gui_mmpose_model.checkbutton_redownload_model_zoo.instate(['selected'])
        )

    def monitor_fetch_thread(self, thread):
        if thread.is_alive():
            self.gui_mmpose_model.root.after(50, lambda: self.monitor_fetch_thread(thread))
        else:
            self._gui_set_models()
            self._gui_enable_button_refresh()
            self.status_manager.remove_status(Status.FETCHING_MMPOSE_MODELS)

    def _gui_clear_listbox_models(self):
        self.gui_mmpose_model.listbox_models.delete(0, tk.END)

    def _gui_disable_button_refresh(self):
        self.gui_mmpose_model.button_refresh['state'] = 'disabled'

    def _gui_enable_button_refresh(self):
        self.gui_mmpose_model.button_refresh['state'] = 'normal'

    def _gui_set_models(self):
        for model in self.models:
            self.gui_mmpose_model.listbox_models.insert(tk.END, model)

    def _gui_set_details(self):
        self.gui_mmpose_model.details_section_var.set(self.selected_model.section)
        self.gui_mmpose_model.details_arch_var.set(self.selected_model.arch)
        self.gui_mmpose_model.details_dataset_var.set(self.selected_model.dataset)
        self.gui_mmpose_model.details_input_size_var.set(self.selected_model.input_size)
        self.gui_mmpose_model.details_key_metric_var.set(self.selected_model.key_metric)
        self.gui_mmpose_model.details_checkpoint_var.set(self.selected_model.checkpoint)
        self.gui_mmpose_model.details_config_var.set(self.selected_model.config)

    def model_selected(self, event=None):
        current_selection = self.gui_mmpose_model.listbox_models.curselection()
        selection_str = self.gui_mmpose_model.listbox_models.get(current_selection[0])
        selected_model = MMPoseModel.get_from_selection_string(selection_str)
        self.selected_model = next(model for model in self.models if model == selected_model)

        self._gui_set_details()
