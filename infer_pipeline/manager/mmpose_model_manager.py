import os
import copy

import tkinter as tk
from threading import Thread

from gui.gui_mmpose_model import GUIMMPoseModel
from manager.status_manager import Status
from data_types.mmpose_model import MMPoseModel
from model_zoo import ModelZoo
from common import INFER_PIPELINE_MMDPOSE_CONFIGS_DIR
from common import MMPOSE_CHECKPOINTS_DIR

class MMPoseModelManager():
    def __init__(self, root, status_manager):
        self.gui_mmpose_model = GUIMMPoseModel(
            root,
            combobox_model_preset_callback=self.filter,
            button_refresh_callback=self.fetch_models,
            listbox_models_callback=self.model_selected
        )
        self.status_manager = status_manager
        self.model_zoo = ModelZoo(redownload_model_zoo=False)
        self.zoo_models = []
        self.custom_models = []
        self.models_show = []
        self.selected_model = None
        self.filter_models = None
        self.fetch_models()
        self._gui_set_presets()

        self.custom_configs = [
            {
                'original_config': 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py',
                'custom_config': os.path.join(INFER_PIPELINE_MMDPOSE_CONFIGS_DIR, 'td-hm_hrnet-w48_udp-8xb32-210e_sc-384x288.py')
            },
            {
                'original_config': 'configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w48_8xb10-140e_coco-640x640.py',
                'custom_config': os.path.join(INFER_PIPELINE_MMDPOSE_CONFIGS_DIR, 'dekr_hrnet-w48_8xb10-140e_sc-640x640.py')
            },
            {
                'original_config': 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_mobilenetv2_8xb64-210e_coco-256x192.py',
                'custom_config': os.path.join(INFER_PIPELINE_MMDPOSE_CONFIGS_DIR, 'td-hm_mobilenetv2_8xb64-210e_sc-256x192.py')
            },
        ]

    def fetch_models(self):
        if not self.status_manager.has_status(Status.FETCHING_MMPOSE_MODELS):
            self.status_manager.add_status(Status.FETCHING_MMPOSE_MODELS)
            self._gui_disable_button_refresh()
            self._gui_clear_listbox_models()
            fetch_thread = Thread(target=self._fetch_models)
            fetch_thread.start()
            self.monitor_fetch_thread(fetch_thread)

    def _fetch_models(self):
        self.zoo_models = self.model_zoo.get_models(
            dataset='coco',
            redownload_model_zoo=self.gui_mmpose_model.checkbutton_redownload_model_zoo.instate(['selected'])
        )

    def monitor_fetch_thread(self, thread):
        if thread.is_alive():
            self.gui_mmpose_model.root.after(50, lambda: self.monitor_fetch_thread(thread))
        else:
            self.fetch_custom_models()
            self.selected_model = None
            self.filter()
            self._gui_enable_button_refresh()
            self.status_manager.remove_status(Status.FETCHING_MMPOSE_MODELS)

    def fetch_custom_models(self):
        for config in self.custom_configs:
            for model in self.zoo_models:
                if model.config == config['original_config']:
                    new_custom_model = copy.deepcopy(model)
                    new_custom_model.config = config['custom_config']
                    new_custom_model.checkpoint = os.path.join(MMPOSE_CHECKPOINTS_DIR, new_custom_model.checkpoint)
                    self.custom_models.append(new_custom_model)

    def _gui_set_presets(self):
        self.gui_mmpose_model.combobox_model_preset['values'] = [
            'Model Zoo',
            'Custom',
        ]

        self.gui_mmpose_model.combobox_model_preset.current(1)

    def _gui_get_filter(self):
        self.filter_models = self.gui_mmpose_model.combobox_model_preset.get()

    def _gui_clear_listbox_models(self):
        self.gui_mmpose_model.listbox_models.delete(0, tk.END)

    def _gui_disable_button_refresh(self):
        self.gui_mmpose_model.button_refresh['state'] = 'disabled'

    def _gui_enable_button_refresh(self):
        self.gui_mmpose_model.button_refresh['state'] = 'normal'

    def _gui_set_models(self):
        self.gui_mmpose_model.listbox_models.delete(0, tk.END)
        for model in self.models_show:
            self.gui_mmpose_model.listbox_models.insert(tk.END, model)

    def _gui_set_details(self):
        self.gui_mmpose_model.details_section_var.set(self.selected_model.section)
        self.gui_mmpose_model.details_arch_var.set(self.selected_model.arch)
        self.gui_mmpose_model.details_dataset_var.set(self.selected_model.dataset)
        self.gui_mmpose_model.details_input_size_var.set(self.selected_model.input_size)
        self.gui_mmpose_model.details_key_metric_var.set(self.selected_model.key_metric)
        self.gui_mmpose_model.details_checkpoint_var.set(self.selected_model.checkpoint)
        self.gui_mmpose_model.details_config_var.set(self.selected_model.config)

    def _gui_clear_details(self):
        self.gui_mmpose_model.details_section_var.set('')
        self.gui_mmpose_model.details_arch_var.set('')
        self.gui_mmpose_model.details_dataset_var.set('')
        self.gui_mmpose_model.details_input_size_var.set('')
        self.gui_mmpose_model.details_key_metric_var.set('')
        self.gui_mmpose_model.details_checkpoint_var.set('')
        self.gui_mmpose_model.details_config_var.set('')

    def filter(self, event=None):
        self.selected_model = None
        self.models_show.clear()
        self._gui_clear_details()
        self._gui_get_filter()

        if self.filter_models == 'Model Zoo':
            self.models_show = self.zoo_models.copy()

        if self.filter_models == 'Custom':
            self.models_show = self.custom_models.copy()

        self._gui_set_models()

    def model_selected(self, event=None):
        current_selection = self.gui_mmpose_model.listbox_models.curselection()
        selection_str = self.gui_mmpose_model.listbox_models.get(current_selection[0])
        selected_model = MMPoseModel.get_from_selection_string(selection_str)
        self.selected_model = next(model for model in self.models_show if model == selected_model)

        self._gui_set_details()
