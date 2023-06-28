import os
import imp

import tkinter as tk
from threading import Thread

from gui.gui_mmpose_model import GUIMMPoseModel
from manager.status_manager import Status
from data_types.mmpose_model import MMPoseModel
from model_zoo import ModelZoo
from common import INFER_PIPELINE_MMDPOSE_CONFIGS_DIR
from common import MMPOSE029_CHECKPOINTS_DIR, MMPOSE029_CONFIGS_DIR
from common import MMPOSE_DIR, MMPOSE_CHECKPOINTS_DIR


class MMPoseModelManager():
    def __init__(self, root, status_manager):
        self.gui_mmpose_model = GUIMMPoseModel(
            root,
            button_refresh_callback=self.fetch_models,
            listbox_models_callback=self.model_selected
        )
        self.status_manager = status_manager
        self.model_zoo = ModelZoo(redownload_model_zoo=False)
        self.zoo_models = []
        self.custom_models_init = [
            MMPoseModel(
                section='Posewarper + Hrnet + Posetrack18',
                arch='pose_hrnet_w48',
                dataset='posetrack18',
                input_size='384x288',
                key_metric_value=85.0,
                key_metric_name='Total',
                checkpoint=os.path.join(MMPOSE029_CHECKPOINTS_DIR,
                                        'hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth'),
                config=os.path.join(
                    MMPOSE029_CONFIGS_DIR,
                    'body',
                    '2d_kpt_sview_rgb_vid',
                    'posewarper',
                    'posetrack18',
                    'hrnet_w48_posetrack18_384x288_posewarper_stage2.py'
                ),
                transfer_learned=False,
                multi_frame_mmpose029=True
            ),
            # add self-trained model (best model according to COCO or to our metrics)
        ]
        self.custom_models = self.custom_models_init.copy()
        self.models_show = []
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
        redownload_model_zoo = self.gui_mmpose_model.checkbutton_redownload_model_zoo.instate(['selected'])
        self.zoo_models = self.model_zoo.get_models(
            dataset='coco',
            redownload_model_zoo=redownload_model_zoo
        )

    def monitor_fetch_thread(self, thread):
        if thread.is_alive():
            self.gui_mmpose_model.root.after(50, lambda: self.monitor_fetch_thread(thread))
        else:
            self.custom_models = self.custom_models_init.copy()
            self.fetch_custom_models()
            self.selected_model = None
            self.models_show = self.custom_models.copy()
            self._gui_set_models()
            self._gui_enable_button_refresh()
            self.status_manager.remove_status(Status.FETCHING_MMPOSE_MODELS)

    def fetch_custom_models(self):
        unavailable_checkpoints = ['td-hm_hrnet-w32_8xb64-210e_coco-aic-256x192-merge-b05435b9_20221025.pth']

        sections = set()
        for model in self.zoo_models:
            sections.add(model.section)

        models_per_section = {}
        for section in sections:
            models_per_section[section] = []

        for model in self.zoo_models:
            models_per_section[model.section].append(model)

        best_models = []
        for _, models in models_per_section.items():
            models.sort(key=lambda x: x.key_metric_value, reverse=True)
            best_models.append(models[0])

        best_models.sort(key=lambda x: x.key_metric_value, reverse=True)
        bottom_up_models = []
        top_down_models = []
        for model in best_models:
            config_name = os.path.basename(model.config)
            config = imp.load_source(config_name, os.path.join(MMPOSE_DIR, model.config))
            if hasattr(config, 'data_mode'):
                data_mode = config.data_mode
                if data_mode == 'topdown':
                    top_down_models.append(model)
                if data_mode == 'bottomup':
                    bottom_up_models.append(model)

        top_down_models = [model for model in top_down_models if model.checkpoint not in unavailable_checkpoints]
        top_down_models = top_down_models[:10]

        bottom_up_models = [model for model in bottom_up_models if model.checkpoint not in unavailable_checkpoints]
        bottom_up_models = bottom_up_models[:10]

        for model in top_down_models:
            model.config = self.create_custom_config(model, data_mode='topdown')
            model.checkpoint = os.path.join(MMPOSE_CHECKPOINTS_DIR, model.checkpoint)
            self.custom_models.append(model)

        for model in bottom_up_models:
            model.config = self.create_custom_config(model, data_mode='bottomup')
            model.checkpoint = os.path.join(MMPOSE_CHECKPOINTS_DIR, model.checkpoint)
            self.custom_models.append(model)

    def create_custom_config(self, model, data_mode):
        config_name = os.path.basename(model.config)
        config = imp.load_source(config_name, os.path.join(MMPOSE_DIR, model.config))
        if hasattr(config, 'test_dataloader'):
            batch_size = config.test_dataloader['batch_size']
            if batch_size != 1:
                batch_size = 64
        else:
            batch_size = 64

        custom_config = os.path.join(INFER_PIPELINE_MMDPOSE_CONFIGS_DIR,
                                     os.path.basename(model.config).replace('coco', 'sc'))

        with open(custom_config, 'w', encoding='utf8') as config_file:
            config_file.write(f"_base_ = ['{os.path.join(MMPOSE_DIR, model.config)}']\n")
            config_file.write(
                f"test_dataloader = dict(batch_size={batch_size}, dataset=dict(data_root='', ann_file='', bbox_file=None, data_prefix=dict(img='')))\n")
            config_file.write("test_evaluator = dict(format_only=True)\n")
            config_file.write("default_hooks = dict(logger=dict(interval=1))\n")
            config_file.write(f"data_mode = '{data_mode}'")

        config_file.close()
        return custom_config

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
        self.gui_mmpose_model.details_key_metric_var.set(str(self.selected_model.key_metric_value)
                                                         + ' ' + self.selected_model.key_metric_name)
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

    def model_selected(self, event=None):
        current_selection = self.gui_mmpose_model.listbox_models.curselection()
        selection_str = self.gui_mmpose_model.listbox_models.get(current_selection[0])
        selected_model = MMPoseModel.get_from_selection_string(selection_str)
        self.selected_model = next(model for model in self.models_show if model == selected_model)

        self._gui_set_details()
