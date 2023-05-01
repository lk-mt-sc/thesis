import tkinter as tk

from gui.gui_mmdetection_model import GUIMMDetectionModel
from manager.status_manager import Status
from data_types.mmdetection_model import MMDetectionModel


class MMDetectionModelManager():
    def __init__(self, root, status_manager):
        self.gui_mmdetection_model = GUIMMDetectionModel(root, listbox_models_callback=self.model_selected)
        self.status_manager = status_manager
        self.models = []
        self.selected_model = None
        self.fetch_models()

    def fetch_models(self):
        if not self.status_manager.has_status(Status.FETCHING_MMDETECTION_MODELS):
            self.status_manager.add_status(Status.FETCHING_MMDETECTION_MODELS)

            # manually selected models based on box AP
            self.models.append(MMDetectionModel(
                name='Faster R-CNN',
                key_metric='box AP (55.8)',
                checkpoint='faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth',
                config='configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_sc-person.py'
            ))
            self.models.append(MMDetectionModel(
                name='YOLOX',
                key_metric='box AP (50.9)',
                checkpoint='yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth',
                config='configs/yolox/yolox_x_8xb8-300e_sc.py'
            ))
            self.models.append(MMDetectionModel(
                name='RTMDet (default)',
                key_metric='box AP (49.4)',
                checkpoint='rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth',
                config='configs/rtmdet/rtmdet_m_8xb32-300e_sc.py'
            ))
            self._gui_set_models()
            self.status_manager.remove_status(Status.FETCHING_MMDETECTION_MODELS)

    def _gui_set_models(self):
        self.gui_mmdetection_model.listbox_models.delete(0, tk.END)
        for model in self.models:
            self.gui_mmdetection_model.listbox_models.insert(tk.END, model)

    def _gui_set_details(self):
        self.gui_mmdetection_model.details_name_var.set(self.selected_model.name)
        self.gui_mmdetection_model.details_key_metric_var.set(self.selected_model.key_metric)
        self.gui_mmdetection_model.details_checkpoint_var.set(self.selected_model.checkpoint)
        self.gui_mmdetection_model.details_config_var.set(self.selected_model.config)

    def model_selected(self, event=None):
        current_selection = self.gui_mmdetection_model.listbox_models.curselection()
        selection_str = self.gui_mmdetection_model.listbox_models.get(current_selection[0])
        selected_model = MMDetectionModel.get_from_selection_string(selection_str)
        self.selected_model = next(model for model in self.models if model == selected_model)

        self._gui_set_details()
