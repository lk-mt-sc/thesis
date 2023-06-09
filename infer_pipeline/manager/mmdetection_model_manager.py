import os
import tkinter as tk

from gui.gui_mmdetection_model import GUIMMDetectionModel
from manager.status_manager import Status
from data_types.mmdetection_model import MMDetectionModel

from common import TRAIN_PIPELINE_MMDETECTION_TRAININGS_DIR
from common import INFER_PIPELINE_MMDETECTION_CONFIGS_DIR


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

            self.models.append(MMDetectionModel(
                name='Faster R-CNN',
                key_metric_value=0.921,
                key_metric_name='box AP',
                checkpoint=os.path.join(TRAIN_PIPELINE_MMDETECTION_TRAININGS_DIR, 'faster_rcnn',
                                        'faster-rcnn', 'train', 'best_coco_bbox_mAP_epoch_14.pth'),
                config=os.path.join(INFER_PIPELINE_MMDETECTION_CONFIGS_DIR, 'faster-rcnn.py')))
            self.models.append(MMDetectionModel(
                name='RTMDet',
                key_metric_value=0.952,
                key_metric_name='box AP',
                checkpoint=os.path.join(TRAIN_PIPELINE_MMDETECTION_TRAININGS_DIR, 'rtmdet',
                                        'rtmdet', 'train', 'best_coco_bbox_mAP_epoch_37.pth'),
                config=os.path.join(INFER_PIPELINE_MMDETECTION_CONFIGS_DIR, 'rtmdet.py')))
            self.models.append(MMDetectionModel(
                name='YOLOX',
                key_metric_value=0.947,
                key_metric_name='box AP',
                checkpoint=os.path.join(TRAIN_PIPELINE_MMDETECTION_TRAININGS_DIR, 'yolox', 'yolox',
                                        'train', 'best_coco_bbox_mAP_epoch_62.pth'),
                config=os.path.join(INFER_PIPELINE_MMDETECTION_CONFIGS_DIR, 'yolox.py')))
            self.models.append(MMDetectionModel(
                name='VarifocalNet',
                key_metric_value=0.966,
                key_metric_name='box AP',
                checkpoint=os.path.join(TRAIN_PIPELINE_MMDETECTION_TRAININGS_DIR, 'vfnet',
                                        'vfnet', 'train', 'best_coco_bbox_mAP_epoch_20.pth'),
                config=os.path.join(INFER_PIPELINE_MMDETECTION_CONFIGS_DIR, 'vfnet.py')))
            self.models.append(MMDetectionModel(
                name='TOOD',
                key_metric_value=0.966,
                key_metric_name='box AP',
                checkpoint=os.path.join(TRAIN_PIPELINE_MMDETECTION_TRAININGS_DIR, 'tood',
                                        'tood', 'train', 'best_coco_bbox_mAP_epoch_22.pth'),
                config=os.path.join(INFER_PIPELINE_MMDETECTION_CONFIGS_DIR, 'tood.py')))

            self.default_model = self.models[3]

            self._gui_set_models()
            self.status_manager.remove_status(Status.FETCHING_MMDETECTION_MODELS)

    def _gui_set_models(self):
        self.gui_mmdetection_model.listbox_models.delete(0, tk.END)
        for model in self.models:
            self.gui_mmdetection_model.listbox_models.insert(tk.END, model)

    def _gui_set_details(self):
        self.gui_mmdetection_model.details_name_var.set(self.selected_model.name)
        self.gui_mmdetection_model.details_key_metric_var.set(str(self.selected_model.key_metric_value)
                                                              + ' ' + self.selected_model.key_metric_name)
        self.gui_mmdetection_model.details_checkpoint_var.set(self.selected_model.checkpoint)
        self.gui_mmdetection_model.details_config_var.set(self.selected_model.config)

    def model_selected(self, event=None):
        current_selection = self.gui_mmdetection_model.listbox_models.curselection()
        selection_str = self.gui_mmdetection_model.listbox_models.get(current_selection[0])
        selected_model = MMDetectionModel.get_from_selection_string(selection_str)
        self.selected_model = next(model for model in self.models if model == selected_model)

        self._gui_set_details()
