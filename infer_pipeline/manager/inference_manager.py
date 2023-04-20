import os
import glob
import json
import tkinter as tk
from threading import Thread

from common import INFERENCES_DIR
from gui.gui_infer import GUIInfer
from gui.gui_inference import GUIInference
from manager.status_manager import Status
from data_types.inference import Inference


class InferenceManager():
    def __init__(self, root, status_manager):
        self.gui_infer = GUIInfer(root, self.infer)
        self.gui_inference = GUIInference(
            root,
            button_compare_callback=self.compare_inference,
            button_delete_callback=self.delete_inference,
            button_refresh_callback=self.fetch_inferences,
            listbox_inferences_callback=self.inference_selected,
            listbox_data_callback=self.data_selected)
        self.status_manager = status_manager
        self.inferences = []
        self.fetch_inferences()

    def fetch_inferences(self):
        if not self.status_manager.has_status(Status.FETCHING_INFERENCES):
            self.status_manager.add_status(Status.FETCHING_INFERENCES)
            fetch_thread = Thread(target=self._fetch_inferences)
            fetch_thread.start()
            self.monitor_fetch_inferences(fetch_thread)

    def _fetch_inferences(self):
        inferences = sorted(glob.glob(os.path.join(INFERENCES_DIR, '*')))
        for inference in inferences:
            metadata_path = os.path.join(inference, 'metadata.json')
            if os.path.exists(metadata_path):
                metadata = open(metadata_path, 'r', encoding='utf8')
                self.inferences.append(Inference(json.load(metadata)))

    def monitor_fetch_inferences(self, thread):
        if thread.is_alive():
            self.gui_inference.root.after(50, lambda: self.monitor_fetch_inferences(thread))
        else:
            self._gui_set_inferences()
            self.status_manager.remove_status(Status.FETCHING_INFERENCES)

    def _gui_set_inferences(self):
        self.gui_inference.listbox_inferences.delete(0, tk.END)
        for inference in self.inferences:
            self.gui_inference.listbox_inferences.insert(tk.END, inference)

    def infer(self):
        pass

    def inference_selected(self):
        pass

    def data_selected(self):
        pass

    def delete_inference(self):
        pass

    def compare_inference(self):
        pass
