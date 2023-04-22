import os
import glob
import json
import tkinter as tk
from tkinter import messagebox
from threading import Thread

import torch

from utils import id_generator
from common import INFERENCES_DIR
from gui.gui_inference_queue import GUIInferenceQueue
from gui.gui_inference import GUIInference
from manager.status_manager import Status
from data_types.inference import Inference


class InferenceManager():
    def __init__(self, root, status_manager, mmpose_model_manager, mmdetection_model_manager, data_manager):
        self.gui_inference_queue = GUIInferenceQueue(
            root,
            button_add_callback=self.queue_inference_add,
            listbox_inferences_queue_callback=self.queue_inference_selected,
            button_infer_callback=self.infer,
            button_delete_callback=self.queue_inference_delete)
        self.gui_inference = GUIInference(
            root,
            button_compare_callback=self.compare_inference,
            button_delete_callback=self.delete_inference,
            button_refresh_callback=self.fetch_inferences,
            listbox_inferences_callback=self.inference_selected,
            listbox_data_callback=self.data_selected)
        self.status_manager = status_manager
        self.mmpose_model_manager = mmpose_model_manager
        self.mmdetection_model_manager = mmdetection_model_manager
        self.data_manager = data_manager
        self.active_processes = []
        self.pending_processes = []
        self.parallel_processes = 2
        self.inferences = []
        self.selected_inferences = []
        self.queue_inferences = []
        self.queue_selected_inferences = []
        self.fetch_inferences()
        torch.multiprocessing.set_start_method('spawn', force=True)

    def fetch_inferences(self):
        if not self.status_manager.has_status(Status.FETCHING_INFERENCES):
            self.status_manager.add_status(Status.FETCHING_INFERENCES)
            fetch_thread = Thread(target=self._fetch_inferences)
            fetch_thread.start()
            self.monitor_fetch_inferences(fetch_thread)

    def _fetch_inferences(self):
        inferences = sorted(glob.glob(os.path.join(INFERENCES_DIR, '*')))
        self.inferences.clear()
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

    def _gui_set_queue_inferences(self):
        self.gui_inference_queue.listbox_inferences.delete(0, tk.END)
        for inference in self.queue_inferences:
            self.gui_inference_queue.listbox_inferences.insert(tk.END, inference)

    def _gui_get_name_text(self):
        return self.gui_inference_queue.text_infer_name.get(1.0, tk.END).strip()[:50]

    def _gui_get_description_text(self):
        return self.gui_inference_queue.text_infer_description.get(1.0, tk.END).strip()[:320]

    def _gui_enable_button_infer(self):
        self.gui_inference_queue.button_infer['state'] = 'normal'

    def _gui_disable_button_infer(self):
        self.gui_inference_queue.button_infer['state'] = 'disabled'

    def _gui_enable_button_queue_delete(self):
        self.gui_inference_queue.button_delete['state'] = 'normal'

    def _gui_disable_button_queue_delete(self):
        self.gui_inference_queue.button_delete['state'] = 'disabled'

    def _gui_enable_button_queue_add(self):
        self.gui_inference_queue.button_add['state'] = 'normal'

    def _gui_disable_button_queue_add(self):
        self.gui_inference_queue.button_add['state'] = 'disabled'

    def _gui_update_inference_progress(self, inference_id, inference_progress):
        for i, inference in enumerate(self.queue_inferences):
            if inference.id == inference_id:
                self.gui_inference_queue.listbox_inferences.delete(i)

                progress_str = str(inference)
                if inference_progress == 0:
                    progress_str += ' - Starting...'
                else:
                    progress_str += f' - {inference_progress}%'

                self.gui_inference_queue.listbox_inferences.insert(i, progress_str)

    def queue_inference_add(self):
        inference_mmpose_model = self.mmpose_model_manager.selected_model
        inference_mmdetection_model = self.mmdetection_model_manager.selected_model
        inference_data = self.data_manager.selected_data.copy()

        if inference_mmpose_model is None or inference_mmdetection_model is None or not inference_data:
            messagebox.showerror(title='', message='Model or data selection missing.')
            return

        inference_id = id_generator()
        while (self.inference_id_taken(inference_id)):
            inference_id = id_generator()

        inference_name = self._gui_get_name_text()
        if not inference_name:
            inference_name = inference_id

        inference_description = self._gui_get_description_text()
        if not inference_description:
            inference_description = 'No description provided.'

        inference_duration = []

        inference = Inference(metadata={
            'id': inference_id,
            'name': inference_name,
            'mmpose_model': inference_mmpose_model,
            'mmdetection_model': inference_mmdetection_model,
            'data': inference_data,
            'duration': inference_duration,
            'description': inference_description
        })

        self.queue_inferences.append(inference)
        self._gui_set_queue_inferences()

    def queue_inference_delete(self):
        for inference in self.queue_selected_inferences:
            self.queue_inferences.remove(inference)

        self._gui_set_queue_inferences()
        self._gui_disable_button_queue_delete()

    def queue_inference_selected(self, event=None):
        if not self.status_manager.has_status(Status.INFERING):
            current_selection = self.gui_inference_queue.listbox_inferences.curselection()
            if current_selection:
                self.queue_selected_inferences.clear()
                self._gui_enable_button_queue_delete()
                for selection in current_selection:
                    selection_str = self.gui_inference_queue.listbox_inferences.get(selection)
                    inference_id = selection_str.split(' | ')[-1]
                    self.queue_selected_inferences.append(
                        next(inference for inference in self.queue_inferences if inference.id == inference_id))

    def infer(self):
        if self.queue_inferences:
            self._gui_disable_button_infer()
            self._gui_disable_button_queue_delete()
            self._gui_disable_button_queue_add()
            self._gui_set_queue_inferences()
            self.queue_selected_inferences.clear()
            self.status_manager.add_status(Status.INFERING)

            for inference in self.queue_inferences:
                inference_progress = torch.multiprocessing.Value('i')
                inference_process = torch.multiprocessing.Process(
                    target=inference.infer, args=(inference_progress, ))
                pending_process = {
                    'process': inference_process,
                    'progress': inference_progress,
                    'inference_id': inference.id
                }
                self.pending_processes.append(pending_process)

            for i in range(0, self.parallel_processes):
                if len(self.pending_processes) >= i + 1:
                    self.active_processes.append(self.pending_processes[i])

            del self.pending_processes[:len(self.active_processes)]

            for process in self.active_processes:
                self.start_process(process)

    def start_process(self, process):
        process['process'].start()
        self.monitor_inference_process(process)

    def monitor_inference_process(self, process):
        inference_process = process['process']
        if inference_process.is_alive():
            self.gui_inference.root.after(50, lambda: self.monitor_inference_process(process))
            self._gui_update_inference_progress(process['inference_id'], process['progress'].value)
        else:
            self.fetch_inferences()
            self.active_processes.remove(process)
            if self.pending_processes:
                self.active_processes.append(self.pending_processes.pop(0))
                self.start_process(self.active_processes[-1])
            if not self.active_processes:
                self.queue_inferences.clear()
                self._gui_enable_button_infer()
                self._gui_enable_button_queue_add()
                self._gui_set_queue_inferences()
                self.status_manager.remove_status(Status.INFERING)

    def inference_id_taken(self, id_):
        for inference in self.inferences:
            if inference.id == id_:
                return True
        return False

    def inference_selected(self, event=None):
        pass

    def data_selected(self):
        pass

    def delete_inference(self):
        pass

    def compare_inference(self):
        pass
