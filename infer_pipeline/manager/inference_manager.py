import os
import glob
import json
import shutil
import ctypes
import subprocess
import tkinter as tk
from tkinter import messagebox
from threading import Thread

import torch

from utils import id_generator
from common import INFERENCES_DIR, WSL_PREFIX, MMPOSE_DATASET_DIR
from gui.gui_inference_queue import GUIInferenceQueue
from gui.gui_inference import GUIInference
from manager.status_manager import Status
from manager.dataset_manager import Datasets
from data_types.inference import Inference


class InferenceManager():
    def __init__(
            self,
            root,
            dataset_manager,
            status_manager,
            mmpose_model_manager,
            mmdetection_model_manager,
            data_manager,
            metric_manager,
            plot_manager,
            feature_manager
    ):
        self.gui_inference_queue = GUIInferenceQueue(
            root,
            button_add_callback=self.queue_inference_add,
            listbox_inferences_queue_callback=self.queue_inference_selected,
            button_infer_callback=self.infer,
            button_delete_callback=self.queue_inference_delete)
        self.gui_inference = GUIInference(
            root,
            button_delete_callback=self.ask_for_ok_delete_inferences,
            button_refresh_callback=self.fetch_inferences,
            listbox_inferences_select_callback=self.inference_selected,
            listbox_inferences_double_click_callback=self.on_inference_double_click,
            listbox_data_select_callback=self.data_selected,
            listbox_data_double_click_callback=self.on_data_double_click,
            listbox_data_drag_callback=self.on_drag,
            listbox_data_drop_callback=self.on_drop)
        self.dataset_manager = dataset_manager
        self.status_manager = status_manager
        self.mmpose_model_manager = mmpose_model_manager
        self.mmdetection_model_manager = mmdetection_model_manager
        self.data_manager = data_manager
        self.metric_manager = metric_manager
        self.plot_manager = plot_manager
        self.feature_manager = feature_manager
        self.active_processes = []
        self.pending_processes = []
        self.parallel_processes = 1
        self.inferences = []
        self.selected_inferences = []
        self.queue_inferences = []
        self.queue_selected_inferences = []
        self.selected_run = None
        self.dataset_type = self.dataset_manager.datasets[Datasets.COCO.value]
        self.multiprocessing_manager = torch.multiprocessing.Manager()
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
        self.inferences.sort(key=lambda x: x.name)

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

    def _gui_enable_button_delete(self):
        self.gui_inference.button_delete['state'] = 'normal'

    def _gui_disable_button_delete(self):
        self.gui_inference.button_delete['state'] = 'disabled'

    def _gui_update_inference_progress(self, inference_id, inference_progress):
        for i, inference in enumerate(self.queue_inferences):
            if inference.id == inference_id:
                self.gui_inference_queue.listbox_inferences.delete(i)
                progress_str = str(inference)
                if inference_progress:
                    progress_str += ' - ' + inference_progress
                self.gui_inference_queue.listbox_inferences.insert(i, progress_str)

    def _gui_set_details(self):
        selected_inference = self.selected_inferences[0]
        self.gui_inference.details_id_var.set(selected_inference.id)
        self.gui_inference.details_name_var.set(selected_inference.name)
        self.gui_inference.details_date_var.set(
            f'{selected_inference.start_datetime} - {selected_inference.end_datetime}')
        self.gui_inference.details_model_mmpose_var.set(selected_inference.mmpose_model)
        self.gui_inference.details_model_mmdetection_var.set(selected_inference.mmdetection_model)

        durations = selected_inference.detection_duration
        duration = '{:.4f} min'.format(round(durations[0], 4))
        duration += ' - {:.4f} sec'.format(round(durations[1], 4))
        duration += ' - {:.4f} sec'.format(round(durations[2], 4))
        self.gui_inference.details_duration_bb_detection_var.set(duration)

        durations = selected_inference.pose_estimation_duration
        duration = '{:.4f} min'.format(round(durations[0], 4))
        duration += ' - {:.4f} sec'.format(round(durations[1], 4))
        duration += ' - {:.4f} sec'.format(round(durations[2], 4))
        self.gui_inference.details_duration_pose_estimation_var.set(duration)

        self.gui_inference.details_description_var.set(selected_inference.description)

        score = '{:.4f} (Detection)'.format(round(selected_inference.score_detection, 4))
        score += ' - {:.4f} (Pose Estimation)'.format(round(selected_inference.score_pose_estimation, 4))
        self.gui_inference.details_score_var.set(score)

        self.gui_inference.details_listbox_data.delete(0, tk.END)
        for data in selected_inference.data:
            self.gui_inference.details_listbox_data.insert(tk.END, self.data_manager.get_data(data))

    def _gui_clear_details(self):
        self.gui_inference.details_id_var.set('')
        self.gui_inference.details_name_var.set('')
        self.gui_inference.details_date_var.set('')
        self.gui_inference.details_model_mmpose_var.set('')
        self.gui_inference.details_model_mmdetection_var.set('')
        self.gui_inference.details_duration_bb_detection_var.set('')
        self.gui_inference.details_duration_pose_estimation_var.set('')
        self.gui_inference.details_description_var.set('')
        self.gui_inference.details_score_var.set('')

        self.gui_inference.details_listbox_data.delete(0, tk.END)

    def queue_inference_add(self):
        inference_mmpose_model = self.mmpose_model_manager.selected_model
        inference_mmdetection_model = self.mmdetection_model_manager.selected_model
        inference_data = self.data_manager.selected_data.copy()

        if inference_mmpose_model is None or not inference_data:
            messagebox.showerror(title='', message='MMPose model or data selection missing.')
            return

        if inference_mmdetection_model is None:
            inference_mmdetection_model = self.mmdetection_model_manager.default_model

        inference_id = id_generator()
        while (self.inference_id_taken(inference_id)):
            inference_id = id_generator()

        inference_name = self._gui_get_name_text()
        if not inference_name:
            inference_name = inference_id
        else:
            for inference in self.inferences:
                if inference.name == inference_name:
                    messagebox.showerror(title='', message='Inference name already used.')
                    return
            for inference in self.queue_inferences:
                if inference.name == inference_name:
                    messagebox.showerror(title='', message='Inference name already used.')
                    return

        inference_description = self._gui_get_description_text()
        if not inference_description:
            inference_description = 'No description provided.'

        inference = Inference(metadata={
            'id': inference_id,
            'name': inference_name,
            'mmpose_model': inference_mmpose_model,
            'mmdetection_model': inference_mmdetection_model,
            'data': inference_data,
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

            existing_dataset = self.data_manager.get_existing_dataset()

            for inference in self.queue_inferences:
                inference_progress = self.multiprocessing_manager.Value(ctypes.c_wchar_p, '')
                inference_process = torch.multiprocessing.Process(
                    target=inference.infer, args=(inference_progress, existing_dataset, self.dataset_type))
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
        self.selected_inferences.clear()
        current_selection = self.gui_inference.listbox_inferences.curselection()

        for selection in current_selection:
            selection_str = self.gui_inference.listbox_inferences.get(selection)
            selection_str_split = selection_str.split(' | ')
            inference_id = selection_str_split[-1]
            self.selected_inferences.append(
                next(inference for inference in self.inferences if inference.id == inference_id))

        if len(self.selected_inferences) == 1:
            self._gui_set_details()
            self.metric_manager.set_inference(self.selected_inferences[0])
        else:
            self._gui_clear_details()

        self.feature_manager.clear()
        self.selected_run = None

        self._gui_enable_button_delete()

    def data_selected(self, event=None):
        if not self.selected_inferences or len(self.selected_inferences) > 1:
            return

        selected_inference = self.selected_inferences[0]
        selection = self.gui_inference.details_listbox_data.curselection()
        if not selection:
            return
        selection_str = self.gui_inference.details_listbox_data.get(selection)
        data_id = int(selection_str[0:3])
        run_id = data_id

        run = next(run for run in selected_inference.runs if run.id == run_id)
        self.selected_run = run
        self.feature_manager.set_data(self.selected_run, selected_inference.name)
        self.metric_manager.set_data(self.selected_run)

    def ask_for_ok_delete_inferences(self):
        if len(self.selected_inferences) == 1:
            title = 'Delete inference'
            message = 'Are you sure to delete the selected inference?'
        else:
            title = 'Delete inferences'
            message = 'Are you sure to delete the selected inferences?'

        answer = messagebox.askyesno(title, message)
        if answer:
            self.delete_inferences()

    def delete_inferences(self):
        for inference in self.selected_inferences:
            shutil.rmtree(os.path.join(INFERENCES_DIR, inference.id))

            dataset_dir = MMPOSE_DATASET_DIR + f'_{inference.id}'
            if os.path.exists(dataset_dir):
                shutil.rmtree(dataset_dir)

        self.fetch_inferences()
        self._gui_clear_details()
        self._gui_disable_button_delete()
        self.plot_manager.clear_images()
        self.plot_manager.clear_plots()
        self.feature_manager.clear()
        self.metric_manager.clear_compared_inferences()
        self.metric_manager.clear_all_metrics()
        self.selected_run = None

    def on_drag(self, event=None):
        if self.selected_run is None:
            return

        self.gui_inference.root.configure(cursor='plus')

    def on_drop(self, event=None):
        self.gui_inference.root.configure(cursor='')

        if not self.selected_inferences:
            return

        selected_inference = self.selected_inferences[0]
        selected_run = self.selected_run
        if selected_run is None:
            return

        title = self.selected_inferences[0].name + ' - Run ' + str(selected_run.id).zfill(2)
        x, y = event.widget.winfo_pointerxy()
        self.plot_manager.plot_image(x, y, selected_inference, selected_run, title, self.dataset_type)

    def on_inference_double_click(self, event=None):
        if not self.selected_inferences:
            return

        path = self.selected_inferences[0].path.replace('/', '\\')
        subprocess.run([
            'explorer.exe',
            f'\\{WSL_PREFIX}{path}'
        ], check=False)

    def on_data_double_click(self, event=None):
        if self.selected_run is None:
            return

        path = self.selected_run.path.replace('/', '\\')
        subprocess.run([
            'explorer.exe',
            '/select,'
            f'\\{WSL_PREFIX}{path}'
        ], check=False)
