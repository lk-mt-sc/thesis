import os
import glob
import json
import subprocess
import tkinter as tk
from threading import Thread

from common import MMPOSE_RUNS_DIR
from gui.gui_data import GUIData
from manager.status_manager import Status
from data_types.data import Data

from common import MMPOSE_DATA_DIR, WSL_PREFIX


class DataManager():
    def __init__(self, root, status_manager):
        self.gui_data = GUIData(
            root,
            combobox_data_callback=self.filter,
            button_select_all_callback=self.select_all_data,
            button_refresh_callback=self.fetch_data,
            listbox_data_select_callback=self.data_selected,
            listbox_data_double_click_callback=self.on_double_click)
        self.status_manager = status_manager
        self.data_all = []
        self.data_show = []
        self.selected_data = []
        self.filter_data_base = None
        self.filter_data_spotlight = None
        self.filter_data_starts = None
        self.fetch_data()
        self._gui_set_presets()

    def fetch_data(self):
        if not self.status_manager.has_status(Status.FETCHING_DATA):
            self.status_manager.add_status(Status.FETCHING_DATA)
            fetch_thread = Thread(target=self._fetch_data)
            fetch_thread.start()
            self.monitor_fetch_thread(fetch_thread)

    def _fetch_data(self):
        data = sorted(glob.glob(os.path.join(MMPOSE_RUNS_DIR, '*')))
        self.data_all.clear()
        for d in data:
            self.data_all.append(Data(d))

    def monitor_fetch_thread(self, fetch_thread):
        if fetch_thread.is_alive():
            self.gui_data.root.after(50, lambda: self.monitor_fetch_thread(fetch_thread))
        else:
            self.data_show = self.data_all.copy()
            self.selected_data.clear()
            self.filter()
            self._gui_set_data()
            self.status_manager.remove_status(Status.FETCHING_DATA)

    def _gui_set_presets(self):
        self.gui_data.combobox_data_base['values'] = [
            'No Preset',
            'Standard',
            'Deblurred',
            'Interpolated',
            'Debl.-Interp.',
            'Interp.-Debl.'
        ]

        self.gui_data.combobox_data_spotlight['values'] = [
            'All Spotlight',
            'With Spotlight',
            'Without Spotlight'
        ]

        self.gui_data.combobox_data_start['values'] = [
            'All Starts',
            'Starts at Rest',
            'Starts in Motion'
        ]

        self.gui_data.combobox_data_base.current(0)
        self.gui_data.combobox_data_spotlight.current(0)
        self.gui_data.combobox_data_start.current(0)

    def _gui_set_data(self):
        self.gui_data.listbox_data.delete(0, tk.END)
        for data in self.data_show:
            self.gui_data.listbox_data.insert(tk.END, data)

    def _gui_select_all_data(self):
        self.gui_data.listbox_data.select_set(0, tk.END)
        self.gui_data.listbox_data.event_generate("<<ListboxSelect>>")

    def _gui_get_filter(self):
        self.filter_data_base = self.gui_data.combobox_data_base.get()
        self.filter_data_spotlight = self.gui_data.combobox_data_spotlight.get()
        self.filter_data_starts = self.gui_data.combobox_data_start.get()

    def filter(self, event=None):
        self.selected_data.clear()
        self.data_show.clear()
        self._gui_get_filter()

        show_standard = self.filter_data_base in ('Standard', 'No Preset')
        show_deblurred = self.filter_data_base in ('Deblurred', 'No Preset')
        show_interpolated = self.filter_data_base in ('Interpolated', 'No Preset')
        show_deblurred_interpolated = self.filter_data_base in ('Debl.-Interp.', 'No Preset')
        show_interpolated_deblurred = self.filter_data_base in ('Interp.-Debl.', 'No Preset')
        hide_spotlight = self.filter_data_spotlight == 'Without Spotlight'
        hide_no_spotlight = self.filter_data_spotlight == 'With Spotlight'
        hide_starts_at_rest = self.filter_data_starts == 'Starts in Motion'
        hide_starts_in_motion = self.filter_data_starts == 'Starts at Rest'

        for data in self.data_all:
            if show_standard and not (data.deblurred or data.interpolated or data.deblurred_interpolated or data.interpolated_deblurred):
                self.data_show.append(data)
                continue

            if show_deblurred and data.deblurred:
                self.data_show.append(data)
                continue

            if show_interpolated and data.interpolated:
                self.data_show.append(data)
                continue

            if show_deblurred_interpolated and data.deblurred_interpolated:
                self.data_show.append(data)
                continue

            if show_interpolated_deblurred and data.interpolated_deblurred:
                self.data_show.append(data)
                continue

        data_to_remove = []
        for data in self.data_show:
            if hide_spotlight and data.spotlight:
                data_to_remove.append(data)
                continue

            if hide_no_spotlight and not data.spotlight:
                data_to_remove.append(data)
                continue

            if hide_starts_at_rest and data.start_at_rest:
                data_to_remove.append(data)
                continue

            if hide_starts_in_motion and not data.start_at_rest:
                data_to_remove.append(data)
                continue

        for data in data_to_remove:
            self.data_show.remove(data)

        self._gui_set_data()

    def select_all_data(self, event=None):
        self._gui_select_all_data()

    def get_existing_dataset(self):
        persistent_datasets_file_path = os.path.join(MMPOSE_DATA_DIR, 'persistent_datasets.json')
        with open(persistent_datasets_file_path, 'r', encoding='utf8') as persistent_datasets_file:
            persistent_datasets = json.load(persistent_datasets_file)
            for persistent_dataset in persistent_datasets:
                runs = persistent_dataset['runs']
                if len(runs) != len(self.selected_data):
                    continue
                for run in runs:
                    if run not in self.selected_data:
                        break
                return persistent_dataset['path']
        return None

    def data_selected(self, event=None):
        self.selected_data.clear()
        current_selection = self.gui_data.listbox_data.curselection()
        for selection in current_selection:
            selection_str = self.gui_data.listbox_data.get(selection)
            data_id = int(selection_str[:3])
            self.selected_data.append(next(data for data in self.data_show if data.id == data_id))

    def get_data(self, id_):
        return next((data for data in self.data_all if data.id == id_), None)

    def on_double_click(self, event=None):
        if not self.selected_data:
            return

        selected_data = self.selected_data[0]
        path = selected_data.path.replace('/', '\\')
        subprocess.run([
            'explorer.exe',
            f'\\{WSL_PREFIX}{path}'
        ], check=False)
