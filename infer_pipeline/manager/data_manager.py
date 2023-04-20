import os
import glob
import tkinter as tk
from threading import Thread

from common import DATA_DIR
from gui.gui_data import GUIData
from manager.status_manager import Status


class Data:
    def __init__(self, path):
        data = path.split('/')[-1]
        data = data.split('_')
        self.path = path
        self.id = int(data[1])
        self.fps = int(data[2])
        self.n_images = int(data[3])
        if len(data) > 4:
            self.keys = data[4:]
        else:
            self.keys = None

    def __str__(self):
        data_str = f'{str(self.id).zfill(3)} | {str(self.fps).zfill(2)} FPS | {str(self.n_images).zfill(3)} IMG'
        if self.keys is not None:
            for key in self.keys:
                data_str += ' | ' + key
        return data_str

    def get_images(self):
        return sorted(glob.glob(os.path.join(self.path, '*')))


class DataManager():
    def __init__(self, root, status_manager):
        self.gui_data = GUIData(
            root,
            button_refresh_callback=self.fetch_data,
            listbox_data_callback=self.data_selected)
        self.status_manager = status_manager
        self.data = []
        self.fetch_data()

    def fetch_data(self):
        if not self.status_manager.has_status(Status.FETCHING_DATA):
            self.status_manager.add_status(Status.FETCHING_DATA)
            fetch_thread = Thread(target=self._fetch_data)
            fetch_thread.start()
            self.monitor_fetch_thread(fetch_thread)

    def _fetch_data(self):
        data = sorted(glob.glob(os.path.join(DATA_DIR, '*')))
        for d in data:
            self.data.append(Data(d))

    def monitor_fetch_thread(self, fetch_thread):
        if fetch_thread.is_alive():
            self.gui_data.root.after(50, lambda: self.monitor_fetch_thread(fetch_thread))
        else:
            self._gui_set_data()
            self.status_manager.remove_status(Status.FETCHING_DATA)

    def _gui_set_data(self):
        self.gui_data.listbox_data.delete(0, tk.END)
        for data in self.data:
            self.gui_data.listbox_data.insert(tk.END, data)

    def data_selected(self):
        pass
