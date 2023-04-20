import os
import glob
import tkinter as tk
from threading import Thread

from common import MMPOSE_DATA_DIR
from gui.gui_data import GUIData
from manager.status_manager import Status
from data_types.data import Data


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
        data = sorted(glob.glob(os.path.join(MMPOSE_DATA_DIR, '*')))
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
