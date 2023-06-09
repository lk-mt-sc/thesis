import tkinter as tk
from tkinter import ttk


class GUIFeature():
    def __init__(
            self,
            root,
            listbox_feature_select_callback,
            listbox_feature_drag_callback,
            listbox_feature_drop_callback):
        self.root = root

        self.frame = ttk.Frame(self.root, padding=(10, 10))
        self.frame.place(x=480, y=864, width=480, height=576)

        self.title = ttk.Label(self.frame, text='Features', font=self.root.font_title)
        self.title.place(x=0, y=0)

        self.features_var = tk.Variable()
        self.listbox_features = tk.Listbox(
            self.frame,
            height=32,
            listvariable=self.features_var,
            selectmode=tk.SINGLE,
            font=self.root.font_small
        )
        self.listbox_features.configure(exportselection=False)
        self.listbox_features.bind('<<ListboxSelect>>', listbox_feature_select_callback)
        self.listbox_features.bind('<B1-Motion>', listbox_feature_drag_callback)
        self.listbox_features.bind('<ButtonRelease-1>', listbox_feature_drop_callback)
        self.listbox_features.place(x=0, y=30, width=440)
        self.listbox_features_scrollbar = ttk.Scrollbar(self.frame)
        self.listbox_features_scrollbar.place(x=440, y=30, width=20, height=450)
        self.listbox_features.config(yscrollcommand=self.listbox_features_scrollbar.set)
        self.listbox_features_scrollbar.config(command=self.listbox_features.yview)
