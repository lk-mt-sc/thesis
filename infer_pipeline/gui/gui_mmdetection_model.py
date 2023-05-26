import tkinter as tk
from tkinter import ttk

from data_types.custom_labels import ExplorerLabel


class GUIMMDetectionModel():
    def __init__(self, root, listbox_models_callback):
        self.root = root

        self.frame = ttk.Frame(self.root, padding=(10, 10))
        self.frame.place(x=0, y=576, width=480, height=288)

        self.title = ttk.Label(self.frame, text='MMDetection Models', font=self.root.font_title)
        self.title.place(x=0, y=0)

        self.listbox_models_var = tk.StringVar()
        self.listbox_models = tk.Listbox(
            self.frame,
            height=5,
            listvariable=self.listbox_models_var,
            selectmode=tk.SINGLE,
            font=self.root.font_small
        )
        self.listbox_models.configure(exportselection=False)
        self.listbox_models.bind('<<ListboxSelect>>', listbox_models_callback)
        self.listbox_models.place(x=0, y=30, width=440)
        self.listbox_models_scrollbar = ttk.Scrollbar(self.frame)
        self.listbox_models_scrollbar.place(x=440, y=30, width=20, height=68)
        self.listbox_models.config(yscrollcommand=self.listbox_models_scrollbar.set)
        self.listbox_models_scrollbar.config(command=self.listbox_models.yview)

        ttk.Label(self.frame, text='Model Details', font=self.root.font_title).place(x=0, y=110)
        ttk.Label(self.frame, text='Name', font=self.root.font_bold).place(x=0, y=140)
        ttk.Label(self.frame, text='Key Metric', font=self.root.font_bold).place(x=0, y=170)
        ttk.Label(self.frame, text='Checkpoint', font=self.root.font_bold).place(x=0, y=200)
        ttk.Label(self.frame, text='Configuration', font=self.root.font_bold).place(x=0, y=230)

        self.details_name_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.details_name_var).place(x=120, y=140)

        self.details_key_metric_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.details_key_metric_var).place(x=120, y=170)

        self.details_checkpoint_var = tk.StringVar()
        checkpoint_label = ExplorerLabel(self.frame, textvariable=self.details_checkpoint_var, wraplength=320)
        checkpoint_label.place(x=120, y=200)

        self.details_config_var = tk.StringVar()
        config_label = ExplorerLabel(self.frame, textvariable=self.details_config_var, wraplength=320)
        config_label.place(x=120, y=230)
