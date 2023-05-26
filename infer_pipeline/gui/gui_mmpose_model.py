import tkinter as tk
from tkinter import ttk

from data_types.custom_labels import ExplorerLabel


class GUIMMPoseModel():
    def __init__(
            self,
            root,
            combobox_model_preset_callback,
            button_refresh_callback,
            listbox_models_callback):
        self.root = root

        self.frame = ttk.Frame(self.root, padding=(10, 10))
        self.frame.place(x=0, y=0, width=480, height=576)

        self.title = ttk.Label(self.frame, text='MMPose Models', font=self.root.font_title)
        self.title.place(x=0, y=0)

        self.combobox_model_preset = ttk.Combobox(self.frame, width=10, font=self.root.font_small)
        self.combobox_model_preset['values'] = ['No preset']
        self.combobox_model_preset['state'] = 'readonly'
        self.combobox_model_preset.current(0)
        self.combobox_model_preset.bind('<<ComboboxSelected>>', combobox_model_preset_callback)
        self.combobox_model_preset.place(x=180, y=1)

        self.button_refresh = ttk.Button(
            self.frame,
            text='Refresh',
            style='Button.TButton',
            width=7,
            command=button_refresh_callback
        )
        self.button_refresh.place(x=270, y=0, height=20)

        self.root.style.configure('RedownloadModelZoo.TCheckbutton', font=self.root.font_small)
        self.checkbutton_redownload_model_zoo = ttk.Checkbutton(
            self.frame,
            text='Re-download model zoo',
            style='RedownloadModelZoo.TCheckbutton'
        )
        self.checkbutton_redownload_model_zoo.place(x=323, y=0)
        self.checkbutton_redownload_model_zoo.state(['!alternate'])
        self.checkbutton_redownload_model_zoo.state(['!selected'])

        self.listbox_models_var = tk.StringVar()
        self.listbox_models = tk.Listbox(
            self.frame,
            height=20,
            listvariable=self.listbox_models_var,
            selectmode=tk.SINGLE,
            font=self.root.font_small
        )
        self.listbox_models.configure(exportselection=False)
        self.listbox_models.bind('<<ListboxSelect>>', listbox_models_callback)
        self.listbox_models.place(x=0, y=30, width=440)
        self.listbox_models_scrollbar = ttk.Scrollbar(self.frame)
        self.listbox_models_scrollbar.place(x=440, y=30, width=20, height=263)
        self.listbox_models.config(yscrollcommand=self.listbox_models_scrollbar.set)
        self.listbox_models_scrollbar.config(command=self.listbox_models.yview)

        ttk.Label(self.frame, text='Model Details', font=self.root.font_title).place(x=0, y=305)
        ttk.Label(self.frame, text='Section').place(x=0, y=335)
        ttk.Label(self.frame, text='Architecture').place(x=0, y=365)
        ttk.Label(self.frame, text='Dataset').place(x=0, y=395)
        ttk.Label(self.frame, text='Input Size').place(x=0, y=425)
        ttk.Label(self.frame, text='Key Metric').place(x=0, y=455)
        ttk.Label(self.frame, text='Checkpoint').place(x=0, y=485)
        ttk.Label(self.frame, text='Configuration').place(x=0, y=515)

        self.details_section_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.details_section_var).place(x=120, y=335)

        self.details_arch_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.details_arch_var).place(x=120, y=365)

        self.details_dataset_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.details_dataset_var).place(x=120, y=395)

        self.details_input_size_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.details_input_size_var).place(x=120, y=425)

        self.details_key_metric_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.details_key_metric_var).place(x=120, y=455)

        self.details_checkpoint_var = tk.StringVar()
        checkpoint_label = ExplorerLabel(self.frame, textvariable=self.details_checkpoint_var, wraplength=320)
        checkpoint_label.place(x=120, y=485)

        self.details_config_var = tk.StringVar()
        config_label = ExplorerLabel(self.frame, textvariable=self.details_config_var, wraplength=320)
        config_label.place(x=120, y=515)
