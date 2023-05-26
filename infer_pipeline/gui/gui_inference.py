import tkinter as tk
from tkinter import ttk


class GUIInference():
    def __init__(
            self,
            root,
            button_delete_callback,
            button_refresh_callback,
            listbox_inferences_callback,
            listbox_data_select_callback,
            listbox_data_drag_callback,
            listbox_data_drop_callback):
        self.root = root

        self.frame = ttk.Frame(self.root, padding=(10, 10))
        self.frame.place(x=480, y=0, width=480, height=864)

        self.title = ttk.Label(self.frame, text='Inferences', font=self.root.font_title)
        self.title.place(x=0, y=0)

        self.button_delete = ttk.Button(
            self.frame,
            text='Delete',
            style='Button.TButton',
            width=6,
            command=button_delete_callback)
        self.button_delete.place(x=360, y=0, height=20)
        self.button_delete['state'] = 'disabled'

        self.button_refresh = ttk.Button(
            self.frame,
            text='Refresh',
            style='Button.TButton',
            width=7,
            command=button_refresh_callback)
        self.button_refresh.place(x=410, y=0, height=20)

        self.listbox_inferences_var = tk.Variable()
        self.listbox_inferences = tk.Listbox(
            self.frame,
            height=20,
            listvariable=self.listbox_inferences_var,
            selectmode=tk.EXTENDED,
            font=self.root.font_small
        )
        self.listbox_inferences.configure(exportselection=False)
        self.listbox_inferences.bind('<<ListboxSelect>>', listbox_inferences_callback)
        self.listbox_inferences.place(x=0, y=30, width=440)
        self.listbox_inferences_scrollbar = ttk.Scrollbar(self.frame)
        self.listbox_inferences_scrollbar.place(x=440, y=30, width=20, height=263)
        self.listbox_inferences.config(yscrollcommand=self.listbox_inferences_scrollbar.set)
        self.listbox_inferences_scrollbar.config(command=self.listbox_inferences.yview)

        ttk.Label(self.frame, text='Inference Details', font=self.root.font_title).place(x=0, y=305)
        ttk.Label(self.frame, text='ID').place(x=0, y=335)
        ttk.Label(self.frame, text='Name').place(x=0, y=365)
        ttk.Label(self.frame, text='Date').place(x=0, y=395)
        ttk.Label(self.frame, text='Model MMPose').place(x=0, y=425)
        ttk.Label(self.frame, text='Model MMDetection').place(x=0, y=455)
        ttk.Label(self.frame, text='Data').place(x=0, y=485)
        ttk.Label(self.frame, text='Duration (total/avg. per data/avg. per image)').place(x=0, y=655)
        ttk.Label(self.frame, text='Bounding Box Detection').place(x=20, y=675)
        ttk.Label(self.frame, text='Pose Estimation').place(x=20, y=695)
        ttk.Label(self.frame, text='Description').place(x=0, y=725)

        self.details_listbox_data_var = tk.Variable()
        self.details_listbox_data = tk.Listbox(
            self.frame,
            height=10,
            listvariable=self.details_listbox_data_var,
            selectmode=tk.SINGLE,
            font=self.root.font_small
        )
        self.details_listbox_data.configure(exportselection=False)
        self.details_listbox_data.bind('<<ListboxSelect>>', listbox_data_select_callback)
        self.details_listbox_data.bind('<B1-Motion>', listbox_data_drag_callback)
        self.details_listbox_data.bind('<ButtonRelease-1>', listbox_data_drop_callback)
        self.details_listbox_data.bind("<B1-Leave>", lambda event: "break")
        self.details_listbox_data.place(x=0, y=510, width=440)
        self.details_listbox_data_scrollbar = ttk.Scrollbar(self.frame)
        self.details_listbox_data_scrollbar.place(x=440, y=510, width=20, height=132)
        self.details_listbox_data.config(yscrollcommand=self.details_listbox_data_scrollbar.set)
        self.details_listbox_data_scrollbar.config(command=self.details_listbox_data.yview)

        self.details_id_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.details_id_var).place(x=120, y=335)

        self.details_name_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.details_name_var).place(x=120, y=365)

        self.details_date_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.details_date_var).place(x=120, y=395)

        self.details_model_mmpose_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.details_model_mmpose_var, wraplength=320).place(x=120, y=425)

        self.details_model_mmdetection_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.details_model_mmdetection_var, wraplength=320).place(x=120, y=455)

        self.details_duration_bb_detection_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.details_duration_bb_detection_var).place(x=180, y=675)

        self.details_duration_pose_estimation_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.details_duration_pose_estimation_var).place(x=180, y=695)

        self.details_description_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.details_description_var, wraplength=320).place(x=120, y=725)
