import tkinter as tk
from tkinter import ttk


class GUIInferenceQueue():
    def __init__(
        self,
        root,
        button_add_callback,
        listbox_inferences_queue_callback,
        button_infer_callback,
        button_delete_callback,
    ):
        self.root = root

        self.frame = ttk.Frame(self.root, padding=(10, 10))
        self.frame.place(x=0, y=1008, width=480, height=288)

        self.title_new_inference = ttk.Label(self.frame, text='New Inference', font=self.root.font_title)
        self.title_new_inference.place(x=0, y=0)

        ttk.Label(self.frame, text='Name \n(max. 50)').place(x=0, y=30)
        self.text_infer_name = tk.Text(self.frame, height=1, font=self.root.font_small)
        self.text_infer_name.insert('1.0', 'New Inference')
        self.text_infer_name.place(x=80, y=30, width=250)

        ttk.Label(self.frame, text='Description \n(max. 300)').place(x=0, y=60)
        self.text_infer_description = tk.Text(self.frame, height=3, font=self.root.font_small)
        self.text_infer_description.place(x=80, y=60, width=250)

        self.button_add = ttk.Button(
            self.frame,
            text='Add to queue',
            style='Button.TButton',
            command=button_add_callback)
        self.button_add.place(x=350, y=30, width=100, height=70)

        self.title_inference_queue = ttk.Label(self.frame, text='Inference Queue', font=self.root.font_title)
        self.title_inference_queue.place(x=0, y=110)

        self.listbox_inferences_var = tk.Variable()
        self.listbox_inferences = tk.Listbox(
            self.frame,
            height=7,
            listvariable=self.listbox_inferences_var,
            selectmode=tk.EXTENDED,
            font=self.root.font_small
        )
        self.listbox_inferences.configure(exportselection=False)
        self.listbox_inferences.bind('<<ListboxSelect>>', listbox_inferences_queue_callback)
        self.listbox_inferences.place(x=0, y=140, width=440)
        self.listbox_inferences_scrollbar = ttk.Scrollbar(self.frame)
        self.listbox_inferences_scrollbar.place(x=440, y=140, width=20, height=90)
        self.listbox_inferences.config(yscrollcommand=self.listbox_inferences_scrollbar.set)
        self.listbox_inferences_scrollbar.config(command=self.listbox_inferences.yview)

        self.button_infer = ttk.Button(
            self.frame,
            text='Start Inferences',
            style='Button.TButton',
            command=button_infer_callback)
        self.button_infer.place(x=0, y=250, width=100, height=25)

        self.button_delete = ttk.Button(
            self.frame,
            text='Delete',
            style='Button.TButton',
            width=6,
            command=button_delete_callback)
        self.button_delete.place(x=110, y=250, width=50, height=25)
        self.button_delete['state'] = 'disabled'
