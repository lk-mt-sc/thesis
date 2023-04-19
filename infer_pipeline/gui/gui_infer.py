import tkinter as tk
from tkinter import ttk


class GUIInfer():
    def __init__(self, root, button_infer_callback):
        self.root = root

        self.frame = ttk.Frame(self.root, padding=(10, 10))
        self.frame.place(x=0, y=1152, width=480, height=144)

        self.title = ttk.Label(self.frame, text='Start Inference', font=self.root.font_title)
        self.title.place(x=0, y=0)

        ttk.Label(self.frame, text='Name \n(max. 50)').place(x=0, y=30)
        self.text_infer_name = tk.Text(self.frame, height=1, font=self.root.font_small)
        self.text_infer_name.insert('1.0', 'New Inference')
        self.text_infer_name.place(x=80, y=30, width=250)

        ttk.Label(self.frame, text='Description \n(max. 320)').place(x=0, y=60)
        self.text_infer_description = tk.Text(self.frame, height=3, font=self.root.font_small)
        self.text_infer_description.place(x=80, y=60, width=250)

        self.button_infer = ttk.Button(self.frame, text='Infer', command=button_infer_callback)
        self.button_infer.place(x=350, y=30, height=70)
