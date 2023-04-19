import tkinter as tk
from tkinter import ttk


class GUIStatus():
    def __init__(self, root):
        self.root = root

        self.frame = ttk.Frame(self.root, padding=(10, 10), style='Status.TFrame')
        self.frame.place(x=0, y=1296, width=480, height=144)

        self.title = ttk.Label(self.frame, text='Status', font=self.root.font_title)
        self.title.place(x=0, y=0)

        self.status_var = tk.StringVar()
        ttk.Label(self.frame, textvariable=self.status_var, font=self.root.font_status).place(x=110, y=28)

        self.progressbar = ttk.Progressbar(
            self.frame,
            orient='horizontal',
            mode='determinate',
            length=100
        )
        self.progressbar.place(x=0, y=30)
