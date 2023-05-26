import subprocess

from tkinter import ttk


class ExplorerLabel():
    def __init__(self, frame, textvariable, wraplength=None):
        self.frame = frame
        self.textvariable = textvariable
        self.label = ttk.Label(self.frame, textvariable=self.textvariable,
                               wraplength=wraplength or 10000, foreground='#058AFF')
        self.label.bind('<Button-1>', self.on_click)
        self.label.bind("<Enter>", self.on_enter)
        self.label.bind("<Leave>", self.on_leave)

    def place(self, x, y):
        self.label.place(x=x, y=y)

    def on_click(self, event=None):
        prefix = '\\wsl.localhost\\Ubuntu-22.04'
        path = self.textvariable.get().replace('/', '\\')
        subprocess.run([
            'explorer.exe',
            '/select,',
            f'\\{prefix}{path}'
        ], check=False)

    def on_enter(self, event=None):
        self.frame.configure(cursor='hand2')

    def on_leave(self, event=None):
        self.frame.configure(cursor='')
