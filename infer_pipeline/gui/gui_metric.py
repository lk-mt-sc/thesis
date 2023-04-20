from tkinter import ttk


class GUIMetric():
    def __init__(self, root):
        self.root = root

        self.frame = ttk.Frame(self.root, padding=(10, 10))
        self.frame.place(x=960, y=1152, width=2480, height=288)

        self.title = ttk.Label(self.frame, text='Metrics', font=self.root.font_title)
        self.title.place(x=0, y=0)
