from tkinter import ttk


class GUIPlot():
    def __init__(self, root):
        self.root = root

        self.frame = ttk.Frame(self.root, padding=(10, 10))
        self.frame.place(x=960, y=0, width=2480, height=1152)

        self.title = ttk.Label(self.frame, text='Plots', font=self.root.font_title)
        self.title.place(x=0, y=0)

        self.notebook = ttk.Notebook(self.frame)

        self.frame_image_plot = ttk.Frame(self.notebook, width=2445, height=1080, padding=(0, 0))
        self.frame_plot_1 = ttk.Frame(self.notebook, width=2445, height=1080, padding=(0, 0))

        self.frame_image_plot.place(x=0, y=0)
        self.frame_plot_1.place(x=0, y=0)

        self.notebook.add(self.frame_image_plot, text="Image Plot")
        self.notebook.add(self.frame_plot_1, text="Plot 1")

        self.notebook.place(x=0, y=30)
