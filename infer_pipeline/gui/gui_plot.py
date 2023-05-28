import numpy as np
from tkinter import ttk

from data_types.image_plot import ImagePlot


class GUIPlot():
    def __init__(
        self,
        root,
        notebook_tab_change_callback,
        notebook_tab_double_click_callback,
        notebook_tab_middle_click_callback,
        image_plot_middle_click_callback
    ):
        self.root = root

        self.frame = ttk.Frame(self.root, padding=(10, 10))
        self.frame.place(x=960, y=0, width=2480, height=1152)

        self.title = ttk.Label(self.frame, text='Plots', font=self.root.font_title)
        self.title.place(x=0, y=0)

        self.notebook = ttk.Notebook(self.frame)
        self.notebook.bind('<<NotebookTabChanged>>', notebook_tab_change_callback)
        self.notebook.bind('<Double-Button-1>', notebook_tab_double_click_callback)
        self.notebook.bind('<Button-2>', notebook_tab_middle_click_callback)

        w = 960
        h = 1080
        ratio = w/h

        self.default_image = np.full((100, int(ratio * 100), 3), (255, 255, 255))
        self.frame_image_plots = ttk.Frame(self.notebook, width=2445, height=1080, padding=(0, 0))
        self.notebook.add(self.frame_image_plots, text='Image Plot')

        self.frame_image_plot_1 = ttk.Frame(self.frame_image_plots, width=815, height=1080, padding=(0, 0))
        self.frame_image_plot_1.place(x=0, y=0)

        self.frame_image_plot_2 = ttk.Frame(self.frame_image_plots, width=815, height=1080, padding=(0, 0))
        self.frame_image_plot_2.place(x=815, y=0)

        self.frame_image_plot_3 = ttk.Frame(self.frame_image_plots, width=815, height=1080, padding=(0, 0))
        self.frame_image_plot_3.place(x=1630, y=0)

        self.image_plot_1 = ImagePlot(
            self.frame_image_plot_1,
            ratio,
            self.default_image,
            image_plot_middle_click_callback
        )

        self.image_plot_2 = ImagePlot(
            self.frame_image_plot_2,
            ratio,
            self.default_image,
            image_plot_middle_click_callback
        )

        self.image_plot_3 = ImagePlot(
            self.frame_image_plot_3,
            ratio,
            self.default_image,
            image_plot_middle_click_callback
        )

        self.frame_add_plot = ttk.Frame(self.notebook, width=2445, height=1080, padding=(0, 0))
        self.frame_add_plot.place(x=0, y=0)
        self.notebook.add(self.frame_add_plot, text='+')

        self.notebook.place(x=0, y=30)
