import matplotlib
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from common import BACKGROUND_COLOR_HEX, BACKGROUND_COLOR_RGB


class GUIPlot():
    def __init__(
        self,
        root,
        button_add_plot_callback,
        button_delete_plot_callback,
        notebook_tab_changed_callback,
        slider_callback
    ):
        self.root = root

        self.frame = ttk.Frame(self.root, padding=(10, 10))
        self.frame.place(x=960, y=0, width=2480, height=1152)

        self.title = ttk.Label(self.frame, text='Plots', font=self.root.font_title)
        self.title.place(x=0, y=0)

        self.combobox_plot_type = ttk.Combobox(self.frame, width=12, font=self.root.font_small)
        self.combobox_plot_type['values'] = ['']
        self.combobox_plot_type['state'] = 'readonly'
        self.combobox_plot_type.bind('<<ComboboxSelected>>', self.combobox_plot_type_selection_clear)
        self.combobox_plot_type.place(x=2282, y=6, width=40)

        self.button_add_plot = ttk.Button(
            self.frame,
            text='Add Plot',
            style='Button.TButton',
            width=7,
            command=button_add_plot_callback)
        self.button_add_plot.place(x=2330, y=5, height=20)

        self.button_delete_plot = ttk.Button(
            self.frame,
            text='Delete Plot',
            style='Button.TButton',
            width=9,
            command=button_delete_plot_callback)
        self.button_delete_plot.place(x=2387, y=5, height=20)

        self.notebook = ttk.Notebook(self.frame)
        self.notebook.bind('<<NotebookTabChanged>>', notebook_tab_changed_callback)

        self.frame_image_plot = ttk.Frame(self.notebook, width=2445, height=1080, padding=(0, 0))

        # temporary:
        w = 960
        h = 1080
        ratio = w/h
        self.default_image = np.full((100, int(ratio * 100), 3), BACKGROUND_COLOR_RGB)
        self.image = self.default_image
        self.image_figure = Figure(figsize=(ratio * 10, 10), dpi=96, facecolor=BACKGROUND_COLOR_HEX)
        self.image_plot = self.image_figure.add_subplot(111)
        self.image_plot.axis('off')
        self.image_plot.format_coord = lambda x, y: ''
        self.image_imshow = self.image_plot.imshow(self.image)
        self.image_imshow.format_cursor_data = lambda e: ''
        self.image_canvas = FigureCanvasTkAgg(self.image_figure, master=self.frame_image_plot)
        self.image_canvas.draw()

        self.image_toolbar = NavigationToolbar2Tk(self.image_canvas,
                                                  self.frame_image_plot,
                                                  pack_toolbar=False)
        self.image_toolbar.update()
        self.image_toolbar.place(x=0, y=1040)
        self.image_canvas.get_tk_widget().place(x=0, y=0, width=2445, height=1080)

        self.image_counter_var = tk.StringVar()
        self.image_counter_var.set('000/000')
        label_image_counter = ttk.Label(self.frame_image_plot, textvariable=self.image_counter_var)
        label_image_counter.place(x=5, y=1025)

        self.image_slider = ttk.Scale(
            self.frame_image_plot,
            from_=0,
            to=99,
            orient='horizontal',
            command=slider_callback
        )
        self.image_slider.place(x=60, y=1025, width=165)
        self.image_slider.configure(state='disabled')

        self.frame_image_plot.place(x=0, y=0)

        self.notebook.add(self.frame_image_plot, text="Image Plot")
        self.notebook.place(x=0, y=30)

    def combobox_plot_type_selection_clear(self, event=None):
        self.combobox_plot_type.selection_clear()
