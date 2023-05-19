import matplotlib
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk


class GUIPlot():
    def __init__(self, root, slider_callback, button_clear_callback):
        self.root = root

        self.background_color_hex = '#DCDAD5'
        self.background_color_rgb = matplotlib.colors.to_rgb(self.background_color_hex)

        self.frame = ttk.Frame(self.root, padding=(10, 10))
        self.frame.place(x=960, y=0, width=2480, height=1152)

        self.title = ttk.Label(self.frame, text='Plots', font=self.root.font_title)
        self.title.place(x=0, y=0)

        self.notebook = ttk.Notebook(self.frame)

        self.frame_image_plot = ttk.Frame(self.notebook, width=2445, height=1080, padding=(0, 0))

        # temporary:
        w = 960
        h = 1080
        ratio = w/h
        self.default_image = np.full((100, int(ratio * 100), 3), self.background_color_rgb)
        self.image = self.default_image
        self.image_figure = Figure(figsize=(ratio * 10, 10), dpi=96, facecolor=self.background_color_hex)
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
        self.image_toolbar.config(background=self.background_color_hex)
        for button in self.image_toolbar.winfo_children():
            button.config(background=self.background_color_hex, highlightbackground=self.background_color_hex)
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

        self.frame_plot_1 = ttk.Frame(self.notebook, width=2445, height=1080, padding=(0, 0))
        self.feature_figure = Figure(figsize=(23, 10), dpi=96, facecolor=self.background_color_hex)
        self.feature_plot = self.feature_figure.add_subplot(111)
        self.feature_plot.grid()
        self.feature_canvas = FigureCanvasTkAgg(self.feature_figure, master=self.frame_plot_1)
        self.feature_canvas.draw()

        self.feature_toolbar = NavigationToolbar2Tk(self.feature_canvas,
                                                    self.frame_plot_1,
                                                    pack_toolbar=False)
        self.feature_toolbar.config(background=self.background_color_hex)
        for button in self.feature_toolbar.winfo_children():
            button.config(background=self.background_color_hex, highlightbackground=self.background_color_hex)
        self.feature_toolbar.update()
        self.feature_toolbar.place(x=0, y=1040)
        self.feature_canvas.get_tk_widget().place(x=0, y=0, width=2445, height=1080)

        self.feature_button_clear = ttk.Button(
            self.frame_plot_1,
            text='Clear Plot',
            style='Button.TButton',
            width=8,
            command=button_clear_callback)
        self.feature_button_clear.place(x=5, y=1010, height=30)

        self.frame_image_plot.place(x=0, y=0)
        self.frame_plot_1.place(x=0, y=0)

        self.notebook.add(self.frame_image_plot, text="Image Plot")
        self.notebook.add(self.frame_plot_1, text="Plot 1")

        self.notebook.place(x=0, y=30)
