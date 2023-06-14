import copy
from enum import Enum
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from common import BACKGROUND_COLOR_HEX
from data_types.image_plot import ImagePlot


class PlotLayouts(Enum):
    PLOT1X1 = '1x1'
    PLOT2X1 = '2x1'
    PLOT3X1 = '3x1'
    PLOT2X2 = '2x2'
    PLOT3X2 = '3x2'
    PLOT3X3 = '3x3'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class Tracker():
    def __init__(self, window, plot, subplot):
        self.window = window
        self.plot = plot
        self.subplot = subplot
        self.image_plot = ImagePlot(self.window, title='', tracker_plot=True,
                                    tracker_update_callback=self.update)
        self.step = 0

    def update(self, step):
        self.step = step
        self.plot._update_subplot(self.subplot)


class Plot:
    def __init__(self, root, frame, plot_layout):
        self.root = root
        self.frame = frame
        self.figure = Figure(figsize=(23, 10), dpi=96, facecolor=BACKGROUND_COLOR_HEX)
        self.subplots = []
        rows = int(plot_layout.value[0])
        columns = int(plot_layout.value[2])
        for i in range(0, rows * columns):
            self._add_subplot(rows, columns, i + 1)
        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame, pack_toolbar=False)
        self.toolbar.config(background=BACKGROUND_COLOR_HEX)
        for button in self.toolbar.winfo_children():
            button.config(background=BACKGROUND_COLOR_HEX, highlightbackground=BACKGROUND_COLOR_HEX)
        self.toolbar.update()
        self.button_clear = ttk.Button(
            self.frame,
            text='Clear Plot',
            style='Button.TButton',
            width=8,
            command=self.clear_plot)

        self.figure.subplots_adjust(left=0.05, bottom=0.07, right=0.95, top=0.95, wspace=0.1, hspace=0.25)

        self.trackers = []

    def _add_subplot(self, rows, columns, index):
        self.subplots.append(
            {
                'plot': self.figure.add_subplot(rows, columns, index),
                'plottables': [],
                'lines': []
            }
        )

    def draw(self):
        self.canvas.draw()

    def clear_plot(self):
        for subplot in self.subplots:
            subplot['plot'].cla()
            subplot['plottables'].clear()
            subplot['lines'].clear()
        self.draw()

    def place_canvas(self, x, y, width, height):
        self.canvas.get_tk_widget().place(x=x, y=y, width=width, height=height)

    def place_toolbar(self, x, y):
        self.toolbar.place(x=x, y=y)

    def place_button_clear(self, x, y, height):
        self.button_clear.place(x=x, y=y, height=height)

    def on_pick(self, event=None):
        for subplot in self.subplots:
            for line in subplot['lines']:
                legend_line = line['legend_line']
                if legend_line == event.artist:
                    button = event.mouseevent.button
                    if button == 1:
                        plot_line = line['plot_line']
                        visible = plot_line.get_alpha() == 1.0
                        if visible:
                            plot_line.set_alpha(0.2)
                        else:
                            plot_line.set_alpha(1.0)
                        if visible:
                            legend_line.set_alpha(0.2)
                        else:
                            legend_line.set_alpha(1.0)
                        self.draw()
                    if button == 3:
                        plottable = line['plottable']
                        self._delete_from_subplot(subplot, plottable)

    def on_click(self, event=None):
        button = event.button
        double_click = event.dblclick
        in_axes = event.inaxes
        if button == 1 and double_click and in_axes is not None:
            subplot = self._find_subplot(axes=in_axes)
            old_title = subplot['plot'].title
            new_title = simpledialog.askstring('Subplot Title', '', parent=self.frame,
                                               initialvalue=old_title.get_text())
            if new_title:
                subplot['plot'].title.set_text(new_title)
            self.draw()
        if button == 2 and in_axes is not None:
            subplot = self._find_subplot(axes=in_axes)
            self.add_tracker(subplot)

    def add_tracker(self, subplot):
        window = tk.Toplevel(self.root)
        tracker = Tracker(window=window, plot=self, subplot=subplot)
        center_x = self.root.winfo_screenwidth() / 2 - window.winfo_reqwidth() / 2
        center_y = self.root.winfo_screenheight() / 2 - window.winfo_reqheight() / 2
        window.geometry(f'815x1180+{int(center_x - 480)}+{int(center_y - 540)}')
        window.title('Tracker')
        window.protocol("WM_DELETE_WINDOW", lambda arg=window: self.remove_tracker(arg))
        self.trackers.append(tracker)

    def remove_tracker(self, window):
        subplot = None
        for tracker in self.trackers:
            if tracker.window == window:
                subplot = tracker.subplot
                self.trackers.remove(tracker)
                window.destroy()
                self._update_subplot(subplot)
                break

    def plot_on_tracker_plot(self, toplevel, x, y, inference, run, dataset_type):
        for tracker in self.trackers:
            if toplevel != str(tracker.window)[1:]:
                continue
            image_plot = tracker.image_plot
            frame = image_plot.frame
            fx, fy, fw, fh = frame.winfo_rootx(), frame.winfo_rooty(), frame.winfo_width(), frame.winfo_height()
            if x in range(fx, fx + fw) and y in range(fy, fy + fh):
                title = f'Tracker {inference.name} - Run {str(run.id).zfill(2)}'
                tracker.window.title(title)
                image_plot.plot_image(run, '', dataset_type)

    def add_to_plot(self, x, y, plottables):
        subplot = self._find_subplot(x=x, y=y)
        self._add_to_subplot(subplot, plottables)

    def _find_subplot(self, x=None, y=None, axes=None):
        if axes is not None:
            for subplot in self.subplots:
                if subplot['plot'] == axes:
                    return subplot

        if not None in (x, y):
            for subplot in self.subplots:
                # matplotlib bboxes coordinates have lower left origin
                bbox = subplot['plot'].bbox.get_points().tolist()
                bbox = [i for j in bbox for i in j]
                bbox[0] += 978
                bbox[1] = 1172 - bbox[1]
                bbox[2] += 978
                bbox[3] = 1172 - bbox[3]
                bbox = [int(i) for i in bbox]

                if x in range(bbox[0], bbox[2]) and y in range(bbox[3], bbox[1]):
                    return subplot

    def _add_to_subplot(self, subplot, plottables):
        if subplot is None:
            return

        plottables = copy.deepcopy(plottables)
        for plottable in plottables:
            subplot['plottables'].append(plottable)

        self._update_subplot(subplot)

    def _delete_from_subplot(self, subplot, plottable):
        subplot['plottables'].remove(plottable)
        self._update_subplot(subplot)

    def _update_subplot(self, subplot):
        subplot['plot'].cla()
        subplot['lines'].clear()
        self.draw()

        if not subplot['plottables']:
            return

        max_s = 0
        min_s = 0
        max_v = 0
        min_v = 0
        log_x_axis = False
        log_y_axis = False
        for plottable in subplot['plottables']:
            values = plottable.values
            steps = plottable.steps
            cur_max_s = max(steps)
            cur_min_s = min(steps)
            cur_max_v = max(values)
            cur_min_v = min(values)

            if cur_max_s > max_s:
                max_s = cur_max_s

            if cur_min_s < min_s:
                min_s = cur_min_s

            if cur_max_v > max_v:
                max_v = cur_max_v

            if cur_min_v < min_v:
                min_v = cur_min_v

        for plottable in subplot['plottables']:
            values = plottable.values
            steps = plottable.steps

            if max(steps) < max_s or min(steps) > min_s:
                steps = np.interp(steps, (min(steps), max(steps)), (min_s, max_s))

            plot_function = subplot['plot'].plot if not plottable.step_plot else subplot['plot'].step
            plot_function(steps,
                          values,
                          linewidth=plottable.linewidth,
                          linestyle=plottable.linestyle,
                          marker=plottable.marker,
                          markersize=plottable.markersize,
                          markerfacecolor=plottable.markerfacecolor,
                          label=plottable.legend.upper())

            if plottable.log_x_axis:
                log_x_axis = True

            if plottable.log_y_axis:
                log_y_axis = True

        view_s = int(max(abs(min_s), abs(max_s)) * 0.01)
        view_v = int(max(abs(min_v), abs(max_v)) * 0.05)
        if min_s < 0:
            xlim_min = min_s - view_s
        else:
            xlim_min = 0

        if min_v < 0:
            ylim_min = min_v - view_v
        else:
            ylim_min = 0

        if log_x_axis:
            subplot['plot'].set_xscale('log')

        if log_y_axis:
            subplot['plot'].set_yscale('log')

        subplot['plot'].set_xlim(xlim_min, max_s + view_s)
        subplot['plot'].set_ylim(ylim_min, max_v + view_v)
        subplot['plot'].grid()

        legend = subplot['plot'].legend()
        legend_lines = legend.get_lines()
        for legend_line in legend_lines:
            legend_line.set_picker(5)

        plot_lines = subplot['plot'].get_lines()
        for plot_line in plot_lines:
            plot_line.set_alpha(1.0)

        for i, _ in enumerate(legend_lines):
            subplot['lines'].append(
                {
                    'legend_line': legend_lines[i],
                    'plot_line': plot_lines[i],
                    'plottable': subplot['plottables'][i]
                }
            )

        for tracker in self.trackers:
            if tracker.subplot == subplot:
                tracker.subplot['plot'].axvline(x=tracker.step, color='k')

        self.draw()
