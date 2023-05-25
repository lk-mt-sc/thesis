import copy
from enum import Enum

import numpy as np
from tkinter import ttk
from tkinter import simpledialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from common import BACKGROUND_COLOR_HEX


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


class Plot:
    def __init__(self, frame, plot_layout):
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

        self.figure.subplots_adjust(left=0.02, bottom=0.075, right=0.99, top=0.975, wspace=0.1, hspace=0.25)

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
                        plot_lines = line['plot_lines']
                        visible = plot_lines[0].get_alpha() == 1.0
                        for plot_line in plot_lines:
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

    def add_to_plot(self, x, y, plottable):
        subplot = self._find_subplot(x=x, y=y)
        self._add_to_subplot(subplot, plottable)

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

    def _add_to_subplot(self, subplot, plottable):
        if subplot is None:
            return

        plottable = copy.deepcopy(plottable)
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
        max_v = 0
        for plottable in subplot['plottables']:
            v = plottable['values']
            cur_s = len(v)
            cur_v = max(v)

            if cur_s > max_s:
                max_s = cur_s

            if cur_v > max_v:
                max_v = cur_v

        for plottable in subplot['plottables']:
            v = plottable['values']
            s = list(range(0, len(v)))
            no_data_indices = [i for i, x in enumerate(v) if x == -1]

            if no_data_indices:
                v_ = v.copy()
                s_ = s.copy()

                v_ = [i for j, i in enumerate(v_) if j not in no_data_indices]
                s_ = [i for j, i in enumerate(s_) if j not in no_data_indices]

                v_interp = np.interp(no_data_indices, s_, v_)
                for i, _ in enumerate(v_interp):
                    v_.insert(no_data_indices[i], v_interp[i])

                s_ = list(range(0, len(v_)))

            if no_data_indices:
                subplot['plot'].plot(s_, v_, label=plottable['name'])
                subplot['plot'].plot(no_data_indices, v_interp, 'kx', markersize=8)
            else:
                subplot['plot'].plot(s, v, label=plottable['name'])

        subplot['plot'].set_xlim(0, max_s + 1)
        subplot['plot'].set_ylim(0, max_v + 100)
        subplot['plot'].grid()

        legend = subplot['plot'].legend()
        legend_lines = legend.get_lines()
        for legend_line in legend_lines:
            legend_line.set_picker(5)

        plot_lines = subplot['plot'].get_lines()
        for plot_line in plot_lines:
            plot_line.set_alpha(1.0)

        split_indices = [i for i, plot_line in enumerate(plot_lines) if plot_line._linestyle != 'None']
        plot_lines = [plot_line.tolist() for plot_line in np.split(plot_lines, split_indices)]
        del plot_lines[0]

        for i, _ in enumerate(legend_lines):
            subplot['lines'].append(
                {
                    'legend_line': legend_lines[i],
                    'plot_lines': plot_lines[i],
                    'plottable': subplot['plottables'][i]
                }
            )

        self.draw()
