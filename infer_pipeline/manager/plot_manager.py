import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog

from gui.gui_plot import GUIPlot
from data_types.plot import Plot, PlotLayouts


class PlotDialog(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)
        self.title('')
        self.geometry('+%d+%d' % (self.winfo_screenwidth() / 2 - self.winfo_reqwidth() / 2,
                                  self.winfo_screenheight() / 2 - self.winfo_reqheight() / 2))

        tk.Label(self, text='Plot Title').pack(side="top", fill="x")
        self.var_title = tk.StringVar()
        tk.Entry(self, textvariable=self.var_title).pack()

        tk.Label(self, text='Plot Layout').pack(side="top", fill="x")
        self.var_layout = tk.StringVar()
        for layout in PlotLayouts:
            ttk.Radiobutton(self, variable=self.var_layout, value=layout.value,
                            text=layout.value, command=self.destroy).pack()

    def show(self):
        self.wm_deiconify()
        self.wait_window()
        layout = self.var_layout.get()
        plot_layout = PlotLayouts(layout) if PlotLayouts.has_value(layout) else None
        return plot_layout, self.var_title.get()


class PlotManager():
    def __init__(self, root, status_manager, metric_manager):
        self.gui_plot = GUIPlot(
            root,
            notebook_tab_change_callback=self.notebook_tab_changed,
            notebook_tab_double_click_callback=self.rename_plot,
            notebook_tab_middle_click_callback=self.delete_plot,
            image_plot_middle_click_callback=self.clear_image
        )
        self.status_manager = status_manager
        self.metric_manager = metric_manager
        self.plots = []
        self.image_plots = [
            self.gui_plot.image_plot_1,
            self.gui_plot.image_plot_2,
            self.gui_plot.image_plot_3,
        ]
        self.manual_tab_changed = False
        self.old_tab = 0

    def plot_image(self, x, y, inference, run, title, dataset_type):
        notebook = self.gui_plot.notebook
        selected_tab = notebook.index(notebook.select())
        if selected_tab != 0:
            return

        image_plot, position = self._find_image_plot(x, y)
        if not None in (image_plot, position):
            image_plot.plot_image(
                run.data.get_images(),
                run.features.copy(),
                run.bboxes.copy(),
                run.bboxes_bottomup.copy(),
                run.ious.copy(),
                run.detection_scores.copy(),
                run.pose_estimation_scores.copy(),
                title,
                dataset_type
            )
            self.metric_manager.add_to_compared_inferences(inference, position)

    def clear_image(self, event=None, position=None):
        if position is not None:
            self.image_plots[position].clear()

        if event is not None:
            x = self.gui_plot.root.winfo_pointerx()
            y = self.gui_plot.root.winfo_pointery()
            image_plot, position = self._find_image_plot(x, y)
            if not None in (image_plot, position):
                image_plot.clear()
                self.metric_manager.remove_from_compared_inferences(position)

    def clear_images(self):
        for i, _ in enumerate(self.image_plots):
            self.clear_image(position=i)

    def _find_image_plot(self, x, y):
        for i, image_plot in enumerate(self.image_plots):
            frame = image_plot.frame
            fx, fy, fw, fh = frame.winfo_rootx(), frame.winfo_rooty(), frame.winfo_width(), frame.winfo_height()
            if x in range(fx, fx + fw) and y in range(fy, fy + fh):
                return image_plot, i

        return None, None

    def _gui_add_plot(self, plot_layout, title):
        notebook = self.gui_plot.notebook
        frame = ttk.Frame(notebook, width=2445, height=1080, padding=(0, 0))
        new_plot = Plot(frame, plot_layout=plot_layout)
        new_plot.place_canvas(x=0, y=0, width=2445, height=1080)
        new_plot.place_toolbar(x=65, y=1039)
        new_plot.place_button_clear(x=5, y=1045, height=30)
        new_plot.draw()
        frame.place(x=0, y=0)
        notebook.insert(len(notebook.tabs()) - 1, frame, text=title)
        n_tabs = len(notebook.tabs())
        self.plots.append({'tab_id': n_tabs - 2, 'internal_tab_id': n_tabs - 1, 'plot': new_plot})
        self.manual_tab_changed = True
        notebook.select(n_tabs - 2)

    def _gui_delete_plot(self, event=None):
        notebook = self.gui_plot.notebook
        clicked_tab = notebook.tk.call(notebook._w, 'identify', 'tab', event.x, event.y)

        if clicked_tab in (0, len(notebook.tabs()) - 1):
            return

        for plot in self.plots:
            if plot['tab_id'] == clicked_tab:
                if clicked_tab == notebook.index(notebook.select()):
                    self.manual_tab_changed = True
                    notebook.select(0)

                notebook.winfo_children()[plot['internal_tab_id']].destroy()

                self.plots.remove(plot)
                break

        for plot in self.plots:
            if plot['tab_id'] > clicked_tab:
                plot['tab_id'] -= 1
                plot['internal_tab_id'] -= 1

    def add_plot(self):
        plot_layout, title = PlotDialog(self.gui_plot.root).show()
        if plot_layout is None:
            return
        if not title:
            title = 'Unnamed Plot'
        self._gui_add_plot(plot_layout, title)

    def delete_plot(self, event=None):
        self._gui_delete_plot(event)

    def notebook_tab_changed(self, event=None):
        notebook = self.gui_plot.notebook
        selected_tab = notebook.index(notebook.select())

        if selected_tab != len(notebook.tabs()) - 1:
            self.old_tab = selected_tab

        if self.manual_tab_changed:
            self.manual_tab_changed = False
            return

        if notebook.select() == notebook.tabs()[-1]:
            self.manual_tab_changed = True
            notebook.select(self.old_tab)
            self.add_plot()

    def add_to_plot(self, x, y, plottables):
        notebook = self.gui_plot.notebook
        selected_tab = notebook.index(notebook.select())
        if selected_tab == 0:
            return

        current_plot = next(plot['plot'] for plot in self.plots if plot['tab_id'] == selected_tab)
        current_plot.add_to_plot(x, y, plottables=plottables)

    def clear_plots(self):
        for plot in self.plots:
            plot['plot'].clear_plot()

    def rename_plot(self, event=None):
        notebook = self.gui_plot.notebook
        clicked_tab = notebook.tk.call(notebook._w, 'identify', 'tab', event.x, event.y)

        if clicked_tab in (0, len(notebook.tabs()) - 1):
            return

        old_title = notebook.tab(clicked_tab, 'text')
        new_title = simpledialog.askstring('Plot Title', '', parent=self.gui_plot.frame, initialvalue=old_title)
        if new_title:
            notebook.tab(clicked_tab, text=new_title)
