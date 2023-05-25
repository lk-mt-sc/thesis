import cv2 as cv
import matplotlib
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog

from gui.gui_plot import GUIPlot
from data_types.plot import Plot, PlotLayouts
from manager.dataset_manager import HiddenKeypointsImagePlot


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
    def __init__(self, root, status_manager):
        self.gui_plot = GUIPlot(
            root,
            notebook_tab_changed_callback=self.notebook_tab_changed,
            notebook_tab_double_clicked_callback=self.rename_plot,
            notebook_tab_middle_click_callback=self.delete_plot,
            slider_callback=self.slider_changed
        )
        self.status_manager = status_manager
        self.images = []
        self.image = None
        self.n_images = 0
        self.features = []
        self.bboxes = []
        self.detection_scores = []
        self.pose_estimation_scores = []
        self.slider_value = 0
        self.dataset_type = None
        self.plots = []
        self.manual_tab_changed = False
        self.old_tab = 0

    def set_data(self, run, dataset_type):
        self.images = run.data.get_images()
        self.n_images = len(self.images)
        self.features = run.features.copy()
        self.bboxes = run.bboxes.copy()
        self.detection_scores = run.detection_scores.copy()
        self.pose_estimation_scores = run.pose_estimation_scores.copy()
        self.slider_value = 0
        self.image = self.images[self.slider_value]
        self.dataset_type = dataset_type
        self._gui_enable_slider()
        self._gui_set_slider_from_to(from_=0, to=self.n_images-1)
        self._gui_set_slider_value(self.slider_value)

    def clear_image(self):
        self.images.clear()
        self.image = None
        self.n_images = 0
        self.features.clear()
        self.dataset_type = None
        self._gui_clear_image()

    def _gui_set_image_counter(self, value=0, clear=False):
        if clear:
            a = 0
            b = 0
        else:
            a = value + 1
            b = self.n_images
        self.gui_plot.image_counter_var.set(f'{str(a).zfill(3)}/{str(b).zfill(3)}')

    def _gui_set_slider_value(self, value):
        self.gui_plot.image_slider.set(value)

    def _gui_set_slider_from_to(self, from_, to):
        self.gui_plot.image_slider.configure(from_=from_, to=to)

    def _gui_enable_slider(self):
        self.gui_plot.image_slider.configure(state='normal')

    def _gui_disable_slider(self):
        self.gui_plot.image_slider.configure(state='disabled')

    def _gui_clear_image(self):
        self._gui_set_slider_value(0)
        self._gui_disable_slider()
        self._gui_set_image_counter(clear=True)
        self._gui_set_image(self.gui_plot.default_image)

    def _gui_set_image(self, image):
        self.gui_plot.image_imshow.set_data(image)
        self.gui_plot.image_canvas.draw()

    def slider_changed(self, value):
        if not self.images:
            return

        value = int(float(value))
        self.slider_value = value

        self.image = self.images[self.slider_value]
        image = matplotlib.image.imread(self.image)
        bbox = self.bboxes[self.slider_value]
        detection_score = self.detection_scores[self.slider_value]
        pose_estimation_score = self.pose_estimation_scores[self.slider_value]

        for feature in self.features:
            if -1 in (feature.x[self.slider_value], feature.y[self.slider_value]):
                image = cv.putText(image, 'no pose estimations', org=(10, 30), color=(255, 255, 255),
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, lineType=cv.LINE_AA)
                break
        else:
            self._draw_keypoints(image)
            self._draw_skeleton(image)
            self._draw_bounding_box_and_scores(image, bbox, detection_score, pose_estimation_score)

        self._gui_set_image(image)
        self._gui_set_image_counter(value=value)

    def _draw_keypoints(self, image):
        for key, keypoint in self.dataset_type.keypoints.items():
            for feature in self.features:
                feature_name = feature.name
                if feature_name == key and not HiddenKeypointsImagePlot.has_value(feature_name):
                    x = int(feature.x[self.slider_value])
                    y = int(feature.y[self.slider_value])
                    color = keypoint['color']
                    image = cv.circle(image, center=(x, y), radius=5, thickness=-1, color=color)

    def _draw_skeleton(self, image):
        for key, limb in self.dataset_type.skeleton.items():
            first_keypoint = next(feature for feature in self.features if feature.name ==
                                  limb['first_keypoint']['name'])
            second_keypoint = next(feature for feature in self.features if feature.name ==
                                   limb['second_keypoint']['name'])
            if HiddenKeypointsImagePlot.has_value(first_keypoint.name) \
                    or HiddenKeypointsImagePlot.has_value(second_keypoint.name):
                continue
            color = limb['color']
            x1 = int(first_keypoint.x[self.slider_value])
            y1 = int(first_keypoint.y[self.slider_value])
            x2 = int(second_keypoint.x[self.slider_value])
            y2 = int(second_keypoint.y[self.slider_value])
            image = cv.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=2)

    def _draw_bounding_box_and_scores(self, image, bbox, detection_score, pose_estimation_score):
        bbox = [int(i) for i in bbox]
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        image = cv.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
        score = '{:.4f}'.format(round(detection_score, 4)) + '/' + '{:.4f}'.format(round(pose_estimation_score, 4))
        image = cv.putText(image, score, org=(x, y - 7), color=(0, 0, 255),
                           fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.65, thickness=1, lineType=cv.LINE_AA)

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

    def add_to_plot(self, x, y, plottable):
        notebook = self.gui_plot.notebook
        selected_tab = notebook.index(notebook.select())
        if selected_tab == 0:
            return

        current_plot = next(plot['plot'] for plot in self.plots if plot['tab_id'] == selected_tab)
        current_plot.add_to_plot(x, y, plottable=plottable)

    def rename_plot(self, event=None):
        notebook = self.gui_plot.notebook
        clicked_tab = notebook.tk.call(notebook._w, 'identify', 'tab', event.x, event.y)

        if clicked_tab in (0, len(notebook.tabs()) - 1):
            return

        old_title = notebook.tab(clicked_tab, 'text')
        new_title = simpledialog.askstring('Plot Title', '', parent=self.gui_plot.frame, initialvalue=old_title)
        if new_title:
            notebook.tab(clicked_tab, text=new_title)
