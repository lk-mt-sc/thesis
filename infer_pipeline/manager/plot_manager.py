import cv2 as cv
import matplotlib
from tkinter import ttk

from gui.gui_plot import GUIPlot
from data_types.plot import Plot, PlotTypes


class PlotManager():
    def __init__(self, root, status_manager):
        self.gui_plot = GUIPlot(
            root,
            button_add_plot_callback=self.add_plot,
            button_delete_plot_callback=self.delete_plot,
            notebook_tab_changed_callback=self.notebook_tab_changed,
            slider_callback=self.slider_changed
        )
        self.status_manager = status_manager
        self.images = []
        self.image = None
        self.n_images = 0
        self.features = []
        self.slider_value = 0
        self.dataset_type = None
        self.plots = []
        self._gui_set_plot_types()

    def set_data(self, run, dataset_type):
        self.images = run.data.get_images()
        self.n_images = len(self.images)
        self.features = run.features.copy()
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

        for feature in self.features:
            if -1 in (feature.x[self.slider_value], feature.y[self.slider_value]):
                image = cv.putText(image, 'no pose estimations', org=(10, 30), color=(255, 255, 255),
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, lineType=cv.LINE_AA)
                break
        else:
            self._draw_keypoints(image)
            self._draw_skeleton(image)

        self._gui_set_image(image)
        self._gui_set_image_counter(value=value)

    def _draw_keypoints(self, image):
        for key, keypoint in self.dataset_type.keypoints.items():
            for feature in self.features:
                if feature.name == key:
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
            color = limb['color']
            x1 = int(first_keypoint.x[self.slider_value])
            y1 = int(first_keypoint.y[self.slider_value])
            x2 = int(second_keypoint.x[self.slider_value])
            y2 = int(second_keypoint.y[self.slider_value])
            image = cv.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=2)

    def _gui_set_plot_types(self):
        self.gui_plot.combobox_plot_type['values'] = [plot_type.value for plot_type in PlotTypes]
        self.gui_plot.combobox_plot_type.current(0)

    def _gui_add_plot(self, plot_type):
        notebook = self.gui_plot.notebook
        frame = ttk.Frame(notebook, width=2445, height=1080, padding=(0, 0))
        new_plot = Plot(frame, plot_type=plot_type)
        new_plot.place_canvas(x=0, y=0, width=2445, height=1080)
        new_plot.place_toolbar(x=0, y=1040)
        new_plot.place_button_clear(x=5, y=1010, height=30)
        new_plot.draw()
        frame.place(x=0, y=0)
        notebook.add(frame, text='Unnamed Plot')
        n_tabs = len(notebook.tabs())
        notebook.select(n_tabs - 1)
        self.plots.append({'tab_id': n_tabs - 1, 'plot': new_plot})

        if n_tabs == 16:
            self._gui_disable_button_add()

    def _gui_delete_plot(self):
        notebook = self.gui_plot.notebook
        for i, tab in enumerate(notebook.winfo_children()):
            if i == notebook.index(notebook.select()):
                tab.destroy()
                for plot in self.plots:
                    if plot['tab_id'] == i:
                        self.plots.remove(plot)
                        break
                for plot in self.plots:
                    if plot['tab_id'] > i:
                        plot['tab_id'] -= 1
                if len(notebook.tabs()) < 16:
                    self._gui_enable_button_add()
                return

    def _gui_enable_button_add(self):
        self.gui_plot.button_add_plot['state'] = 'normal'

    def _gui_disable_button_add(self):
        self.gui_plot.button_add_plot['state'] = 'disabled'

    def _gui_enable_button_delete(self):
        self.gui_plot.button_delete_plot['state'] = 'normal'

    def _gui_disable_button_delete(self):
        self.gui_plot.button_delete_plot['state'] = 'disabled'

    def _gui_get_plot_type(self):
        return self.gui_plot.combobox_plot_type.get()

    def _gui_get_selected_tab(self):
        notebook = self.gui_plot.notebook
        return notebook.index(notebook.select())

    def add_plot(self):
        plot_type = PlotTypes(self._gui_get_plot_type())
        self._gui_add_plot(plot_type)

    def delete_plot(self):
        self._gui_delete_plot()

    def notebook_tab_changed(self, event=None):
        selected_tab = self._gui_get_selected_tab()

        if selected_tab == 0:
            self._gui_disable_button_delete()
        else:
            self._gui_enable_button_delete()

    def add_to_plot(self, x, y, plottable):
        selected_tab = self._gui_get_selected_tab()
        if selected_tab == 0:
            return

        current_plot = next(plot['plot'] for plot in self.plots if plot['tab_id'] == selected_tab)
        current_plot.add_to_plot(x, y, plottable=plottable)
