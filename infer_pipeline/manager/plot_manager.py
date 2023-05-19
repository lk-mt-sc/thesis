import copy

import cv2 as cv
import matplotlib

from gui.gui_plot import GUIPlot


class PlotManager():
    def __init__(self, root, status_manager):
        self.gui_plot = GUIPlot(root, slider_callback=self.slider_changed,
                                button_clear_callback=self.clear_feature_plot)
        self.status_manager = status_manager
        self.images = []
        self.image = None
        self.n_images = 0
        self.features = []
        self.slider_value = 0
        self.dataset_type = None
        self.feature_plots = []

    def toggle_feature_plot(self, name, feature):
        name = copy.deepcopy(name)
        feature = copy.deepcopy(feature)
        for feature_plot in self.feature_plots:
            if feature_plot['name'] == name:
                self._remove_feature_plot(feature_plot['name'])
                break
        else:
            self._add_feature_plot(name, feature)
        self._update_feature_plot()

    def _add_feature_plot(self, name, feature):
        self.feature_plots.append({
            'name': name,
            'feature': feature,
        })

    def _remove_feature_plot(self, name):
        for i, _ in enumerate(self.feature_plots):
            if self.feature_plots[i]['name'] == name:
                del self.feature_plots[i]
                break

    def _update_feature_plot(self):
        self._gui_clear_feature_plot()

        if not self.feature_plots:
            return

        max_s = 0
        max_v = 0
        for feature_plot in self.feature_plots:
            x = feature_plot['feature'].x
            y = feature_plot['feature'].y
            cur_s = len(x)
            cur_v = max(max(x), max(y))

            if cur_s > max_s:
                max_s = cur_s

            if cur_v > max_v:
                max_v = cur_v

        for feature_plot in self.feature_plots:
            x = feature_plot['feature'].x
            y = feature_plot['feature'].y
            s = range(0, len(x))
            label_x = feature_plot['name'] + '_x'
            label_y = feature_plot['name'] + '_y'
            self.gui_plot.feature_plot.plot(s, x, label=label_x)
            self.gui_plot.feature_plot.plot(s, y, label=label_y)

        self.gui_plot.feature_plot.set_xlim(0, max_s)
        self.gui_plot.feature_plot.set_ylim(0, max_v)
        # self.gui_plot.feature_plot.set_xticks(range(0, 1000, 100))
        # self.gui_plot.feature_plot.set_yticks(range(0, 1000, 100))

        self.gui_plot.feature_plot.legend()
        self.gui_plot.feature_plot.grid()
        self.gui_plot.feature_canvas.draw()

    def _gui_clear_feature_plot(self):
        self.gui_plot.feature_plot.cla()
        self.gui_plot.feature_canvas.draw()

    def clear_feature_plot(self):
        self.feature_plots.clear()
        self._gui_clear_feature_plot()

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
