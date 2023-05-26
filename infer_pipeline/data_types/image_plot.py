import cv2 as cv
import matplotlib
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from common import BACKGROUND_COLOR_HEX
from manager.dataset_manager import HiddenKeypointsImagePlot


class ImagePlot():
    def __init__(self, frame, ratio, default_image):
        self.frame = frame
        self.ratio = ratio
        self.default_image = default_image
        self.image = self.default_image
        self.title = tk.StringVar()
        self.figure = Figure(figsize=(self.ratio * 10, 10), dpi=96, facecolor=BACKGROUND_COLOR_HEX)
        self.plot = self.figure.add_subplot()
        self.figure.subplots_adjust(left=0.03, bottom=0, right=0.97, top=1)
        self.plot.axis('off')
        self.plot.format_coord = lambda x, y: ''
        self.imshow = self.plot.imshow(self.image)
        self.imshow.format_cursor_data = lambda e: ''
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.draw()

        self.label_title = ttk.Label(self.frame, textvariable=self.title)
        self.label_title.place(x=25, y=80)

        self.toolbar = NavigationToolbar2Tk(self.canvas,
                                            self.frame,
                                            pack_toolbar=False)
        self.toolbar.config(background=BACKGROUND_COLOR_HEX)
        for button in self.toolbar.winfo_children():
            button.config(background=BACKGROUND_COLOR_HEX, highlightbackground=BACKGROUND_COLOR_HEX)
        self.toolbar.update()
        self.toolbar.place(x=25, y=1000)

        self.canvas.get_tk_widget().place(x=0, y=0, width=815, height=1080)

        self.counter_var = tk.StringVar()
        self.counter_var.set('000/000')
        label_counter = ttk.Label(self.frame, textvariable=self.counter_var)
        label_counter.place(x=30, y=985)

        self.slider = ttk.Scale(
            self.frame,
            from_=0,
            to=99,
            orient='horizontal',
            command=self.slider_changed
        )
        self.slider.place(x=85, y=985, width=165)
        self.slider.configure(state='disabled')

        self.images = []
        self.image = None
        self.n_images = 0
        self.features = []
        self.bboxes = []
        self.detection_scores = []
        self.pose_estimation_scores = []
        self.slider_value = 0
        self.dataset_type = None

    def plot_image(self, images, features, bboxes, detection_scores, pose_estimation_scores, title, dataset_type):
        self.images = images
        self.n_images = len(self.images)
        self.features = features
        self.bboxes = bboxes
        self.detection_scores = detection_scores
        self.pose_estimation_scores = pose_estimation_scores
        self.slider_value = 0
        self.image = self.images[self.slider_value]
        self.title.set(title)
        self.dataset_type = dataset_type
        self._gui_enable_slider()
        self._gui_set_slider_from_to(from_=0, to=self.n_images-1)
        self._gui_set_slider_value(self.slider_value)

    def clear(self):
        self.images.clear()
        self.image = None
        self.n_images = 0
        self.features.clear()
        self.bboxes.clear()
        self.detection_scores.clear()
        self.pose_estimation_scores.clear()
        self.slider_value = 0
        self.title.set('')
        self.dataset_type = None
        self._gui_clear_image()

    def _gui_set_slider_value(self, value):
        self.slider.set(value)

    def _gui_set_slider_from_to(self, from_, to):
        self.slider.configure(from_=from_, to=to)

    def _gui_enable_slider(self):
        self.slider.configure(state='normal')

    def _gui_disable_slider(self):
        self.slider.configure(state='disabled')

    def _gui_clear_image(self):
        self._gui_set_slider_value(0)
        self._gui_disable_slider()
        self._gui_set_image_counter(clear=True)
        self._gui_set_image(self.default_image)

    def _gui_set_image(self, image):
        self.imshow.set_data(image)
        self.canvas.draw()

    def _gui_set_image_counter(self, value=0, clear=False):
        if clear:
            a = 0
            b = 0
        else:
            a = value + 1
            b = self.n_images
        self.counter_var.set(f'{str(a).zfill(3)}/{str(b).zfill(3)}')

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
