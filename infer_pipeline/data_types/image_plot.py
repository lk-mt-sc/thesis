import tkinter as tk
from tkinter import ttk

import cv2 as cv
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from common import BACKGROUND_COLOR_HEX
from manager.dataset_manager import KeyPointsImagePlot


class ImagePlot():
    def __init__(self, frame, ratio, default_image, image_plot_middle_click_callback):
        self.frame = frame
        self.ratio = ratio
        self.default_image = default_image
        self.image = self.default_image
        self.title = tk.StringVar()
        self.title.set('No Inference Selected')
        self.figure = Figure(figsize=(self.ratio * 10, 10), dpi=96, facecolor=BACKGROUND_COLOR_HEX)
        self.plot = self.figure.add_subplot()
        self.figure.subplots_adjust(left=0.03, bottom=0, right=0.97, top=1)
        self.plot.axis('off')
        self.plot.format_coord = lambda x, y: ''
        self.imshow = self.plot.imshow(self.image)
        self.imshow.format_cursor_data = lambda e: ''
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.mpl_connect('button_press_event', self.on_click)
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
        self.bboxes_bottomup = []
        self.ious = []
        self.detection_scores = []
        self.pose_estimation_scores = []
        self.slider_value = 0
        self.dataset_type = None
        self.image_plot_middle_click_callback = image_plot_middle_click_callback

    def plot_image(self, images, features, bboxes, bboxes_bottomup, ious,
                   detection_scores, pose_estimation_scores, title, dataset_type):
        self.images = images
        self.n_images = len(self.images)
        self.features = features
        self.bboxes = bboxes
        self.bboxes_bottomup = bboxes_bottomup
        self.ious = ious
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
        self.bboxes_bottomup.clear()
        self.ious.clear()
        self.detection_scores.clear()
        self.pose_estimation_scores.clear()
        self.slider_value = 0
        self.title.set('No Inference Selected')
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

        for feature in self.features:
            if feature.values[self.slider_value] == -1:
                image = cv.putText(image, 'no pose estimations', org=(10, 30), color=(1.0, 1.0, 1.0),
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, lineType=cv.LINE_AA)
                break
        else:
            bbox = self.bboxes[self.slider_value]
            detection_score = self.detection_scores[self.slider_value]
            pose_estimation_score = self.pose_estimation_scores[self.slider_value]

            if self.bboxes_bottomup:
                bbox_bottomup = self.bboxes_bottomup[self.slider_value]
                iou = self.ious[self.slider_value]
            else:
                bbox_bottomup = None
                iou = None

            self._draw_keypoints(image)
            self._draw_skeleton(image)
            self._draw_bounding_box_and_scores(image, bbox, bbox_bottomup, iou, detection_score, pose_estimation_score)

        self._gui_set_image(image)
        self._gui_set_image_counter(value=value)

    def _draw_keypoints(self, image):
        drawn_keypoints = []
        for keypoint_name, keypoint in self.dataset_type.keypoints.items():
            keypoint_name = keypoint_name[:-2]
            if KeyPointsImagePlot.has_value(keypoint_name) and not keypoint_name in drawn_keypoints:
                x = int(next(f for f in self.features if f.name == keypoint_name + '_x').values[self.slider_value])
                y = int(next(f for f in self.features if f.name == keypoint_name + '_y').values[self.slider_value])
                color = [c / 256 for c in keypoint['color']]
                image = cv.circle(image, center=(x, y), radius=5, thickness=-1, color=color)
                drawn_keypoints.append(keypoint_name)

    def _draw_skeleton(self, image):
        for key, limb in self.dataset_type.skeleton.items():
            keypoint_1_name = limb['keypoint_1'][0]['name']
            keypoint_2_name = limb['keypoint_2'][0]['name']

            if False in (KeyPointsImagePlot.has_value(keypoint_1_name), KeyPointsImagePlot.has_value(keypoint_2_name)):
                continue

            x1 = int(next(f for f in self.features if f.name == keypoint_1_name + '_x').values[self.slider_value])
            y1 = int(next(f for f in self.features if f.name == keypoint_1_name + '_y').values[self.slider_value])
            x2 = int(next(f for f in self.features if f.name == keypoint_2_name + '_x').values[self.slider_value])
            y2 = int(next(f for f in self.features if f.name == keypoint_2_name + '_y').values[self.slider_value])
            color = [c / 256 for c in limb['color']]
            image = cv.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=2)

    def _draw_bounding_box_and_scores(self, image, bbox, bbox_bottomup, iou, detection_score, pose_estimation_score):
        bbox = [int(i) for i in bbox]
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        image = cv.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=(0.0, 0.0, 1.0), thickness=2)
        score = '{:.4f}'.format(round(detection_score, 4)) + '/' + '{:.4f}'.format(round(pose_estimation_score, 4))
        image = cv.putText(image, score, org=(x1, y1 - 7), color=(0.0, 0.0, 1.0),
                           fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.65, thickness=1, lineType=cv.LINE_AA)

        if not None in (bbox_bottomup, iou):
            bbox = [int(i) for i in bbox_bottomup]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            image = cv.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=(1.0, 0.0, 0.43), thickness=2)
            iou = '{:.4f}'.format(round(iou, 4))
            image = cv.putText(image, iou, org=(x1 + 5, y1 + 20), color=(1.0, 0.0, 0.43),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.65, thickness=1, lineType=cv.LINE_AA)

    def on_click(self, event=None):
        if event.button == 2:
            self.image_plot_middle_click_callback(event=event)
