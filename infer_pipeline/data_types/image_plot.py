import os
import glob
import json
import shutil
import tkinter as tk
from tkinter import ttk

import cv2 as cv
import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from common import BACKGROUND_COLOR_HEX, MMPOSE_DATA_EXPORT_DIR
from manager.dataset_manager import KeyPointsImagePlot, KeyPointsFeatureList, KeypointsDefaultCOCO
from metrics.all_metrics import AllMetrics


class ImagePlot():
    def __init__(self, frame, title=None, image_plot_middle_click_callback=None, tracker_plot=False, tracker_update_callback=None):
        self.frame = frame
        self.tracker_plot = tracker_plot
        self.tracker_update_callback = tracker_update_callback
        self.ratio = 960 / 1080
        self.default_image = np.full((100, int(self.ratio * 100), 3), (255, 255, 255))
        self.image = self.default_image
        self.title = tk.StringVar()
        self.title.set(title or 'No Inference Selected')
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

        self.toolbar = NavigationToolbar2Tk(self.canvas,
                                            self.frame,
                                            pack_toolbar=False)
        self.toolbar.config(background=BACKGROUND_COLOR_HEX)
        for button in self.toolbar.winfo_children():
            button.config(background=BACKGROUND_COLOR_HEX, highlightbackground=BACKGROUND_COLOR_HEX)
        self.toolbar.update()

        self.counter_var = tk.StringVar()
        self.counter_var.set('000/000')
        label_counter = ttk.Label(self.frame, textvariable=self.counter_var)

        self.slider = ttk.Scale(
            self.frame,
            from_=0,
            to=99,
            orient='horizontal',
            command=self.slider_changed
        )

        self.slider.configure(state='disabled')

        if self.tracker_plot:
            label_counter.place(x=10, y=10)
            self.slider.place(x=65, y=10, width=165)
            self.toolbar.place(x=10, y=30)
            self.button_export = ttk.Button(
                self.frame,
                text='Export to Dataset',
                style='Button.TButton',
                width=20,
                command=self.export_to_dataset)
            self.button_export.place(x=250, y=7, height=60)
            self.counter_dataset_images_var = tk.StringVar()
            self.counter_dataset_images_var.set('Dataset images from this run: 0')
            ttk.Label(self.frame, textvariable=self.counter_dataset_images_var).place(x=385, y=10)
            self.current_image_in_dataset_var = tk.StringVar()
            self.current_image_in_dataset_var.set('The current image is not in the dataset.')
            ttk.Label(self.frame, textvariable=self.current_image_in_dataset_var).place(x=385, y=30)
            self._gui_disable_button_export()
            self.canvas.get_tk_widget().place(x=0, y=85, width=815, height=1080)
        else:
            self.label_title.place(x=25, y=80)
            self.canvas.get_tk_widget().place(x=0, y=0, width=815, height=1080)
            self.toolbar.place(x=25, y=1000)
            label_counter.place(x=30, y=985)
            self.slider.place(x=85, y=985, width=165)

        self.run = None
        self.slider_value = 0
        self.dataset_type = None
        self.image_plot_middle_click_callback = image_plot_middle_click_callback

    def set_dataset_information(self):
        counter_dataset_images = len(
            sorted(glob.glob(os.path.join(MMPOSE_DATA_EXPORT_DIR, f'{str(self.run.id).zfill(3)}_*.png'))))
        self.counter_dataset_images_var.set(f'Dataset images from this run: {counter_dataset_images}')

        current_image_filename = self.image.split('/')[-1]
        export_image_filename = str(self.run.id).zfill(3) + '_' + current_image_filename
        export_image_path = os.path.join(MMPOSE_DATA_EXPORT_DIR, export_image_filename)
        if os.path.exists(export_image_path):
            self.current_image_in_dataset_var.set('The current image is in the dataset.')
            self._gui_disable_button_export()
        else:
            self.current_image_in_dataset_var.set('The current image is not in the dataset.')
            self._gui_enable_button_export()

    def export_to_dataset(self):
        current_image_filename = self.image.split('/')[-1]
        export_image_filename = str(self.run.id).zfill(3) + '_' + current_image_filename
        export_image_path = os.path.join(MMPOSE_DATA_EXPORT_DIR, export_image_filename)

        shutil.copyfile(self.image, export_image_path)

        keypoints = [0 for i in range(0, 3 * len(KeypointsDefaultCOCO))]
        processed_keypoints = []
        for keypoint_name, keypoint in self.dataset_type.keypoints.items():
            keypoint_name = keypoint_name[:-2]
            if KeyPointsFeatureList.has_value(keypoint_name) and not keypoint_name in processed_keypoints:
                feature_x = next(f for f in self.run.features if f.name == keypoint_name + '_x')
                feature_y = next(f for f in self.run.features if f.name == keypoint_name + '_y')
                x = feature_x.values[self.slider_value]
                y = feature_y.values[self.slider_value]
                v = 2

                for i, keypoint in enumerate(KeypointsDefaultCOCO):
                    if keypoint.value == keypoint_name:
                        keypoints[i * 3], keypoints[i * 3 + 1], keypoints[i * 3 + 2] = x, y, v
                processed_keypoints.append(keypoint_name)

        bbox = self.run.bboxes[self.slider_value]

        export_json_path = export_image_path.split('.')[0] + '.json'
        with open(export_json_path, 'w', encoding='utf8') as annotations_file:
            json.dump(
                {
                    'image': export_image_path,
                    'keypoints': keypoints,
                    'bbox': bbox
                },
                annotations_file)

        self.set_dataset_information()

    def plot_image(self, run, title, dataset_type):
        self.run = run
        self.title.set(title)
        self.dataset_type = dataset_type

        if self.tracker_plot:
            self._gui_enable_button_export()

        self._gui_enable_slider()
        self._gui_set_slider_from_to(from_=0, to=len(self.run.data.get_images())-1)
        self._gui_set_slider_value(self.slider_value)

    def clear(self):
        self.image = None
        self.slider_value = 0
        self.title.set('No Inference Selected')
        self.dataset_type = None
        self._gui_clear_image()

    def _gui_enable_button_export(self):
        self.button_export['state'] = 'normal'

    def _gui_disable_button_export(self):
        self.button_export['state'] = 'disabled'

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
            a = value
            b = len(self.run.data.get_images()) - 1
        self.counter_var.set(f'{str(a).zfill(3)}/{str(b).zfill(3)}')

    def slider_changed(self, value):
        if self.run is None:
            return

        value = int(float(value))
        self.slider_value = value

        if self.tracker_update_callback is not None:
            self.tracker_update_callback(self.slider_value)

        self.image = self.run.data.get_images()[self.slider_value]
        image = matplotlib.image.imread(self.image)

        for feature in self.run.features:
            if feature.values[self.slider_value] == -1:
                image = cv.putText(image, 'no pose estimations', org=(10, 30), color=(1.0, 1.0, 1.0),
                                   fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, lineType=cv.LINE_AA)
                break
        else:
            bbox = self.run.bboxes[self.slider_value]
            detection_score = self.run.detection_scores[self.slider_value]
            pose_estimation_score = self.run.pose_estimation_scores[self.slider_value]

            if self.run.bboxes_bottomup:
                bbox_bottomup = self.run.bboxes_bottomup[self.slider_value]
                iou = self.ious[self.slider_value]
            else:
                bbox_bottomup = None
                iou = None

            self._draw_keypoints(image)
            self._draw_skeleton(image)
            self._draw_bounding_box_and_scores(image, bbox, bbox_bottomup, iou, detection_score, pose_estimation_score)

        self._gui_set_image(image)
        self._gui_set_image_counter(value=value)
        self.set_dataset_information()

    def _draw_keypoints(self, image):
        drawn_keypoints = []
        for keypoint_name, keypoint in self.dataset_type.keypoints.items():
            keypoint_name = keypoint_name[:-2]
            if KeyPointsImagePlot.has_value(keypoint_name) and not keypoint_name in drawn_keypoints:
                feature_x = next(f for f in self.run.features if f.name == keypoint_name + '_x')
                feature_y = next(f for f in self.run.features if f.name == keypoint_name + '_y')
                x = int(feature_x.values[self.slider_value])
                y = int(feature_y.values[self.slider_value])
                color = [c / 256 for c in keypoint['color']]
                image = cv.circle(image, center=(x, y), radius=5, thickness=-1, color=color)
                drawn_keypoints.append(keypoint_name)

                if self.tracker_plot:
                    metric_x = next(
                        (m for m in self.run.metrics[AllMetrics.HIGHPASS.value] if m.feature == feature_x), None)
                    metric_y = next(
                        (m for m in self.run.metrics[AllMetrics.HIGHPASS.value] if m.feature == feature_y), None)

                    for _, metrics in self.run.metrics.items():
                        metrics[0].tracker_plot(image, self.slider_value, metric_x, metric_y)

    def _draw_skeleton(self, image):
        for key, limb in self.dataset_type.skeleton.items():
            keypoint_1_name = limb['keypoint_1'][0]['name']
            keypoint_2_name = limb['keypoint_2'][0]['name']

            if False in (KeyPointsImagePlot.has_value(keypoint_1_name), KeyPointsImagePlot.has_value(keypoint_2_name)):
                continue

            x1 = int(next(f for f in self.run.features if f.name == keypoint_1_name + '_x').values[self.slider_value])
            y1 = int(next(f for f in self.run.features if f.name == keypoint_1_name + '_y').values[self.slider_value])
            x2 = int(next(f for f in self.run.features if f.name == keypoint_2_name + '_x').values[self.slider_value])
            y2 = int(next(f for f in self.run.features if f.name == keypoint_2_name + '_y').values[self.slider_value])
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
        if event.button == 2 and self.image_plot_middle_click_callback is not None:
            self.image_plot_middle_click_callback(event=event)
