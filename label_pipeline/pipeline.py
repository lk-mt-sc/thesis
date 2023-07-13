import os
import sys
import glob
import pickle
import random
import tkinter as tk

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from gui import GUI
from dataset import DatasetManager, Datasets, KeyPointsImagePlot
from stats import Statistics
from labeled_data import LabeledData
sys.path.append('/home/lukas/thesis/infer_pipeline')  # nopep8


class Pipeline():
    def __init__(self):
        self.root = tk.Tk()
        self.root.bind('<KeyPress>', self.on_key_press)

        self.gui = GUI(
            self.root,
            data_selected_callback=self.data_selected,
            prev_button_callback=self.prev_image,
            next_button_callback=self.next_image,
            submit_button_callback=self.submit_labels,
            reset_button_callback=self.reset_labels,
            feature_plot_clicked_callback=self.set_image,
            calculate_high_pass_button_callback=self.calculate_high_pass_properties,
            toggle_high_pass_plot_button_callback=None
        )

        self.dataset_manager = DatasetManager()
        self.dataset_manager.create_datasets()

        self.runs = sorted(glob.glob(os.path.join(DATA_DIR, 'run_*.pkl')))
        self.data = []
        self.selected_data = None
        self.images = []

        keypoints_scores = {
            'left_shoulder': [],
            'right_shoulder': [],
            'left_hip': [],
            'right_hip': [],
            'left_elbow': [],
            'right_elbow': [],
            'left_ankle': [],
            'right_ankle': [],
            'left_knee': [],
            'right_knee': [],
            'left_wrist': [],
            'right_wrist': [],
        }

        for i, run in enumerate(self.runs):
            with open(run, 'rb') as run_file:
                run_data = pickle.load(run_file)

            features = run_data.features[10:34]
            for feature in features[0::2]:
                keypoints_scores[feature.name[:-2]] += list(np.clip(feature.scores, 0.0, 1.0))

        data_points_low_confidence = {
            'left_shoulder': 0,
            'right_shoulder': 0,
            'left_hip': 0,
            'right_hip': 0,
            'left_elbow': 0,
            'right_elbow': 0,
            'left_ankle': 0,
            'right_ankle': 0,
            'left_knee': 0,
            'right_knee': 0,
            'left_wrist': 0,
            'right_wrist': 0,
        }

        fig, axes = plt.subplots(2, 6, sharey=True, tight_layout=True)
        for i, key in enumerate(keypoints_scores):
            x = i % 2
            y = i // 2
            values, _, patches = axes[x, y].hist(keypoints_scores[key], bins=100, range=(0.0, 1.0))
            n = 80
            values_low_confidence = sum(values[:n])
            data_points_low_confidence[key] = values_low_confidence
            axes[x, y].set_title(f'{key}, {values_low_confidence} data points with confidence < {n}%')
            axes[x, y].set_xlim(0, 1)
            axes[x, y].set_ylim(0, 2000)

            for i in range(0, 80):
                patches[i].set_facecolor('orange')

        if SHOW_HISTOGRAM:
            plt.show()

        keypoints_distribution = {
            'left_shoulder': 0,
            'right_shoulder': 0,
            'left_hip': 0,
            'right_hip': 0,
            'left_elbow': 0,
            'right_elbow': 0,
            'left_ankle': 0,
            'right_ankle': 0,
            'left_knee': 0,
            'right_knee': 0,
            'left_wrist': 0,
            'right_wrist': 0,
        }

        total_data_points_low_confidence = sum(k for _, k in data_points_low_confidence.items())
        for key in data_points_low_confidence:
            keypoints_distribution[key] = round(data_points_low_confidence[key] / total_data_points_low_confidence * 64)

        if SHOW_HISTOGRAM:
            n_features = sum([i for _, i in keypoints_distribution.items()])
            print(f'Calculated distribution ({n_features} total features):')
            for key, item in keypoints_distribution.items():
                print(key, item)

        keypoints_distribution['left_shoulder'] += 1
        keypoints_distribution['right_shoulder'] += 1
        keypoints_distribution['left_wrist'] -= 1
        keypoints_distribution['right_wrist'] -= 1

        if SHOW_HISTOGRAM:
            n_features = sum([i for _, i in keypoints_distribution.items()])
            print(f'\nAdjusted distribution ({n_features} total features):')
            for key, item in keypoints_distribution.items():
                print(key, item)

        keypoint_ids = {
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16,
        }

        self.random_features = []
        for key, n_keypoint in keypoints_distribution.items():
            self.random_features += [keypoint_ids[key]] * n_keypoint
        random.shuffle(self.random_features)

        for i, run in enumerate(self.runs):
            with open(run, 'rb') as run_file:
                run_data = pickle.load(run_file)

            id_ = i + 1
            feature = self.random_features[i]
            feature_x = run_data.features[feature * 2]
            feature_y = run_data.features[feature * 2 + 1]
            assert len(feature_x.steps) == len(feature_y.steps)

            values_x = feature_x.values
            values_y = feature_y.values

            diff_x = np.diff(values_x)
            diff_y = np.diff(values_y)

            diff_x = np.insert(diff_x, 0, 0)
            diff_y = np.insert(diff_y, 0, 0)

            lower_quantile = np.percentile(diff_x, 15)
            upper_quantile = np.percentile(diff_x, 85)
            candidates_x = [i for i, d in enumerate(diff_x) if d <= lower_quantile or d >= upper_quantile]

            lower_quantile = np.percentile(diff_y, 15)
            upper_quantile = np.percentile(diff_y, 85)
            candidates_y = [i for i, d in enumerate(diff_y) if d <= lower_quantile or d >= upper_quantile]

            keypoint_name = ' '.join(feature_x.name.split('_')[:2]).upper()
            name = f'{str(id_).zfill(3)} ID - KEYPOINT: {keypoint_name}'

            out_filename = os.path.join(OUT_DIR, 'labels_' + os.path.basename(run))
            if os.path.exists(out_filename):
                labeled_data = LabeledData.load(out_filename)
            else:
                labeled_data = LabeledData(
                    feature_x=feature_x,
                    feature_y=feature_y,
                    candidates_x=candidates_x,
                    candidates_y=candidates_y
                )
                labeled_data.save(out_filename)

            self.data.append(
                {
                    'id': id_,
                    'name': name,
                    'run': run_data,
                    'feature_x': feature_x,
                    'feature_y': feature_y,
                    'labeled_data': labeled_data,
                    'out_filename': out_filename,
                }
            )

        data_list_strings = [d['name'] for d in self.data]
        self.gui.data_list.set_data(data_list_strings)

        self.image_counter = 0
        self.n_images = 0

        self.statistics = Statistics(self.gui.statistics)

    def calculate_high_pass_properties(self):
        labeled_data = self.get_already_labeled_data()
        self.statistics.calculate_high_pass_properties(labeled_data)

    def get_already_labeled_data(self):
        return [LabeledData.load(labeled_data) for labeled_data in glob.glob(os.path.join(OUT_DIR, '*.pkl'))]

    def data_selected(self, event=None):
        selected_index = self.gui.data_list.data_listbox.curselection()

        if selected_index:
            self.gui.control.prev_button.configure(state=tk.NORMAL)
            self.gui.control.next_button.configure(state=tk.NORMAL)
            self.gui.labels.label_0_radiobutton.configure(state=tk.NORMAL)
            self.gui.labels.label_1_radiobutton.configure(state=tk.NORMAL)
            self.gui.labels.label_2_radiobutton.configure(state=tk.NORMAL)
            self.gui.labels.mark_x.configure(state=tk.NORMAL)
            self.gui.labels.mark_y.configure(state=tk.NORMAL)
            self.gui.labels.bounding_box_cuts_climber.configure(state=tk.NORMAL)
            self.gui.labels.bounding_box_cuts_climber_tv.configure(state=tk.NORMAL)
            self.gui.labels.bounding_box_cuts_climber_dark.configure(state=tk.NORMAL)
            self.gui.labels.side_swap.configure(state=tk.NORMAL)
            self.gui.labels.reset_button.configure(state=tk.NORMAL)

            selection = self.gui.data_list.data_listbox.get(selected_index[0])

            self.gui.image_plots.clear()
            self.gui.feature_plots.clear()

            self.selected_data = next(d for d in self.data if d['name'] == selection)
            self.gui.feature_plots.set_features(self.selected_data['labeled_data'])
            self.gui.feature_plots.set_labels(self.selected_data['labeled_data'])
            self.images = [cv.imread(image) for image in self.selected_data['run'].data.get_images()]
            self.image_counter = 0
            self.set_labels_selection()
            self.update_plots()
            already_labeled_data = self.get_already_labeled_data()
            if already_labeled_data:
                self.statistics.calculate_statistics(already_labeled_data, self.selected_data['labeled_data'])

    def set_labels_selection(self):
        labels = self.selected_data['labeled_data'].labels[self.image_counter]
        self.gui.labels.label_var.set(labels[0])
        self.gui.labels.bounding_box_cuts_climber_var.set(labels[1])
        self.gui.labels.bounding_box_cuts_climber_tv_var.set(labels[2])
        self.gui.labels.bounding_box_cuts_climber_dark_var.set(labels[3])
        self.gui.labels.side_swap_var.set(labels[4])
        self.gui.labels.mark_x_var.set(labels[5])
        self.gui.labels.mark_y_var.set(labels[6])

        if labels[0] == -1:
            self.gui.labels.submit_button.configure(state=tk.DISABLED)
        else:
            self.gui.labels.submit_button.configure(state=tk.NORMAL)

    def save_labels(self):
        filename = self.selected_data['out_filename']
        self.selected_data['labeled_data'].save(filename)

    def submit_labels(self, event=None):
        label = self.gui.labels.label_var.get()
        bounding_box_cuts_climber = self.gui.labels.bounding_box_cuts_climber_var.get()
        bounding_box_cuts_climber_tv = self.gui.labels.bounding_box_cuts_climber_tv_var.get()
        bounding_box_cuts_climber_dark = self.gui.labels.bounding_box_cuts_climber_dark_var.get()
        side_swap = self.gui.labels.side_swap_var.get()
        mark_x = self.gui.labels.mark_x_var.get()
        mark_y = self.gui.labels.mark_y_var.get()

        labeled_data = self.selected_data['labeled_data']
        labeled_data.add_labels(
            self.image_counter,
            (
                label,
                bounding_box_cuts_climber,
                bounding_box_cuts_climber_tv,
                bounding_box_cuts_climber_dark,
                side_swap,
                mark_x,
                mark_y
            )
        )
        self.gui.feature_plots.set_labels(self.selected_data['labeled_data'])

        self.save_labels()
        already_labeled_data = self.get_already_labeled_data()
        if already_labeled_data:
            self.statistics.calculate_statistics(already_labeled_data, self.selected_data['labeled_data'])

    def reset_labels(self):
        self.gui.labels.label_var.set(-1)
        self.gui.labels.bounding_box_cuts_climber_var.set(False)
        self.gui.labels.bounding_box_cuts_climber_tv_var.set(False)
        self.gui.labels.bounding_box_cuts_climber_dark_var.set(False)
        self.gui.labels.side_swap_var.set(False)
        self.gui.labels.mark_x_var.set(False)
        self.gui.labels.mark_y_var.set(False)
        self.submit_labels()

    def next_image(self, candidates_only=False):
        if self.selected_data is None:
            return

        if self.image_counter < self.n_images - 1:
            if candidates_only:
                possible_next_candidate = self.image_counter + 1
                candidates_x = self.selected_data['labeled_data'].candidates_x
                candidates_y = self.selected_data['labeled_data'].candidates_y
                while (possible_next_candidate not in candidates_x + candidates_y):
                    if possible_next_candidate + 1 < self.n_images - 1:
                        possible_next_candidate += 1
                    else:
                        possible_next_candidate = None
                        break
            else:
                possible_next_candidate = None

            self.image_counter = possible_next_candidate or self.image_counter + 1

            if self.gui.labels.auto_submit_checkbutton_var.get():
                self.submit_labels()

            self.set_labels_selection()
            self.update_plots()

    def prev_image(self, candidates_only=False):
        if self.selected_data is None:
            return

        if self.image_counter > 0:
            if candidates_only:
                possible_next_candidate = self.image_counter - 1
                candidates_x = self.selected_data['labeled_data'].candidates_x
                candidates_y = self.selected_data['labeled_data'].candidates_y
                while (possible_next_candidate not in candidates_x + candidates_y):
                    if possible_next_candidate - 1 > 0:
                        possible_next_candidate -= 1
                    else:
                        possible_next_candidate = None
                        break
            else:
                possible_next_candidate = None

            self.image_counter = possible_next_candidate or self.image_counter - 1

            if self.gui.labels.auto_submit_checkbutton_var.get():
                self.submit_labels()

            self.set_labels_selection()
            self.update_plots()

    def set_image(self, image_nr):
        if self.selected_data is None:
            return

        if image_nr in range(0, self.n_images):
            self.image_counter = image_nr

            if self.gui.labels.auto_submit_checkbutton_var.get():
                self.submit_labels()

            self.set_labels_selection()
            self.update_plots()

    def update_plots(self):
        if self.selected_data is None:
            return

        self.n_images = len(self.images)
        self.gui.control.counter_var.set(f'{str(self.image_counter).zfill(4)}/{str(self.n_images - 1).zfill(4)}')

        all_indices = list(range(0, self.n_images))
        indices = all_indices[max(0, self.image_counter - 3):self.image_counter + 4]
        if self.image_counter < 3:
            indices = [None] * (3 - self.image_counter) + indices
        if self.image_counter > len(all_indices) - 4:
            indices = indices + [None] * (self.image_counter - len(all_indices) + 4)

        images = []
        bboxes = self.selected_data['run'].bboxes

        for index in indices:
            if index is None:
                images.append(None)
            else:
                image = self.images[index]

                for key, limb in self.dataset_manager.datasets[Datasets.COCO.value].skeleton.items():
                    keypoint_1_name = limb['keypoint_1'][0]['name']
                    keypoint_2_name = limb['keypoint_2'][0]['name']

                    if False in (KeyPointsImagePlot.has_value(keypoint_1_name), KeyPointsImagePlot.has_value(keypoint_2_name)):
                        continue

                    x1 = int(next(f for f in self.selected_data['run'].features if f.name ==
                                  keypoint_1_name + '_x').values[index])
                    y1 = int(next(f for f in self.selected_data['run'].features if f.name ==
                                  keypoint_1_name + '_y').values[index])
                    x2 = int(next(f for f in self.selected_data['run'].features if f.name ==
                                  keypoint_2_name + '_x').values[index])
                    y2 = int(next(f for f in self.selected_data['run'].features if f.name ==
                                  keypoint_2_name + '_y').values[index])
                    color = [c for c in limb['color']]
                    image = cv.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=2)

                drawn_keypoints = []
                for keypoint_name, keypoint in self.dataset_manager.datasets[Datasets.COCO.value].keypoints.items():
                    keypoint_name = keypoint_name[:-2]
                    if KeyPointsImagePlot.has_value(keypoint_name) and not keypoint_name in drawn_keypoints:
                        feature_x = next(
                            f for f in self.selected_data['run'].features if f.name == keypoint_name + '_x')
                        feature_y = next(
                            f for f in self.selected_data['run'].features if f.name == keypoint_name + '_y')
                        x = int(feature_x.values[index])
                        y = int(feature_y.values[index])

                        if keypoint_name in self.selected_data['feature_x'].name:
                            color = [255, 0, 220]
                        else:
                            color = [c for c in keypoint['color']]

                        image = cv.circle(image, center=(x, y), radius=5, thickness=-1, color=color)
                        drawn_keypoints.append(keypoint_name)

                bbox = bboxes[index]
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                image = cv.rectangle(image, pt1=(x, y), pt2=(x+w, y+h), thickness=2, color=(255, 0, 0))

                margin = 50
                x_min = max(0, x - margin)
                y_min = max(0, y - margin)
                x_max = min(image.shape[1], x + w + margin)
                y_max = min(image.shape[0], y + h + margin)

                images.append(image[y_min:y_max, x_min:x_max])

        self.gui.image_plots.set_images(images)
        self.gui.feature_plots.set_tracker(self.image_counter)

    def on_key_press(self, event):
        key = event.keysym
        if key == 'q':
            self.root.destroy()

        if key == 'a':
            self.prev_image()

        if key == 'd':
            self.next_image()

        if key == 'w':
            self.prev_image(candidates_only=True)

        if key == 'e':
            self.next_image(candidates_only=True)

        if key == 's':
            if str(self.gui.labels.submit_button['state']) == 'normal':
                self.submit_labels()

    def mainloop(self):
        self.root.mainloop()


if __name__ == '__main__':

    random.seed(0)

    WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
    DATA_DIR = os.path.join(WORKING_DIR, 'data')
    OUT_DIR = os.path.join(WORKING_DIR, 'out')
    SHOW_HISTOGRAM = False

    pipeline = Pipeline()
    pipeline.mainloop()
