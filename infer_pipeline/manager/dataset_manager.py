from enum import Enum

from data_types.dataset import Dataset
from datasets.coco import dataset_info as coco_dataset_info


class Datasets(Enum):
    COCO = 'coco'


class HiddenKeypointsImagePlot(Enum):
    NOSE = 'nose'
    LEFT_EYE = 'left_eye'
    RIGHT_EYE = 'right_eye'
    LEFT_EAR = 'left_ear'
    RIGHT_EAR = 'right_ear'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class HiddenKeypointsFeaturePlot(Enum):
    NOSE = 'nose'
    LEFT_EYE = 'left_eye'
    RIGHT_EYE = 'right_eye'
    LEFT_EAR = 'left_ear'
    RIGHT_EAR = 'right_ear'
    HEAD = 'head'
    NECK = 'neck'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class InterpolationKeypoints(Enum):
    HEAD = 'head'
    NECK = 'neck'
    LEFT_SHOULDER = 'left_shoulder'
    RIGHT_SHOULDER = 'right_shoulder'
    LEFT_EAR = 'left_ear'
    RIGHT_EAR = 'right_ear'


class DatasetManager():
    def __init__(self):
        self.datasets = {}

    def create_datasets(self):
        self.datasets[Datasets.COCO.value] = self._create_dataset(coco_dataset_info)

    def _create_dataset(self, dataset):
        name = dataset['dataset_name']

        keypoints = {}
        for _, keypoint in dataset['keypoint_info'].items():
            keypoints[keypoint['name']] = keypoint

        skeleton = {}
        for key, limb in dataset['skeleton_info'].items():
            skeleton[key] = {'first_keypoint': keypoints[limb['link'][0]],
                             'second_keypoint': keypoints[limb['link'][1]],
                             'color': limb['color']}

        return Dataset(
            name=name,
            keypoints=keypoints,
            skeleton=skeleton
        )
