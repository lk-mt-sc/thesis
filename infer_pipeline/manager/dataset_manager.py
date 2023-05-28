from enum import Enum

from data_types.dataset import Dataset
from datasets.coco import dataset_info as coco_dataset_info


class Datasets(Enum):
    COCO = 'coco'


class KeyPointsImagePlot(Enum):
    HEAD = 'head'
    NECK = 'neck'
    LEFT_SHOULDER = 'left_shoulder'
    RIGHT_SHOULDER = 'right_shoulder'
    LEFT_ELBOW = 'left_elbow'
    RIGHT_ELBOW = 'right_elbow'
    LEFT_WRIST = 'left_wrist'
    RIGHT_WRIST = 'right_wrist'
    LEFT_HIP = 'left_hip'
    RIGHT_HIP = 'right_hip'
    LEFT_KNEE = 'left_knee'
    RIGHT_KNEE = 'right_knee'
    LEFT_ANKLE = 'left_ankle'
    RIGHT_ANKLE = 'right_ankle'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class KeyPointsFeatureList(Enum):
    LEFT_SHOULDER = 'left_shoulder'
    RIGHT_SHOULDER = 'right_shoulder'
    LEFT_ELBOW = 'left_elbow'
    RIGHT_ELBOW = 'right_elbow'
    LEFT_WRIST = 'left_wrist'
    RIGHT_WRIST = 'right_wrist'
    LEFT_HIP = 'left_hip'
    RIGHT_HIP = 'right_hip'
    LEFT_KNEE = 'left_knee'
    RIGHT_KNEE = 'right_knee'
    LEFT_ANKLE = 'left_ankle'
    RIGHT_ANKLE = 'right_ankle'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class KeypointsInterpolation(Enum):
    HEAD = 'head'
    NECK = 'neck'
    LEFT_SHOULDER = 'left_shoulder'
    RIGHT_SHOULDER = 'right_shoulder'
    LEFT_EAR = 'left_ear'
    RIGHT_EAR = 'right_ear'


class KeypointsNoMetric(Enum):
    HEAD = 'head'
    NECK = 'neck'
    NOSE = 'nose'
    LEFT_EYE = 'left_eye'
    RIGHT_EYE = 'right_eye'
    LEFT_EAR = 'left_ear'
    RIGHT_EAR = 'right_ear'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class DatasetManager():
    def __init__(self):
        self.datasets = {}

    def create_datasets(self):
        self.datasets[Datasets.COCO.value] = self._create_dataset(coco_dataset_info)

    def _create_dataset(self, dataset):
        name = dataset['dataset_name']

        keypoints = {}
        for _, keypoint in dataset['keypoint_info'].items():
            keypoints[keypoint['name'] + '_x'] = keypoint
            keypoints[keypoint['name'] + '_y'] = keypoint

        skeleton = {}
        for key, limb in dataset['skeleton_info'].items():
            skeleton[key] = {'keypoint_1': (keypoints[limb['link'][0] + '_x'], keypoints[limb['link'][0] + '_y']),
                             'keypoint_2': (keypoints[limb['link'][1] + '_x'], keypoints[limb['link'][1] + '_y']),
                             'color': limb['color']}

        return Dataset(
            name=name,
            keypoints=keypoints,
            skeleton=skeleton
        )
