from enum import Enum

from data_types.dataset import Dataset
from datasets.aic import dataset_info as aic_dataset_info
from datasets.coco import dataset_info as coco_dataset_info
from datasets.crowdpose import dataset_info as crowdpose_dataset_info
from datasets.mpii import dataset_info as mpii_dataset_info


class Datasets(Enum):
    AIC = 'aic'
    COCO = 'coco'
    CROWDPOSE = 'crowdpose'
    MPII = 'mpii'


class DatasetManager():
    def __init__(self):
        self.datasets = {}

    def create_datasets(self):
        self.datasets[Datasets.AIC.value] = self._create_dataset(aic_dataset_info)
        self.datasets[Datasets.COCO.value] = self._create_dataset(coco_dataset_info)
        self.datasets[Datasets.CROWDPOSE.value] = self._create_dataset(crowdpose_dataset_info)
        self.datasets[Datasets.MPII.value] = self._create_dataset(mpii_dataset_info)

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
