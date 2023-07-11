from enum import Enum


class KeyPointsImagePlot(Enum):
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


class Datasets(Enum):
    COCO = 'coco'


class Dataset():
    def __init__(self, name, keypoints, skeleton):
        self.name = name
        self.keypoints = keypoints
        self.skeleton = skeleton

    def __str__(self):
        return self.name


class DatasetManager():
    def __init__(self):
        self.datasets = {}

    def create_datasets(self):
        coco_dataset_info = dict(
            dataset_name='coco',
            paper_info=dict(
                author='Lin, Tsung-Yi and Maire, Michael and '
                'Belongie, Serge and Hays, James and '
                'Perona, Pietro and Ramanan, Deva and '
                r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
                title='Microsoft coco: Common objects in context',
                container='European conference on computer vision',
                year='2014',
                homepage='http://cocodataset.org/',
            ),
            keypoint_info={
                0:
                dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
                1:
                dict(
                    name='left_eye',
                    id=1,
                    color=[51, 153, 255],
                    type='upper',
                    swap='right_eye'),
                2:
                dict(
                    name='right_eye',
                    id=2,
                    color=[51, 153, 255],
                    type='upper',
                    swap='left_eye'),
                3:
                dict(
                    name='left_ear',
                    id=3,
                    color=[51, 153, 255],
                    type='upper',
                    swap='right_ear'),
                4:
                dict(
                    name='right_ear',
                    id=4,
                    color=[51, 153, 255],
                    type='upper',
                    swap='left_ear'),
                5:
                dict(
                    name='left_shoulder',
                    id=5,
                    color=[0, 255, 0],
                    type='upper',
                    swap='right_shoulder'),
                6:
                dict(
                    name='right_shoulder',
                    id=6,
                    color=[255, 128, 0],
                    type='upper',
                    swap='left_shoulder'),
                7:
                dict(
                    name='left_elbow',
                    id=7,
                    color=[0, 255, 0],
                    type='upper',
                    swap='right_elbow'),
                8:
                dict(
                    name='right_elbow',
                    id=8,
                    color=[255, 128, 0],
                    type='upper',
                    swap='left_elbow'),
                9:
                dict(
                    name='left_wrist',
                    id=9,
                    color=[0, 255, 0],
                    type='upper',
                    swap='right_wrist'),
                10:
                dict(
                    name='right_wrist',
                    id=10,
                    color=[255, 128, 0],
                    type='upper',
                    swap='left_wrist'),
                11:
                dict(
                    name='left_hip',
                    id=11,
                    color=[0, 255, 0],
                    type='lower',
                    swap='right_hip'),
                12:
                dict(
                    name='right_hip',
                    id=12,
                    color=[255, 128, 0],
                    type='lower',
                    swap='left_hip'),
                13:
                dict(
                    name='left_knee',
                    id=13,
                    color=[0, 255, 0],
                    type='lower',
                    swap='right_knee'),
                14:
                dict(
                    name='right_knee',
                    id=14,
                    color=[255, 128, 0],
                    type='lower',
                    swap='left_knee'),
                15:
                dict(
                    name='left_ankle',
                    id=15,
                    color=[0, 255, 0],
                    type='lower',
                    swap='right_ankle'),
                16:
                dict(
                    name='right_ankle',
                    id=16,
                    color=[255, 128, 0],
                    type='lower',
                    swap='left_ankle'),
                17:
                dict(
                    name='neck',
                    id=17,
                    color=[51, 153, 255],
                    type='upper',
                    swap=''),
                18:
                dict(
                    name='head',
                    id=18,
                    color=[51, 153, 255],
                    type='upper',
                    swap='')
            },
            skeleton_info={
                0:
                dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
                1:
                dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
                2:
                dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
                3:
                dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
                4:
                dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
                5:
                dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
                6:
                dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
                7:
                dict(
                    link=('left_shoulder', 'right_shoulder'),
                    id=7,
                    color=[51, 153, 255]),
                8:
                dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
                9:
                dict(
                    link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
                10:
                dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
                11:
                dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
                12:
                dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
                13:
                dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
                14:
                dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
                15:
                dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
                16:
                dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
                17:
                dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
                18:
                dict(
                    link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255]),
                19:
                dict(
                    link=('left_shoulder', 'neck'), id=19, color=[51, 153, 255]),
                20:
                dict(
                    link=('right_shoulder', 'neck'), id=20, color=[51, 153, 255]),
                21:
                dict(
                    link=('neck', 'head'), id=21, color=[51, 153, 255]),
            },
            joint_weights=[
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
                1.5
            ],
            sigmas=[
                0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
                0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
            ])

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
