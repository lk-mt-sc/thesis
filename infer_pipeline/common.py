import os

import matplotlib

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))

THESIS_DIR = os.environ['THESIS_DIR']
INFER_PIPELINE_DIR = os.path.join(THESIS_DIR, 'infer_pipeline')
INFER_PIPELINE_MMDETECTION_CONFIGS_DIR = os.path.join(INFER_PIPELINE_DIR, 'configs', 'mmdet')
INFER_PIPELINE_MMDPOSE_CONFIGS_DIR = os.path.join(INFER_PIPELINE_DIR, 'configs', 'mmpose')
INFERENCES_DIR = os.path.join(INFER_PIPELINE_DIR, 'inferences')

TRAIN_PIPELINE_DIR = os.path.join(THESIS_DIR, 'train_pipeline')
TRAIN_PIPELINE_MMDETECTION_DIR = os.path.join(TRAIN_PIPELINE_DIR, 'mmdet')
TRAIN_PIPELINE_MMDETECTION_CONFIGS_DIR = os.path.join(TRAIN_PIPELINE_MMDETECTION_DIR, 'configs')
TRAIN_PIPELINE_MMDETECTION_TRAININGS_DIR = os.path.join(TRAIN_PIPELINE_MMDETECTION_DIR, 'trainings')

MMPOSE_DIR = os.environ['MMPOSE_DIR']
MMPOSE_DATA_DIR = os.path.join(MMPOSE_DIR, 'data', 'sc')
MMPOSE_RUNS_DIR = os.path.join(MMPOSE_DATA_DIR, 'runs')
MMPOSE_DATASET_DIR = os.path.join(MMPOSE_DATA_DIR, 'dataset')
MMPOSE_CHECKPOINTS_DIR = os.path.join(MMPOSE_DIR, 'checkpoints')
MMPOSE_TEST_SCRIPT = os.path.join(MMPOSE_DIR, 'tools', 'test.py')

MMDETECTION_DIR = os.environ['MMDETECTION_DIR']
MMDETECTION_DATA_DIR = os.path.join(MMDETECTION_DIR, 'data', 'sc')
MMDETECTION_CHECKPOINTS_DIR = os.path.join(MMDETECTION_DIR, 'checkpoints')
MMDETECTION_TEST_SCRIPT = os.path.join(MMDETECTION_DIR, 'tools', 'test.py')

BACKGROUND_COLOR_HEX = '#DCDAD5'
BACKGROUND_COLOR_RGB = matplotlib.colors.to_rgb(BACKGROUND_COLOR_HEX)

WSL_PREFIX = '\\wsl.localhost\\Ubuntu-22.04'
