import os

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
INFERENCES_DIR = os.path.join(WORKING_DIR, 'inferences')

MMPOSE_DIR = os.environ['MMPOSE_DIR']
MMPOSE_DATA_DIR = os.path.join(MMPOSE_DIR, 'data', 'sc')
MMPOSE_CHECKPOINTS_DIR = os.path.join(MMPOSE_DIR, 'checkpoints')

MMDETECTION_DIR = os.environ['MMDETECTION_DIR']
MMDETECTION_DATA_DIR = os.path.join(MMDETECTION_DIR, 'data', 'sc')
MMDETECTION_CHECKPOINTS_DIR = os.path.join(MMDETECTION_DIR, 'checkpoints')
