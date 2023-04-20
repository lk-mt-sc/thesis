import os

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
MMPOSE_DIR = os.environ['MMPOSE_DIR']
MMDETECTION_DIR = os.environ['MMDETECTION_DIR']

CHECKPOINTS_DIR = os.path.join(MMPOSE_DIR, 'checkpoints')
DATA_DIR = os.path.join(MMPOSE_DIR, 'data', 'sc')
INFERENCES_DIR = os.path.join(WORKING_DIR, 'infer')
