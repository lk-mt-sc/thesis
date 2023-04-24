import os

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
INFERENCES_DIR = os.path.join(WORKING_DIR, 'inferences')

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
MMDETECTION_IMAGES2COCO_SCRIPT = os.path.join(MMDETECTION_DIR, 'tools', 'dataset_converters', 'images2coco.py')
