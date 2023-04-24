# Copyright (c) OpenMMLab. All rights reserved.
from .mmpose_inferencer import MyMMPoseInferencer
from .pose2d_inferencer import MyPose2DInferencer
from .utils import get_model_aliases

__all__ = ['MyPose2DInferencer', 'MyMMPoseInferencer', 'get_model_aliases']
