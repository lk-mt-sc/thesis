from enum import Enum


class AllMetrics(Enum):
    MISSING_POSE_ESTIMATIONS = 'Missing Pose Estimations'
    DELTAS = 'Deltas'
    PEAKS = 'Peaks'
    LOWPASS = 'Low-Pass (Butterw.)'
    HIGHPASS = 'High-Pass (Butterw.)'
