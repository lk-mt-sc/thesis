from enum import Enum


class AllMetrics(Enum):
    MISSING_POSE_ESTIMATIONS = 'Missing Pose Estimations'
    DELTAS = 'Deltas'
    PEAKS = 'Peaks'
    LOWPASS = 'Low-Pass'
    HIGHPASS = 'High-Pass'
    FFT = 'FFT'
    INSTANTANEOUS_FREQUENCY = 'Instantaneous Frequency'
