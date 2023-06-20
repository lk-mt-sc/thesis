from enum import Enum

from gui.gui_status import GUIStatus


class Status(Enum):
    IDLE = 'Idle'
    FETCHING_DATA = 'Fetching data'
    FETCHING_MMPOSE_MODELS = 'Fetching MMPose models'
    FETCHING_MMDETECTION_MODELS = 'Fetching MMDetection models'
    FETCHING_INFERENCES = 'Fetching inferences'
    INFERING = 'Infering'
    LOADING_IMAGE = 'Loading image'
    LOADING_VISUALISATION = 'Loading visualisation'


class StatusManager():
    def __init__(self, root):
        self.gui_status = GUIStatus(root)
        self.status = []
        self._gui_set_status()

    def add_status(self, status):
        self.status.append(status)
        self._gui_set_status()

    def remove_status(self, status):
        self.status.remove(status)
        self._gui_set_status()

    def has_status(self, status):
        return status in self.status

    def _gui_set_status(self):
        if not self.status:
            self.status.append(Status.IDLE)

        if len(self.status) > 1 and Status.IDLE in self.status:
            self.status.remove(Status.IDLE)

        status_str = [status.value for status in self.status]
        self.gui_status.status_var.set(' | '.join(status_str))
