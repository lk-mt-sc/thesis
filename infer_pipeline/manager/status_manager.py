from enum import Enum

from gui.gui_status import GUIStatus


class Status(Enum):
    IDLE = 'Idle'
    FETCHING_DATA = 'Fetching data'
    FETCHING_MODELS = 'Fetching models'
    FETCHING_INFERENCES = 'Fetching inferences'
    INFERING = 'Infering'
    LOADING_IMAGE = 'Loading image'


class StatusManager():
    def __init__(self, root):
        self.gui_status = GUIStatus(root)
        self.status = []
        self._gui_set_status()
        self._gui_set_progress()

    def add_status(self, status):
        self.status.append(status)
        self._gui_set_status()

    def remove_status(self, status):
        self.status.remove(status)
        self._gui_set_status()

    def has_status(self, status):
        return status in self.status

    def update_progress(self, value):
        self._gui_set_progress(value)

    def _gui_set_status(self):
        if not self.status:
            self.status.append(Status.IDLE)

        if len(self.status) > 1 and Status.IDLE in self.status:
            self.status.remove(Status.IDLE)

        status_str = [status.value for status in self.status]
        self.gui_status.status_var.set(' | '.join(status_str))

    def _gui_set_progress(self, value=0):
        self.gui_status.progressbar['value'] = value
