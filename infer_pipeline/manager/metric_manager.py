from gui.gui_metric import GUIMetric


class MetricManager():
    def __init__(self, root, status_manager):
        self.gui_metric = GUIMetric(root)
        self.status_manager = status_manager
