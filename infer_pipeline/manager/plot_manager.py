from gui.gui_plot import GUIPlot


class PlotManager():
    def __init__(self, root, status_manager):
        self.gui_plot = GUIPlot(root)
        self.status_manager = status_manager
