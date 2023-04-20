import tkinter as tk
from tkinter import ttk
from tkinter import font

import matplotlib

from gui.gui_mmpose_model import GUIMMPoseModel
from gui.gui_mmdetection_model import GUIMMDetectionModel
from gui.gui_data import GUIData
from gui.gui_infer import GUIInfer
from gui.gui_inference import GUIInference
from gui.gui_feature import GUIFeature
from gui.gui_plot import GUIPlot
from gui.gui_metric import GUIMetric

from manager.status_manager import StatusManager


class Pipeline():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Pipeline')
        self.root.attributes('-zoomed', True)
        self.root.bind('<KeyPress>', self.on_key_press)

        self.root.style = ttk.Style()
        self.root.style.theme_use('clam')

        self.root.background_color_hex = '#DCDAD5'
        self.root.background_color_rgb = matplotlib.colors.to_rgb(self.root.background_color_hex)

        self.root.font = font.nametofont('TkDefaultFont')
        self.root.font.configure(family='Noto', size=8)
        self.root.font_title = self.root.font.copy()
        self.root.font_title.configure(weight='bold', size=10)
        self.root.font_status = self.root.font.copy()
        self.root.font_status.configure(slant='italic', size=10)
        self.root.font_small = self.root.font.copy()
        self.root.font_small.configure(size=7)
        self.root.option_add('*TCombobox*Listbox.font', self.root.font_small)

        self.root.style.configure('Compare.TButton', font=self.root.font_small, padding=(0, 0))
        self.root.style.configure('Delete.TButton', font=self.root.font_small, padding=(0, 0))
        self.root.style.configure('Refresh.TButton', font=self.root.font_small, padding=(0, 0))

        self.gui_mmpose_model = GUIMMPoseModel(self.root, button_refresh_callback=None, listbox_models_callback=None)
        self.gui_detection_model = GUIMMDetectionModel(self.root, listbox_models_callback=None)
        self.gui_data = GUIData(self.root, button_refresh_callback=None, listbox_data_callback=None)
        self.gui_infer = GUIInfer(self.root, button_infer_callback=None)
        self.gui_inference = GUIInference(
            self.root,
            button_compare_callback=None,
            button_delete_callback=None,
            button_refresh_callback=None,
            listbox_inferences_callback=None,
            listbox_data_callback=None
        )
        self.gui_feature = GUIFeature(self.root, listbox_feature_callback=None)
        self.gui_plot = GUIPlot(self.root)
        self.gui_metric = GUIMetric(self.root)

        ttk.Separator(self.root, orient='horizontal').place(x=10, y=576, width=460)
        ttk.Separator(self.root, orient='horizontal').place(x=10, y=864, width=460)
        ttk.Separator(self.root, orient='horizontal').place(x=490, y=864, width=460)
        ttk.Separator(self.root, orient='horizontal').place(x=970, y=1152, width=2445)

        ttk.Separator(self.root, orient='vertical').place(x=480, y=10, height=1350)
        ttk.Separator(self.root, orient='vertical').place(x=960, y=10, height=1350)

        self.status_manager = StatusManager(self.root)

    def on_key_press(self, event):
        key = event.keysym
        if key == 'q':
            self.root.destroy()

    def mainloop(self):
        self.root.mainloop()


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.mainloop()
