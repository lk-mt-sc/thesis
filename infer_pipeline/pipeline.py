import tkinter as tk
from tkinter import ttk
from tkinter import font

import matplotlib

from manager.dataset_manager import DatasetManager
from manager.status_manager import StatusManager
from manager.data_manager import DataManager
from manager.mmpose_model_manager import MMPoseModelManager
from manager.mmdetection_model_manager import MMDetectionModelManager
from manager.inference_manager import InferenceManager
from manager.feature_manager import FeatureManager
from manager.plot_manager import PlotManager
from manager.metric_manager import MetricManager


class Pipeline():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Inference Pipeline')
        self.root.attributes('-zoomed', True)
        self.root.bind('<KeyPress>', self.on_key_press)

        self.root.style = ttk.Style()
        self.root.style.theme_use('clam')

        self.root.background_color_hex = '#DCDAD5'
        self.root.background_color_rgb = matplotlib.colors.to_rgb(self.root.background_color_hex)

        self.root.font = font.nametofont('TkDefaultFont')
        self.root.font.configure(family='Noto', size=8)
        self.root.font_bold = self.root.font.copy()
        self.root.font_bold.configure(weight='bold')
        self.root.font_title = self.root.font.copy()
        self.root.font_title.configure(weight='bold', size=10)
        self.root.font_status = self.root.font.copy()
        self.root.font_status.configure(slant='italic', size=10)
        self.root.font_small = self.root.font.copy()
        self.root.font_small.configure(size=7)
        self.root.option_add('*TCombobox*Listbox.font', self.root.font_small)

        self.root.style.configure('Button.TButton', font=self.root.font_small, padding=(0, 0))

        self.dataset_manager = DatasetManager()
        self.dataset_manager.create_datasets()

        self.status_manager = StatusManager(self.root)
        self.mmpose_model_manager = MMPoseModelManager(self.root, self.status_manager)
        self.mmdetection_model_manager = MMDetectionModelManager(self.root, self.status_manager)
        self.data_manager = DataManager(self.root, self.status_manager)
        self.metric_manager = MetricManager(self.root, self.status_manager)
        self.plot_manager = PlotManager(self.root, self.status_manager, self.metric_manager)
        self.feature_manager = FeatureManager(self.root, self.status_manager, self.metric_manager, self.plot_manager)
        self.inference_manager = InferenceManager(
            self.root,
            self.dataset_manager,
            self.status_manager,
            self.mmpose_model_manager,
            self.mmdetection_model_manager,
            self.data_manager,
            self.metric_manager,
            self.plot_manager,
            self.feature_manager)

        self.metric_manager.plot_manager = self.plot_manager

        ttk.Separator(self.root, orient='horizontal').place(x=10, y=576, width=460)
        ttk.Separator(self.root, orient='horizontal').place(x=10, y=864, width=460)
        ttk.Separator(self.root, orient='horizontal').place(x=490, y=864, width=460)
        ttk.Separator(self.root, orient='horizontal').place(x=970, y=1152, width=2445)
        ttk.Separator(self.root, orient='vertical').place(x=480, y=10, height=1350)
        ttk.Separator(self.root, orient='vertical').place(x=960, y=10, height=1350)

    def on_key_press(self, event):
        key = event.keysym
        if key == 'q':
            self.root.destroy()

    def mainloop(self):
        self.root.mainloop()


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.mainloop()
