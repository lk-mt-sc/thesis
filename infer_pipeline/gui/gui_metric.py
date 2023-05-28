import tkinter as tk
from tkinter import ttk


class GUIMetric():
    def __init__(
        self,
        root,
        calculable_metrics,
        listbox_metrics_select_callback,
        listbox_metrics_drag_callback,
        listbox_metrics_drop_callback,
        button_calculate_callback
    ):
        self.root = root

        self.frame = ttk.Frame(self.root, padding=(10, 10))
        self.frame.place(x=960, y=1152, width=2480, height=288)

        self.title_left = ttk.Label(self.frame, text='Metrics on Inferences in Comparison', font=self.root.font_title)
        self.title_left.place(x=0, y=0)

        ttk.Label(self.frame, text='Standard Metrics', font=self.root.font_bold).place(x=0, y=30)
        ttk.Label(self.frame, text='Missing Pose Estimations').place(x=0, y=50)
        ttk.Label(self.frame, text='Outlier @XY').place(x=0, y=70)
        ttk.Label(self.frame, text='Avg. Error Low-Pass @X Hz').place(x=0, y=90)
        ttk.Label(self.frame, text='Avg. Error High-Pass @X Hz').place(x=0, y=110)
        ttk.Label(self.frame, text='...').place(x=0, y=130)
        ttk.Label(self.frame, text='...').place(x=0, y=150)
        ttk.Label(self.frame, text='...').place(x=0, y=170)

        self.inference_1_title_var = tk.StringVar()
        self.inference_1_title_var.set('No Inference Selected')
        ttk.Label(self.frame, textvariable=self.inference_1_title_var, font=self.root.font_bold).place(x=180, y=30)

        self.inference_2_title_var = tk.StringVar()
        self.inference_2_title_var.set('No Inference Selected')
        ttk.Label(self.frame, textvariable=self.inference_2_title_var, font=self.root.font_bold).place(x=360, y=30)

        self.inference_3_title_var = tk.StringVar()
        self.inference_3_title_var.set('No Inference Selected')
        ttk.Label(self.frame, textvariable=self.inference_3_title_var, font=self.root.font_bold).place(x=530, y=30)

        ttk.Separator(self.frame, orient='vertical').place(x=679, y=0, height=198)

        self.title_right = ttk.Label(self.frame, text='Metrics on Selected Data', font=self.root.font_title)
        self.title_right.place(x=689, y=0)

        ttk.Label(self.frame, text='Calculated Metrics', font=self.root.font_bold).place(x=689, y=30)

        self.listbox_metrics_var = tk.Variable()
        self.listbox_metrics = tk.Listbox(
            self.frame,
            height=10,
            listvariable=self.listbox_metrics_var,
            selectmode=tk.SINGLE
        )
        self.listbox_metrics.configure(exportselection=False)
        self.listbox_metrics.bind('<<ListboxSelect>>', listbox_metrics_select_callback)
        self.listbox_metrics.bind('<B1-Motion>', listbox_metrics_drag_callback)
        self.listbox_metrics.bind('<ButtonRelease-1>', listbox_metrics_drop_callback)
        self.listbox_metrics.place(x=689, y=50, width=250)
        self.listbox_metrics_scrollbar = ttk.Scrollbar(self.frame)
        self.listbox_metrics_scrollbar.place(x=939, y=50, width=20, height=142)
        self.listbox_metrics.config(yscrollcommand=self.listbox_metrics_scrollbar.set)
        self.listbox_metrics_scrollbar.config(command=self.listbox_metrics.yview)

        ttk.Label(self.frame, text='Type', font=self.root.font_bold).place(x=989, y=30)
        self.add_label_var = tk.StringVar()
        self.add_label_var.set(calculable_metrics.OUTLIER.value)
        for i, metric in enumerate(calculable_metrics):
            ttk.Radiobutton(self.frame, variable=self.add_label_var, value=metric.value,
                            text=metric.value).place(x=989, y=50 + i * 20)

        ttk.Label(self.frame, text='Properties', font=self.root.font_bold).place(x=1139, y=30)

        ttk.Label(self.frame, text='Name:').place(x=1139, y=50)
        self.name_var = tk.StringVar()
        self.name_var.set('Outlier @XY')
        ttk.Label(self.frame, textvariable=self.name_var).place(x=1209, y=50)

        ttk.Label(self.frame, text='Parameter:').place(x=1139, y=75)
        self.parameter_var = tk.StringVar()
        ttk.Entry(self.frame, textvariable=self.parameter_var).place(x=1209, y=73, width=100, height=20)

        self.button_calculate = ttk.Button(
            self.frame,
            text='Calculate Metric',
            style='Button.TButton',
            width=15,
            command=button_calculate_callback)
        self.button_calculate.place(x=989, y=115, height=25)
