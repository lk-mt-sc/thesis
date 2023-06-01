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

        self.title_inferences = ttk.Label(
            self.frame,
            text='Metrics on Inferences in Comparison',
            font=self.root.font_title)
        self.title_inferences.place(x=0, y=0)

        n_metrics = 7

        ttk.Label(self.frame, text='Standard Metrics', font=self.root.font_bold).place(x=0, y=30)
        self.inference_metrics_names = []
        self.inference_metrics_values = [[] for _ in range(0, 3)]
        for i in range(0, n_metrics):
            var = tk.StringVar()
            var.set('...')
            self.inference_metrics_names.append(var)
            ttk.Label(self.frame, textvariable=var).place(x=0, y=50 + i * 20)

            var = tk.StringVar()
            self.inference_metrics_values[0].append(var)
            ttk.Label(self.frame, textvariable=var).place(x=180, y=50 + i * 20)

            var = tk.StringVar()
            self.inference_metrics_values[1].append(var)
            ttk.Label(self.frame, textvariable=var).place(x=345, y=50 + i * 20)

            var = tk.StringVar()
            self.inference_metrics_values[2].append(var)
            ttk.Label(self.frame, textvariable=var).place(x=510, y=50 + i * 20)

        self.inference_metrics_titles = []
        for i in range(0, 3):
            var = tk.StringVar()
            var.set('No Inference Selected')
            self.inference_metrics_titles.append(var)
            ttk.Label(self.frame, textvariable=var, font=self.root.font_bold).place(x=180 + i * 165, y=30)

        ttk.Separator(self.frame, orient='vertical').place(x=679, y=0, height=198)

        self.title_data = ttk.Label(self.frame, text='Metrics on Selected Data', font=self.root.font_title)
        self.title_data.place(x=689, y=0)

        ttk.Label(self.frame, text='Standard Metrics', font=self.root.font_bold).place(x=689, y=30)
        self.data_metrics_names = []
        self.data_metrics_values = []
        for i in range(0, n_metrics):
            var = tk.StringVar()
            var.set('...')
            self.data_metrics_names.append(var)
            ttk.Label(self.frame, textvariable=var).place(x=689, y=50 + i * 20)

            var = tk.StringVar()
            self.data_metrics_values.append(var)
            ttk.Label(self.frame, textvariable=var).place(x=859, y=50 + i * 20)

        ttk.Separator(self.frame, orient='vertical').place(x=1039, y=0, height=198)

        self.title_feature = ttk.Label(self.frame, text='Metrics on Selected Feature', font=self.root.font_title)
        self.title_feature.place(x=1049, y=0)

        ttk.Label(self.frame, text='Calculated Metrics', font=self.root.font_bold).place(x=1049, y=30)

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
        self.listbox_metrics.place(x=1049, y=50, width=250)
        self.listbox_metrics_scrollbar = ttk.Scrollbar(self.frame)
        self.listbox_metrics_scrollbar.place(x=1299, y=50, width=20, height=142)
        self.listbox_metrics.config(yscrollcommand=self.listbox_metrics_scrollbar.set)
        self.listbox_metrics_scrollbar.config(command=self.listbox_metrics.yview)

        ttk.Label(self.frame, text='Type', font=self.root.font_bold).place(x=1339, y=30)
        self.add_label_var = tk.StringVar()
        self.add_label_var.set(calculable_metrics.PEAKS.value)
        for i, metric in enumerate(calculable_metrics):
            ttk.Radiobutton(self.frame, variable=self.add_label_var, value=metric.value,
                            text=metric.value).place(x=1339, y=50 + i * 20)

        ttk.Label(self.frame, text='Properties', font=self.root.font_bold).place(x=1489, y=30)

        parameter_rows = 4
        parameter_columns = 2
        self.parameter_name_vars = []
        self.parameter_value_vars = []
        for i in range(parameter_columns):
            for j in range(parameter_rows):
                name_var = tk.StringVar()
                name_var.set(f'Parameter {(i + 1) * (j + 1)}:')
                value_var = tk.StringVar()
                self.parameter_name_vars.append(name_var)
                self.parameter_value_vars.append(value_var)
                ttk.Label(self.frame, textvariable=name_var).place(x=1489 + i * 191, y=50 + j * 25)
                ttk.Entry(self.frame, textvariable=value_var).place(
                    x=1569 + i * 191, y=48 + j * 25, width=100, height=20)

        ttk.Label(self.frame, text='Name:').place(x=1523, y=150)
        self.name_var = tk.StringVar()
        self.name_var.set('')
        ttk.Entry(self.frame, textvariable=self.name_var).place(x=1569, y=150, width=291, height=20)

        self.button_calculate = ttk.Button(
            self.frame,
            text='Calculate Metric',
            style='Button.TButton',
            width=15,
            command=button_calculate_callback)
        self.button_calculate.place(x=1339, y=115, height=25)
        self.button_calculate['state'] = 'disabled'
