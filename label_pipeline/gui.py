import tkinter as tk
from tkinter import ttk
from tkinter import font

import imutils
import cv2 as cv
import PIL.Image
import PIL.ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GUI():
    def __init__(
            self,
            root,
            data_selected_callback,
            prev_button_callback,
            next_button_callback,
            submit_button_callback,
            reset_button_callback,
            feature_plot_clicked_callback,
            calculate_high_pass_button_callback,
            toggle_high_pass_plot_button_callback):
        self.root = root
        self.root.title('Label Pipeline')
        self.root.attributes('-zoomed', True)

        self.root.style = ttk.Style()
        self.root.style.theme_use('clam')
        self.root.style.configure('Button.TButton', padding=(0, 0))

        self.root.background_color = '#DCDAD5'
        self.root.border_color = '#969696'
        self.root.configure(background=self.root.background_color)

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

        self.image_plots = ImagePlots(self.root)
        self.image_plots.clear()

        self.feature_plots = FeaturePlots(self.root, feature_plot_clicked_callback)
        self.feature_plots.clear()

        self.data_list = DataList(self.root, data_selected_callback)
        self.data_list.clear()

        self.control = Control(self.root, prev_button_callback, next_button_callback)
        self.statistics = Statistics(self.root)
        self.labels = Labels(self.root, submit_button_callback, reset_button_callback)
        self.metrics = Metrics(self.root, calculate_high_pass_button_callback, toggle_high_pass_plot_button_callback)


class ImagePlots():
    def __init__(self, root):
        self.root = root
        self.frame = ttk.Frame(self.root)
        self.frame.place(x=0, y=10, width=3440, height=540)

        self.frames = [ttk.Frame(self.frame) for _ in range(0, 7)]
        for i, frame in enumerate(self.frames):
            frame.place(x=i * (480 + 10) + 4, y=0, width=480, height=540)

        self.canvas = [tk.Canvas(self.frames[i]) for i in range(0, 7)]
        for canvas in self.canvas:
            canvas.configure(bg='#E5E5E5', highlightbackground=self.root.border_color, highlightthickness=1)
            canvas.place(x=0, y=0, width=480, height=540)
        self.canvas[3].configure(highlightbackground='black', highlightthickness=3)

        self.images = [None for _ in range(0, 7)]

    def set_images(self, images):
        assert len(images) == len(self.canvas)
        for i, image in enumerate(images):
            if image is None:
                image = np.full((540, 480, 3), (229, 229, 229), dtype=np.uint8)

            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            h, w = image.shape[0], image.shape[1]
            if h > w:
                image = imutils.resize(image, height=540)
            if w > h:
                image = imutils.resize(image, width=480)
            self.images[i] = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
            self.canvas[i].create_image(240, 270, image=self.images[i], anchor=tk.CENTER)

    def clear(self):
        images = [np.full((540, 480, 3), (229, 229, 229), dtype=np.uint8) for _ in range(0, 7)]
        self.set_images(images)


class FeaturePlots():
    def __init__(self, root, feature_plot_clicked_callback):
        self.root = root
        self.frame = ttk.Frame(self.root)
        self.frame.place(x=1474, y=560, width=1944, height=797)
        self.feature_plot_clicked_callback = feature_plot_clicked_callback

        self.figure, self.axes = plt.subplots(2, 1)
        self.figure.set_facecolor(self.root.background_color)
        self.figure.subplots_adjust(left=0.04, bottom=0.06, right=0.96, top=0.94, hspace=0.4)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().place(x=0, y=0, width=1944, height=797)
        self.canvas.mpl_connect('button_press_event', self.plot_clicked)

        self.tracker_x = None
        self.tracker_y = None
        self.labels_x = []
        self.labels_y = []
        self.candidates_x = []
        self.candidates_y = []
        self.twin_x = self.axes[0].twinx()
        self.twin_y = self.axes[1].twinx()

        self.canvas.draw()

    def set_features(self, labeled_data):
        steps_x = labeled_data.feature_x.steps
        steps_y = labeled_data.feature_y.steps

        values_x = labeled_data.feature_x.values
        values_y = labeled_data.feature_y.values

        scores = labeled_data.scores

        self.twin_x.plot(steps_x, scores, color='red',
                         linestyle='dashed', linewidth=1.0, label='Confidence')
        self.twin_y.plot(steps_y, scores, color='red',
                         linestyle='dashed', linewidth=1.0, label='Confidence')

        self.twin_x.set_ylim(0, 1.1)
        self.twin_y.set_ylim(0, 1.1)

        self.twin_x.set_yticks([x / 10.0 for x in range(0, 11, 1)])
        self.twin_y.set_yticks([x / 10.0 for x in range(0, 11, 1)])

        self.twin_x.set_ylabel('Confidence/Normalized Metric')
        self.twin_y.set_ylabel('Confidence/Normalized Metric')

        self.twin_x.yaxis.set_label_position("right")
        self.twin_y.yaxis.set_label_position("right")

        self.axes[0].plot(steps_x, values_x, linewidth=1.5, label='Feature')
        self.axes[1].plot(steps_y, values_y, linewidth=1.5, label='Feature')

        self.axes[0].set_xlim(-25, ((max(steps_x) // 10) + 1) * 10)
        self.axes[1].set_xlim(-25, ((max(steps_y) // 10) + 1) * 10)

        self.axes[0].set_ylim(0, 1100)
        self.axes[1].set_ylim(0, 1100)

        self.axes[0].set_xticks(range(-10, ((max(steps_x) // 10) + 1) * 10 + 1, 10))
        self.axes[1].set_xticks(range(-10, ((max(steps_y) // 10) + 1) * 10 + 1, 10))

        self.axes[0].set_yticks(range(0, 1101, 100))
        self.axes[1].set_yticks(range(0, 1101, 100))

        self.axes[0].set_xlabel('Image')
        self.axes[1].set_xlabel('Image')

        self.axes[0].set_ylabel('Pixel')
        self.axes[1].set_ylabel('Pixel')

        self.axes[0].set_title(labeled_data.title)
        self.axes[1].set_title(labeled_data.title)

        self.axes[0].grid(True)
        self.axes[1].grid(True)

        self.tracker_x = self.axes[0].axvline(x=0, color='black')
        self.tracker_y = self.axes[1].axvline(x=0, color='black')

        self.labels_x = [Line2D([0], [0], linestyle='', marker='o', color='blue') for _ in range(0, len(steps_x))]
        self.labels_y = [Line2D([0], [0], linestyle='', marker='o', color='blue') for _ in range(0, len(steps_y))]
        for label in self.labels_x:
            label.set_visible(False)
            label.set_markerfacecolor('none')
            self.axes[0].add_line(label)
        for label in self.labels_y:
            label.set_visible(False)
            label.set_markerfacecolor('none')
            self.axes[1].add_line(label)

        for candidate in labeled_data.candidates_x:
            self.candidates_x.append(Line2D([candidate], [values_x[candidate]],
                                     marker='o', color='black', markersize=2))

        for candidate in labeled_data.candidates_y:
            self.candidates_y.append(Line2D([candidate], [values_y[candidate]],
                                     marker='o', color='black', markersize=2))

        for candidate in self.candidates_x:
            self.axes[0].add_line(candidate)

        for candidate in self.candidates_y:
            self.axes[1].add_line(candidate)

        self.labels_x = [Line2D([0], [0], linestyle='', marker='o', color='blue')for _ in range(0, len(steps_x))]
        self.labels_y = [Line2D([0], [0], linestyle='', marker='o', color='blue')for _ in range(0, len(steps_y))]
        for label in self.labels_x:
            label.set_visible(False)
            label.set_markerfacecolor('none')
            self.axes[0].add_line(label)
        for label in self.labels_y:
            label.set_visible(False)
            label.set_markerfacecolor('none')
            self.axes[1].add_line(label)

        handles, labels = [], []
        for axis in [self.axes[0], self.twin_x]:
            h, l = axis.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        handles.extend([Line2D([0], [0], marker='o', markeredgecolor='black', label='Candidate', markerfacecolor='black'),
                        Line2D([0], [0], marker='o', markeredgecolor='green', label='Correct', markerfacecolor='none'),
                        Line2D([0], [0], marker='o', markeredgecolor='red', label='Incorrect', markerfacecolor='none'),
                        Line2D([0], [0], marker='o', markeredgecolor='darkorange', label='Incorrect\n(Hidden)', markerfacecolor='none')])
        labels.extend(['Candidate', 'Correct', 'Incorrect', 'Incorrect\n(Hidden)'])
        self.axes[0].legend(handles, labels, loc='upper left')

        handles, labels = [], []
        for axis in [self.axes[1], self.twin_y]:
            h, l = axis.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        handles.extend([Line2D([0], [0], marker='o', markeredgecolor='black', label='Candidate', markerfacecolor='black'),
                        Line2D([0], [0], marker='o', markeredgecolor='green', label='Correct', markerfacecolor='none'),
                        Line2D([0], [0], marker='o', markeredgecolor='red', label='Incorrect', markerfacecolor='none'),
                        Line2D([0], [0], marker='o', markeredgecolor='darkorange', label='Incorrect\n(Hidden)', markerfacecolor='none')])
        labels.extend(['Candidate', 'Correct', 'Incorrect', 'Incorrect\n(Hidden)'])

        self.axes[1].legend(handles, labels, loc='upper left')

        self.canvas.draw()

    def set_tracker(self, step):
        self.tracker_x.set_xdata([step])
        self.tracker_y.set_xdata([step])
        self.canvas.draw()

    def set_labels(self, labeled_data):
        feature_x_values = labeled_data.feature_x.values
        feature_y_values = labeled_data.feature_y.values

        for i, labels in enumerate(labeled_data.labels):
            label = labels[0]

            if label == -1:
                continue

            self.labels_x[i].set_xdata([i])
            self.labels_x[i].set_ydata(feature_x_values[i])
            self.labels_x[i].set_visible(True)

            self.labels_y[i].set_xdata([i])
            self.labels_y[i].set_ydata(feature_y_values[i])
            self.labels_y[i].set_visible(True)

            if label == 0:
                self.labels_x[i].set_color('green')
                self.labels_y[i].set_color('green')
            if label == 1:
                self.labels_x[i].set_color('red')
                self.labels_y[i].set_color('red')
            if label == 2:
                self.labels_x[i].set_color('darkorange')
                self.labels_y[i].set_color('darkorange')

        self.canvas.draw()

    def plot_clicked(self, event=None):
        if event.inaxes:
            image_nr = int(round(event.xdata))
            self.feature_plot_clicked_callback(image_nr=image_nr)

    def clear(self):
        self.twin_x.cla()
        self.twin_y.cla()
        self.axes[0].cla()
        self.axes[1].cla()
        self.labels_x.clear()
        self.labels_y.clear()
        self.candidates_x.clear()
        self.candidates_y.clear()
        self.canvas.draw()


class DataList():
    def __init__(self, root, data_selected_callback):
        self.root = root
        self.frame = ttk.Frame(self.root)
        self.frame.place(x=5, y=560, width=478, height=750)

        self.canvas = tk.Canvas(self.frame, background=self.root.background_color,
                                highlightbackground=self.root.border_color, highlightthickness=1)
        self.canvas.place(x=0, y=0, relwidth=1, relheight=1)

        ttk.Label(self.frame, text='Data', font=self.root.font_title).place(x=10, y=10)

        self.listbox_data_var = tk.Variable()
        self.data_listbox = tk.Listbox(
            self.frame,
            height=53,
            listvariable=self.listbox_data_var,
            selectmode=tk.SINGLE,
            font=self.root.font_small
        )
        self.data_listbox.configure(exportselection=False)
        self.data_listbox.bind('<<ListboxSelect>>', data_selected_callback)
        self.data_listbox.place(x=10, y=40, width=440)
        self.listbox_data_scrollbar = ttk.Scrollbar(self.frame)
        self.listbox_data_scrollbar.place(x=450, y=40, width=20, height=692)
        self.data_listbox.config(yscrollcommand=self.listbox_data_scrollbar.set)
        self.listbox_data_scrollbar.config(command=self.data_listbox.yview)

    def set_data(self, data):
        self.data_listbox.delete(0, tk.END)
        for d in data:
            self.data_listbox.insert(tk.END, d)

    def clear(self):
        self.data_listbox.delete(0, tk.END)


class Control():
    def __init__(self, root, prev_button_callback, next_button_callback):
        self.root = root
        self.frame = ttk.Frame(self.root)
        self.frame.place(x=5, y=1320, width=478, height=40)

        self.counter_var = tk.StringVar()
        self.counter_var.set('0000/0000')
        ttk.Label(self.frame, textvariable=self.counter_var).place(x=10, y=12)

        self.prev_button = ttk.Button(self.frame, text='<--', command=prev_button_callback, style='Button.TButton')
        self.prev_button.place(x=90, y=11, width=50, height=20)
        self.next_button = ttk.Button(self.frame, text='-->', command=next_button_callback, style='Button.TButton')
        self.next_button.place(x=145, y=11, width=50, height=20)

        self.prev_button.configure(state=tk.DISABLED)
        self.next_button.configure(state=tk.DISABLED)


class Statistics():
    def __init__(self, root):
        self.root = root
        self.frame = ttk.Frame(self.root)
        self.frame.place(x=495, y=560, width=478, height=800)

        self.canvas = tk.Canvas(self.frame, background=self.root.background_color,
                                highlightbackground=self.root.border_color, highlightthickness=1)
        self.canvas.place(x=0, y=0, relwidth=1, relheight=1)

        ttk.Label(self.frame, text='Statistics', font=self.root.font_title).place(x=10, y=10)

        ttk.Label(self.frame, text='Statistic', font=self.root.font_bold).place(x=10, y=40)
        ttk.Label(self.frame, text='Overall', font=self.root.font_bold).place(x=320, y=40)
        ttk.Label(self.frame, text='Current', font=self.root.font_bold).place(x=390, y=40)

        self.statistics = []

    def create_statistics(self, n):
        for i in range(n):
            var_0 = tk.StringVar()
            var_1 = tk.StringVar()
            var_2 = tk.StringVar()
            ttk.Label(self.frame, textvariable=var_0).place(x=10, y=60 + i * 20)
            ttk.Label(self.frame, textvariable=var_1).place(x=320, y=60 + i * 20)
            ttk.Label(self.frame, textvariable=var_2).place(x=390, y=60 + i * 20)
            self.statistics.append([var_0, var_1, var_2])
        return self.statistics


class Labels():
    def __init__(self, root, submit_button_callback, reset_button_callback):
        self.root = root
        self.frame = ttk.Frame(self.root)
        self.frame.place(x=985, y=560, width=478, height=395)

        self.canvas = tk.Canvas(self.frame, background=self.root.background_color,
                                highlightbackground=self.root.border_color, highlightthickness=1)
        self.canvas.place(x=0, y=0, relwidth=1, relheight=1)

        ttk.Label(self.frame, text='Labels', font=self.root.font_title).place(x=10, y=10)

        self.auto_submit_checkbutton_var = tk.BooleanVar()
        self.auto_submit_checkbutton_var.set(False)
        self.auto_submit_checkbutton = ttk.Checkbutton(
            self.frame,
            text='Auto Submit on Image Change',
            variable=self.auto_submit_checkbutton_var,
            onvalue=True,
            offvalue=False,
        )
        self.auto_submit_checkbutton.place(x=163, y=365, height=20)

        self.submit_button = ttk.Button(self.frame, text='Submit',
                                        command=submit_button_callback, style='Button.TButton')
        self.submit_button.place(x=417, y=364, width=50, height=20)
        self.submit_button.configure(state=tk.DISABLED)

        self.reset_button = ttk.Button(self.frame, text='Reset',
                                       command=reset_button_callback, style='Button.TButton')
        self.reset_button.place(x=360, y=364, width=50, height=20)
        self.reset_button.configure(state=tk.DISABLED)

        ttk.Label(self.frame, text='Label', font=self.root.font_bold).place(x=10, y=40)
        self.label_var = tk.IntVar()
        self.label_var.set(-1)
        self.label_0_radiobutton = ttk.Radiobutton(
            self.frame,
            text='Correct (Keypoint Not Hidden)',
            variable=self.label_var,
            value=0,
            command=self.enable_submit_button
        )
        self.label_0_radiobutton.place(x=10, y=60)
        self.label_0_radiobutton.configure(state=tk.DISABLED)

        self.label_1_radiobutton = ttk.Radiobutton(
            self.frame,
            text='Incorrect (Keypoint Not Hidden)',
            variable=self.label_var,
            value=1,
            command=self.enable_submit_button
        )
        self.label_1_radiobutton.place(x=10, y=80)
        self.label_1_radiobutton.configure(state=tk.DISABLED)

        self.label_2_radiobutton = ttk.Radiobutton(
            self.frame,
            text='Incorrect (Keypoint Hidden)',
            variable=self.label_var,
            value=2,
            command=self.enable_submit_button
        )
        self.label_2_radiobutton.place(x=10, y=100)
        self.label_2_radiobutton.configure(state=tk.DISABLED)

        self.mark_x_var = tk.BooleanVar()
        self.mark_x_var.set(False)
        self.mark_x = ttk.Checkbutton(
            self.frame,
            text='Mark X for Metrics',
            variable=self.mark_x_var,
            onvalue=True,
            offvalue=False,
        )
        self.mark_x.place(x=220, y=60, height=20)
        self.mark_x.configure(state=tk.DISABLED)

        self.mark_y_var = tk.BooleanVar()
        self.mark_y_var.set(False)
        self.mark_y = ttk.Checkbutton(
            self.frame,
            text='Mark Y for Metrics',
            variable=self.mark_y_var,
            onvalue=True,
            offvalue=False,
        )
        self.mark_y.place(x=220, y=80, height=20)
        self.mark_y.configure(state=tk.DISABLED)

        ttk.Label(self.frame, text='Other', font=self.root.font_bold).place(x=10, y=130)
        self.bounding_box_cuts_climber_var = tk.BooleanVar()
        self.bounding_box_cuts_climber_var.set(False)
        self.bounding_box_cuts_climber = ttk.Checkbutton(
            self.frame,
            text='Bounding Box Cuts Climber',
            variable=self.bounding_box_cuts_climber_var,
            onvalue=True,
            offvalue=False,
        )
        self.bounding_box_cuts_climber.place(x=10, y=150, height=20)
        self.bounding_box_cuts_climber.configure(state=tk.DISABLED)

        self.bounding_box_cuts_climber_tv_var = tk.BooleanVar()
        self.bounding_box_cuts_climber_tv_var.set(False)
        self.bounding_box_cuts_climber_tv = ttk.Checkbutton(
            self.frame,
            text='Bounding Box Cuts Climber (due to TV overlay)',
            variable=self.bounding_box_cuts_climber_tv_var,
            onvalue=True,
            offvalue=False,
        )
        self.bounding_box_cuts_climber_tv.place(x=10, y=170, height=20)
        self.bounding_box_cuts_climber_tv.configure(state=tk.DISABLED)

        self.bounding_box_cuts_climber_dark_var = tk.BooleanVar()
        self.bounding_box_cuts_climber_dark_var.set(False)
        self.bounding_box_cuts_climber_dark = ttk.Checkbutton(
            self.frame,
            text='Bounding Box Cuts Climber (due to unilluminated area)',
            variable=self.bounding_box_cuts_climber_dark_var,
            onvalue=True,
            offvalue=False,
        )
        self.bounding_box_cuts_climber_dark.place(x=10, y=190, height=20)
        self.bounding_box_cuts_climber_dark.configure(state=tk.DISABLED)

        self.side_swap_var = tk.BooleanVar()
        self.side_swap_var.set(False)
        self.side_swap = ttk.Checkbutton(
            self.frame,
            text='Side Swap (Each Swap Counted Individually)',
            variable=self.side_swap_var,
            onvalue=True,
            offvalue=False,
        )
        self.side_swap.place(x=10, y=210, height=20)
        self.side_swap.configure(state=tk.DISABLED)

        ttk.Label(self.frame, text="* hidden Keypoints due to: \n- behind climber's body \n- behind TV overlay \n- in unilluminated area").place(x=10, y=330)

    def enable_submit_button(self):
        self.submit_button.configure(state=tk.NORMAL)


class Metrics():
    def __init__(self, root, calculate_high_pass_button_callback, toggle_high_pass_plot_button_callback):
        self.root = root
        self.frame = ttk.Frame(self.root)
        self.frame.place(x=985, y=965, width=478, height=395)

        self.canvas = tk.Canvas(self.frame, background=self.root.background_color,
                                highlightbackground=self.root.border_color, highlightthickness=1)
        self.canvas.place(x=0, y=0, relwidth=1, relheight=1)

        ttk.Label(self.frame, text='Metrics', font=self.root.font_title).place(x=10, y=10)

        ttk.Label(self.frame, text='High-Pass', font=self.root.font_bold).place(x=10, y=40)

        self.high_pass_sampling_frequency_var = tk.StringVar()
        self.high_pass_sampling_frequency_var.set('Sampling Freq.: 25 Hz')
        ttk.Label(self.frame, textvariable=self.high_pass_sampling_frequency_var).place(x=10, y=60)

        self.high_pass_order_var = tk.StringVar()
        self.high_pass_order_var.set('Order: 10')
        ttk.Label(self.frame, textvariable=self.high_pass_order_var).place(x=160, y=60)

        self.high_pass_cutoff_frequency_var = tk.StringVar()
        self.high_pass_cutoff_frequency_var.set('Cutoff Freq.: 10 Hz')
        ttk.Label(self.frame, textvariable=self.high_pass_cutoff_frequency_var).place(x=250, y=60)

        self.calculate_high_pass_button = ttk.Button(
            self.frame,
            text='Calculate High Pass Properties',
            command=calculate_high_pass_button_callback,
            style='Button.TButton'
        )
        self.calculate_high_pass_button.place(x=10, y=80, width=180, height=20)

        self.toggle_high_pass_plot_button = ttk.Button(
            self.frame,
            text='Toggle High Pass Plot',
            command=toggle_high_pass_plot_button_callback,
            style='Button.TButton'
        )
        self.toggle_high_pass_plot_button.place(x=200, y=80, width=180, height=20)
