import tkinter as tk
from tkinter import ttk


class GUIData():
    def __init__(self, root, button_refresh_callback, listbox_data_callback):
        self.root = root

        self.frame = ttk.Frame(self.root, padding=(10, 10))
        self.frame.place(x=0, y=864, width=480, height=144)

        self.title = ttk.Label(self.frame, text='Data', font=self.root.font_title)
        self.title.place(x=0, y=0)

        self.combobox_dataset = ttk.Combobox(self.frame, width=10, font=self.root.font_small)
        self.combobox_dataset['values'] = ['No preset', 'Dark', 'Deblurred', 'Interpolated']
        self.combobox_dataset['state'] = 'readonly'
        self.combobox_dataset.current(0)
        self.combobox_dataset.bind('<<ComboboxSelected>>', None)
        self.combobox_dataset.place(x=320, y=1)

        self.button_refresh = ttk.Button(
            self.frame,
            text='Refresh',
            style='Button.TButton',
            width=7,
            command=button_refresh_callback)
        self.button_refresh.place(x=410, y=0, height=20)

        self.listbox_data_var = tk.Variable()
        self.listbox_data = tk.Listbox(
            self.frame,
            height=7,
            listvariable=self.listbox_data_var,
            selectmode=tk.EXTENDED
        )
        self.listbox_data.configure(exportselection=False)
        self.listbox_data.bind('<<ListboxSelect>>', listbox_data_callback)
        self.listbox_data.place(x=0, y=30, width=440)
        self.listbox_data_scrollbar = ttk.Scrollbar(self.frame)
        self.listbox_data_scrollbar.place(x=440, y=30, width=20, height=100)
        self.listbox_data.config(yscrollcommand=self.listbox_data_scrollbar.set)
        self.listbox_data_scrollbar.config(command=self.listbox_data.yview)
