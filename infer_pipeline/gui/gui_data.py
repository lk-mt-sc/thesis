import tkinter as tk
from tkinter import ttk


class GUIData():
    def __init__(
            self,
            root,
            combobox_data_callback,
            button_select_all_callback,
            button_refresh_callback,
            listbox_data_select_callback,
            listbox_data_double_click_callback
    ):
        self.root = root

        self.frame = ttk.Frame(self.root, padding=(10, 10))
        self.frame.place(x=0, y=864, width=480, height=144)

        self.title = ttk.Label(self.frame, text='Data', font=self.root.font_title)
        self.title.place(x=0, y=0)

        self.combobox_data_base = ttk.Combobox(self.frame, width=12, font=self.root.font_small)
        self.combobox_data_base['values'] = ['']
        self.combobox_data_base['state'] = 'readonly'
        self.combobox_data_base.bind('<<ComboboxSelected>>', combobox_data_callback)
        self.combobox_data_base.place(x=50, y=1)

        self.combobox_data_spotlight = ttk.Combobox(self.frame, width=12, font=self.root.font_small)
        self.combobox_data_spotlight['values'] = ['']
        self.combobox_data_spotlight['state'] = 'readonly'
        self.combobox_data_spotlight.bind('<<ComboboxSelected>>', combobox_data_callback)
        self.combobox_data_spotlight.place(x=150, y=1)

        self.combobox_data_start = ttk.Combobox(self.frame, width=12, font=self.root.font_small)
        self.combobox_data_start['values'] = ['']
        self.combobox_data_start['state'] = 'readonly'
        self.combobox_data_start.bind('<<ComboboxSelected>>', combobox_data_callback)
        self.combobox_data_start.place(x=250, y=1)

        self.button_select_all = ttk.Button(
            self.frame,
            text='Select all',
            style='Button.TButton',
            width=8,
            command=button_select_all_callback)
        self.button_select_all.place(x=350, y=0, height=20)

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
        self.listbox_data.bind('<<ListboxSelect>>', listbox_data_select_callback)
        self.listbox_data.bind('<Double-Button-1>', listbox_data_double_click_callback)
        self.listbox_data.place(x=0, y=30, width=440)
        self.listbox_data_scrollbar = ttk.Scrollbar(self.frame)
        self.listbox_data_scrollbar.place(x=440, y=30, width=20, height=100)
        self.listbox_data.config(yscrollcommand=self.listbox_data_scrollbar.set)
        self.listbox_data_scrollbar.config(command=self.listbox_data.yview)
