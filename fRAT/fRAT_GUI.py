try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import ast
import re
import os
import sys
import logging
import xmltodict
from pathlib import Path
from PIL import ImageTk, Image

from fRAT import fRAT
from statmap import main as statmap_calc
from printResults import printResults
import dash_report
from utils import *
from utils.fRAT_config_setup import *
from utils.statmap_config_setup import *

w = None
VERSION = "0.15.0"


def start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    Config_GUI(root)

    root.mainloop()


def create_Config_GUI(rt):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_Config_GUI(root, *args, **kwargs)' .'''
    global w, root
    root = rt

    w = tk.Toplevel(root)
    top = Config_GUI(w)

    return w, top


class Config_GUI:
    def __init__(self, window=None, page='Home', load_initial_values=True):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        self.widgets = {}
        self.dynamic_widgets = {}
        self.frames = []
        self.page = page
        self.background = '#d9d9d9'

        if load_initial_values:
            for toml_file in ['fRAT_config.toml', 'statmap_config.toml']:
                self.load_initial_values(toml_file)

            self.style = ttk.Style()
            self.style.theme_use('clam')

            _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
            _fgcolor = '#000000'  # X11 color: 'black'
            _compcolor = '#d9d9d9'  # X11 color: 'gray85'
            _ana1color = '#d9d9d9'  # X11 color: 'gray85'
            _ana2color = '#ececec'  # Closest X11 color: 'gray92'
            window.geometry("512x780+100+100")
            window.minsize(512, 780)
            window.maxsize(512, 780)
            window.resizable(1, 1)
            window.title("fRAT GUI")
            window.configure(background=self.background)
            window.configure(highlightbackground=self.background)
            window.configure(highlightcolor="black")

        if page == 'Statistical_maps':
            self.current_info = Statistical_maps

        else:
            try:
                self.current_info = eval(self.page)
            except NameError:
                self.current_info = pages

        self.settings_frame_create(window)

        if self.page == 'Home':
            self.banner_draw(window)
            self.Options_frame_draw(self.settings_frame)
            self.Atlas_frame_draw(self.settings_frame)
            self.Run_frame_draw(self.settings_frame)

            self.statmap_frame_create(window)

    def statmap_frame_create(self, window):
        self.statmap_frame = tk.LabelFrame(window)
        self.frames.append(self.statmap_frame)

        self.statmap_frame.place(relx=0.02, rely=0.77, relheight=0.22, relwidth=0.973)
        self.statmap_frame.configure(text='Statistical maps', font='Helvetica 18 bold')
        self.format_frame(self.statmap_frame)

        self.statmap_run_frame_create(self.statmap_frame)
        self.statmap_settings_frame_create(self.statmap_frame)

    def statmap_settings_frame_create(self, window):
        self.statmap_settings_frame = tk.LabelFrame(window)
        self.frames.append(self.statmap_settings_frame)
        self.statmap_settings_frame.place(relx=0.02, rely=0, height=140, relwidth=0.467)
        self.statmap_settings_frame.configure(text='Settings', font='Helvetica 18 bold')
        self.format_frame(self.statmap_settings_frame)

        self.statmap_save_button = ttk.Button(self.statmap_settings_frame)
        self.statmap_save_button.place(relx=0.02, rely=0.08, height=42, width=105)
        self.statmap_save_button.configure(command=lambda: Save_settings(['Statistical_maps'], 'statmap_config.toml'))
        self.statmap_save_button.configure(text='''Save settings''')
        Tooltip.CreateToolTip(self.statmap_save_button, 'Save all statistical map settings')

        self.statmap_reset_button = ttk.Button(self.statmap_settings_frame)
        self.statmap_reset_button.place(relx=0.5, rely=0.08, height=42, width=105)
        self.statmap_reset_button.configure(command=lambda: Reset_settings(['Statistical_maps']))
        self.statmap_reset_button.configure(text='''Reset settings''')
        Tooltip.CreateToolTip(self.statmap_reset_button, 'Reset statistical map settings')

        self.statmap_settings_button = ttk.Button(self.statmap_settings_frame)
        self.statmap_settings_button.place(relx=0.17, rely=0.55, height=42, width=150)
        self.statmap_settings_button.configure(command=lambda: self.change_frame('Statistical_maps'))
        self.statmap_settings_button.configure(text='''Settings''')

    def statmap_run_frame_create(self, window):
        self.statmap_run_frame = tk.LabelFrame(window)
        self.frames.append(self.statmap_run_frame)
        self.statmap_run_frame.place(relx=0.5, rely=0, height=140, relwidth=0.467)
        self.statmap_run_frame.configure(text='Run', font='Helvetica 18 bold')
        self.format_frame(self.statmap_run_frame)

        statmap_options = ('Image SNR', 'Temporal SNR')
        state = tk.StringVar()
        state.set(statmap_options[1])
        self.statmap_option = tk.OptionMenu(self.statmap_run_frame, state, *statmap_options)
        self.statmap_option.place(relx=0.08, rely=0.27, width=200, bordermode='ignore')
        self.statmap_option.configure(bg=self.background)
        self.statmap_option.val = state

        self.statmap_run_button = ttk.Button(self.statmap_run_frame)
        self.statmap_run_button.place(relx=0.17, rely=0.55, height=42, width=150)
        self.statmap_run_button.configure(command=lambda: Button_handler('Make maps', self.statmap_option.val.get()))
        self.statmap_run_button.configure(text='''Make maps''')  # TODO: Change these lines
        Tooltip.CreateToolTip(self.statmap_run_button, 'Print ROIs from selected atlas. This can be used to find which '
                                                 'numbers to input in the "Plotting" menu to plot specific regions.'
                                                 '\nNOTE: This does not change the atlas to be used for analysis.')

    @staticmethod
    def load_initial_values(toml_file):
        with open(toml_file, 'r') as f:
            for line in f.readlines():
                if line[0] == '#':
                    curr_page = re.split('# |\n', line)[1]

                elif curr_page == 'Version Info':
                    continue

                elif line is not '\n':
                    setting = [x.replace("'", "").strip() for x in re.split(" = |\[|\]|\n|(?<!')#.*", line) if x]

                    try:
                        if setting[1] in ['true', 'false']:
                            setting[1] = setting[1].title()

                        setting[1] = ast.literal_eval(setting[1])

                        if isinstance(setting[1], tuple):
                            setting[1] = list(setting[1])

                    except (ValueError, SyntaxError):
                        pass

                    eval(curr_page)[setting[0]]['Current'] = setting[1]

    def refresh_frame(self):
        """ refresh the content of the label every second """
        for widget in self.widgets:
            try:
                eval(self.page)[widget]['Current'] = self.widgets[widget].get()
            except AttributeError:
                eval(self.page)[widget]['Current'] = self.widgets[widget].val.get()

        self.settings_frame.destroy()
        self.settings_frame_create(root)

        # request tkinter to call self.refresh after 1s (the delay is given in ms)
        self.settings_frame.after(10000, self.refresh_frame)

    def banner_draw(self, window):
        img = Image.open(f'{os.getcwd()}/fRAT.gif')

        zoom = 0.8
        pixels_x, pixels_y = tuple([int(zoom * x) for x in img.size])
        img = img.resize((pixels_x, pixels_y))

        width, height = img.size
        new_width = width + 88
        result = Image.new('RGB', (new_width, height-20), (0, 0, 0))
        result.paste(img, (44, -10))

        result = ImageTk.PhotoImage(result)

        panel = tk.Label(window, image=result, borderwidth=0)
        panel.photo = result
        panel.place(relx=0.02, y=10)

        self.frames.append(panel)

    def Options_frame_draw(self, window):
        self.Options_frame = tk.LabelFrame(window)
        self.Options_frame.place(relx=0.5, rely=0.01, height=150, relwidth=0.345)
        self.Options_frame.configure(text=f'''Options''', font='Helvetica 18 bold')
        self.format_frame(self.Options_frame)
        self.frames.append(self.Options_frame)

        self.Save_button = ttk.Button(self.Options_frame)
        self.Save_button.place(relx=0.05, y=10, height=42, width=150)
        self.Save_button.configure(command=lambda: Save_settings(pages, 'fRAT_config.toml'))
        self.Save_button.configure(text='''Save settings''')
        Tooltip.CreateToolTip(self.Save_button, 'Save all fRAT settings')

        self.Reset_button = ttk.Button(self.Options_frame)
        self.Reset_button.place(relx=0.05, y=70, height=42, width=150)
        self.Reset_button.configure(command=lambda: Reset_settings(pages))
        self.Reset_button.configure(text='''Reset settings''')
        Tooltip.CreateToolTip(self.Reset_button, 'Reset fRAT settings to recommended values')

    def Atlas_frame_draw(self, window):
        self.Atlas_frame = tk.LabelFrame(window)
        self.Atlas_frame.place(relx=0.5, rely=0.348, height=128, relwidth=0.467)
        self.Atlas_frame.configure(text=f'''Atlas information''', font='Helvetica 18 bold')
        self.format_frame(self.Atlas_frame)
        self.frames.append(self.Atlas_frame)

        atlas_options = ('Cerebellum_MNIflirt',
                         'Cerebellum_MNIfnirt',
                         'HarvardOxford-Cortical',
                         'HarvardOxford-Subcortical',
                         'JHU-labels',
                         'JHU-tracts',
                         'Juelich',
                         'MNI',
                         'SMATT',
                         'STN',
                         'Striatum-Structural',
                         'Talairach',
                         'Thalamus')

        state = tk.StringVar()
        state.set(atlas_options[2])
        self.Atlas_option = tk.OptionMenu(self.Atlas_frame, state, *atlas_options)
        self.Atlas_option.place(relx=0.08, rely=0.2, width=200, bordermode='ignore')
        self.Atlas_option.configure(bg=self.background)
        self.Atlas_option.val = state

        self.Atlas_button = ttk.Button(self.Atlas_frame)
        self.Atlas_button.place(relx=0.17, rely=0.45, height=42, width=150)
        self.Atlas_button.configure(command=lambda: Print_atlas_ROIs(self.Atlas_option.val.get()))
        self.Atlas_button.configure(text='''Print Atlas ROIs''')
        Tooltip.CreateToolTip(self.Atlas_button, 'Print ROIs from selected atlas. This can be used to find which '
                                                 'numbers to input in the "Plotting" menu to plot specific regions.'
                                                 '\nNOTE: This does not change the atlas to be used for analysis.')

    def Run_frame_draw(self, window):
        self.Run_frame = tk.LabelFrame(window)
        self.Run_frame.place(relx=0.025, rely=0.65, height=145, relwidth=0.945)
        self.Run_frame.configure(text=f'''Run''', font='Helvetica 18 bold')
        self.format_frame(self.Run_frame)
        self.frames.append(self.Run_frame)

        self.paramValues_button = ttk.Button(self.Run_frame)
        self.paramValues_button.place(x=2, y=12, height=42, width=150)
        self.paramValues_button.configure(command=lambda: Button_handler('Make paramValues.csv'))
        self.paramValues_button.configure(text='''Setup parameters''')
        Tooltip.CreateToolTip(self.paramValues_button, 'Creates a csv table with prefilled parameter info for each file. Can be set using a command line flag instead')

        self.Print_button = ttk.Button(self.Run_frame)
        self.Print_button.place(x=156, y=12, height=42, width=150)
        self.Print_button.configure(command=lambda: Button_handler('Print_results'))
        self.Print_button.configure(text='''Print results''')
        Tooltip.CreateToolTip(self.Print_button, 'Print results of fRAT to the terminal')

        self.Dash_button = ttk.Button(self.Run_frame)
        self.Dash_button.place(x=309, y=12, height=42, width=150)
        self.Dash_button.configure(command=lambda: Button_handler('Run_dash'))
        self.Dash_button.configure(text='''Interactive table''')
        Tooltip.CreateToolTip(self.Dash_button, 'Create an interactive table to display fRAT results')

        self.Run_button = ttk.Button(self.Run_frame)
        self.Run_button.place(x=137, y=66, height=42, width=181)
        self.Run_button.configure(command=lambda: Button_handler('Run_fRAT'))
        self.Run_button.configure(text='''Run fRAT''')
        Tooltip.CreateToolTip(self.Run_button, 'Run fRAT with current settings')

    def settings_frame_create(self, window):
        self.settings_frame = tk.LabelFrame(window)
        self.frames.append(self.settings_frame)

        if self.page == 'Home':
            self.settings_frame.place(relx=0.02, rely=0.15, relheight=0.61, relwidth=0.973)
            self.settings_frame.configure(text='fRAT', font='Helvetica 18 bold')
            self.format_frame(self.settings_frame)

            self.General_settings_frame = tk.LabelFrame(self.settings_frame)
            self.frames.append(self.General_settings_frame)
            self.General_settings_frame.place(relx=0.025, rely=0.01, height=280, relwidth=0.41)
            self.General_settings_frame.configure(text=f'''Settings''', font='Helvetica 18 bold')
            self.format_frame(self.General_settings_frame)
            current_frame = self.General_settings_frame

            y_loc = 10
            relx = 0.03
            for page in pages:
                if page == 'Home':
                    continue

                if page == 'Violin_plot':
                    break

                self.index_setup(page, current_frame, y_loc, relx)

                y_loc += 60

        elif self.page == 'Plotting':
            y_loc = self.widget_create('Home')

            self.Plot_settings_frame = tk.LabelFrame(window)
            self.frames.append(self.Plot_settings_frame)
            self.Plot_settings_frame.place(relx=0.31, y=y_loc + 58, height=270, relwidth=0.394)
            self.Plot_settings_frame.configure(text=f'''Specific plot settings''', font='Helvetica 18 bold')
            self.format_frame(self.Plot_settings_frame)

            continue_loop = True

            for page in pages:
                if page == 'Violin_plot':
                    continue_loop = False
                    y_loc = 10
                    relx = 0.03
                    current_frame = self.Plot_settings_frame
                elif continue_loop:
                    continue

                self.index_setup(page, current_frame, y_loc, relx)

                y_loc += 60

        elif self.page in ['General', 'Analysis', 'Parsing', 'Statistical_maps']:
            self.widget_create('Home')

        else:
            self.widget_create('Plotting')

    def widget_create(self, previous_frame):
        y_loc = 40
        for setting in self.current_info:
            self.label_create(setting, y_loc, self.current_info[setting])

            if self.current_info[setting]['type'] == 'Scale':
                widget = self.scale_create(setting, self.current_info[setting], y_loc)
                self.widgets = {**self.widgets, **widget}

            elif self.current_info[setting]['type'] == 'CheckButton':
                widget = self.checkbutton_create(setting, self.current_info[setting], y_loc)
                self.widgets = {**self.widgets, **widget}

            elif self.current_info[setting]['type'] == 'OptionMenu':
                widget = self.optionmenu_create(setting, self.current_info[setting], y_loc)
                self.widgets = {**self.widgets, **widget}

            elif self.current_info[setting]['type'] == 'Entry':
                widget = self.entry_create(setting, self.current_info[setting], y_loc)
                self.widgets = {**self.widgets, **widget}

            elif self.current_info[setting]['type'] == 'Dynamic':
                y_loc, widget = self.dynamic_widget(setting, self.current_info[setting], y_loc)
                self.dynamic_widgets = {**self.dynamic_widgets, **widget}
                y_loc -= 40

            y_loc += 40

        self.settings_frame.place(relx=0.02, rely=0.014, height=y_loc + 40, relwidth=0.973)
        self.settings_frame.configure(text=f'''{self.page.replace('_', ' ')}''', font='Helvetica 18 bold')
        self.format_frame(self.settings_frame)

        self.Index_button = tk.Button(self.settings_frame, command=lambda: self.change_frame(previous_frame))
        self.Index_button.place(relx=0.35, y=y_loc - 30, height=42, width=150)
        self.Index_button.configure(activebackground="#ececec")
        self.Index_button.configure(activeforeground="#000000")
        self.Index_button.configure(background=self.background)
        self.Index_button.configure(foreground="#000000")
        self.Index_button.configure(highlightbackground=self.background)
        self.Index_button.configure(highlightcolor="black")
        self.Index_button.configure(text=f'''Back to {previous_frame}''')

        return y_loc

    def format_frame(self, frame):
        frame.configure(borderwidth="2")
        frame.configure(relief='groove')
        frame.configure(foreground="black")
        frame.configure(background=self.background)
        frame.configure(highlightbackground=self.background)
        frame.configure(highlightcolor="black")

    def index_setup(self, page, frame, y_loc, relx):
        self.__setattr__(page, ttk.Button(frame, command=lambda: self.change_frame(page)))
        index_button = getattr(self, page)
        index_button.place(relx=relx, y=y_loc, width=185, height=42)
        index_button.configure(text=f'''{page.replace('_', ' ')}''')

    def dynamic_widget(self, name, info, y_loc):
        try:
            text = eval(info['Options'])['Current']
            text = [value.strip() for value in text.split(',')]
        except KeyError:
            text = ""

        dynamic_widgets = {}

        if info['subtype'] == 'OptionMenu':
            if not text:
                text = " "

            info['DynamOptions'] = [*text]

            if info['Current'] not in info['DynamOptions']:
                try:
                    info['Current'] = info['DynamOptions'][info['DefaultNumber']]
                except IndexError:
                    info['Current'] = ''

            if len(info['DynamOptions']) == 1:
                info['DynamOptions'].append('')

            widget = self.optionmenu_create(name, info, y_loc)

            dynamic_widgets = {**dynamic_widgets, **widget}

            y_loc += 40

        elif info['subtype'] == 'Checkbutton':
            for value in text:
                widget = self.checkbutton_create(f"{name}_{value}", info, y_loc)
                [widget.configure(text=value) for widget in widget.values()]

                dynamic_widgets = {**dynamic_widgets, **widget}
                y_loc += 30

        return y_loc, dynamic_widgets

    def label_create(self, name, y_loc, info=None, relx=0.08, font=None):
        self.__setattr__(name, tk.Label(self.settings_frame))
        label_name = getattr(self, name)

        try:
            name = info['label']
        except KeyError:
            name = name.capitalize()

        label_name.place(relx=relx, y=y_loc, height=22, bordermode='ignore')
        label_name.configure(background=self.background)
        label_name.configure(foreground="#000000")
        label_name.configure(text=f'''{name.replace("_", " ")}:''', font=font)

        if info is not None:
            Tooltip.CreateToolTip(label_name, info['Description'])

    def scale_create(self, name, info, y_loc):
        self.__setattr__(name, tk.Scale(self.settings_frame, from_=info['From'], to=info['To']))
        widget = getattr(self, name)

        widget.place(x=300, y=y_loc - 15, relwidth=0.205, relheight=0.0, height=39, bordermode='ignore')
        widget.configure(resolution=info['Resolution'])

        self.scale_default_settings(widget)

        widget.set(info['Current'])

        return {name: widget}

    def scale_default_settings(self, widget):
        widget.configure(activebackground="#ececec")
        widget.configure(background=self.background)
        widget.configure(bigincrement="0.05")
        widget.configure(foreground="#000000")
        widget.configure(highlightbackground=self.background)
        widget.configure(highlightcolor="black")
        widget.configure(orient="horizontal")
        widget.configure(troughcolor=self.background)

    def checkbutton_create(self, name, info, y_loc):
        state = tk.BooleanVar()

        self.__setattr__(name, tk.Checkbutton(self.settings_frame, variable=state))
        widget = getattr(self, name)

        widget.place(x=295, y=y_loc, bordermode='ignore')
        widget.val = state

        self.checkbutton_default_settings(widget)

        if info['Current'] in ['true', 'false']:
            info['Current'] = ast.literal_eval(info['Current'].title())

        current_val = info['Current']

        # Dynamic widget handler
        if not isinstance(info['Current'], bool) and name.rsplit('_', 1)[1] in info['Current']:
            current_val = True
        elif not isinstance(info['Current'], bool) and not name.rsplit('_', 1)[1] in info['Current']:
            current_val = False

        if current_val:
            widget.select()
        else:
            widget.deselect()

        return {name: widget}

    def checkbutton_default_settings(self, widget):
        widget.configure(activebackground="#ececec")
        widget.configure(activeforeground="#000000")
        widget.configure(background=self.background)
        widget.configure(foreground="#000000")
        widget.configure(highlightbackground=self.background)
        widget.configure(highlightcolor="black")
        widget.configure(justify='left')

    def optionmenu_create(self, name, info, y_loc):
        state = tk.StringVar()

        state.set(info['Current'])

        try:
            options = info['DynamOptions']
        except KeyError:
            options = info['Options']

        self.__setattr__(name, tk.OptionMenu(self.settings_frame, state, *options))
        widget = getattr(self, name)

        widget.place(x=295, y=y_loc, width=183, bordermode='ignore')
        widget.configure(bg=self.background)
        widget.val = state

        return {name: widget}

    def entry_create(self, name, info, y_loc):
        self.__setattr__(name, tk.Entry(self.settings_frame))
        widget = getattr(self, name)

        widget.place(x=300, y=y_loc, height=25, bordermode='ignore')

        self.entry_default_settings(widget)

        if info['Current'] is None:
            info['Current'] = str(info['Current'])

        if isinstance(info['Current'], (list, tuple)):
            info['Current'] = [str(x) for x in info['Current']]
            info['Current'] = ", ".join(info['Current'])

        widget.insert(0, info['Current'])

        return {name: widget}

    @staticmethod
    def entry_default_settings(widget):
        widget.configure(background="white")
        widget.configure(font="Helvetica")
        widget.configure(foreground="#000000")
        widget.configure(insertbackground="black")
        widget.configure(selectforeground="white")

    def change_frame(self, page):
        for widget in self.widgets:
            try:
                eval(self.page)[widget]['Current'] = self.widgets[widget].get()
            except AttributeError:
                eval(self.page)[widget]['Current'] = self.widgets[widget].val.get()

        for widget in self.dynamic_widgets:
            if self.dynamic_widgets[widget].winfo_class() == 'Checkbutton':
                params = widget.rsplit('_', 1)

                if isinstance(eval(self.page)[params[0]]['Current'], str):
                        eval(self.page)[params[0]]['Current'] = [x.strip() for x in eval(self.page)[params[0]]['Current'].split(',')]

                if self.dynamic_widgets[widget].val.get():  # If checkbutton is true
                    if params[1] not in eval(self.page)[params[0]]['Current']:
                        if eval(self.page)[params[0]]['Current'][0] == '':
                            eval(self.page)[params[0]]['Current'][0] = (params[1])

                        else:
                            eval(self.page)[params[0]]['Current'].append(params[1])

                else:  # If checkbutton is false
                    if params[1] in eval(self.page)[params[0]]['Current']:
                        eval(self.page)[params[0]]['Current'].remove(params[1])

                    if len(eval(self.page)[params[0]]['Current']) == 0:
                        eval(self.page)[params[0]]['Current'].append('')  # If list is empty, set first element to blank string

            else:
                eval(self.page)[widget]['Current'] = self.dynamic_widgets[widget].val.get()

        for frame in self.frames:
            frame.destroy()
        self.frames.clear()

        self.__init__(root, page, load_initial_values=False)


class Tooltip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() +27
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font=("tahoma", "12", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

    @staticmethod
    def CreateToolTip(widget, text):
        toolTip = Tooltip(widget)

        def enter(event):
            toolTip.showtip(text)

        def leave(event):
            toolTip.hidetip()

        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)


def Button_handler(command, *args):
    try:
        if command == 'Run_fRAT':
            Save_settings(pages, 'fRAT_config.toml')

            stale_pages = check_stale_state()
            if stale_pages:
                print(f"\nPlot settings for the following plots have not been updated to reflect new critical parameter settings: "
                      f"\n * {stale_pages}"
                      f"\nOpening these pages in the GUI will update these settings. Also make sure to update labels for these plots in their pages.\n")
                return

            print('----- Running fRAT -----')
            fRAT()

        elif command == 'Print_results':
            printResults()

        elif command == "Make paramValues.csv":
            Save_settings(pages, 'fRAT_config.toml')
            make_table()
            sys.exit()

        elif command == "Run_dash":
            dash_report.main()

        elif command == "Make maps":
            Save_settings(['Statistical_maps'], 'statmap_config.toml')
            statmap_calc(args[0])

    except Exception as err:
        if err.args[0] == 'No folder selected.':
            print('----- Exiting -----\n')
        else:
            log = logging.getLogger(__name__)
            log.exception(err)
            sys.exit()


def check_stale_state():
    current_critical_params = [value.strip() for value in Parsing['parameter_dict1']['Current'].split(',')]
    dynamic_widgets = (Violin_plot['table_cols'], Violin_plot['table_rows'],
                       Brain_table['brain_table_cols'], Brain_table['brain_table_rows'],
                       Region_barchart['single_roi_fig_colour'], Region_barchart['single_roi_fig_x_axis'],
                       Region_histogram['histogram_fig_x_facet'], Region_histogram['histogram_fig_y_facet'])
    plot_pages = {0: 'Scatterplot', 1: 'Scatterplot',
                  2: 'Brain table', 3: 'Brain table',
                  4: 'Regional barchart', 5: 'Regional barchart',
                  6: 'Regional histogram', 7: 'Regional histogram'}

    stale_pages = []
    for counter, widget in enumerate(dynamic_widgets):
        if widget['Current'] not in current_critical_params and plot_pages[counter] not in stale_pages:
            stale_pages.append(plot_pages[counter])

    if stale_pages:
        return "\n * ".join(stale_pages)


def Print_atlas_ROIs(selection):
    """Extract labels from specified FSL atlas XML file."""
    try:
        fsl_path = os.environ['FSLDIR']
    except OSError:
        raise Exception('FSL environment variable not set.')

    atlas_path = f"{fsl_path}/data/atlases/{selection}.xml"

    with open(atlas_path) as fd:
        atlas_label_dict = xmltodict.parse(fd.read())

    roiArray = []
    roiArray.append('No ROI')

    for roiLabelLine in atlas_label_dict['atlas']['data']['label']:
        roiArray.append(roiLabelLine['#text'])

    roiArray.append('Overall')

    print(f"----------------------------\n{selection} Atlas:\n----------------------------")

    for roi_num, roi in enumerate(roiArray):
        print("{roi_num}: {roi}".format(roi_num=roi_num, roi=roi))

    print("----------------------------\n")


def Save_settings(page_list, file):
    with open(f'{Path(os.path.abspath(__file__)).parents[0]}/{file}', 'w') as f:
        f.write(f"# Version Info\n")
        f.write(f"version = '{VERSION}'\n")
        f.write("\n")

        for page in page_list:
            if page == 'Home':
                continue

            f.write(f"# {page}\n")

            for key in eval(page).keys():
                description = eval(page)[key]['Description'].replace('\n', ' ')

                if eval(page)[key]['type'] == 'OptionMenu':
                    description += f" Options: {eval(page)[key]['Options']}."

                convert = 'default'

                try:
                    if eval(page)[key]['save_as'] == 'string':
                        convert = 'string'

                    elif eval(page)[key]['save_as'] == 'list':
                        if eval(page)[key]['Current'] == 'Runtime':
                            convert = 'string'

                        else:
                            try:
                                eval(page)[key]['Current'] = [int(x) for x in eval(page)[key]['Current'].split(',')]
                            except AttributeError:  # Handle exception if already a list so cannot split
                                pass
                            except ValueError:  # Handle exception if list items cannot be converted into ints
                                convert = 'split_list'

                    elif eval(page)[key]['save_as'] == 'string_or_list':
                        try:
                            eval(page)[key]['Current'] = list(ast.literal_eval(eval(page)[key]['Current']))  # Try to convert to list

                        except (ValueError, TypeError):  # Handles error if input is None or is string
                            convert = 'string'

                except KeyError:  # Handle exception if save_as does not exist as key
                    eval(page)[key]['Current'] = str(eval(page)[key]['Current']).lower() \
                        if type(eval(page)[key]['Current']) is bool else eval(page)[key]['Current']

                    if eval(page)[key]['Current'] in ['None', None]:
                        convert = 'string'

                if convert == 'default':  # Save as is
                    offset = ' ' * (80 - len(f"{key} = {eval(page)[key]['Current']}"))
                    f.write(f"{key} = {eval(page)[key]['Current']}  "
                            f"{offset}# {description}\n")

                elif convert == 'string':  # Convert to string
                    offset = ' ' * (80 - len(f"{key} = '{eval(page)[key]['Current']}'"))
                    f.write(f"{key} = '{eval(page)[key]['Current']}'  "
                            f"{offset}# {description}\n")

                elif convert == 'split_list':  # Split items then convert to list
                    offset = ' ' * (80 - len(f"{key} = {[val.strip() for val in eval(page)[key]['Current'].split(',')]}"))
                    f.write(f"{key} = {[val.strip() for val in eval(page)[key]['Current'].split(',')]}  "
                            f"{offset}# {description}\n")

            f.write("\n")

        f.flush()
        f.close()

    print('----- Saved settings -----')


def Reset_settings(pages):
    for page in pages:
        if page == 'Home':
            continue

        for key in eval(page).keys():
            eval(page)[key]['Current'] = eval(page)[key]['Recommended']

    print('----- Reset fRAT settings to recommended values, save them to retain these settings -----')


def make_table():
    config = Utils.load_config(Path(os.path.abspath(__file__)).parents[0], 'fRAT_config.toml')  # Load config file

    print('--- Creating paramValues.csv ---')
    print('Select the base directory.')
    base_directory = Utils.file_browser(title='Select the base directory', chdir=True)

    participant_dirs = find_participant_dirs(config)

    data = []
    for participant in participant_dirs:
        brain_file_list = Utils.find_files(f"{participant}/func", "hdr", "nii.gz", "nii")
        brain_file_list = [os.path.splitext(brain)[0] for brain in brain_file_list]
        brain_file_list.sort()

        for file in brain_file_list:
            # Try to find parameters to prefill table
            brain_file_params = parse_params_from_file_name(file, config)
            data.append([participant, file, *brain_file_params, np.NaN])

    df = pd.DataFrame(columns=['Participant', 'File name',
                               *config.parameter_dict.keys(), 'Ignore file? (y for yes, otherwise blank)'],
                      data=data)

    df.to_csv('paramValues.csv', index=False)

    print(f"\nparamValues.csv saved in {base_directory}.\n\nInput parameter values in paramValues.csv and change "
          f"make_table_only to False in the config file to continue analysis. \nIf analysis has already been "
          f"conducted, move paramValues.csv into the ROI report folder. \nIf the csv file contains unexpected "
          f"parameters, update the parsing options in the GUI or parameter_dict2 in fRAT_config.toml.")


def find_participant_dirs(config):
    participant_dirs = [direc for direc in glob("*") if re.search("^p[0-9]+", direc)]

    if len(participant_dirs) == 0:
        raise FileNotFoundError('Participant directories not found.')
    elif config.verbose:
        print(f'Found {len(participant_dirs)} participant folders.')

    return participant_dirs


def parse_params_from_file_name(json_file_name, cfg=config):
    """Search for MRI parameters in each json file name for use in table headers and created the combined json."""
    param_nums = []

    for key in cfg.parameter_dict:
        parameter = cfg.parameter_dict[key]  # Extract search term

        if parameter in cfg.binary_params:
            param = re.search("{}".format(parameter), json_file_name, flags=re.IGNORECASE)

            if param is not None:
                param_nums.append('On')  # Save 'on' if parameter is found in file name
            else:
                param_nums.append('Off')  # Save 'off' if parameter not found in file name

        else:
            # Float search
            param = re.search("{}[0-9]p[0-9]".format(parameter), json_file_name, flags=re.IGNORECASE)
            if param is not None:
                param_nums.append(param[0][1] + "." + param[0][-1])
                continue

            # If float search didnt work then Integer search
            param = re.search("{}[0-9]".format(parameter), json_file_name, flags=re.IGNORECASE)
            if param is not None:
                param_nums.append(param[0][-1])  # Extract the number from the parameter

            else:
                param_nums.append(str(param))  # Save None if parameter not found in file name

    return param_nums


if __name__ == '__main__':
    print('----------------------------\n----- Running fRAT_GUI -----\n----------------------------')
    start_gui()
