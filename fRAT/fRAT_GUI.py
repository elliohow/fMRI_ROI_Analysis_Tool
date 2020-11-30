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
from pathlib import Path

from fRAT import fRAT
from printResults import printResults
from utils import *
from utils.config_setup import *

w = None


def vp_start_gui():
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
    def __init__(self, window=None, page='Settings', load_initial_values=True):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        self.widgets = {}
        self.dynamic_widgets = {}
        self.frames = []
        self.page = page
        self.background = '#d9d9d9'

        if load_initial_values:
            self.load_initial_widget_values()

            self.style = ttk.Style()
            self.style.theme_use('clam')

            _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
            _fgcolor = '#000000'  # X11 color: 'black'
            _compcolor = '#d9d9d9'  # X11 color: 'gray85'
            _ana1color = '#d9d9d9'  # X11 color: 'gray85'
            _ana2color = '#ececec'  # Closest X11 color: 'gray92'
            window.geometry("512x780+50+50")
            window.minsize(72, 15)
            window.maxsize(2048, 1028)
            window.resizable(1, 1)
            window.title("fRAT GUI")
            window.configure(background=self.background)
            window.configure(highlightbackground=self.background)
            window.configure(highlightcolor="black")

        try:
            self.current_info = eval(self.page)
        except NameError:
            self.current_info = pages

        self.Setting_frame_create(window)

        if self.page == 'Settings':
            self.Options_frame_draw(window)
            self.Run_frame_draw(window)

    def load_initial_widget_values(self):
        with open('config.toml', 'r') as f:
            for page in pages:
                if page == 'Settings':
                    continue

                for line in f.readlines():
                    if line[0] == '#':
                        curr_page = re.split('# |\n', line)[1]

                    elif line is not '\n':
                        setting = [x.replace("'", "").strip() for x in re.split(" = |\[|\]|\n|(?<!')#.*", line) if x]

                        try:
                            if setting[1] in ['true', 'false']:
                                setting[1] = setting[1].title()

                            setting[1] = ast.literal_eval(setting[1])

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

        self.Setting_frame.destroy()
        self.Setting_frame_create(root)

        # request tkinter to call self.refresh after 1s (the delay is given in ms)
        self.Setting_frame.after(10000, self.refresh_frame)

    def Options_frame_draw(self, window):
        self.Options_frame = tk.LabelFrame(window)
        self.Options_frame.place(relx=0.5, rely=0.06, height=150, relwidth=0.333)
        self.Options_frame.configure(text=f'''Options''', font='Helvetica 18 bold')
        self.frame_setup(self.Options_frame)
        self.frames.append(self.Options_frame)

        self.Save_button = ttk.Button(self.Options_frame)
        self.Save_button.place(relx=0.05, y=10, height=42, width=150)
        self.Save_button.configure(command=Save_settings)
        self.Save_button.configure(text='''Save preferences''')
        Tooltip.CreateToolTip(self.Save_button, 'Save all preferences')

        self.Reset_button = ttk.Button(self.Options_frame)
        self.Reset_button.place(relx=0.05, y=70, height=42, width=150)
        self.Reset_button.configure(command=Reset_settings)
        self.Reset_button.configure(text='''Reset preferences''')
        Tooltip.CreateToolTip(self.Reset_button, 'Reset preferences to recommended values')

    def Run_frame_draw(self, window):
        self.Run_frame = tk.LabelFrame(window)
        self.Run_frame.place(relx=0.056, rely=0.42, height=90, relwidth=0.91)
        self.Run_frame.configure(text=f'''Run''', font='Helvetica 18 bold')
        self.frame_setup(self.Run_frame)
        self.frames.append(self.Run_frame)

        self.paramValues_button = ttk.Button(self.Run_frame)
        self.paramValues_button.place(x=2, y=12, height=42, width=150)
        self.paramValues_button.configure(command=ParamParser.make_table)
        self.paramValues_button.configure(text='''Setup parameters''')
        Tooltip.CreateToolTip(self.paramValues_button, 'Creates a csv table with prefilled parameter info for each file. Can be set using a command line flag instead')

        self.Print_button = ttk.Button(self.Run_frame)
        self.Print_button.place(x=156, y=12, height=42, width=150)
        self.Print_button.configure(command=lambda:Button_handler('Print_results'))
        self.Print_button.configure(text='''Print results''')
        Tooltip.CreateToolTip(self.Print_button, 'Print results of fRAT to the terminal.')

        self.Run_button = ttk.Button(self.Run_frame)
        self.Run_button.place(x=309, y=12, height=42, width=150)
        self.Run_button.configure(command=lambda:Button_handler('Run_fRAT'))
        self.Run_button.configure(text='''Run fRAT''')
        Tooltip.CreateToolTip(self.Run_button, 'Run fRAT with current settings.')

    def Setting_frame_create(self, window):
        self.Setting_frame = tk.LabelFrame(window)
        self.frames.append(self.Setting_frame)

        if self.page == 'Settings':
            self.Setting_frame.place(relx=0.02, rely=0.014, relheight=0.54, relwidth=0.973)
            self.Setting_frame.configure(text=f'''{self.page.replace('_', ' ')}''', font='Helvetica 18 bold')
            self.frame_setup(self.Setting_frame)

            self.General_settings_frame = tk.LabelFrame(window)
            self.frames.append(self.General_settings_frame)
            self.General_settings_frame.place(relx=0.056, rely=0.06, height=280, relwidth=0.4)
            self.General_settings_frame.configure(text=f'''Preferences''', font='Helvetica 18 bold')
            self.frame_setup(self.General_settings_frame)
            current_frame = self.General_settings_frame

            y_loc = 10
            relx = 0.03
            for page in pages:
                if page == 'Settings':
                    continue

                if page == 'Scatter_plot':
                    break

                self.index_setup(page, current_frame, y_loc, relx)

                y_loc += 60

        elif self.page == 'Plotting':
            y_loc = self.widget_create('Settings')

            self.Plot_settings_frame = tk.LabelFrame(window)
            self.frames.append(self.Plot_settings_frame)
            self.Plot_settings_frame.place(relx=0.31, y=y_loc + 70, height=270, relwidth=0.394)
            self.Plot_settings_frame.configure(text=f'''Specific plot settings''', font='Helvetica 18 bold')
            self.frame_setup(self.Plot_settings_frame)

            continue_loop = True

            for page in pages:
                if page == 'Scatter_plot':
                    continue_loop = False
                    y_loc = 10
                    relx = 0.03
                    current_frame = self.Plot_settings_frame
                elif continue_loop:
                    continue

                self.index_setup(page, current_frame, y_loc, relx)

                y_loc += 60

        elif self.page in ['General', 'Analysis', 'Parsing']:
            self.widget_create('Settings')

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

        self.Setting_frame.place(relx=0.02, rely=0.014, height=y_loc + 40, relwidth=0.973)
        self.Setting_frame.configure(text=f'''{self.page.replace('_', ' ')}''', font='Helvetica 18 bold')
        self.frame_setup(self.Setting_frame)

        self.Index_button = tk.Button(self.Setting_frame, command=lambda: self.change_frame(previous_frame))
        self.Index_button.place(relx=0.35, y=y_loc - 30, height=42, width=150)
        self.Index_button.configure(activebackground="#ececec")
        self.Index_button.configure(activeforeground="#000000")
        self.Index_button.configure(background=self.background)
        self.Index_button.configure(foreground="#000000")
        self.Index_button.configure(highlightbackground=self.background)
        self.Index_button.configure(highlightcolor="black")
        self.Index_button.configure(text=f'''Back to {previous_frame}''')

        return y_loc

    def frame_setup(self, frame):
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
                    info['Current'] = info['DynamOptions'][info['DynamNumber']]
                except IndexError:
                    info['Current'] = ''

            if len(info['DynamOptions']) == 1:
                info['DynamOptions'].append('')

            widget = self.optionmenu_create(name, info, y_loc)

            dynamic_widgets = {**dynamic_widgets, **widget}

        elif info['subtype'] == 'Checkbutton':

            for value in text:
                widget = self.checkbutton_create(f"{name}_{value}", info, y_loc)
                [widget.configure(text=value) for widget in widget.values()]

                dynamic_widgets = {**dynamic_widgets, **widget}
                y_loc += 30

        return y_loc, dynamic_widgets

    def label_create(self, name, y_loc, info=None, relx=0.08, font=None):
        self.__setattr__(name, tk.Label(self.Setting_frame))
        label_name = getattr(self, name)
        label_name.place(relx=relx, y=y_loc, height=22, bordermode='ignore')
        label_name.configure(background=self.background)
        label_name.configure(foreground="#000000")
        label_name.configure(text=f'''{name.replace("_", " ")}''', font=font)

        if info is not None:
            Tooltip.CreateToolTip(label_name, info['Description'])

    def scale_create(self, name, info, y_loc):
        self.__setattr__(name, tk.Scale(self.Setting_frame, from_=info['From'], to=info['To']))
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

        self.__setattr__(name, tk.Checkbutton(self.Setting_frame, variable=state))
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

        self.__setattr__(name, tk.OptionMenu(self.Setting_frame, state, *options))
        widget = getattr(self, name)

        widget.place(x=295, y=y_loc, width=183, bordermode='ignore')
        widget.configure(bg=self.background)
        widget.val = state

        return {name: widget}

    def entry_create(self, name, info, y_loc):
        self.__setattr__(name, tk.Entry(self.Setting_frame))
        widget = getattr(self, name)

        widget.place(x=300, y=y_loc, height=25, bordermode='ignore')

        self.entry_default_settings(widget)

        if info['Current'] is None:
            info['Current'] = str(info['Current'])

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


def Button_handler(command):
    try:
        if command == 'Run_fRAT':
            Save_settings()
            print('----- Running fRAT -----')
            fRAT()

        elif command == 'Print_results':
            printResults()

    except Exception as err:
        if err.args[0] == 'No folder selected.':
            print('----- Exiting -----\n')
        else:
            log = logging.getLogger(__name__)
            log.exception(err)
            sys.exit()


def Save_settings():
    with open(f'{Path(os.path.abspath(__file__)).parents[0]}/config.toml', 'w') as f:
        for page in pages:
            if page == 'Settings':
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
                        try:
                            if eval(page)[key]['Current'] == 'Runtime':
                                convert = 'string'
                            else:
                                [val.strip() for val in eval(page)[key]['Current'].split(',')]
                                convert = 'split_list'
                        except AttributeError:
                            pass

                    elif eval(page)[key]['save_as'] == 'string_or_list':
                        try:
                            eval(page)[key]['Current'] = list(ast.literal_eval(eval(page)[key]['Current']))  # Try to convert to list

                        except (ValueError, TypeError):  # Handles error if input is None or is string
                            convert = 'string'

                except KeyError:  # Handle exception for if save_as does not exist as key
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


def Reset_settings():
    for page in pages:
        if page == 'Settings':
            continue

        for key in eval(page).keys():
            eval(page)[key]['Current'] = eval(page)[key]['Recommended']

    print('----- Reset settings to recommended values, save them to retain these settings -----')


if __name__ == '__main__':
    print('----------------------------\n----- Running fRAT_GUI -----\n----------------------------\n')
    vp_start_gui()
