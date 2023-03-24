import platform
import toml
import tkinter as tk
import tkinter.ttk as ttk

import ast
import sys
import textwrap

from PIL import ImageTk
from operator import itemgetter

from fRAT.nogui import fRAT
from fRAT._version import __version__

from fRAT.utils import *
from fRAT.utils.directory_comparison import *
from fRAT.utils.fRAT_config_setup import *
from fRAT.utils.statmap_config_setup import *

from fRAT.utils import dash_report
from fRAT.utils.printResults import printResults
from fRAT.utils.statmap import main as statmap_calc

w = None
WIDGET_Y_PADDING = 9
WIDGET_X_PADDING = 10


def start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root, top

    path_check()

    Utils.checkversion(__version__)

    print('----------------------------\n----- Running fRAT_GUI -----\n----------------------------')

    root = tk.Tk()
    top = GUI(root)

    root.mainloop()


class GUI:
    def __init__(self, window=None, page='Home', load_initial_values=True):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        self.widgets = {}
        self.dynamic_widgets = {}
        self.frames = []
        self.page = page
        self.background = '#d9d9d9'

        if load_initial_values:
            # Load config file last used
            with open(f'{Path(os.path.abspath(__file__)).parents[0]}/configuration_profiles/latest_settings.toml',
                      'r') as tomlfile:
                parse = tomlfile.readlines()
                parse = toml.loads(''.join(parse))

            for toml_file in [f'{folder}/{config_file}' for folder, config_file in parse.items()]:
                self.load_initial_values(toml_file)

            self.style = ttk.Style()
            self.style.theme_use('clam')

            window.geometry("+100+100")
            window.resizable(0, 0)
            window.title("fRAT GUI")
            window.configure(background=self.background)
            window.configure(highlightbackground=self.background)
            window.configure(highlightcolor="black")

        self.change_page_specific_settings(window)

        if page == 'Statistical_maps':
            self.current_info = Statistical_maps
        else:
            try:
                self.current_info = eval(self.page)
            except NameError:
                self.current_info = pages

        self.create_settings_frame(window)

        if self.page == 'Home':
            self.banner_draw(window)
            self.create_home_options_frame(self.settings_frame)
            self.create_home_atlas_frame(self.settings_frame)
            self.create_homestar_runner_frame(self.settings_frame)
            self.create_statmap_frame(window)

    def change_page_specific_settings(self, window):
        if platform.system() == 'Darwin':
            statistics_width = "530"
            home_width = "512"
            statmap_width = "500"
            general_width = "480"
            parsing_width = "450"
        else:
            statistics_width = "550"
            home_width = "530"
            statmap_width = "520"
            general_width = "500"
            parsing_width = "480"

        if self.page == 'Home':
            window.geometry(f"{home_width}x860")
        elif self.page == 'General':
            window.geometry(f"{general_width}x860")
        elif self.page == 'Analysis':
            window.geometry(f"{general_width}x860")
        elif self.page == 'Parsing':
            window.geometry(f"{parsing_width}x280")
        elif self.page == 'Plotting':
            window.geometry(f"{home_width}x860")
        elif self.page == 'Statistics':
            window.geometry(f"{statistics_width}x860")
        elif self.page == 'Statistical_maps':
            window.geometry(f"{statmap_width}x860")

    def create_statmap_frame(self, window):
        self.statmap_frame = tk.LabelFrame(window)
        self.frames.append(self.statmap_frame)

        self.statmap_frame.place(relx=0.02, rely=0.77, relheight=0.22, relwidth=0.973)
        self.statmap_frame.configure(text='Statistical maps', font='Helvetica 18 bold')
        self.format_frame(self.statmap_frame, borderwidth=1)

        self.statmap_run_frame_create(self.statmap_frame)
        self.statmap_settings_frame_create(self.statmap_frame)

    def statmap_settings_frame_create(self, window):
        self.statmap_settings_frame = tk.LabelFrame(window)
        self.frames.append(self.statmap_settings_frame)
        self.statmap_settings_frame.place(relx=0.02, rely=0, height=140, relwidth=0.467)
        self.statmap_settings_frame.configure(text='Settings', font='Helvetica 18 bold')
        self.format_frame(self.statmap_settings_frame, borderwidth=1)

        self.statmap_save_button = ttk.Button(self.statmap_settings_frame)
        self.statmap_save_button.place(relx=0.02, rely=0.08, height=42, width=105)
        self.statmap_save_button.configure(
            command=lambda: Save_settings(['Statistical_maps'], 'maps/statmap_config.toml'), text='''Save settings''')
        Tooltip.CreateToolTip(self.statmap_save_button, 'Save all statistical map settings')

        self.statmap_reset_button = ttk.Button(self.statmap_settings_frame)
        self.statmap_reset_button.place(relx=0.5, rely=0.08, height=42, width=105)
        self.statmap_reset_button.configure(command=lambda: Reset_settings(['Statistical_maps']),
                                            text='''Reset settings''')
        Tooltip.CreateToolTip(self.statmap_reset_button, 'Reset statistical map settings')

        self.statmap_settings_button = ttk.Button(self.statmap_settings_frame)
        self.statmap_settings_button.place(relx=0.17, rely=0.55, height=42, width=150)
        self.statmap_settings_button.configure(command=lambda: self.change_frame('Statistical_maps'),
                                               text='''Settings''')

    def statmap_run_frame_create(self, window):
        self.statmap_run_frame = tk.LabelFrame(window)
        self.frames.append(self.statmap_run_frame)
        self.statmap_run_frame.place(relx=0.5, rely=0, height=140, relwidth=0.467)
        self.statmap_run_frame.configure(text='Run', font='Helvetica 18 bold')
        self.format_frame(self.statmap_run_frame, borderwidth=1)

        statmap_options = ('Image SNR', 'Temporal SNR', 'Add Gaussian noise')
        state = tk.StringVar()
        state.set(statmap_options[1])
        self.statmap_option = tk.OptionMenu(self.statmap_run_frame, state, *statmap_options)
        self.statmap_option.place(relx=0.08, rely=0.27, width=200, bordermode='ignore')
        self.statmap_option.configure(bg=self.background)
        self.statmap_option.val = state

        self.statmap_run_button = ttk.Button(self.statmap_run_frame)
        self.statmap_run_button.place(relx=0.17, rely=0.55, height=42, width=150)
        self.statmap_run_button.configure(command=lambda: button_handler('Make maps', self.statmap_option.val.get()))
        self.statmap_run_button.configure(text='''Make maps''')
        Tooltip.CreateToolTip(self.statmap_run_button, 'Make statistical maps using method chosen in the drop down '
                                                       'menu.')

    @staticmethod
    def load_initial_values(toml_file):
        with open(f"{Path(os.path.abspath(__file__)).parents[0]}/configuration_profiles/{toml_file}", 'r') as f:
            for line in f.readlines():
                if line[:2] == '##':  # Subheadings
                    continue

                elif line[0] == '#':  # Headings
                    curr_page = re.split('# |\n', line)[1]

                elif curr_page == 'Version Info':
                    continue

                elif line != '\n':
                    setting = [x.replace("'", "").strip() for x in re.split(" = |\[|\]|\n|(?<!')#.*", line) if x]

                    current_setting = Utils.convert_toml_input_to_python_object(setting[1])

                    try:
                        eval(curr_page)[setting[0]]['Current'] = current_setting
                    except KeyError:
                        warnings.warn(
                            f'"{setting[0]}" not present in config setup file. Configuration file may be outdated.')

    def banner_draw(self, window):
        img = Image.open(f'{Path(os.path.abspath(__file__)).parents[0]}/images/fRAT.gif')

        zoom = 0.8
        pixels_x, pixels_y = tuple([int(zoom * x) for x in img.size])
        img = img.resize((pixels_x, pixels_y))

        width, height = img.size
        new_width = width + 88
        result = Image.new('RGB', (new_width, height - 20), (0, 0, 0))
        result.paste(img, (44, -10))

        result = ImageTk.PhotoImage(result)

        panel = tk.Label(window, image=result, borderwidth=0)
        panel.photo = result
        panel.place(relx=0.02, y=10)

        self.frames.append(panel)

    def create_home_options_frame(self, window):
        self.Options_frame = tk.LabelFrame(window)
        self.Options_frame.place(relx=0.5, rely=0.01, height=150, relwidth=0.345)
        self.Options_frame.configure(text=f'''Options''', font='Helvetica 18 bold')
        self.format_frame(self.Options_frame, borderwidth=1)
        self.frames.append(self.Options_frame)

        self.Save_button = ttk.Button(self.Options_frame)
        self.Save_button.place(relx=0.05, y=10, height=42, width=150)
        self.Save_button.configure(command=lambda: Save_settings(pages, 'roi_analysis/fRAT_config.toml'))
        self.Save_button.configure(text='''Save settings''')
        Tooltip.CreateToolTip(self.Save_button, 'Save all fRAT settings')

        self.Reset_button = ttk.Button(self.Options_frame)
        self.Reset_button.place(relx=0.05, y=70, height=42, width=150)
        self.Reset_button.configure(command=lambda: Reset_settings(pages))
        self.Reset_button.configure(text='''Reset settings''')
        Tooltip.CreateToolTip(self.Reset_button, 'Reset fRAT settings to recommended values')

    def create_home_atlas_frame(self, window):
        self.Atlas_frame = tk.LabelFrame(window)
        self.Atlas_frame.place(relx=0.5, rely=0.348, height=128, relwidth=0.467)
        self.Atlas_frame.configure(text=f'''Atlas information''', font='Helvetica 18 bold')
        self.format_frame(self.Atlas_frame, borderwidth=1)
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

    def create_homestar_runner_frame(self, window):
        self.Run_frame = tk.LabelFrame(window)
        self.Run_frame.place(relx=0.025, rely=0.69, height=145, relwidth=0.945)
        self.Run_frame.configure(text=f'''Run''', font='Helvetica 18 bold')
        self.format_frame(self.Run_frame, borderwidth=1)
        self.frames.append(self.Run_frame)

        self.paramValues_button = ttk.Button(self.Run_frame)
        self.paramValues_button.place(x=2, y=12, height=42, width=150)
        self.paramValues_button.configure(command=lambda: button_handler('Make paramValues.csv'))
        self.paramValues_button.configure(text='''Setup parameters''')
        Tooltip.CreateToolTip(self.paramValues_button,
                              'Creates a csv table with prefilled parameter info for each file. Can be set using a command line flag instead')

        self.Print_button = ttk.Button(self.Run_frame)
        self.Print_button.place(x=156, y=12, height=42, width=150)
        self.Print_button.configure(command=lambda: button_handler('Print_results'))
        self.Print_button.configure(text='''Print results''')
        Tooltip.CreateToolTip(self.Print_button, 'Print results of fRAT to the terminal')

        self.Dash_button = ttk.Button(self.Run_frame)
        self.Dash_button.place(x=309, y=12, height=42, width=150)
        self.Dash_button.configure(command=lambda: button_handler('Run_dash'))
        self.Dash_button.configure(text='''Interactive table''')
        Tooltip.CreateToolTip(self.Dash_button, 'Create an interactive table to display fRAT results')

        self.Run_button = ttk.Button(self.Run_frame)
        self.Run_button.place(x=137, y=66, height=42, width=181)
        self.Run_button.configure(command=lambda: button_handler('Run_fRAT'))
        self.Run_button.configure(text='''Run fRAT''')
        Tooltip.CreateToolTip(self.Run_button, 'Run fRAT with current settings')

    def create_settings_frame(self, window):
        if self.page == 'Home':
            self.settings_frame = tk.LabelFrame(window)
            self.frames.append(self.settings_frame)

            self.settings_frame.place(relx=0.02, rely=0.15, relheight=0.61, relwidth=0.973)
            self.settings_frame.configure(text='fRAT', font='Helvetica 18 bold')
            self.format_frame(self.settings_frame, borderwidth=1)

            self.General_settings_frame = tk.LabelFrame(self.settings_frame)
            self.frames.append(self.General_settings_frame)
            self.General_settings_frame.place(relx=0.025, rely=0.01, height=330, relwidth=0.41)
            self.General_settings_frame.configure(text=f'''Settings''', font='Helvetica 18 bold')
            self.format_frame(self.General_settings_frame, borderwidth=1)
            current_frame = self.General_settings_frame

            y_loc = 10
            relx = 0.03
            for page in pages:
                if page == 'Home':
                    continue

                if page == 'Violin_plot':
                    break

                self.page_switch_button_setup(page, current_frame, y_loc, relx)

                y_loc += 60

        else:
            self.create_scrollable_frame(window)
            self.create_widgets(previous_frame='Home')

    def create_plot_settings_frame(self, row):
        self.plot_settings_frame = tk.LabelFrame(self.settings_frame)
        self.frames.append(self.plot_settings_frame)
        self.plot_settings_frame.grid(row=row, column=0, columnspan=1)
        self.plot_settings_frame.configure(text=f'''Specific plot settings''', font='Helvetica 18 bold')
        self.format_frame(self.plot_settings_frame, borderwidth=1)

        return row + 1

    def create_scrollable_frame(self, window):
        self.sbf = ScrollbarFrame(window)
        self.sbf.pack(side="top", fill="both", expand=True)
        self.settings_frame = self.sbf.scrolled_frame

        self.frames.extend([self.sbf, self.settings_frame])

    def create_widgets(self, previous_frame, create_return_button=True):
        row = 0

        for setting in self.current_info:
            self.label_create(setting, row, self.current_info[setting])

            if self.current_info[setting]['type'] == 'Scale':
                widget = self.scale_create(setting, self.current_info[setting], row)
                self.widgets = {**self.widgets, **widget}

            elif self.current_info[setting]['type'] == 'CheckButton':
                widget = self.checkbutton_create(setting, self.current_info[setting], row)
                self.widgets = {**self.widgets, **widget}

            elif self.current_info[setting]['type'] == 'OptionMenu':
                widget = self.optionmenu_create(setting, self.current_info[setting], row)
                self.widgets = {**self.widgets, **widget}

            elif self.current_info[setting]['type'] == 'Entry':
                widget = self.entry_create(setting, self.current_info[setting], row)
                self.widgets = {**self.widgets, **widget}

            elif self.current_info[setting]['type'] == 'Button':
                widget = self.button_create(setting, self.current_info[setting], row)
                self.widgets = {**self.widgets, **widget}

            elif self.current_info[setting]['type'] == 'Dynamic':
                widget, row = self.dynamic_widget(setting, self.current_info[setting], row)
                self.dynamic_widgets = {**self.dynamic_widgets, **widget}

            row += 1

        self.settings_frame.configure(text=f'''{self.page.replace('_', ' ')}''', font='Helvetica 18 bold')
        self.format_frame(self.settings_frame)

        if create_return_button == True:
            self.return_button_create_and_format(previous_frame, row)

    def button_create(self, name, info, row):
        func = eval(info['Command'])

        if info['Pass self']:
            self.__setattr__(name, ttk.Button(self.settings_frame, command=lambda: func(self)))
        else:
            self.__setattr__(name, ttk.Button(self.settings_frame, command=lambda: func()))

        widget = getattr(self, name)
        widget.configure(text=info['Text'])

        widget.grid(row=row, column=1, pady=WIDGET_Y_PADDING, padx=WIDGET_X_PADDING, sticky='W')

        return {name: widget}

    def return_button_create_and_format(self, previous_frame, row):
        self.return_button = ttk.Button(self.settings_frame, command=lambda: self.change_frame(previous_frame))
        self.return_button.grid(column=0, row=row, columnspan=2, pady=15)
        self.return_button.configure(text=f'''Back to {previous_frame}''')

    def format_frame(self, frame, borderwidth=0):
        frame.configure(borderwidth=borderwidth)
        frame.configure(relief='raised')
        frame.configure(background=self.background)
        frame.configure(highlightbackground=self.background)
        frame.configure(highlightcolor="black")

    def page_switch_button_setup(self, page, frame, y_loc, relx):
        self.__setattr__(page, ttk.Button(frame, command=lambda: self.change_frame(page)))
        page_switch_button = getattr(self, page)
        page_switch_button.place(relx=relx, y=y_loc, width=185, height=42)
        page_switch_button.configure(text=f'''{page.replace('_', ' ')}''')

    def dynamic_widget(self, name, info, row):
        try:
            text = eval(info['Options'])['Current']
            text = [value.strip() for value in text.split(',')]
        except KeyError:
            text = ""

        dynamic_widgets = {}

        if info['subtype'] == 'OptionMenu':
            info['DynamOptions'] = [*text]

            # If not currently set by user, set to default value
            if info['Current'] not in info['DynamOptions']:
                try:
                    info['Current'] = info['DynamOptions'][info['DefaultNumber']]
                except IndexError:
                    info['Current'] = ''

            if len(info['DynamOptions']) == 1:
                info['DynamOptions'].append('')

            widget = self.optionmenu_create(name, info, row)

            dynamic_widgets = {**dynamic_widgets, **widget}

        elif info['subtype'] == 'OptionMenu2':
            if info['Current'] == 'FILL IV TYPE AS BETWEEN-SUBJECTS':
                info['Current'] = info['Options2'][info['DefaultNumber']]

            try:
                info['Current'] = info['Current'].split(', ')
            except AttributeError:
                # Error occurs if string has already been converted to list
                pass

            if len(info['Current']) != len(text):
                info['Current'] = info['Current'] * len(text)

            for counter, value in enumerate(text):
                widget = self.optionmenu_create(f"{name}_{value}", info, row, counter)

                dynamic_widgets = {**dynamic_widgets, **widget}

                # Replace current labels and create one for each row
                self.label_create(f"{info['label']} - {value}", row, info, fixed_name=True)

                row += 1

        elif info['subtype'] == 'Checkbutton':
            if info['Current'] == 'INCLUDE ALL VARIABLES':
                info['Current'] = Parsing['parameter_dict1']['Current'].split(', ')

            for value in text:
                widget = self.checkbutton_create(f"{name}_{value}", info, row)
                [widget.configure(text=value) for widget in widget.values()]

                dynamic_widgets = {**dynamic_widgets, **widget}

                row += 1

        return dynamic_widgets, row

    def label_create(self, name, row, info=None, font=None, fixed_name=False):
        self.__setattr__(name, tk.Label(self.settings_frame))
        label_name = getattr(self, name)

        if not fixed_name:
            try:
                name = info['label']
            except KeyError:
                if info['type'] != 'subheading':
                    name = name.capitalize()

        label_name.configure(background=self.background)
        label_name.configure(foreground="#000000")

        if info['type'] != 'subheading':
            Tooltip.CreateToolTip(label_name, info['Description'])
            label_name.grid(row=row, column=0, pady=WIDGET_Y_PADDING, padx=WIDGET_X_PADDING, sticky='W')
            label_name.configure(text=f'''{name.replace("_", " ")}:''', font=font)
        else:
            label_name.grid(row=row, column=0, pady=(30, 5), ipadx=WIDGET_X_PADDING, sticky='SW')
            label_name.configure(text=name, font=('Helvetica', 17, 'bold'))

        try:
            if info['status'] == 'important':
                label_name.configure(font=('Helvetica', 14, 'bold'))
        except KeyError:
            pass

    def scale_create(self, name, info, row):
        self.__setattr__(name, tk.Scale(self.settings_frame, from_=info['From'], to=info['To']))
        widget = getattr(self, name)

        widget.grid(row=row, column=1, pady=WIDGET_Y_PADDING, padx=WIDGET_X_PADDING, sticky='W')
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

    def checkbutton_create(self, name, info, row):
        state = tk.BooleanVar()

        self.__setattr__(name, tk.Checkbutton(self.settings_frame, variable=state))
        widget = getattr(self, name)

        widget.grid(row=row, column=1, pady=WIDGET_Y_PADDING, padx=WIDGET_X_PADDING, sticky='W')
        widget.val = state

        self.checkbutton_default_settings(widget)

        if info['Current'] in ['true', 'false']:
            info['Current'] = ast.literal_eval(info['Current'].title())

        current_val = info['Current']

        # Dynamic widget handler
        if not isinstance(info['Current'], bool) and info['Current'] == 'all':
            current_val = True
        elif not isinstance(info['Current'], bool) and name.rsplit('_', 1)[1] in info['Current']:
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

    def optionmenu_create(self, name, info, row, counter=None):
        state = tk.StringVar()
        if counter is not None:
            state.set(info['Current'][counter])
        else:
            state.set(info['Current'])

        try:
            options = info['DynamOptions']
        except KeyError:
            try:
                options = info['Options2']
            except KeyError:
                options = info['Options']

        self.__setattr__(name, tk.OptionMenu(self.settings_frame, state, *options))
        widget = getattr(self, name)

        widget.grid(row=row, column=1, pady=WIDGET_Y_PADDING, padx=WIDGET_X_PADDING, sticky='W')
        widget.configure(bg=self.background)
        widget.val = state

        return {name: widget}

    def entry_create(self, name, info, row):
        self.__setattr__(name, tk.Entry(self.settings_frame))
        widget = getattr(self, name)
        widget.grid(row=row, column=1, pady=WIDGET_Y_PADDING, padx=WIDGET_X_PADDING, sticky='W')

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
        current_options2_menu = {}

        for widget in self.widgets:
            if eval(self.page)[widget]['type'] != 'Button':
                try:
                    eval(self.page)[widget]['Current'] = self.widgets[widget].get()
                except AttributeError:
                    eval(self.page)[widget]['Current'] = self.widgets[widget].val.get()

        for widget in self.dynamic_widgets:
            if self.dynamic_widgets[widget].winfo_class() == 'Checkbutton':
                params = widget.rsplit('_', 1)

                if isinstance(eval(self.page)[params[0]]['Current'], str):
                    eval(self.page)[params[0]]['Current'] = [x.strip() for x in
                                                             eval(self.page)[params[0]]['Current'].split(',')]

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
                        eval(self.page)[params[0]]['Current'].append(
                            '')  # If list is empty, set first element to blank string

            else:
                try:
                    eval(self.page)[widget]['Current'] = self.dynamic_widgets[widget].val.get()
                except KeyError:
                    # Will raise KeyError for OptionsMenu2
                    current_widget = eval(self.page)[widget.rpartition('_')[0]]

                    if current_widget['label'] in current_options2_menu:
                        current_options2_menu[current_widget['label']].append(self.dynamic_widgets[widget].val.get())
                    else:
                        current_options2_menu[current_widget['label']] = []

                    current_widget['Current'][len(current_options2_menu[current_widget['label']])] = \
                    self.dynamic_widgets[widget].val.get()

        for frame in self.frames:
            frame.destroy()

        self.frames.clear()

        self.__init__(root, page, load_initial_values=False)


class ConfigurationFiles:
    """Class container for processing configuration files."""

    analysis_config = 'fRAT_config.toml'
    statmap_config = 'statmap_config.toml'

    def update_latest_settings_file(self):
        """Update latest_settings.toml when configuration file changes"""
        pass


class Tooltip:
    def __init__(self, widget):
        self.widget = widget
        self.widget_type = widget.winfo_class()
        self.tooltip_window = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text

        if self.tooltip_window or not self.text:
            return

        if self.widget_type == 'TButton':
            x_offset = 120
            y_offset = 32
        else:
            x_offset = 57
            y_offset = 27

        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + x_offset
        y = y + cy + self.widget.winfo_rooty() + y_offset

        self.tooltip_window = tooltip_window = tk.Toplevel(self.widget)
        tooltip_window.overrideredirect(True)
        tooltip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(tooltip_window, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "12", "normal"))
        label.pack(ipadx=1)

        tooltip_window.update_idletasks()
        tooltip_window.lift()

    def hidetip(self):
        tooltip_window = self.tooltip_window
        self.tooltip_window = None

        if tooltip_window:
            tooltip_window.destroy()

    @staticmethod
    def CreateToolTip(widget, text):
        toolTip = Tooltip(widget)
        text = '\n'.join(['\n'.join(textwrap.wrap(line, 120, break_long_words=False, replace_whitespace=False))
                          for line in text.splitlines() if line.strip() != ''])

        def enter(event):
            toolTip.showtip(text)

        def leave(event):
            toolTip.hidetip()

        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)


class AutoHidingScrollbar(tk.Scrollbar):
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            # grid_remove is currently missing from Tkinter
            self.pack_forget()
        else:
            if self.cget("orient") == 'horizontal':
                self.pack(fill='x')
            else:
                self.pack(fill='y')
        tk.Scrollbar.set(self, lo, hi)


class ScrollbarFrame(tk.LabelFrame):
    """
    Extends class tk.Frame to support a scrollable Frame
    This class is independent from the widgets to be scrolled and
    can be used to replace a standard tk.Frame
    """

    def __init__(self, parent, **kwargs):
        tk.LabelFrame.__init__(self, parent, **kwargs)

        # The Scrollbar, layout to the right
        vsb = AutoHidingScrollbar()
        vsb.pack(side="right", fill="y")

        # The Canvas which supports the Scrollbar Interface, layout to the left
        self.canvas = tk.Canvas(self, borderwidth=0, background='#d9d9d9', highlightbackground='#d9d9d9')
        self.canvas.pack(side="left", fill="both", expand=True)

        # Bind the Scrollbar to the self.canvas Scrollbar Interface
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.configure(command=self.canvas.yview)

        # The Frame to be scrolled, layout into the canvas
        # All widgets to be scrolled have to use this Frame as parent
        self.scrolled_frame = tk.LabelFrame(self.canvas, background=self.canvas.cget('bg'))
        self.canvas.create_window((0, 0), window=self.scrolled_frame, anchor='center')

        # Configures the scroll region of the Canvas dynamically
        self.scrolled_frame.bind("<Configure>", self.on_configure)

        top.frames.append(vsb)

    def on_configure(self, event):
        """Set the scroll region to encompass the scrolled frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


def button_handler(command, *args):
    try:
        if command == 'Run_fRAT':
            check_stale_state()

            Save_settings(pages, f'roi_analysis/{ConfigurationFiles.analysis_config}')

            print('----- Running fRAT -----')
            fRAT(ConfigurationFiles.analysis_config)

        elif command == 'Print_results':
            printResults(ConfigurationFiles.analysis_config)

        elif command == "Make paramValues.csv":
            Save_settings(pages, f'roi_analysis/{ConfigurationFiles.analysis_config}')
            make_table()

        elif command == "Run_dash":
            dash_report.main(f'roi_analysis/{ConfigurationFiles.analysis_config}')

        elif command == "Make maps":
            Save_settings(['Statistical_maps'], f'maps/{ConfigurationFiles.statmap_config}')
            statmap_calc(args[0], __version__, ConfigurationFiles.statmap_config)

    except Exception as err:
        if err.args[0] == 'No folder selected.':
            print('----- Exiting -----\n')
        else:
            log = logging.getLogger(__name__)
            log.exception(err)
            sys.exit()


def run_tests(GUI):
    print('----- Running tests -----')
    # Hack to save all settings on general settings page, to be changed in the future when callbacks are implemented
    GUI.change_frame('General')
    Save_settings(pages, f'roi_analysis/{ConfigurationFiles.analysis_config}')

    path_to_example_data = find_example_dataset()

    # Create tSNR maps and run ROI analysis
    statmap_calc('Temporal SNR', __version__, 'test_config.toml', path_to_example_data)
    fRAT('test_config.toml', path_to_example_data)

    # Run tests to check if output of fRAT matches the example data
    roi_output_test = TestDifferences([f'{path_to_example_data}/sub-02/statmaps/test_maps',
                                       f'{path_to_example_data}/sub-02/statmaps/temporalSNR_report'],
                                      General['verbose_errors']['Current'])

    voxelwise_map_test = TestDifferences([f'{path_to_example_data}/test_ROI_report',
                                          f'{path_to_example_data}/HarvardOxford-Cortical_ROI_report'],
                                         General['verbose_errors']['Current'])

    # Delete files
    if General['delete_test_folder']['Current'] == 'Always' \
            or (General['delete_test_folder']['Current'] == 'If completed without error'
                and voxelwise_map_test.status == 'No errors'
                and roi_output_test.status == 'No errors'):
        shutil.rmtree(f'{path_to_example_data}/test_ROI_report')
        shutil.rmtree(f'{path_to_example_data}/sub-01/statmaps/test_maps')
        shutil.rmtree(f'{path_to_example_data}/sub-02/statmaps/test_maps')

        print('\nDeleted test folders.')

    else:
        print('\nRetaining test folders.')

    if voxelwise_map_test.status == 'No errors' and roi_output_test.status == 'No errors':
        print("\n--- End of installation testing, no errors found ---")
    else:
        warnings.warn("\n--- End of installation testing, errors found ---")


def find_example_dataset():
    example_data_folders = glob(os.path.expanduser('~/Documents/fRAT/*example_data*'))

    if not os.path.exists(os.path.expanduser('~/Documents/fRAT/')):
        raise FileNotFoundError(f'fRAT folder not present in documents directory.\n'
                                'Run mkdir ~/Documents/fRAT in the terminal, then download the example dataset from '
                                'https://osf.io/pbm3d/, extract it and place it into this folder.')

    if not example_data_folders:
        raise FileNotFoundError(f'No "example_data" folder in fRAT directory.\n'
                                f'Download it from https://osf.io/pbm3d/ and place it into the fRAT directory '
                                f'("~/Documents/fRAT/").')

    if len(example_data_folders) > 1:
        raise FileExistsError('More than one example dataset folder found in fRAT directory. '
                              'Only keep the version that matches your fRAT version.')

    return example_data_folders[0]


def check_stale_state():
    current_critical_params = Parsing['parameter_dict1']['Current'].split(', ')

    dynamic_widgets = itemgetter('table_cols', 'table_rows',
                                 'brain_table_cols', 'brain_table_rows',
                                 'single_roi_fig_colour', 'single_roi_fig_x_axis',
                                 'histogram_fig_x_facet', 'histogram_fig_y_facet',
                                 'brain_table_col_labels', 'brain_table_row_labels')(Plotting)

    dynamic_widgets += itemgetter('IV_type', 'include_as_variable')(Statistics)

    for counter, widget in enumerate(dynamic_widgets):
        if widget['Current'] == '' and len(current_critical_params) > 1:
            # If current value is blank but shouldn't be as number of critical params is above 1
            dynamic_widgets[counter]['Current'] = current_critical_params[dynamic_widgets[counter]['DefaultNumber']]

        elif widget['Recommended'] == 'CHANGE TO DESIRED LABEL' and widget['Current'] != 'CHANGE TO DESIRED LABEL':
            # If brain table column and row labels have been changed, do not change again
            # This allows these labels to be updated automatically if user never enters the brain table menu
            # But does not change to the default value if labels have been set either manually or automatically
            pass

        elif widget['Current'] == 'FILL IV TYPE AS BETWEEN-SUBJECTS':
            # Set all IV's to between subjects if not been modified already
            dynamic_widgets[counter]['Current'] = [widget['Options2'][widget['DefaultNumber']]] * len(
                current_critical_params)

        elif widget['Current'] == 'INCLUDE ALL VARIABLES':
            dynamic_widgets[counter]['Current'] = current_critical_params

        elif not widget['Current'] == '' and widget['Current'] not in current_critical_params \
                and not widget['Recommended'] in ['FILL IV TYPE AS BETWEEN-SUBJECTS', 'INCLUDE ALL VARIABLES']:
            try:
                dynamic_widgets[counter]['Current'] = current_critical_params[dynamic_widgets[counter]['DefaultNumber']]
            except IndexError:
                # If only one parameter has been given in critical params
                dynamic_widgets[counter]['Current'] = ''


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
    for roiLabelLine in atlas_label_dict['atlas']['data']['label']:
        roiArray.append(roiLabelLine['#text'])

    roiArray.extend(['No ROI', 'Overall'])
    roiArray = sorted(roiArray)

    print(f"----------------------------\n{selection} Atlas:\n----------------------------")

    for roi_num, roi in enumerate(roiArray):
        print("{roi_num}: {roi}".format(roi_num=roi_num, roi=roi))

    print("----------------------------\n")


def Save_settings(page_list, file):
    with open(f'{Path(os.path.abspath(__file__)).parents[0]}/configuration_profiles/{file}', 'w') as f:
        f.write(f"# Version Info\n")
        f.write(f"version = '{__version__}'\n")
        f.write("\n")

        for page in page_list:
            if page == 'Home':
                continue

            f.write(f"# {page}\n")

            for key in eval(page).keys():
                if eval(page)[key]['type'] == 'subheading':
                    f.write(f"\n## {key}\n")
                    continue
                elif eval(page)[key]['type'] == 'Button':
                    continue

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
                            eval(page)[key]['Current'] = list(
                                ast.literal_eval(eval(page)[key]['Current']))  # Try to convert to list

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
                    # Take out the string 'all' for Dynamic Checkbuttons, which will be the case if Recommended is set
                    # to 'all'
                    if eval(page)[key]['type'] == 'Dynamic' \
                            and eval(page)[key]['subtype'] == 'Checkbutton' \
                            and 'all,' in eval(page)[key]['Current']:
                        eval(page)[key]['Current'] = eval(page)[key]['Current'].replace('all, ', '')

                    offset = ' ' * (
                            80 - len(f"{key} = {[val.strip() for val in eval(page)[key]['Current'].split(',')]}"))
                    f.write(f"{key} = {[val.strip() for val in eval(page)[key]['Current'].split(',')]}  "
                            f"{offset}# {description}\n")

            f.write("\n")

        f.flush()
        f.close()

    print(f'----- Saved config file: {file.split("/")[-1]} -----')


def Reset_settings(pages):
    for page in pages:
        if page == 'Home':
            continue

        for key in eval(page).keys():
            if eval(page)[key]['type'] not in ['subheading', 'Button']:
                eval(page)[key]['Current'] = eval(page)[key]['Recommended']

    print('----- Reset fRAT settings to recommended values, save them to retain these settings -----')


def make_table():
    config = Utils.load_config(f'{Path(os.path.abspath(__file__)).parents[0]}/configuration_profiles/roi_analysis',
                               ConfigurationFiles.analysis_config)  # Load config file

    print('--- Creating paramValues.csv ---')
    print('Select the base directory.')
    base_directory = Utils.file_browser(title='Select the base directory', chdir=False)

    participant_dirs = find_participant_dirs(base_directory, config)

    data = []
    for participant_dir in participant_dirs:
        if config.make_folder_structure:
            file_loc = f'{participant_dir}'

        else:
            file_loc = f'{participant_dir}/{config.parsing_folder}'

        brain_file_list = Utils.find_files(file_loc, "hdr", "nii.gz", "nii")

        if config.make_folder_structure and not brain_file_list:
            raise FileNotFoundError(f'No files found in {participant_dir}.'
                                    f'\nAs make folder structure is set to true, '
                                    f'files to be scanned should be placed into this root directory and not a '
                                    f'subdirectory.')

        brain_file_list = [Utils.strip_ext(brain) for brain in brain_file_list]
        brain_file_list.sort()

        for file in brain_file_list:
            # Try to find parameters to prefill table
            keys, brain_file_params = parse_params_from_file_name(file, config)
            data.append([os.path.split(participant_dir)[-1], file, *brain_file_params, np.NaN, np.NaN])

        if config.make_folder_structure:
            create_folder_structure(participant_dir, config)

    df = pd.DataFrame(columns=['Participant', 'File name',
                               *keys,
                               'Ignore file during analysis? (y for yes, otherwise blank)',
                               'Baseline parameter combination for statistics (y for yes, otherwise blank)'],
                      data=data)

    df.to_csv(f'{base_directory}/paramValues.csv', index=False)

    print(f"\nparamValues.csv saved in {base_directory}."
          f"\n\nMake sure values in paramValues.csv are correct before continuing analysis. Also make sure anatomy "
          f"scans were not present when setting up parameters, or these lines will need to be removed from "
          f"paramValues.csv."
          f"\nIf the csv file contains unexpected parameters, update the parsing options in the GUI and rerun setup "
          f"parameters, or manually update them.")

    if config.make_folder_structure:
        print(f"\nSet up folder structure and moved fMRI volumes into {config.parsing_folder} directory.")

    if config.automatically_create_statistics_options_file and config.parameter_dict1 != ['']:
        create_statistics_file(directory=base_directory)

    elif config.automatically_create_statistics_options_file and config.parameter_dict1 == ['']:
        print(f"\nstatisticsOptions.csv not saved, no independent variables given in parsing menu.")


def create_statistics_file(directory=''):
    Save_settings(pages, f'roi_analysis/{ConfigurationFiles.analysis_config}')

    config = Utils.load_config(f'{Path(os.path.abspath(__file__)).parents[0]}/configuration_profiles/roi_analysis',
                               ConfigurationFiles.analysis_config)

    if not directory:
        directory = Utils.file_browser('Select base directory or report output directory')

    table, folder_type = Utils.load_paramValues_file(directory=directory)

    if folder_type == 'report_folder':
        # Go back to base folder if report folder is selected
        directory += '/..'

    data = []

    for parameter in config.parameter_dict1:
        data.append([parameter, 'Calculate main effect'])

        table.columns = [x.lower() for x in table.columns]

        unique_vals = table[parameter.lower()].unique()
        unique_vals.sort()

        combinations = list(itertools.combinations(sorted(unique_vals), 2))

        for combination in combinations:
            data.append([' v '.join(str(v) for v in combination), np.NaN])

        data.extend([[np.NaN, np.NaN],
                     [parameter, 'Calculate simple effect', 'Exclude from analysis']])

        for value in unique_vals:
            data.append([value, np.NaN, np.NaN])

        data.append([np.NaN, np.NaN, np.NaN])

    df = pd.DataFrame(data=data)
    df.to_csv(f'{directory}/statisticsOptions.csv', index=False, header=False)

    print('--- Created statisticsOptions.csv ---')


def create_noise_file():
    print('--- Creating noiseValues.csv ---')
    Utils.load_config(f'{Path(os.path.abspath(__file__)).parents[0]}/configuration_profiles/maps',
                      ConfigurationFiles.statmap_config)

    directory = Utils.file_browser('Select base directory')
    _, participant_names = Utils.find_participant_dirs(directory=directory)

    data = []
    for participant in participant_names:
        data.append([participant, np.NaN, np.NaN])

    df = pd.DataFrame(columns=['Participant', 'Noise over time', 'Background noise'],
                      data=data)

    df.to_csv(f'{directory}/noiseValues.csv', index=False)
    print('--- Created noiseValues.csv ---')


def create_folder_structure(participant, config):
    direcs = ['func', 'anat', 'fslfast', 'statmaps']

    if config.parsing_folder not in direcs:
        direcs.append(config.parsing_folder)

    for direc in direcs:
        Utils.check_and_make_dir(f"{participant}/{direc}")

    brain_file_list = Utils.find_files(participant, "hdr", "nii.gz", "nii", "json")
    for file in brain_file_list:
        Utils.move_file(file, participant, f'{participant}/{config.parsing_folder}', rename_copy=False)


def find_participant_dirs(base_directory, config):
    participant_dirs = [direc for direc in glob(f"{base_directory}/*") if re.search("sub-[0-9]+", direc)]

    if len(participant_dirs) == 0:
        raise FileNotFoundError('Participant directories not found.\n'
                                'Make sure participant directories are labelled e.g. sub-01 and the selected '
                                'directory contains all participant directories.')
    elif config.verbose:
        print(f'Found {len(participant_dirs)} participant folders.')

    return participant_dirs


def parse_params_from_file_name(json_file_name, cfg=config):
    """Search for MRI parameters in each json file name for use in table headers and created the combined json."""
    param_nums = []
    keys = []

    for key in cfg.parameter_dict:
        parameter = cfg.parameter_dict[key]  # Extract search term

        if parameter == '':
            continue

        elif key in cfg.binary_params:
            param = re.search("{}".format(parameter), json_file_name, flags=re.IGNORECASE)

            if param is not None:
                param_nums.append('On')  # Save 'on' if parameter is found in file name
            else:
                param_nums.append('Off')  # Save 'off' if parameter not found in file name

        else:
            # Float search
            param = re.search("{}[0-9][p|.][0-9]".format(parameter), json_file_name, flags=re.IGNORECASE)
            if param is not None:
                param_nums.append(param[0][1] + "." + param[0][-1])

            # If float search didn't work then integer search
            else:
                param = re.search("{}[0-9]".format(parameter), json_file_name, flags=re.IGNORECASE)
                if param is not None:
                    param_nums.append(param[0][-1])  # Extract the number from the parameter

                else:
                    param_nums.append(str(param))  # Save None if parameter not found in file name

        keys.append(key)

    return keys, param_nums


def path_check():
    try:
        os.environ['FSLDIR']
    except KeyError:
        raise IsADirectoryError("$FSLDIR not in path")

    if os.path.isdir(f"{os.environ['FSLDIR']}/bin"):
        print(f"\nFSL directory found in {os.environ['FSLDIR']}.")
    else:
        raise IsADirectoryError(f"bin folder not found in {os.environ['FSLDIR']}")


if __name__ == '__main__':
    start_gui()
