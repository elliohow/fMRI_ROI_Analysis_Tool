"""Configuration file"""
pages = ['Home', 'General', 'Analysis', 'Statistics', 'Parsing', 'Plotting']

'''General settings'''
General = {
    'run_analysis': {'type': 'CheckButton', 'Recommended': 'true',
                     'status': 'important',
                     'Description': 'true or false. Can skip this step if json files have already been created.',
                     'label': 'Run analysis'},

    'run_statistics': {'type': 'CheckButton', 'Recommended': 'true',
                       'status': 'important',
                       'Description': 'true or false.', 'label': 'Run statistics'},

    'run_plotting': {'type': 'CheckButton', 'Recommended': 'true',
                     'status': 'important',
                     'Description': 'true or false.', 'label': 'Run plotting'},

    'verbose': {'type': 'CheckButton', 'Recommended': 'true',
                'Description': 'true or false.', 'label': 'Verbose fRAT stages'},

    'verbose_cmd_line_args': {'type': 'CheckButton', 'Recommended': 'false',
                              'Description': 'true or false.', 'label': 'Verbose command line arguments'},

    'multicore_processing': {'type': 'CheckButton', 'Recommended': 'true',
                             'Description': 'true or false. Use multicore processing during analysis? Multicore processing currently works within participants not between them. Recommended: true'},

    'max_core_usage': {'type': 'OptionMenu', 'Recommended': 'max', 'Options': ['max', 6, 5, 4, 3, 2, 1],
                       'save_as': 'string',
                       'Description': "'max' to select number of cores available on the system, alternatively an int to manually select number of cores to use. Recommended: 'max'"},

    'brain_file_loc': {'type': 'Entry', 'Recommended': "", 'save_as': 'string',
                       'label': 'Base directory of subjects',
                       'Description': 'Either the absolute location of brain files or blank, if blank then a browser window will allow you to search for the files at runtime. If passing in this information as a command line flag, this will be ignored.'},

    'report_output_folder': {'type': 'Entry', 'Recommended': "", 'save_as': 'string',
                             'label': 'fRAT output directory location',
                             'Description': 'Either the absolute location of json files or blank, if blank then a browser window will allow you to search for the files at runtime. If passing in this information as a command line flag, this will be ignored.'},

    'averaging_type': {'type': 'OptionMenu',
                       'Recommended': 'Participant averaged',
                       'Options': ['Session averaged', 'Participant averaged'],
                       'save_as': 'string',
                       'Description': 'Participant averaged or Session averaged.\n'
                                      'This setting is used to determine which statistics to use for plotting, '
                                      'and when accessing results (for example through the '
                                      'interactive report). \nNote: Histograms will always use the raw results. The '
                                      'linear mixed model from the statistics will always use session averaged data'},

    'parameter_file': {'type': 'Entry', 'Recommended': "paramValues.csv", 'save_as': 'string',
                       'Description': 'Recommended: paramValues.csv\n'
                                      'Name of the file to parse for critical params. Option added to allow quick '
                                      'swapping between different parameter files.'},

    'file_cleanup': {'type': 'OptionMenu', 'Recommended': 'move', 'Options': ['move', 'delete'], 'save_as': 'string',
                     'Description': 'Move or delete intermediate files.', 'label': 'File cleanup method'},

    'Installation testing': {'type': 'subheading'},

    'run_tests': {'type': 'Button', 'Command': 'run_tests', 'Text': 'Run installation tests', 'Pass self': True,
                  'Description': 'true or false. Run tests to verify fRAT output of current installation matches that '
                                 'in the "example_data" folder.'},

    'delete_test_folder': {'type': 'OptionMenu',
                           'Recommended': 'Always',
                           'Options': ['Always', 'If completed without error', 'Never'],
                           'save_as': 'string',
                           'Description': 'Option to choose whether the folder generated while running tests is '
                                          'deleted upon completion.'},

    'verbose_errors': {'type': 'CheckButton', 'Recommended': 'false',
                       'Description': 'true or false. '
                                      'Print all missing files and differences found during testing to the terminal.'}
}

'''Analysis settings'''
Analysis = {
    'atlas_number': {'type': 'OptionMenu', 'Recommended': 'HarvardOxford-cort', 'save_as': 'string', 'label': 'Atlas',
                     'Options': [
                         'Cerebellum-MNIflirt',
                         'Cerebellum-MNIfnirt',
                         'HarvardOxford-cort',
                         'HarvardOxford-sub',
                         'JHU-ICBM-labels',
                         'JHU-ICBM-tracts',
                         'juelich',
                         'MNI',
                         'SMATT-labels',
                         'STN',
                         'striatum-structural',
                         'Talairach-labels',
                         'Thalamus'
                     ],
                     'Description': ''},

    'input_folder_name': {'type': 'Entry', 'Recommended': "func_cleaned", 'save_as': 'string',
                          'Description': 'Folder found in each subjects directory containing the files to be analysed. '
                                         'func_cleaned is the default option as this folder will automatically be '
                                         'created when making statmaps. If the "Noise volume included in time series" '
                                         'option was set to true, or motion outlier removal was used when creating '
                                         'the statmaps, this folder will contain cleaned versions of the original func '
                                         'files. However if these options were not used when creating the statmaps, '
                                         'the folder will still be present, however the files will be identical to '
                                         'those in the "func" folder.'},

    'output_folder': {'type': 'Entry', 'Recommended': 'DEFAULT', 'save_as': 'string', 'label': 'Output directory',
                      'Description': 'Directory to save output. If set to DEFAULT, output directory will be set to '
                                     'the cortical atlas used appended with "_ROI_report". '
                                     '\nExample: HarvardOxford-Cortical_ROI_report/'},

    'dof': {'type': 'Entry', 'Recommended': 12, 'label': 'DOF',
            'Description': 'Degrees of freedom for FLIRT (only used for the fMRI to anatomical alignment when using '
                           'Correlation Ratio cost function). Recommended: 12'},

    'anat_align_cost_function': {'type': 'OptionMenu',
                                 'label': 'fMRI to anatomical cost function',
                                 'Recommended': 'BBR',
                                 'Options': ['BBR', 'Correlation Ratio'],
                                 'save_as': 'string',
                                 'Description': 'BBR or Correlation Ratio. Recommended: BBR.\n'
                                                'Using BBR (Boundary-Based Registration) requires an FSL FAST '
                                                'segmentation (this will be automatically created if necessary if '
                                                'the Run FSL FAST option is set to "Run if files not found") '
                                                'and a wholehead non-brain extracted anatomical placed in the anat '
                                                'folder.'},

    'grey_matter_segment': {'type': 'CheckButton',
                            'Recommended': 'true', 'label': 'Use FAST to only include grey matter',
                            'Description': 'true or false. Recommended: true if using a cortical atlas.'
                                           '\nNote: FSL FAST segmentation files should be placed in the '
                                           'sub-{id}/fslfast/ directory. Only the FSL FAST file appended with pve_1 '
                                           'needs to be in this directory, however if all files output by FAST are '
                                           'placed in this directory, then fRAT will find the necessary file.'},

    'run_fsl_fast': {'type': 'OptionMenu',
                     'Options': ['Run if files not found', 'Never run'],
                     'Recommended': 'Run if files not found',
                     'Description': 'Recommended: "Run if files not found".\n These files will only be searched for '
                                    '(and thus created) if "Use FSL FAST segmentation" is set to true.',
                     'label': 'Run FSL FAST',
                     'save_as': 'string'},

    'fslfast_min_prob': {'type': 'Scale', 'Recommended': 0.1, 'From': 0, 'To': 1, 'Resolution': 0.05,
                         'label': 'FSL FAST minimum probability', 'Description': 'Recommended: 0.1'},

    'stat_map_folder': {'type': 'Entry', 'Recommended': '', 'save_as': 'string',
                        'status': 'important',
                        'label': 'Statistical map folder',
                        'Description': 'Folder name which contains the statistical map files. '
                                       'Example: temporalSNR_report'},

    'stat_map_suffix': {'type': 'Entry', 'Recommended': '_tSNR.nii.gz', 'save_as': 'string',
                        'status': 'important',
                        'label': 'Statistical map suffix',
                        'Description': 'File name suffix of the statistical map files. Include the file extension. '
                                       'Example: _tSNR.img'},

    # 'bootstrap': {'type': 'CheckButton', 'Recommended': 'false',
    #               'Description': 'true or false. Calculate bootstrapped mean and confidence intervals using 10,000 iterations'},

    'conf_level_number': {'type': 'OptionMenu', 'Recommended': '95%, 1.96',
                          'Options': ['80%, 1.28', '85%, 1.44', '90%, 1.64', '95%, 1.96', '98%, 2.33', '99%, 2.58'],
                          'save_as': 'string', 'label': 'Confidence level',
                          'Description': 'Set the confidence level for confidence interval calculations.\n'
                                         'Numbers represent the confidence level and the corresponding critical z value.\n'
                                         'Recommended: 95%, 1.96.'},

    'binary_params': {'type': 'Dynamic', 'Recommended': [''], 'Options': 'Parsing["parameter_dict1"]',
                      'subtype': 'Checkbutton', 'label': 'Binary parameters',
                      'save_as': 'list', 'Description': 'Add parameters here which will either be on or off.'},

    'Outlier detection': {'type': 'subheading'},

    'noise_cutoff': {'type': 'CheckButton', 'Recommended': 'true',
                     'Description': 'true or false. Calculate a noise cutoff based on voxels not assigned an '
                                    'ROI or that have been excluded from analysis. Voxels with values of 0 are '
                                    'not included when calculating the noise cutoff, Recommended: true.'},

    'gaussian_outlier_detection': {'type': 'CheckButton', 'Recommended': 'true',
                                   'label': 'Gaussian outlier (GauO) detection',
                                   'Description': 'true or false. Fit a gaussian to the data to determine outliers using Elliptic Envelope. '
                                                  'Recommended: true.'},

    'gaussian_outlier_contamination': {'type': 'Scale', 'Recommended': 0.1, 'From': 0, 'To': 1, 'Resolution': 0.01,
                                       'label': 'GauO contamination percentage',
                                       'Description': 'Percent of expected outliers in dataset\n'
                                                      f'Recommended: 0.1'},

    'gaussian_outlier_location': {'type': 'OptionMenu', 'Recommended': 'below gaussian',
                                  'Options': ['below gaussian', 'above gaussian', 'both'],
                                  'save_as': 'string', 'label': 'GauO location',
                                  'Description': 'Data to remove (if gaussian outlier detection is true).\n'
                                                 'For example: if set to below gaussian, data below the gaussian will be removed.\n'
                                                 'Recommended: below gaussian.'},
}

'''Statistics settings'''
Statistics = {
    'create_statistics_options_file': {'type': 'Button', 'Command': 'create_statistics_file', 'Pass self': 'false',
                                       'Text': 'Create statisticsOptions.csv',
                                       'Description': 'Create file in base folder to choose statistics options. Such as '
                                                      'which parameters to exclude from analysis and what main effect '
                                                      'contrasts to calculate.'},

    'automatically_create_statistics_options_file': {'type': 'CheckButton',
                                                     'Recommended': 'true',
                                                     'Description': 'true or false.\n'
                                                                    "Usually statisticsOptions.csv is automatically created when creating "
                                                                    "paramValues.csv. Deselect this option if you won't be running "
                                                                    "the statistics step. The create statisticsOptions.csv button "
                                                                    "above can also be used to manually create this file."},

    'statistics_subfolder_name': {'type': 'Entry',
                                  'Recommended': 'stats',
                                  'save_as': 'string',
                                  'Description': 'Directory name for statistics folder.'},

    'print_result': {'type': 'CheckButton', 'Recommended': 'true',
                     'label': 'Print results to terminal',
                     'Description': 'true or false. Prints results to terminal if true in addition to saving results '
                                    'to file.'},

    'minimum_voxels': {'type': 'Entry',
                       'Recommended': 400,
                       'Description': 'For bootstrapped change versus baseline, for each ROI, the average number of '
                                      'voxels per session for an ROI must be above this value to be included in the '
                                      'analysis.'
                                      'For running the linear mixed model, for each ROI, any sessions with a voxel '
                                      'count below this value will be removed.'
                                      '\nHighly recommended to set a value here, as ROIs with a small number of voxels '
                                      'may suggest poor fitting.'
                                      '\nRecommended value 400'},

    'bootstrap_samples': {'type': 'Entry', 'Recommended': 1000,
                          'Description': 'Recommended value 1000. \n'
                                         'Note: Bootstrapping is only used to calculate percentage change versus '
                                         'baseline.'},

    'bootstrap_confidence_interval': {'type': 'Entry', 'Recommended': 95,
                                      'Description': 'Recommended value: 95 \n'
                                                     'Note: Bootstrapping is only used to calculate percentage change '
                                                     'versus baseline.'},

    'regional_stats_rois': {'type': 'Entry', 'Recommended': 'all', 'save_as': 'list',
                            'label': 'ROIs to calculate statistics for',
                            'Description': "Provide a comma-separated list of regions, e.g. '3, 5', the string "
                                           "'all' for all rois or the string 'Runtime' to provide regions at runtime."},

    'include_as_variable': {'type': 'Dynamic', 'Recommended': 'INCLUDE ALL VARIABLES',
                            'status': 'important',
                            'Options': 'Parsing["parameter_dict1"]',
                            'subtype': 'Checkbutton',
                            'save_as': 'list',
                            'Description': 'Select which variables to include in statistical analysis.\nUsed to '
                                           'determine which variables to use as fixed effects in linear mixed models '
                                           'and which variables to take into account when balancing data for the main '
                                           'effect t test data.'},

    'brain_map_p_thresh': {'type': 'Entry', 'Recommended': 0.05,
                           'label': 'Coefficient map p-threshold',
                           'Description': "P-value threshold to use when creating brain coefficient maps. "
                                          "Any fixed effect that doesn't have a p-value equal to or less than this "
                                          "value will not be included in the coefficient map.\n",
                           'status': 'important'},

    'T-tests': {'type': 'subheading'},

    'run_t_tests': {'type': 'CheckButton', 'status': 'important',
                    'Recommended': 'true',
                    'Description': 'true or false.\n'},

    'IV_type': {'type': 'Dynamic', 'Recommended': 'FILL IV TYPE AS BETWEEN-SUBJECTS',
                'Options': 'Parsing["parameter_dict1"]',
                'subtype': 'OptionMenu2', 'save_as': 'list', 'Options2': ['Within-subjects', 'Between-subjects'],
                'label': 'IV type', 'status': 'important', 'DefaultNumber': 1,
                'Description': 'Type of variable collected. Used to choose which t-test to use for pairwise '
                               'comparisons.'},

    # 'glm_statistic': {'type': 'OptionMenu',
    #                   'label': 'GLM statistic for overall effect',
    #                   'Recommended': 'Mean',
    #                   'Options': ['Mean', 'Median'],
    #                   'save_as': 'string',
    #                   'Description': 'Mean or Median.\nNote: this setting is only used when '
    #                                  'looking at the overall effect, as statistics ran on individual ROIs will use raw '
    #                                  'voxel values.'},

    'Linear mixed models': {'type': 'subheading'},

    'run_linear_mixed_models': {'type': 'CheckButton', 'status': 'important',
                                'Recommended': 'true',
                                'Description': 'true or false.\n'},

    'categorical_variables': {'type': 'Dynamic', 'Recommended': [''], 'Options': 'Parsing["parameter_dict1"]',
                              'subtype': 'Checkbutton',
                              'save_as': 'list',
                              'Description': 'Select which variables (if any) are categorical. Used for the LMM.'},

    'main_effects': {'type': 'CheckButton',
                     'Recommended': 'true',
                     'label': 'Compute main effects',
                     'Description': 'true or false.\n'
                                    'Note: This option is independent from the other effect calculations.'},

    'main_and_interaction_effects': {'type': 'CheckButton',
                                     'Recommended': 'true',
                                     'label': 'Compute main and interaction effects',
                                     'Description': 'true or false.\n'
                                                    'Note: This option is independent from the other effect calculations.'},

    'interaction_effects': {'type': 'CheckButton',
                            'Recommended': 'false',
                            'label': 'Compute interaction effects',
                            'Description': 'true or false.\n'
                                           'Note: This option is independent from the other effect calculations.'},

    'R2 vs voxel count LMM': {'type': 'subheading'},

    'max_below_thresh': {'type': 'Entry', 'Recommended': 0, 'label': 'Maximum percent of sessions below thresh',
                             'Description': 'Recommended value 0. \n'
                                            'For a given ROI, this value sets the maximum percent of sessions that can '
                                            'be excluded (due to having insufficient voxel count) before the ROI is not '
                                            'included in r2 vs voxel count statistics.'
                                            '\nWith the default value of 100(%) an ROI will not be included in this '
                                            'calculation if any sessions have been excluded.'},
}

'''Parsing settings'''
Parsing = {
    'parameter_dict1': {'type': 'Entry',
                        'status': 'important',
                        'Recommended': 'MB, SENSE', 'save_as': 'list', 'label': 'Critical parameters',
                        'Description': 'Comma-separated list of independent variables. '
                                       '\n As these critical parameters will also be used when labelling the rows and '
                                       'columns of both the violin plots and histograms, they should be written as '
                                       'you want them to appear in these figures.'
                                       '\nNote: This field can also be blank.'},

    'parameter_dict2': {'type': 'Entry',
                        'status': 'important',
                        'Recommended': 'mb, s', 'save_as': 'list',
                        'label': 'Critical parameter abbreviation',
                        'Description': 'Comma-separated list of terms to parse the file name for. Each entry '
                                       'corresponds to a critical parameter above. \nOptional if using table parameter '
                                       'verification, however if the file name contains this information it can use '
                                       'this information to auto-detect the critical parameters used for each fMRI '
                                       'volume.'
                                       '\nNote: This field can be blank.'},

    'make_folder_structure': {'type': 'CheckButton', 'Recommended': 'true',
                              'Description': 'true or false. Make folder structure when creating paramValues.csv'},

    'parsing_folder': {'type': 'Entry', 'Recommended': "func", 'save_as': 'string',
                       'Description': 'Folder to find files to add to paramValues.csv. If using "Make folder '
                                      'structure" option, this will be the directory the files in the participant '
                                      'folder will be moved to.'},
}

'''General plot settings'''
Plotting = {
    'General plot settings': {'type': 'subheading'},

    'plot_dpi': {'type': 'Entry', 'Recommended': 450, 'label': 'Figure DPI',
                 'Description': 'Recommended value 450'},

    'plot_font_size': {'type': 'Entry', 'Recommended': 40, 'label': 'Figure font size',
                       'Description': 'Recommended value 30'},

    'plot_scale': {'type': 'Entry', 'Recommended': 10, 'label': 'Figure scale',
                   'Description': 'Recommended value 10'},

    'make_violin_plot': {'type': 'CheckButton', 'Recommended': 'true', 'label': 'Make violin plots',
                         'Description': 'true or false.'},

    'make_brain_table': {'type': 'CheckButton', 'Recommended': 'true', 'label': 'Make brain visualisations',
                         'Description': 'true or false.'},

    'make_one_region_fig': {'type': 'CheckButton', 'Recommended': 'true', 'label': 'Make regional barcharts',
                            'Description': 'true or false.'},

    'make_histogram': {'type': 'CheckButton', 'Recommended': 'true', 'label': 'Make regional histograms',
                       'Description': 'true or false.'},

    'colorblind_friendly_plot_colours': {'type': 'Entry', 'Recommended': '#ffeda0, #feb24c, #fc4e2a, #bd0026',
                                         'save_as': 'list',
                                         'Description': 'Hex values of colourblind friendly colour scale.'},

    'regional_fig_rois': {'type': 'Entry', 'Recommended': 'all', 'save_as': 'list', 'label': 'ROIs to plot',
                          'Description': "Provide a comma-separated list of regions to plot e.g. '3, 5', the string "
                                         "'all' for all rois or the string 'Runtime' to provide regions at runtime."},

    'Brain table': {'type': 'subheading'},

    'brain_tight_layout': {'type': 'CheckButton', 'Recommended': 'false',
                           'Description': 'true or false. Use a tight layout when laying out the figure. Recommended: false'},

    'brain_fig_value_min': {'type': 'Entry', 'Recommended': 0, 'label': 'Minimum median and mean value',
                            'Description': 'Provides the minimum value of the colourbar when creating mean and median '
                                           'images. For example, set minimum to 50 to make areas with values below 50 '
                                           'appear black.\n'
                                           'Recommended value: 0'},

    'brain_fig_value_max': {'type': 'Entry', 'Recommended': None, 'label': 'Maximum median and mean value',
                            'Description': 'Provides the maximum value of the colourbar when creating mean and median '
                                           'images. For example, set maximum to 50 to make areas with values above 50 '
                                           'appear as the brighest colour on the colourbar.\n'
                                           'Recommended value: None. Note: will default to 100 for scaled maps.'},

    'brain_x_coord': {'type': 'Entry', 'Recommended': -1,
                      'Description': 'Voxel location to slice the images at in the x axis. '
                                     'Recommended settings for both variables: 91 or 58'},

    'brain_z_coord': {'type': 'Entry', 'Recommended': 19,
                      'Description': 'Voxel location to slice the images at in the z axis. '
                                     'Recommended settings for both variables: 91 or 58'},

    'brain_table_col_labels': {'type': 'Entry', 'Recommended': 'CHANGE TO DESIRED LABEL', 'save_as': 'string',
                               'label': 'Column labels',
                               'DefaultNumber': 0, 'Description': 'Label for columns.'},

    'brain_table_row_labels': {'type': 'Entry', 'Recommended': 'CHANGE TO DESIRED LABEL', 'save_as': 'string',
                               'label': 'Row labels',
                               'DefaultNumber': 1, 'Description': 'Label for rows.'},

    'brain_table_cols': {'type': 'Dynamic', 'Recommended': 'DEFAULT', 'Options': 'Parsing["parameter_dict1"]',
                         'subtype': 'OptionMenu', 'save_as': 'string',
                         'DefaultNumber': 0, 'Description': ''},

    'brain_table_rows': {'type': 'Dynamic', 'Recommended': 'DEFAULT', 'Options': 'Parsing["parameter_dict1"]',
                         'subtype': 'OptionMenu', 'save_as': 'string',
                         'DefaultNumber': 1, 'Description': ''},

    'Violin plot': {'type': 'subheading'},

    'table_x_label': {'type': 'Entry', 'Recommended': 'tSNR mean', 'save_as': 'string',
                      'Description': ''},

    'table_y_label': {'type': 'Entry', 'Recommended': 'ROI', 'save_as': 'string',
                      'Description': ''},

    'violin_show_data': {'type': 'CheckButton', 'Recommended': 'true', 'label': 'Show data points',
                         'Description': 'true or false.'},

    'violin_jitter': {'type': 'CheckButton', 'Recommended': 'true', 'label': 'Jitter data points',
                      'Description': 'true or false.'},

    'violin_colour': {'type': 'Dynamic', 'Recommended': '#fc4e2a',
                      'Options': 'Plotting["colorblind_friendly_plot_colours"]',
                      'subtype': 'OptionMenu', 'save_as': 'string',
                      'Description': 'Hex value of colour blind friendly colour. Value taken from colorblind friendly plot colours.'},

    'boxplot_colour': {'type': 'Dynamic', 'Recommended': '#feb24c',
                       'Options': 'Plotting["colorblind_friendly_plot_colours"]',
                       'subtype': 'OptionMenu', 'save_as': 'string',
                       'Description': 'Hex value of colour blind friendly colour. Value taken from colorblind friendly plot colours.'},

    'table_cols': {'type': 'Dynamic', 'Recommended': 'DEFAULT', 'Options': 'Parsing["parameter_dict1"]',
                   'subtype': 'OptionMenu', 'save_as': 'string', 'DefaultNumber': 0,
                   'Description': ''},

    'table_rows': {'type': 'Dynamic', 'Recommended': 'DEFAULT', 'Options': 'Parsing["parameter_dict1"]',
                   'subtype': 'OptionMenu', 'save_as': 'string', 'DefaultNumber': 1,
                   'Description': ''},

    'Regional bar chart': {'type': 'subheading'},

    'single_roi_fig_label_x': {'type': 'Entry', 'Recommended': 'Multiband factor', 'save_as': 'string',
                               'Description': ""},

    'single_roi_fig_label_y': {'type': 'Entry', 'Recommended': 'temporal Signal to Noise Ratio', 'save_as': 'string',
                               'Description': ""},

    'single_roi_fig_label_fill': {'type': 'Entry', 'Recommended': 'SENSE factor', 'save_as': 'string',
                                  'Description': ""},

    'single_roi_fig_x_axis': {'type': 'Dynamic', 'Recommended': 'DEFAULT', 'Options': 'Parsing["parameter_dict1"]',
                              'subtype': 'OptionMenu', 'save_as': 'string', 'DefaultNumber': 0,
                              'Description': ''},

    'single_roi_fig_colour': {'type': 'Dynamic', 'Recommended': 'DEFAULT', 'Options': 'Parsing["parameter_dict1"]',
                              'subtype': 'OptionMenu', 'save_as': 'string', 'DefaultNumber': 1,
                              'Description': ''},

    'Regional histogram': {'type': 'subheading'},

    'histogram_binwidth': {'type': 'Entry', 'Recommended': 2, 'label': 'Bin width', 'Description': ""},

    'histogram_fig_label_x': {'type': 'Entry', 'Recommended': 'temporal Signal to Noise Ratio',
                              'save_as': 'string', 'label': 'x-axis label',
                              'Description': ''},

    'histogram_fig_label_y': {'type': 'Entry', 'Recommended': 'Frequency',
                              'save_as': 'string', 'label': 'y-axis label',
                              'Description': ''},

    'histogram_stat_line_size': {'type': 'Entry', 'Recommended': 1.5, 'label': 'Statistic line size',
                                 'Description': ''},

    'histogram_show_mean': {'type': 'CheckButton', 'Recommended': 'true', 'label': 'Show mean',
                            'Description': 'true or false.'},

    'histogram_show_median': {'type': 'CheckButton', 'Recommended': 'true', 'label': 'Show median',
                              'Description': 'true or false.'},

    'histogram_show_legend': {'type': 'CheckButton', 'Recommended': 'true', 'label': 'Show legend',
                              'Description': 'true or false.'},

    'histogram_fig_x_facet': {'type': 'Dynamic', 'Recommended': 'DEFAULT', 'label': 'x-axis facet',
                              'Options': 'Parsing["parameter_dict1"]',
                              'subtype': 'OptionMenu', 'save_as': 'string', 'DefaultNumber': 0,
                              'Description': ''},

    'histogram_fig_y_facet': {'type': 'Dynamic', 'Recommended': 'DEFAULT', 'label': 'y-axis facet',
                              'Options': 'Parsing["parameter_dict1"]',
                              'subtype': 'OptionMenu', 'save_as': 'string',
                              'DefaultNumber': 1, 'Description': ''},

    'histogram_fig_colour': {'type': 'Dynamic', 'Recommended': '#fc4e2a', 'label': 'Bin colour',
                             'Options': 'Plotting["colorblind_friendly_plot_colours"]',
                             'subtype': 'OptionMenu', 'save_as': 'string',
                             'Description': 'Hex value of colour blind friendly colour. Value taken from colorblind friendly plot colours.'}
}
