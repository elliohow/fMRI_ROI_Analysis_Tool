"""Configuration file"""
pages = ['Home', 'General', 'Analysis', 'Parsing', 'Plotting', 'Violin_plot',
         'Brain_table', 'Region_barchart', 'Region_histogram']

'''General settings'''
General = {
    'run_analysis': {'type': 'CheckButton', 'Recommended': 'true',
                     'Description': 'true or false. Can skip this step if json files have already been created.',
                     'label': 'Run analysis'},

    'run_plotting': {'type': 'CheckButton', 'Recommended': 'true',
                     'Description': 'true or false.', 'label': 'Run plotting'},

    'run_statistics': {'type': 'CheckButton', 'Recommended': 'true',
                       'Description': 'true or false.', 'label': 'Run statistics'},

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

    'file_cleanup': {'type': 'OptionMenu', 'Recommended': 'move', 'Options': ['move', 'delete'], 'save_as': 'string',
                     'Description': 'Move or delete intermediate files.', 'label': 'File cleanup method'},
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

    'output_folder': {'type': 'Entry', 'Recommended': 'DEFAULT', 'save_as': 'string', 'label': 'Output directory',
                      'Description': 'Directory to save output. If set to DEFAULT, output directory will be set to '
                                     'the cortical atlas used appended with "_ROI_report". '
                                     '\nExample: HarvardOxford-Cortical_ROI_report/'},

    'dof': {'type': 'Entry', 'Recommended': 12, 'label': 'DOF',
            'Description': 'Degrees of freedom for FLIRT. Recommended: 12'},

    'anat_align': {'type': 'CheckButton', 'Recommended': 'true',
                   'label': 'Align to anatomical volume',
                   'Description': 'true or false. Recommended: true.'

                                  '\nNote: anatomical file should: be placed in the sub-{id}/anat/ directory, already be '
                                  'skull stripped and be the only file in this directory. optiBET is recommended for skull stripping.'},

    'grey_matter_segment': {'type': 'CheckButton', 'Recommended': 'true', 'label': 'Use FSL FAST segmentation',
                            'Description': 'true or false. Recommended: true. '
                                           '\nNote: Requires anatomical align be true to function. FSL FAST '
                                           'segmentation files should be placed in the sub-{id}/fslfast/ '
                                           'directory. Only the FSL FAST file appended with pve_1 needs to be in this '
                                           'directory, however if all files output by FAST are placed in this '
                                           'directory, then fRAT will find the necessary file.'},

    'fslfast_min_prob': {'type': 'Scale', 'Recommended': 0.1, 'From': 0, 'To': 1, 'Resolution': 0.05,
                         'label': 'fslFAST minimum probability', 'Description': 'Recommended: 0.1'},

    'outlier_detection_method': {'type': 'OptionMenu', 'Recommended': 'individual',
                                 'Options': ['individual', 'pooled'],
                                 'save_as': 'string',
                                 'Description': "Method to run outlier detection. Options: 'individual' or 'pooled'\n"
                                                "If set to individual, outlier detection will take place for each subject "
                                                "individual. If set to pooled, data will be pooled before running outlier "
                                                "detection.\n"
                                                "Note: if set to individual, NIFTI files will be created for each "
                                                "subject, showing which files have been excluded using the outlier "
                                                "detection methods selected. However this is not possible for pooled "
                                                "outlier detection.\n"
                                                "Recommended: individual."},

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

    'stat_map_folder': {'type': 'Entry', 'Recommended': 'temporalSNR_report/', 'save_as': 'string',
                        'label': 'Statistical map folder',
                        'Description': 'Folder name which contains the statistical map files. '
                                       'Example: temporalSNR_report/'},

    'stat_map_suffix': {'type': 'Entry', 'Recommended': '_tSNR.nii.gz', 'save_as': 'string',
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
}

'''Parsing settings'''
Parsing = {
    'parameter_dict1': {'type': 'Entry', 'Recommended': 'MB, SENSE', 'save_as': 'list', 'label': 'Critical parameters',
                        'Description': 'Comma-separated list of parameter names to be parsed for and plotted. '
                                       '\n As these critical parameters will also be used when labelling the rows and '
                                       'columns of both the violin plots and histograms, they should be written as '
                                       'you want them to appear in these figures.'},

    'parameter_dict2': {'type': 'Entry', 'Recommended': 'mb, s', 'save_as': 'list',
                        'label': 'Critical parameter abbreviation',
                        'Description': 'Comma-separated list of terms to parse the file name for. Each entry '
                                       'corresponds to a critical parameter above. \nOptional if using table parameter '
                                       'verification, however if the file name contains this information it can use '
                                       'this information to auto-detect the critical parameters used for each fMRI '
                                       'volume.'},

    'make_folder_structure': {'type': 'CheckButton', 'Recommended': 'true',
                              'Description': 'true or false. Make folder structure when creating paramValues.csv'},
}

'''General plot settings'''
Plotting = {
    'plot_dpi': {'type': 'Entry', 'Recommended': 200, 'label': 'Figure DPI',
                 'Description': 'Recommended value 600'},

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
                          'Description': "Provide a comma-separated list of regions to plot e.g. [3, 5], the string "
                                         "'all' for all rois or the string 'Runtime' to provide regions at runtime."},
}

'''Brain table settings'''
Brain_table = {
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

    'brain_x_coord': {'type': 'Entry', 'Recommended': 91,
                      'Description': 'Voxel location to slice the images at in the x axis. Recommended settings for both variables: 91 or 58'},

    'brain_z_coord': {'type': 'Entry', 'Recommended': 91,
                      'Description': 'Voxel location to slice the images at in the z axis. Recommended settings for both variables: 91 or 58'},

    'brain_table_col_labels': {'type': 'Entry', 'Recommended': 'MB', 'save_as': 'string', 'label': 'Column labels',
                               'Description': 'Label for columns.'},

    'brain_table_row_labels': {'type': 'Entry', 'Recommended': 'SENSE', 'save_as': 'string', 'label': 'Row labels',
                               'Description': 'Label for rows.'},

    'brain_table_cols': {'type': 'Dynamic', 'Recommended': 'MB', 'Options': 'Parsing["parameter_dict1"]',
                         'subtype': 'OptionMenu', 'save_as': 'string',
                         'DefaultNumber': 0, 'Description': ''},

    'brain_table_rows': {'type': 'Dynamic', 'Recommended': 'SENSE', 'Options': 'Parsing["parameter_dict1"]',
                         'subtype': 'OptionMenu', 'save_as': 'string',
                         'DefaultNumber': 1, 'Description': ''}
}

'''Violin plot settings'''
Violin_plot = {

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

    'table_cols': {'type': 'Dynamic', 'Recommended': 'MB', 'Options': 'Parsing["parameter_dict1"]',
                   'subtype': 'OptionMenu', 'save_as': 'string', 'DefaultNumber': 0,
                   'Description': ''},

    'table_rows': {'type': 'Dynamic', 'Recommended': 'SENSE', 'Options': 'Parsing["parameter_dict1"]',
                   'subtype': 'OptionMenu', 'save_as': 'string', 'DefaultNumber': 1,
                   'Description': ''}
}

'''One region bar chart'''
Region_barchart = {

    'single_roi_fig_label_x': {'type': 'Entry', 'Recommended': 'Multiband factor', 'save_as': 'string',
                               'Description': ""},

    'single_roi_fig_label_y': {'type': 'Entry', 'Recommended': 'temporal Signal to Noise Ratio', 'save_as': 'string',
                               'Description': ""},

    'single_roi_fig_label_fill': {'type': 'Entry', 'Recommended': 'SENSE factor', 'save_as': 'string',
                                  'Description': ""},

    'single_roi_fig_x_axis': {'type': 'Dynamic', 'Recommended': 'MB', 'Options': 'Parsing["parameter_dict1"]',
                              'subtype': 'OptionMenu', 'save_as': 'string', 'DefaultNumber': 0,
                              'Description': ''},

    'single_roi_fig_colour': {'type': 'Dynamic', 'Recommended': 'SENSE', 'Options': 'Parsing["parameter_dict1"]',
                              'subtype': 'OptionMenu', 'save_as': 'string', 'DefaultNumber': 1,
                              'Description': ''},

}

'''Region histogram'''
Region_histogram = {

    'histogram_binwidth': {'type': 'Entry', 'Recommended': 5, 'Description': ""},

    'histogram_fig_label_x': {'type': 'Entry', 'Recommended': 'temporal Signal to Noise Ratio', 'save_as': 'string',
                              'Description': ''},

    'histogram_fig_label_y': {'type': 'Entry', 'Recommended': 'Frequency', 'save_as': 'string',
                              'Description': ''},

    'histogram_stat_line_size': {'type': 'Entry', 'Recommended': 1.5,
                                 'Description': ''},

    'histogram_show_mean': {'type': 'CheckButton', 'Recommended': 'true',
                            'Description': 'true or false.'},

    'histogram_show_median': {'type': 'CheckButton', 'Recommended': 'true',
                              'Description': 'true or false.'},

    'histogram_show_legend': {'type': 'CheckButton', 'Recommended': 'true',
                              'Description': 'true or false.'},

    'histogram_fig_x_facet': {'type': 'Dynamic', 'Recommended': 'MB', 'Options': 'Parsing["parameter_dict1"]',
                              'subtype': 'OptionMenu', 'save_as': 'string', 'DefaultNumber': 0,
                              'Description': ''},

    'histogram_fig_y_facet': {'type': 'Dynamic', 'Recommended': 'SENSE',
                              'Options': 'Parsing["parameter_dict1"]',
                              'subtype': 'OptionMenu', 'save_as': 'string',
                              'DefaultNumber': 1, 'Description': ''},

    'histogram_fig_colour': {'type': 'Dynamic', 'Recommended': '#fc4e2a',
                             'Options': 'Plotting["colorblind_friendly_plot_colours"]',
                             'subtype': 'OptionMenu', 'save_as': 'string',
                             'Description': 'Hex value of colour blind friendly colour. Value taken from colorblind friendly plot colours.'}
}
