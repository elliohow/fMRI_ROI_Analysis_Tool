"""Configuration file"""
# A copy of this file will be saved in

'''Nipype settings'''
frac_inten = 0.4  # Fractional intensity threshold for BET. Default: 0.5
dof = 12  # Degrees of freedom for FLIRT. Recommended: 12
motion_correct = False  # True or False. Note: This will inflate values such as tSNR.
use_freesurf_file = False  # True or False. Select True to use freesurfer segmentation to only calculate statistics for grey matter voxels.

verbose = True  # True or False
make_table_only = False # True or False. If true, a csv file template containing brain file information is created and then the program is terminated. Can be set using a command line flag instead.
run_steps = "all"  # "all", "analyse", or "plot". "analyse" to only run analysis steps, "plot" if json files have already been created or "all" to run all steps.
save_stats_only = True  # Will save intermediate NiPype files if set to False. Recommended: True
stat_map_folder = 'QA_report/'  # Folder name which contains the statistical map files
stat_map_suffix = '_tSNR.img'  # File name suffix of the statistical map files. Include the file extension.
bootstrap = False
multicore_processing = True  # True or False. Use multicore processing to use during analysis? Recommended: True
max_core_usage = 'max'  # 'max' to select number of cores available on the system, alternatively an int to manually select number of cores to use. Recommended: 'max'

brain_file_loc = ""  # Either the absolute location of brain files or blank, if blank then a browser window will allow you to search for the files at runtime.If passing in this information as a command line flag, this will be ignored.
json_file_loc = ""  # Either the absolute location of json files or blank, if blank then a browser window will allow you to search for the files at runtime. If passing in this information as a command line flag, this will be ignored.

# atlas_number options =
#  0: Cerebellum/Cerebellum-MNIflirt-maxprob-thr0-1mm.nii.gz
#  1: Cerebellum/Cerebellum-MNIfnirt-maxprob-thr0-1mm.nii.gz
#  2: HarvardOxford/HarvardOxford-cort-maxprob-thr0-1mm.nii.gz
#  3: HarvardOxford/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz
#  4: JHU/JHU-ICBM-labels-1mm.nii.gz
#  5: JHU/JHU-ICBM-tracts-maxprob-thr0-1mm.nii.gz
#  6: Juelich/Juelich-maxprob-thr0-1mm.nii.gz
#  7: MNI/MNI-maxprob-thr0-1mm.nii.gz
#  8: SMATT/SMATT-labels-1mm.nii.gz
#  9: STN/STN-maxprob-thr0-0.5mm.nii.gz
#  10: Striatum/striatum-structural-1mm.nii.gz
#  11: Talairach/Talairach-labels-1mm.nii.gz
#  12: Thalamus/Thalamus-maxprob-thr0-1mm.nii.gz
atlas_number = 2  # Number corresponding to the atlas above. e.g. 2 for Harvard oxford cortical atlas

# conf_level_number options (confidence level, corresponding critical z value)=
#  0: 80%, 1.28
#  1: 85%, 1.44
#  2: 90%, 1.64
#  3: 95%, 1.96
#  4: 98%, 2.33
#  5: 99%, 2.58
conf_level_number = 3  # Set the confidence level for confidence interval calculations. Recommended: 3

# roi_stat_list options =
#  0: Voxel number
#  1: Mean
#  2: Standard Deviation
#  3: Confidence Interval
#  4: Min
#  5: Max
roi_stat_number = 1  # Set the statistic to scale the brain figures by. Recommended: 1

# Provide a list of rois to include in analysis e.g. [3, 5] or "all" for all rois. Recommended: "all"
include_rois = "all"
# Provide a list of rois to exclude in analysis e.g. [3, 5] or "none" to exclude none. Recommended: "none"
exclude_rois = "none"

'''Parsing settings'''
# MRI parameters to parse in the format of a dict. Key indicates parameter name and value indicates how it would be represented in the file name (if using name parameter searching) If using table parameter searching, values can be blank.
parameter_dict = {"MB": "mb",
                  "SENSE": "s"}
binary_params = []  # Add parameters here which will either be off or on.
# Set verify_param_method to true if you are confident the file names reflect the MRI parameters. Recommended: False
verify_param_method = "table"  # "table", "name" or "manual". How to find parameter values: "table" finds values from spreadsheet document, "name" finds values from file name, "manual" allows you to manually input parameters at runtime.
param_table_name = "paramValues.csv"  # Optional. Only used if verify_param_method equals "table". Only change if creating a param table manually. Default value: "paramValues.csv"
always_replace_combined_json = True  # True or False. Recommended: True

'''General plot settings'''
plot_dpi = 200  # Recommended value 600
plot_font_size = 40  # Recommended value 30
plot_scale = 10  # Recommended value 10

# Figure colours
colorblind_friendly_plot_colours = ['#ffeda0', '#feb24c', '#fc4e2a', '#bd0026']  # Hex values of colour blind friendly colour scale

make_scatter_table = True  # True or False
make_brain_table = True  # True or False
make_one_region_fig = True  # True or False
make_histogram = True  # True or False

'''Brain facet grid'''
# 'brain_fig_value_min' and 'brain_fig_value_max' can be changed to provide cutoff values. For example, set min to 50
# and max to 100 to make areas with values below 50 to disappear and values over 100 to be set to the same
# bright colour.
brain_fig_value_min = 0  # Recommended value 0
brain_fig_value_max = None  # Recommended value None. Note: will default to 100 for scaled maps.

brain_tight_layout = False  # True or False. Use a tight layout when laying otu the figure. Recommended: False

# brain_plot_file options =
# 0: _Mean.nii.gz
# 1: _Mean_within_roi_scaled.nii.gz",
# 2: _Mean_mixed_roi_scaled.nii.gz"
# 3: Produce all three figures
brain_fig_file = 3  # Number corresponding to the options. e.g. 2 for mixed_roi_scaled.

brain_table_cols = 'MB'  # String should be a key from the parameter_dict
brain_table_rows = 'SENSE'  # String should be a key from the parameter_dict

brain_table_x_size = 40  # Change the size of the x-axis. Recommended: 50
brain_table_y_size = 10  # Change the size of the y-axis. Recommended: 10

# Voxel location to slice the images at in the x and z axes. Recommended settings for both variables: 91 or 58
brain_x_coord = 91
brain_z_coord = 91

'''Two parameter scatter plot'''
table_cols = 'MB'  # String should be a key from the parameter_dict
table_rows = 'SENSE'  # String should be a key from the parameter_dict

table_x_label = 'TSNR mean'
table_y_label = 'ROI'

table_row_order = 'both'  # 'roi', 'stat' or 'both'. Recommended: 'both'

'''One region bar chart'''
# Provide a comma-separated list of regions to plot e.g. 3, 5 or 'all' for all rois. Or use None to provide regions at runtime.
single_roi_fig_regions = None

# Figure 'aesthetics'
single_roi_fig_x_axis = 'MB'
single_roi_fig_colour = 'SENSE'

# Figure labels
single_roi_fig_label_x = "Multiband factor"
single_roi_fig_label_y = "temporal Signal to Noise Ratio"
single_roi_fig_label_fill = "SENSE factor"

'''Region histogram'''
# Provide a comma-separated list of regions to plot e.g. 3, 5 or 'all' for all rois. Or use None to provide regions at runtime.
histogram_fig_regions = None # TODO: Make x scale variable

histogram_binwidth = 5

# Figure faceting
histogram_fig_x_facet = 'MB'
histogram_fig_y_facet = 'SENSE'

# Figure colours
histogram_fig_colour = '#fc4e2a'  # Hex value of colour blind friendly colour

# Figure labels
histogram_fig_label_x = "temporal Signal to Noise Ratio"
histogram_fig_label_y = "Frequency"

histogram_stat_line_size = 1.5
histogram_show_mean = True
histogram_show_median = True
histogram_show_legend = True
