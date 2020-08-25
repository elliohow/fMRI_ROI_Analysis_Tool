"""Configuration file"""
# A copy of this file will be saved in

'''Nipype settings'''
brain_extract = True  # True or false. Select True to use BET. Recommended: True
frac_inten = 0.4  # Fractional intensity threshold for BET. Default: 0.5
dof = 12  # Degrees of freedom for FLIRT. Recommended: 12
use_freesurf_file = True  # True or false. Select True to use freesurfer segmentation to only calculate statistics for grey matter voxels.

run_analysis_steps = "all"  # TODO: True or false. Select false if json files have already been created.
save_stats_only = True  # Will save intermediate NiPype files if set to False. Recommended: True
stat_map_folder = 'stat_map/'  # Folder name which contains the statistical map files
stat_map_suffix = '_statmap.nii'  # File name suffix of the statistical map files. Include the file extension.
bootstrap = False

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
# MRI parameters to parse file names for. Key indicates parameter name and value indicates how it would be represented
# in the file name.
parameter_dict = {"MB": "mb",
                  "SENSE": "s",
                  "Half scan": "half"}
binary_params = ["Half scan"]  # Add parameters here which will either be off or on
# Set skip_verify_params to true if you are confident the file names reflect the MRI parameters. Recommended: False
skip_verify_params = False


'''General plot settings'''
plot_dpi = 100  # Recommended value 600
plot_font_size = 30  # Recommended value 30
plot_scale = 10  # Recommended value 10

make_figure_table = True  # True or False
make_brain_table = True  # True or False
make_one_region_fig = False  # True or False


'''Brain image figure'''
# 'brain_fig_value_min' and 'brain_fig_value_max' can be changed to provide cutoff values. For example, set min to 50
# and max to 100 to make areas with values below 50 to disappear and values over 100 to be set to the same
# bright colour.
brain_fig_value_min = 0  # Recommended value 0
brain_fig_value_max = None  # Recommended value None. Note: will default to 100 for scaled maps.

# brain_plot_file options =
# 0: _Mean_atlas.nii.gz
# 1: _Mean_roi_scaled_atlas.nii.gz",
# 2: _Mean_global_scaled_atlas.nii.gz"
brain_fig_file = 0  # Number corresponding to the options. e.g. 2 for global_scaled_atlas.

brain_table_cols = 'MB'  # String should be a key from the parameter_dict
brain_table_rows = 'SENSE'  # String should be a key from the parameter_dict

brain_table_x_size = 50  # Change the size of the x-axis. Recommended: 50
brain_table_y_size = 10  # Change the size of the y-axis. Recommended: 10

# Voxel location to slice the images at in the x and z axes. Recommended settings for both variables: 91 or 58
brain_x_coord = 91
brain_z_coord = 91

'''Two parameter table'''
table_cols = 'MB'  # String should be a key from the parameter_dict
table_rows = 'SENSE'  # String should be a key from the parameter_dict

table_y_label = 'ROI'
table_x_label = 'TSNR mean'


'''One region figure'''
# Provide a list of regions to plot e.g. [3, 5] or 'all' for all rois. Or use None to provide regions at runtime.
single_roi_fig_regions = None

# Figure 'aesthetics'
single_roi_fig_x_axis = 'MB'
single_roi_fig_colour = 'SENSE'

# Figure labels
single_roi_fig_label_x = "Multiband factor"
single_roi_fig_label_y = "temporal Signal to Noise Ratio"
single_roi_fig_label_fill = "SENSE factor"

# Figure colours
single_roi_fig_colours = ['#ffeda0', '#feb24c', '#fc4e2a', '#bd0026'] # Hex values of colour blind friendly colour scale
