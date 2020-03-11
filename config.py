"""Configuration file"""
'''Nipype settings'''
file_type = "nifti"  # File extension of statistical parameter map files. todo: change

brain_extract = True  # True or false. Select True to use BET. Recommended: True
frac_inten = 0.4  # Fractional intensity threshold for BET. Default: 0.5
dof = 12  # Degrees of freedom for FLIRT. Recommended: 12

run_analysis = False  # True or false. Select false if json files have already been created.
save_stats_only = True  # Will save intermediate NiPype files if set to False. Recommended: True


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

mb_search = "mb"  # TODO: implement this and the rest of the searching

'''Parsing settings'''
# MRI parameters to parse file names for. Key indicates parameter name and value indicates how it would be represented
# in the file name.
parameter_dict = {"MB": "mb",
                  "SENSE": "s",
                  "Half scan": "half"}
skip_verify_params = True  # Set to true if you are confident the file names reflect the MRI parameters. Recommended: False

'''General plot settings'''
plot_dpi = 100  # Recommended value 600
plot_font_size = 30  # Recommended value 30
plot_scale = 10  # Recommended value 10

make_figure_table = False
make_brain_table = False
make_one_region_fig = False

# Parameters to plot in the 2D table todo
plot_MB = "True"
plot_SENSE = "True"
plot_half = "False"

'''Brain image figure'''

# 'brain_fig_value_min' and 'brain_fig_value_max' can be changed to provide cutoff values. For example, set min to 50
# and max to 100 to make areas with values below 50 to disappear and values over 100 to be set to the same
# bright colour.
brain_fig_value_min = 0  # Recommended value 0
brain_fig_value_max = None  # Recommended value None. Note: will default to 100 for scaled maps.

# brain_plot_file_extension options =
#    "_Mean_atlas.nii.gz",
#    "_Mean_roi_scaled_atlas.nii.gz",
#    "_Mean_global_scaled_atlas.nii.gz"
brain_fig_file_extension = "_Mean_global_scaled_atlas.nii.gz" # todo

'''Two parameter table'''
table_cols = 'MB'
table_rows = 'SENSE'
table_y_label = 'ROI'
table_x_label = 'TSNR mean'

'''One region figure'''
# Provide a list of regions to plot e.g. [3, 5] or 'all' for all rois. Or use None to provide regions at runtime.
single_roi_fig_regions = None

single_roi_fig_x_axis = 'MB'
single_roi_fig_colour = 'SENSE'

single_roi_fig_label_x = "Multiband factor"
single_roi_fig_label_y = "temporal Signal to Noise Ratio"
single_roi_fig_label_fill = "SENSE factor"
single_roi_fig_colours = ['#ffeda0', '#feb24c', '#fc4e2a', '#bd0026'] # Hex values of colour blind friendly colour scale