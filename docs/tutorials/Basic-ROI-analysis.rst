==================
Basic ROI analysis
==================
.. note::
    This tutorial will focus on how to use the GUI version of the fRAT, as while many settings and functions can be
    accessed without the GUI, it is suggested that the GUI is used where possible until you already have familiarity
    with fRAT.

File setup
----------
.. tip::
    In the GUI, settings that will most often need changing are **bold**. Additionally, most settings have a tooltip
    giving an explanation of what the setting changes, and if relevant, how the format the setting expects.

Before being able to run the ROI analysis, a few initial setup steps need to be taken. Firstly, the base folder should
be structured with functional files organised into folders named using the format ``sub-{number}`` (e.g. ``sub-42``):

.. image:: images/input_folder_subjects.png
    :width: 200

Next, enter the "Parsing" options menu found under the "Settings" panel in the GUI. The "critical parameters" setting
indicates what are the independent variables of the experiment and will be used when labelling plots, whereas the
"critical parameters abbreviation" setting indicates how these variables are represented in the filenames of the scans
(if at all). For example, the "critical parameters" option may contain "Multiband, Sense", whereas the "critical
parameters abbreviation" option will instead contain "mb, s". In this case, the file name ``P1_MB3_S2_matchBW.nii``
will represent that multiband 3 and SENSE 2 were used.

After filling these options in, select the "Make folder structure" option. This will create the basic folder structure
required for the fRAT and will sort the files into the correct directory. After returning to the home
page, click the "Setup parameters" button in the "Run" panel of the GUI. This will parse files names for critical
parameters using the "critical parameters abbreviation" option set above, with the output being saved in
``paramValues.csv``. This file should be checked before proceeding to make sure the correct parameters have been applied
to each file. Alternatively, when running without the GUI, pass the --make_table flag when running ``fRAT.py``, e.g.
``fRAT.py --make_table``. After using the parsing GUI option, the necessary directories will be created with the files
put into the folder ``func``:

.. image:: images/input_folder_parsed.png
    :width: 300

.. warning::
    Make sure the lists of critical parameters given are in the same order, otherwise the critical parameter names
    will be applied to the wrong abbreviations.

In the newly created ``anat`` folder, place a single skull stripped anatomical volume with the suffix "_brain". The
default BBR cost function for functional to anatomical registration requires a second, whole head, non-brain extracted
volume to also be placed in the ``anat`` folder. This second file should not contain the word "brain" in the file
name. While the BBR cost function also requires segmentation to have been ran using FAST, FAST will be automatically ran
when running the analysis if this is not the case. However if you wish to run the segmentation ahead of time,
the FAST output should be placed in the ``fslfast`` folder. If running the analysis on cortical regions (for example
when using the Harvard-Oxford Cortical atlas), it is recommended that white matter and extracranial voxels are excluded
from the analysis by setting the GUI option "Use FAST to only include grey matter" to true. While there is not
currently support for using FAST to improve analysis of subcortical regions, support may be added in the future.

If FAST needs to be ran (either for BBR registration or to include only grey matter in the analysis) the accuracy of the
skull stripped anatomical scans should still be assessed before running it. As overly conservative skull stripping can
lead to skull being retained in the Resulting image, which FAST may then misidentify as grey matter.
Conversely, overly liberal skull stripping can lead to parts of the brain being removed, meaning that these voxels will
also not be included in any ROIs.

.. note::
    To skull strip the anatomical files, it is highly recommended that optiBET_ is used as it has consistently produced
    the best brain extraction accuracy.

Voxel-wise tSNR map creation
----------------------------
Before creating the tSNR maps, click the "Settings" button in the "Statistical maps" section of the GUI. The default
options will normally be sufficient, however if a noise scan has been added to the functional volumes, make sure under
the "Image SNR calculation" header that information about this noise volume is given. This allows the fRAT to remove it
when creating tSNR maps, and if creating iSNR maps, it will be used to calculate the noise value.

After inspecting the settings, to create the tSNR maps return to the home screen and in the "Run" panel of the
"Statistical maps" section, select "Temporal SNR" from the dropdown menu then click "Make maps". A file explorer will
appear allowing you to navigate to the base folder where your subject folders are located. After selecting this base
folder, the tSNR will be created for each participant. During creation of the maps, the folder ``func_cleaned``
will be created, which contains functional volumes better suited to be used for the ROI analysis.

.. image:: images/input_folder_statistics.png
    :width: 300

.. note::
    The ``changes_made_to_files.txt`` contain details of how the files have been cleaned. While ``func_cleaned`` is the
    default folder that the ROI analysis will search for function volumes in, if you are unhappy with using
    these files over the original files, this option can be changed using the ``Input folder name``
    setting on the analysis screen of the GUI.

Running the ROI analysis
------------------------
The same process for creating voxel-wise maps applies here, check each options menu from the "Settings" panel in the
"fRAT" section of the GUI and then click the "Run fRAT" button to run the ROI analysis when you are ready to run the
analysis. Again, the default options should be sufficient, however the **bolded** options are the ones most likely to
need changing. In particular, the "General" option menu allows ROI analysis pipeline steps to be skipped if desired.
Further, the "Statistical map folder" setting in the "Analysis" option menu should be changed to "temporalSNR_report".
If you wish to analyse any of the other files output by the tSNR map creation, the "Statistical map suffix" option can
for example be changed to "tStd.nii.gz". The ``Atlas information`` option on the home page allows you to print the
ROIs and their corresponding numeric key for the atlas chosen from the dropdown menu. This allows you to both select
which atlas is more appropriate to use for analysis, but also allows you to specify using this numeric tags which
ROIs to produce figures and statistics for, e.g. ``1,5,32`` to choose specific ROIs or ``all`` for all ROIs.

.. note::
    If running the ``Plotting`` or ``Statistics`` steps separately, the folder output by the analysis should be selected
    instead of the base folder.

Exploring ROI analysis output
-----------------------------
After running the analysis, in addition to the folders created before, the base folder will now contain the newly created
output folder:

.. image:: images/input_folder_analysis.png
    :width: 300

Here is the folder structure of the output folder:

.. image:: images/output_folder.png
    :width: 300


In the folder structure above:

- ``additional_info.csv`` contains further information about files such as the displacement values as measured during motion correction
- ``analysis_log.toml`` is the configuration files used to run the analysis step (logs are also output for the statistics and plotting steps)
- ``copy_paramValues.csv`` is the parameter values used for the analysis
- ``index.html`` is used to open the html report output by the plotting step
- ``Statistics`` contains the statistical analysis output
- ``Figures`` contains folders for each plot type created
- ``fRAT_report`` contains the pages of the html report
- ``Overall`` contains the final results, summarised across participants/sessions
- ``sub-{number}`` contains the results computed for each participant

In the ``Overall`` folder, ``Summarised_results`` will contain ``Participant_averaged_results`` and
``Session_averaged_results``, with each folder containing a separate file for each parameter combination and also a
``combined_results.json`` file which combines the data from every other file in this folder.
Participant averaged data first averages data within participants before being averaging between participants.
Whereas session averaged results instead averages data between all sessions,
disregarding which participant was scanned in each session; this can be useful where the statistical map being converted
should be participant agnostic.

The ``NIFTI_ROI`` folder also found in the ``Overall`` folder contains the results from the the
``Participant_averaged_results`` and ``Session_averaged_results`` folders in ``.nii.gz`` format, with a separate file
created for each statistic type and parameter combination. These are used for the brain grid figures during the plotting
step. There are 3 types of ``NIFTI_ROI`` files:

* Standard (no suffix)
* Within ROI scaled
* Between ROI scaled

The "standard" files contain the actual statistic values for each ROI. "Within ROI scaled" and "between ROI scaled"
files scale the values to a range between 0 and 100, with "Within ROI scaled" files have scaled each ROI based on the
maximum value seen within that ROI across all parameters, and "between ROI scaled" files have scaled each ROI based on
the maximum value seen across all ROIs across all parameters.

``Raw_results``, found in both the ``Overall`` and the ``sub-{number}`` folders contains the value of every voxel in
each ROI. This data is used to produce histograms in the plotting step.

In each ``sub-{number}`` folder, ``Summarised_results`` in addition to the results from each session for this participant, there
is also an ``Averaged_results`` folder. This folder allows you to check if any of the participant results are outliers.

Figure creation also makes html file

.. note::
    For plotting, as scaling of brain grid figures are calculated during the analysis step, scaled brain grid figures
    should only be used if all files analysed together are also displayed together, otherwise the scaling will be based
    on files which are not present in the figures. WHAT DOES THIS MEAN

Statistical map creation
------------------------

Both the  and `interactive table` GUI options can be used to explore the data once the analysis has
been ran. The `print results` option prints the results for the selected region of interests to the terminal, whereas
the `interactive table` option opens up the result in a browser window.