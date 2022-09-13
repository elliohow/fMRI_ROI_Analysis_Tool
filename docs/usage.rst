.. include:: links.rst

=====
Usage
=====

.. note::
    It is recommended that the fRAT is first ran using the files in the ``example_data`` folder to test whether the project
    dependencies are correctly installed.

The ``fRAT.py`` or ``fRAT_GUI.py`` files are used to run the non-GUI or GUI versions of fRAT respectively.
Configuration settings can be changed in the GUI, alternatively they can be changed directly in the config.toml files.
For shell scripting multiple analyses/plots, flags can be passed when running fRAT.py to specify the fMRI file locations
(for scriping multiple analyses), or the location of the JSON files outputted by the fRAT (for scripting
plotting/statistics), e.g. `fRAT.py --brain_loc BRAIN_LOC --json_loc JSON_LOC`. Help text for available flags can be
accessed with the command: `fRAT.py --help`.

.. note::
    This documentation will focus on how to use the GUI version of the fRAT, as while many settings and functions can be
    accessed without the GUI, it is suggested that the GUI is used where possible until you already have familiarity
    with fRAT.


ROI analysis pipeline
----------------------
The ROI analysis pipeline has 3 main steps: analysis, statistical analysis and figure creation; the latter two steps
requiring the ROI analysis step to first be ran. Each pipeline step outputs a configuration log file to log what
settings were used during this step.

.. tip::
    In the GUI, settings that will most often need changing are **bold**.

    Additionally, most settings have a tooltip giving an explanation of what the option does, and if relevant, how to
    format the option entry.

To run the ROI analysis, first create subject directories with the format ``sub-<subjectnumber>`` (e.g. ``sub-42``) and
place all fMRI files in the relevant directories. Next, enter the "Parsing" options menu found under the "Settings" panel
in the GUI. The "critical parameters" setting indicates what are the independent variables of the experiment and will be
used when labelling plots, whereas the "critical parameters abbreviation" setting indicates how these variables are
represented in the filenames of the scans (if at all). For example, the "critical parameters" option may contain
"Multiband, Sense", whereas the "critical parameters abbreviation" option will instead contain "mb, s". In this case,
the file name ``P1_MB3_S2_matchBW.nii`` will represent that multiband 3 and SENSE 2 were used.

.. warning::
    Make sure the lists of critical parameters given are in the same order, otherwise the critical parameter names
    will be applied to the wrong abbreviations.

EXPLAIN THE BASIC THINGS THAT ARE NEEDED SUCH AS ANAT, FMRI, VOXEL MAPS

After filling these options in, check the "Make folder structure" option. This will create the basic folder structure
required for the fRAT and will sort the files into the correct directory. After returning to the home
page, click the "Setup parameters" button in the "Run" panel of the GUI. This will parse files names for critical
parameters using the "critical parameters abbreviation" option set above, with the output being saved in
``paramValues.csv``. This file should be checked before proceeding to make sure the correct parameters have been applied
to each file. Alternatively, when running without the GUI, pass the --make_table flag when running ``fRAT.py``, e.g.
``fRAT.py --make_table``.

In the newly created ``anat`` folder, place a single skull stripped anatomical volume with the suffix "_brain". The
default BBR cost function for functional to anatomical registration requires a second, whole head, non-brain extracted
volume should also be placed in the ``anat`` folder. This second file should not contain the word "brain" in the file
name. While the BBR cost function also requires segmentation to have been ran using FAST, FAST will be automatically ran
when running the analysis. However if you wish to run the segmentation ahead of time, the FAST output should be placed
in the ``fslfast`` folder. If running the analysis on cortical regions (for example when using the Harvard-Oxford
Cortical atlas), it is recommended that white matter and extracranial voxels are excluded from the analysis by setting
the GUI option "Use FAST to only include grey matter" to true. While there is not currently support for using FAST to
improve analysis of subcortical regions, support may be added in the future.

If FAST needs to be ran (either for BBR registration or to include only grey matter in the analysis) the accuracy of the
skull stripped anatomical scans should still be assessed before running it. As overly conservative skull stripping can
lead to skull being retained in the Resulting image, which FAST may then misidentify as grey matter.
Conversely, overly liberal skull stripping can lead to parts of the brain being removed, meaning that these voxels will
also not be included in any ROIs.

.. note::
    To skull strip the anatomical files, it is highly recommended that optiBET_ is used as it has consistently produced
    the best brain extraction accuracy.

EXPLAIN VOXEL MAPS

After running the analysis, ``(OUTPUT_FOLDER)/Overall/Summarised_results/`` will contain
``Participant_averaged_results`` and ``Session_averaged_results``. Participant averaged results refers to region of
interest (ROI) results being first averaged within participants before being averaging between participants (i.e. the
more traditional method). Whereas session averaged results instead averages the ROI results between all sessions,
disregarding which participant was scanned in each session; this can be useful where the statistical map being converted
should be participant agnostic. ``combined_results.json`` found in these folders contains the final summary results of the
data. Both the  and `interactive table` GUI options can be used to explore the data once the analysis has
been ran. The `print results` option prints the results for the selected region of interests to the terminal, whereas
the `interactive table` option opens up the result in a browser window.

Figure creation also makes html file

.. note::
    For plotting, as scaling of brain grid figures are calculated during the analysis step, scaled brain grid figures
    should only be used if all files analysed together are also displayed together, otherwise the scaling will be based
    on files which are not present in the figures. WHAT DOES THIS MEAN

Statistical map creation
------------------------




## Potential errors
### Multicore processing
On some Mac OS systems, multicore processing may cause the below issue:

```objc[16599]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork()```

#### Solution
In the terminal, edit your bash_profile with:

```nano ~/.bash_profile```

At the end of the bash_profile file add the line:

```export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES```

Then save and exit the bash_profile. Solution originally found here:
[Link](https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr)

Make sure there are no spaces in the path to the folder you want to analyse