.. include:: links.rst

=====
Usage
=====
.. contents:: :local:

.. note::
    It is recommended that the fRAT is first ran using the files in the ``example_data`` folder to test whether the project
    dependencies are correctly installed. This folder also gives demonstrates the input necessary for the fRAT and
    the output produced by the fRAT.



Key concepts of fRAT
====================
The ROI analysis pipeline has 3 main steps: ROI analysis, statistical analysis and figure creation; the latter two steps
requiring the ROI analysis step to first be ran. Each step outputs a configuration log file to log what
settings were used during this step. As seen in the image below, the fRAT requires: functional volumes, anatomical
volumes, voxelwise statistical maps and optionally (but recommended for cortical ROIs) an FSL FAST segmentation.

.. image:: images/entire_process.png

Running the fRAT
================
The ``fRAT.py`` or ``fRAT_GUI.py`` files are used to run the non-GUI or GUI versions of fRAT respectively.
Configuration settings can be changed in the GUI, alternatively they can be changed directly in the config.toml files.
For shell scripting multiple analyses/plots, flags can be passed when running fRAT.py to specify the fMRI file locations
(for scriping multiple analyses), or the location of the JSON files outputted by the fRAT (for scripting
plotting/statistics), e.g. `fRAT.py --brain_loc BRAIN_LOC --json_loc JSON_LOC`. Help text for available flags can be
accessed with the command: `fRAT.py --help`. To learn the procedure of how to run an ROI analysis, follow this
:doc:`tutorial </tutorials/Basic-ROI-analysis>`.

.. note::
    Using flags to script statistical map creation is currently not possible.


Potential errors
================
Multicore processing
--------------------
On some Mac OS systems, multicore processing may cause the below issue:

```objc[16599]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork()```

Solution
********
In the terminal, edit your bash_profile with:

```nano ~/.bash_profile```

At the end of the bash_profile file add the line:

```export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES```

Then save and exit the bash_profile. Solution originally found here:
[Link](https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr)

FileNotFoundError: No such file or directory
********************************************
While this error can be caused for a number of reasons, such as a file that is present in the  ``paramValues.csv``
table that is no longer present in its subject directory. If this error occurs at the beginning of the ROI
analysis, during the anatomical file alignment step, it is likely that in the path to the chosen base directory, there
is a space which Nipype does not know how to handle. An example can be seen below:

``FileNotFoundError: No such file or directory '/Users/elliohow/spa ce_test/sub-03/to_mni_from_CS_MPRAGE_optiBET_brain.mat' for output 'out_matrix_file' of a FLIRT interface``
