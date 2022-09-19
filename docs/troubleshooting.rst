===============
Troubleshooting
===============
.. contents:: :local:
    :depth: 1

Below you can find solutions for some of the issues that you may come across when using the fRAT. However
if this page does not help resolve your issue please submit an issue report `here. <https://github.com/elliohow/fMRI_ROI_Analysis_Tool/issues>`__

Mac OS multicore processing issue
---------------------------------
On some Mac OS systems, multicore processing may cause the below issue:

```objc[16599]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork()```

Solution
********
In the terminal, edit your bash_profile with:

```nano ~/.bash_profile```

At the end of the bash_profile file add the line:

```export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES```

Then save and exit the bash_profile. Solution originally found `here. <https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr>`__


FileNotFoundError: No such file or directory
--------------------------------------------
While this error can be caused for a number of reasons, such as a file that is present in the  ``paramValues.csv``
table that is no longer present in its subject directory. If this error occurs at the beginning of the ROI
analysis, during the anatomical file alignment step, it is likely that in the path to the chosen base directory, there
is a space which Nipype does not know how to handle. An example can be seen below:

``FileNotFoundError: No such file or directory '/Users/elliohow/spa ce_test/sub-03/to_mni_from_CS_MPRAGE_optiBET_brain.mat' for output 'out_matrix_file' of a FLIRT interface``
