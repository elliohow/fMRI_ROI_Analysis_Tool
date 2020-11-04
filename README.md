# fMRI ROI Analysis Tool (fRAT): A simple pipeline to analyse and plot ROI-level analyses
Make sure that nipype version matches your version of FSL and Freesurfer

Scaled figures should only be used if all files analysed together are also displayed together, otherwise the
scaling could have been carried out based on a file which isn't being presented.

If an FSL's FAST or freesurfer segmentation are used to only include grey matter voxels (recommended for most use cases), 
these segmentations should before running fRAT analysis. FAST is recommended when cortical regions are being examined.
Support for subcortical regions may be added in the future.

The main.py file is used to run the fRAT. Configuration settings can be changed in config.py.
After the analysis has been conducted, printResults.py can be used to print the results to the terminal.

## Folder structure:
### Base folder:
The base folder is the folder which contains all files needed to run the fRAT. Before running an analysis, 
the base folder should contain:
1. The NIFTI/ANALYZE files to analyse
2. The statistical maps for each NIFTI/ANALYZE file should be inside a folder, with the name of the folder being defined in 
the config file.

All optional files/folders mentioned below should be placed in the base folder.

### If scan parameters are extracted from a table (recommended):
* A paramValues.csv file containing the MRI parameter values of each scan should be in the base folder. To create this 
file, set the "make_table_only" line in the config file to "True". 
* To change which keywords to extract from file names when creating this table, edit the "parameter_dict" option in the 
config file. The key for each dict entry represents the name of the parameter and the value represents how each parameter
appears in the file name (if at all). The keys for each parameter are also used for labelling plots.
* If the file names do not contain information about the parameters used, edit the table so it contains the correct 
information.

### If aligning to anatomical (recommended):
* A single anatomical volume should be placed in a folder called 'anat'

### If aligning to freesurfer segmentation:
* The file "aseg.auto_noCCseg.mgz" found in "freesurfer/mri/" is the only necessary file. All other freesurfer files and
folder can be deleted. The path should therefore be: "{base_folder}/freesurfer/mri/aseg.auto_noCCseg.mgz"
* Alternatively, the entire "freesurfer" folder can be placed in the base folder.

### If aligning to FSL FAST segmentation (recommended):
* Output of FAST should be placed in a folder called "fslfast"

## Potential errors
### Multicore processing
On some Mac OS systems, multicore processing may cause the below issue:

`objc[16599]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork()`

#### Solution
In the terminal, edit your bash_profile with `nano ~/.bash_profile`

At the end of the bash_profile file add the line: `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`

Then save and exit the bash_profile. Solution found here: [Link](https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr)