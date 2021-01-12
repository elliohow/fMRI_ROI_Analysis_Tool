# fMRI ROI Analysis Tool (fRAT) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://github.com/elliohow/fMRI_ROI_Analysis_Tool/blob/master/LICENSE)
> **STILL IN ACTIVE DEVELOPMENT**
>
> A GUI-based pipeline to analyse and plot ROI-level analyses for **MacOS** and **Linux**. 
>
> Tested and developed using **Python version 3.7.5**. 

## Table of Contents
<!--ts-->
* [Installing / Getting started](#installing--getting-started)
* [Running](#running)
  * [Folder structure](#folder-structure)
  * [paramValues.csv](#paramvaluescsv)
  * [If aligning to anatomical (recommended)](#if-aligning-to-anatomical-recommended)
  * [If aligning to FSL FAST segmentation (recommended)](#if-aligning-to-fsl-fast-segmentation-recommended)
  * [Shell scripts](#shell-scripts)
* [Potential errors](#potential-errors)
  * [Multicore processing](#multicore-processing)
* [Versioning](#versioning)
* [Licensing](#licensing)
<!--te-->


## Installing / Getting started
To install all required modules use the following command in the project directory: `pip install -r requirements.txt`.

To run the GUI version of the fRAT, use `fRAT_GUI.py`. It is recommended that the fRAT is first ran using the files in 
the 'data' folder to test whether the project dependencies are correctly installed.

## Running
The `fRAT.py` or `fRAT_GUI.py` files are used to run the non-GUI or GUI versions of fRAT respectively. Configuration 
settings can be changed in the GUI, alternatively they can be changed directly in the config.toml file. 
The recommended critical parameter validation method requires the creation of a paramValues.csv file containing 
information about critical parameters ([further information here](#paramvaluescsv)). If the paramValues.csv file is 
created before running the fRAT analysis, it should be placed in the 'base folder' (a copy of this file will then be 
placed in  the output folder for logging purposes); alternatively, if it is created after analysis it can be placed in 
the output folder (whose name changes depending on the atlas used for analysis).

After the analysis has been conducted, the output folder will contain the results for each individual fMRI volume in 
separate JSON files. 
Once the plot step has been run, combined_results.json contains the results of all fMRI volumes combined into one file.
`printResults.py` (or the print results GUI option) can be used to print the desired results to the terminal once
`combined_results.json` has been created. 
During analysis, a `config_log.py` file will be created in the output folder to record which config settings were used 
for analysis.

NOTE: For plotting, as scaling of figures is calculated during the analysis step, scaled figures should only be used if 
all files analysed together are also displayed together, otherwise the scaling will be based on files which are not 
present in the figures.

### Folder structure:
The following section details the folder structure needed to run the fRAT and the structure of the folder outputted by 
running the fRAT. An example of the folder structure needed to run the fRAT is given
[here](https://github.com/elliohow/fMRI_ROI_Analysis_Tool/tree/master/data). In this example 'QA_report' is the name of
the folder containing the statistical map files and 'HarvardOxford-Cortical_ROI_report' is the folder that has been
output by the fRAT.

#### Folder structure for running fRAT
The base folder is the folder which contains all the files to be used by the fRAT. Before running the fRAT analysis,
the base folder should be structured like this:
```
Base folder
├── stat_maps (name can be chosen by user)
│   └── NIFTI/Analyze statistical map files
│
├── anat (optional but recommended)
│   └── anatomy file
│
├── fslfast (optional but recommended)
│   └── fslfast pve_1 output
│
├── NIFTI/Analyze fMRI files
└── paramValues.csv (created through GUI)
```

Therefore, an example folder structure is:
```
Base folder
├── stat_maps (name can be chosen by user)
│   ├── P1_MB3_S2_matchBW_tSNR.nii
│   └── P2_MB1_S1P5_matchBW_tSNR.nii
│
├── anat (optional but recommended)
│   └── CS_MPRAGE.nii
│
├── fslfast (optional but recommended)
│   └── bet_MPRAGE_pve_1.nii.gz
│
├── P1_MB3_S2_matchBW.nii
├── P2_MB1_S1P5_matchBW.nii
└── paramValues.csv (created through GUI)
```

#### Folder structure of fRAT output
```
Output folder
├── Figures
│   ├── Barcharts
│   │   └── ...
│   ├── Brain_grids
│   │   └── ...
│   ├── Brain_images
│   │   └── (Individual images of brains used for the brain grid images)
│   ├── Histograms
│   │   └── ...
│   └── Scatterplots
│       └── ...
│
├── Intermediate files
│   └── (All intermediate files created during analysis)
├── NIFTI_ROI
│   └── (NIFTI-GZ files used to create the files in the 'Brain_images' folder)
├── Raw_results
│   └── (JSON files containing non-summarised results for every ROI. Used to create the histogram figures and can be used for further statistical tests)
├── Summarised_results
│   ├── combined_results.json (created after plot step has been ran. Combines results from all other JSON files in this folder)
│   └── (JSON files containing summarised results for each ROI)
│
├── config_log.toml (log of settings used for analysis)
└── copy_paramValues.csv (will be present if paramValues.csv was created before analysis)
```

### paramValues.csv:
* A `paramValues.csv` file containing the MRI parameter values of each scan should be in the base folder. To create this
  file, select the 'Setup parameters' option in the GUI. Alternatively, when running `fRAT.py`, the "make_table_only"
  variable in config.toml can be set to "True", or pass the --make_table flag when running `fRAT.py`, e.g.
  `fRAT.py --make_table`.
* To change which keywords to extract from file names when creating this table, edit the two critical parameters
  options in the Parsing menu of the GUI. Alternatively edit the "parameter_dict1" and "parameter_dict2" options in
  `config.toml`. Each critical parameter/parameter_dict list entry represents the name of the parameter and how it would
  be represented in file name (if at all). Critical parameters are also used for labelling plots.
* If the file names do not contain information about which parameters were used (such as the file name
  `P1_MB3_S2_matchBW.nii` showing that multiband 3 and SENSE 2 were used), edit paramValues.csv so it contains the correct
  information.
* As an alternative to using paramValues.csv, the critical parameter selection method can be changed to 'manual' where
  the user will be prompted at runtime to enter the correct values, or 'name' if the user is certain the parameters can
  be successfully parsed from the file names.

### If aligning to anatomical (recommended):
* A single (non-brain extracted) anatomical volume should be placed in a folder called "anat".

### If aligning to FSL FAST segmentation (recommended):
* FSL's FAST segmentation can be used to only include grey matter voxels in the analysis, FAST segmentations should be
  completed before running fRAT analysis.
* Output of FAST should be placed in a folder called "fslfast", only the file with the suffix "pve_1" needs to be used.
* FAST segmentation is recommended when cortical regions are being examined. Support for subcortical regions may
  be added in the future.

### Shell scripts
For shell scripting multiple analyses/plots, flags can be passed when running fRAT.py to specify the fMRI file locations
(for scriping multiple analyses), or the location of the JSON files outputted by the fRAT (for scripting plotting),
e.g. `fRAT.py --brain_loc BRAIN_LOC --json_loc JSON_LOC`. Help text for available flags can be accessed with the
command: `fRAT.py --help`.

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

## Versioning
We use [SemVer](http://semver.org/) for versioning. For the versions available, see the 
[link to tags on this repository](https://github.com/elliohow/fMRI_ROI_Analysis_Tool/tags).

## Licensing
This project uses the MIT license. For the text version of the license see 
[here](https://github.com/elliohow/fMRI_ROI_Analysis_Tool/blob/master/LICENSE).
