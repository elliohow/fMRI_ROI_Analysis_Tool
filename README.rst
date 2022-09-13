=============================
fRAT - fMRI ROI Analysis Tool
=============================
fRAT is a GUI-based toolset used for analysis of task-based and resting-state functional MRI (fMRI) data. Primarily fRAT
is used to analyse and plot region-of-interest data, with a number of supplemental functions provided within.

.. note::
    This project is under active development.

    fRAT is written using **Python version 3.8.0** for **MacOS** and is based on Nipype.

.. image:: https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square
  :target: http://makeapullrequest.com
  :alt: PRs Welcome!

.. image:: https://img.shields.io/hexpm/l/plug?style=flat-square
  :target: https://github.com/elliohow/fMRI_ROI_Analysis_Tool/blob/master/LICENSE
  :alt: License information

About
-----

Documentation: https://fmri-roi-analysis-tool.readthedocs.io
Project repository: https://github.com/elliohow/fMRI_ROI_Analysis_Tool


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



### GUI images
.. image:: images/GUI.png
    :width: 700


### HTML report images
.. image:: images/HTML_report.png
    :width: 900


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
├── anat
│   ├── skull stripped anatomy file (should have '_brain' extension)
│   └── anatomical file (necessary if using BBR cost function)
│
├── fslfast (optional but recommended)
│   └── ... (All files output by fslfast)
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
├── anat
│   └── MPRAGE_brain.nii
│
├── fslfast (optional but recommended)
│   └── ... (fslfast files)
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
├── fRAT_report
│   └── (Pages of HTML report accessed using index.html)
├── Intermediate files
│   └── (All intermediate files created during analysis)
├── NIFTI_ROI
│   └── (NIFTI-GZ files used to create the files in the 'Brain_images' folder)
├── Raw_results
│   └── (JSON files containing non-summarised results for every ROI. Used to create the histogram figures and can be used for further statistical tests)
├── Summarised_results
│   ├── combined_results.json (Combines results from all other JSON files in this folder)
│   └── (JSON files containing summarised results for each ROI)
│
├── index.html (Index page of HTML report showing created figures)
├── config_log.toml (log of settings used for analysis)
└── copy_paramValues.csv (will be present if paramValues.csv was created before analysis)
```

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
