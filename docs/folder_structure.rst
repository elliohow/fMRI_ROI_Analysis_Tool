================
Folder structure
================
The following section details the folder structure needed to run the fRAT and the structure of the folder outputted by
running the fRAT. An example of the folder structure needed to run the fRAT is given
`here <https://github.com/elliohow/fMRI_ROI_Analysis_Tool/tree/master/example_data>`_. In this example 'QA_report' is the name of
the folder containing the statistical map files and 'HarvardOxford-Cortical_ROI_report' is the folder that has been
output by the fRAT.

#### Folder structure for running fRAT
The base folder is the folder which contains all the files to be used by the fRAT. Before running the fRAT analysis,
the base folder should be structured like this:

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


Therefore, an example folder structure is:

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


#### Folder structure of fRAT output

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


| simpleble-master
| ├── docs
| │   ├── build
| │   ├── make.bat
| │   ├── Makefile
| │   └── source
| ├── LICENSE
| ├── README.md
| ├── requirements.txt
| └── simpleble
|     └── simpleble.py
|
|