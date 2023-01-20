<img alt="" width=512 src="docs/images/fRAT.gif">

# fRAT - fMRI ROI Analysis Tool
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) [![GitHub license](https://img.shields.io/hexpm/l/plug?style=flat-square)](https://github.com/elliohow/fMRI_ROI_Analysis_Tool/blob/master/LICENSE)
fRAT is an open-source python-based analysis pipeline used to simplify the processing and analysis of fMRI data by
converting voxelwise maps into ROI-wise maps.

> This project is under active development.
>
> fRAT is written using **Python version 3.8.0** for **MacOS** and is based on Nipype.

<img src="docs/images/ROI_example.png" 
  title="A region of interest map created using fRAT, showing the mean temporal Signal-to-Noise for each region. Data is displayed in MNI152 standard space and combines data from multiple subjects." 
height=500>

Documentation: https://fmri-roi-analysis-tool.readthedocs.io

## Testing installation of fRAT
Before running fRAT, it is recommended that the fRAT and project dependency installation is tested.
To test fRAT, download the files provided here: https://osf.io/pbm3d/. This `example_data` folder should be placed
in the base folder, allowing the fRAT tests to find them.
This folder also gives demonstrates the input necessary for the fRAT and the output produced by the fRAT.

The `fRAT.py` or `fRAT_GUI.py` files are used to run the non-GUI or GUI versions of fRAT respectively.
Configuration settings can be changed in the GUI, alternatively they can be changed directly in the config.toml files.
For shell scripting multiple analyses/plots, flags can be passed when running fRAT.py to specify the fMRI file locations
(for scriping multiple analyses), or the location of the JSON files outputted by the fRAT (for scripting
plotting/statistics), e.g. `fRAT.py --brain_loc BRAIN_LOC --json_loc JSON_LOC`, however using flags to script
statistical map creation is not currently possible. Help text for available flags can be
accessed with the command: `fRAT.py --help`. For a tutorial showing how to run an ROI analysis, [see here](https://fmri-roi-analysis-tool.readthedocs.io/en/latest/tutorials/Basic-ROI-analysis.html).


## GUI images
<img src="docs/images/GUI.png" width=700>

## HTML report images
<img src="900" width=900>

## Versioning
We use `Semantic versioning <http://semver.org/>`_ for versioning. For the versions available, see the
`tag list <https://github.com/elliohow/fMRI_ROI_Analysis_Tool/tags>`_ for this project.

## Licensing
This project uses the Apache 2.0 license. For the text version of the license see
`here <https://github.com/elliohow/fMRI_ROI_Analysis_Tool/blob/master/LICENSE>`_.
