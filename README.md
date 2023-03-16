<img src="https://github.com/elliohow/fMRI_ROI_Analysis_Tool/blob/master/docs/images/fRAT.gif?raw=true" width=500>

# fRAT - fMRI ROI Analysis Tool
[![status](https://joss.theoj.org/papers/cc9c0cb3b12abaf30c8381728d3229d7/status.svg)](https://joss.theoj.org/papers/cc9c0cb3b12abaf30c8381728d3229d7)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) 
[![GitHub license](https://img.shields.io/hexpm/l/plug?style=flat-square)](https://github.com/elliohow/fMRI_ROI_Analysis_Tool/blob/master/LICENSE)
[![Github release (latest by date)](https://img.shields.io/github/v/release/elliohow/fmri_roi_analysis_tool?style=flat-square)](https://github.com/elliohow/fmri_roi_analysis_tool/releases/latest)
[![Github issues](https://img.shields.io/github/issues/elliohow/fmri_roi_analysis_tool?style=flat-square)](https://github.com/elliohow/fmri_roi_analysis_tool/issues)
[![Documentation](https://img.shields.io/readthedocs/fmri-roi-analysis-tool)](https://fmri-roi-analysis-tool.readthedocs.io/en/latest/)

fRAT is an open-source python-based GUI application used to simplify the processing and analysis of fMRI data by
converting voxelwise maps into ROI-wise maps. An installation of FSL is required in order to use fRAT.

> fRAT is written using **Python** for **MacOS, Linux and WSL2**.

Documentation:

[Home page](https://fmri-roi-analysis-tool.readthedocs.io)

[Installation instructions](https://fmri-roi-analysis-tool.readthedocs.io/en/latest/installation.html)

[ROI analysis tutorial](https://fmri-roi-analysis-tool.readthedocs.io/en/latest/tutorials/Basic-ROI-analysis.html)

## Reporting bugs

To report a bug, please go to [fRAT's Issues](https://github.com/elliohow/fMRI_ROI_Analysis_Tool/issues/new).

For other questions, issues or discussion please go to [fRAT's Discussions](https://github.com/elliohow/fMRI_ROI_Analysis_Tool/discussions).

## Contributing with development

The [Fork & Pull Request Workflow](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) is used for contributing. Below is a summary of the necessary steps for this workflow:

1. Fork this repository.
2. Clone the repository at your machine.
3. Add your changes in a branch named after the feature (`lower-case-with-hyphens`).
4. Make a pull request to `fRAT`, targeting the `master` branch.

## Images
<img src="https://github.com/elliohow/fMRI_ROI_Analysis_Tool/blob/master/docs/images/GUI.png?raw=true" title="Example of the fRAT GUI" width=700>

<img src="https://github.com/elliohow/fMRI_ROI_Analysis_Tool/blob/master/docs/images/ROI_example.png?raw=true" 
  title="A region of interest map created using fRAT, showing the mean temporal Signal-to-Noise for each region. Data is displayed in MNI152 standard space and combines data from multiple subjects." 
width=700>

<img src="https://github.com/elliohow/fMRI_ROI_Analysis_Tool/blob/master/docs/images/HTML_report.png?raw=true" title="Example of a HTML report output by fRAT" width=600>

## Versioning
We use [Semantic versioning](http://semver.org/) for versioning. For the versions available, see the
[tag list](https://github.com/elliohow/fMRI_ROI_Analysis_Tool/tags) for this project.

## Licensing
This project uses the Apache 2.0 license. For the text version of the license see
[here](https://github.com/elliohow/fMRI_ROI_Analysis_Tool/blob/master/LICENSE). 
Prior to version 1.0.0, this project used an MIT license.
