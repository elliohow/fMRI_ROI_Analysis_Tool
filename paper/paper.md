---
title: 'fRAT: an interactive, Python-based tool for region-of-interest summaries of functional imaging data'
tags:
  - fMRI
  - Python
  - Neuroscience
  - Nipype
  - Pipeline
authors:
  - name: Elliot Howley
    orcid: 0000-0002-3868-2516
    affiliation: 1
    corresponding: true
  - name: Susan Francis
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Denis Schluppeck
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: University of Nottingham, UK
   index: 1
date: 19 December 2022
bibliography: paper.bib
---

# Summary

Functional magnetic resonance imaging (fMRI) is a technique commonly used for neuroimaging. As optimal fMRI parameters and data quality metrics (such as echo time and temporal signal-to-noise ratio [tSNR]) can vary spatially [@clareSingleshotMeasurementEstablish2001], knowledge of how these metrics vary over brain regions can be used to optimise MRI scan parameters for a chosen region-of-interest (ROI). Reporting of these metrics over either the whole brain or for a small number of voxels is common, however this can obscure important inter-regional differences. There is currently a lack of software available to facilitate analysis of these metrics simultaneously across multiple ROIs, therefore the goal of the fMRI ROI Analysis Tool (`fRAT`) is to provide a toolset to address this gap, and provide a straightforward method for conducting ROI-wise analyses of fMRI metrics. 

# Statement of need

`fRAT` is an open-source python-based analysis pipeline used to simplify the processing and analysis of fMRI data by converting voxelwise maps into ROI-wise maps \autoref{fig:brain_images}. The graphical user interface (GUI) of `fRAT` is designed to provide a user-friendly interface to run and customise the settings of `fRAT` \autoref{fig:GUI}. `fRAT` relies heavily on the python library `Nipype` to access the analysis tools provided by `FSL` [@jenkinsonFSL2012]. `fRAT` documentation is available  to provide information on installation and usage of `fRAT` (https://fmri-roi-analysis-tool.readthedocs.io/en/latest/).

The ROI analysis requires anatomical, functional and voxelwise statistical map volumes to run in \autoref{fig:Pipeline}, with the only requirement being the anatomical volumes should be skull stripped (preferably using optiBET [@lutkenhoffOptimizedBrainExtraction2014]). Additionally, the voxelwise tSNR maps to be used in the ROI analysis can be created using `fRAT`. However, as `fRAT` is designed to be used in a wide variety of fMRI research, any voxelwise map can be used as an input to the ROI analysis. As the results for each region is combined between participants, `fRAT` can be used to summarise data quality metrics for the entire dataset. As tSNR provides a rough estimate of activation detection power [@murphyHowLongScan2007; @welvaertDefinitionSignalToNoiseRatio2013], using `fRAT` to calculate tSNR for ROIs increases the interpretability of results.

Hardware and fMRI sequence choice can affect optimal fMRI parameters and data quality metrics. For example, parallel imaging commonly leads to non-uniform signal-to-noise (SNR) degradation over the image, this SNR degradation can be quantified by the local geometry factor and is highly dependent on the coil geometry [@pruessmannSENSESensitivityEncoding1999]. Therefore, aside from using `fRAT` to report data quality metrics across a dataset, the included statistics and visualisation options can be used to allow imaging sites to optimise fMRI sequence or hardware choice for each ROI.



# Figures
![Representation of how fRAT produces ROI-wise maps. Data from a single participant is shown here. **(A)** Voxelwise tSNR map (in native space). **(B)** Harvard-Oxford Cortical atlas regions assigned to participant (in native space). **(C)** Combination of **(A)** and **(B)** to produce final ROI-wise tSNR map (in standard space). \label{fig:brain_images}](brain_images.png)

![Example of fRAT's graphical user interface. **(Left column)** Main menu screen. **(Right column)** Analysis settings screen. \label{fig:GUI}](GUI.png)

![Flowchart showing fRAT's pipeline \label{fig:Pipeline}](process_overview.png)

# Acknowledgements

# References
