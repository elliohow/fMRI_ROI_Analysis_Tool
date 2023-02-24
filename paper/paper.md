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
    orcid: 0000-0003-0903-7507
    affiliation: 1
  - name: Denis Schluppeck
    orcid: 0000-0002-0634-7713
    affiliation: 1
affiliations:
 - name: University of Nottingham, UK
   index: 1
date: 24 February 2023
bibliography: paper.bib
---

# Summary

Functional magnetic resonance imaging (fMRI) is widely used to address basic cognitive as well as clinical neuroscience questions. The specific choice of imaging sequence and parameters used for image acquisition, can have a marked effect on the acquired data. Because the effects of different parameter settings, such as echo time (TE) or parallel imaging acceleration factors, can vary spatially and may also interact [@clareSingleshotMeasurementEstablish2001], assessing how these metrics affect data quality over brain regions is crucial. Such data quality metrics, in particular the temporal signal-to-noise ratio (tSNR), can therefore be used to optimise MRI scan parameters for a chosen region-of-interest (ROI). Reporting of these metrics over either the whole brain or for a small number of voxels is common; however this can obscure important inter-regional differences. There is currently a lack of easy-to-use tools to analyse these metrics simultaneously across multiple ROIs. The goal of the fMRI ROI Analysis Tool (`fRAT`), presented here, is to provide a toolset to address this gap, and provide a straightforward method for conducting ROI-wise
analyses of fMRI metrics.

# Statement of need

`fRAT` is an open-source, python-based analysis application, used to simplify the processing and analysis of fMRI data by converting voxelwise maps into ROI-wise maps (\autoref{fig:brain_images}). The graphical user interface of `fRAT` is designed to provide a user-friendly way to run and customise the settings of `fRAT` (\autoref{fig:GUI}). `fRAT` relies heavily on the python library `Nipype` to access the analysis tools provided by `FSL` [@jenkinsonFSL2012]. [Documentation](https://fmri-roi-analysis-tool.readthedocs.io/en/latest/) is available to provide information on installation and usage of `fRAT`.

The ROI analysis requires anatomical, functional and voxelwise statistical maps in order to run (\autoref{fig:Pipeline}). One pre-requisite is that anatomical volumes should be skull stripped (using FSL's BET [@smithFastRobustAutomated2002], or preferably, using optiBET [@lutkenhoffOptimizedBrainExtraction2014]). Additionally, the voxelwise maps of data quality metrics (e.g. tSNR) can be computed with `fRAT` itself. As tSNR provides a rough estimate of activation detection power in fMRI studies [@murphyHowLongScan2007; @welvaertDefinitionSignalToNoiseRatio2013], calculating this metric for multiple ROIs may be particularly useful for planning studies aimed at specific brain regions. However, as `fRAT` is designed to be used flexibly, any other voxelwise statistical map can be used as an input to the ROI analysis. `fRAT` can also be used to summarize data quality for each region across participants, and is therefore also useful for larger, multi-participant datasets. `fRAT` also includes statistics and visualisation options that allow for quantitative comparisons of the effect of different fMRI sequences or hardware on data quality. This may make it particularly useful for comparisons across datasets obtained at different imaging sites.

One application of the tools provided by `fRAT` is to enable imaging sites to provide guidance on the optimal fMRI parameters to use, taking into account different experimental requirements and the regions being investigated. This is beneficial as the effect of fMRI sequence and hardware on data quality metrics can vary spatially over the brain in a way that is hard to reason about without pilot data. For example, parallel imaging commonly leads to spatially non-uniform signal-to-noise ratio degradation, which is highly dependent on the coil geometry [@pruessmannSENSESensitivityEncoding1999].

# Figures

![Representation of how ROI-wise maps are produced. Data from a single participant is shown here. All figures are in native space apart from **(D),** which is in standard space. **(A)** Original functional volume. **(B)** Voxelwise temporal signal-to-noise ratio (tSNR) map (brighter colours, higher tSNR values). **(C)** Harvard-Oxford Cortical atlas regions assigned to participant anatomy. **(D)** Combination of **(B)** and **(C)** to produce final ROI-wise tSNR map. \label{fig:brain_images}](brain_images.png)

![A screenshot of fRAT's graphical user interface. **(Left column)** Main menu screen. **(Right column)** Analysis settings screen. \label{fig:GUI}](GUI.png)

![Flowchart showing fRAT's processing pipeline. \label{fig:Pipeline}](process_overview.png)

# Acknowledgements

This work was supported by the Engineering and Physical Sciences Research Council [grant number EP/R513283/1].

# References
