.. include:: links.rst

=================
Statement of need
=================
.. contents:: :local:

Overview
--------

There are several tools that provide functionality to report ROI-wise summaries, including the widely used `Freesurfer` infrastructure and packages built on top of them. For example, the `R` packages `ggseg` and `ggseg3d` can be used to show aggregated data such as cortical thickness in atlas-derived regions of interest. However these packages are designed primarily for use with anatomical datasets and would require some additional coding for use with fMRI data quality and statistical metrics. Several tools do provide data quality metrics for fMRI datasets, such as `fMRIPrep` and `MRIQC`. However, these tools either report voxelwise maps or aggregate metrics over the entire brain instead of chosen ROIs. This can obscure important inter-regional differences which may be particularly informative for optimizing scanning parameters for planned experiments

`fRAT` is an open source, python-based application which focuses on ROI-wise analysis of fMRI data, by providing an easy to use and flexible pipeline for converting voxelwise data into ROI-wise data, based on `FSL`'s provided anatomical atlases. The provided plotting options can be used to customize different aspects of the data, such as the spatial distribution of the metric of interest, while the statistical tools facilitate univariate and multivariate analyses within ROIs. The graphical user interface is designed to provide a user-friendly way to run and customize the settings of `fRAT`. `fRAT` relies heavily on the python library `Nipype` to access the analysis tools provided by FSL.

Usage
-----

The user provides a 4D fMRI timeseries as an input, from which the voxelwise maps of data quality metrics (e.g. tSNR) are computed within fRAT. Alternatively, pre-computed data quality maps can be used. In addition, the ROI analysis requires a structural scan (MPRAGE) which should be skull stripped (using FSL’s BET, or preferably, using optiBET) in order to run. As tSNR provides a rough estimate of activation detection power in fMRI studies, calculating this metric for multiple ROIs may be particularly useful for planning studies aimed at specific brain regions. However, as `fRAT` is designed to be used flexibly, any other voxelwise statistical map can be used as an input to the ROI analysis. `fRAT` can also be used to summarize data quality metrics for each region across participants, and is therefore also useful for larger, multi-participant datasets. The statistics and visualisation options provided by `fRAT` allow for quantitative comparisons of the effect of different fMRI sequences or hardware on data quality. This may make it particularly useful for comparisons across datasets obtained at different imaging sites.

Potential use-cases
-------------------

Study planning
^^^^^^^^^^^^^^

`fRAT` is able to be used by imaging sites to provide guidance on the optimal fMRI parameters (such as Multiband (MB) factor, parallel imaging acceleration factor and echo time (TE)), taking into account different experimental requirements and the regions of the brain being investigated. This is beneficial as the effect of fMRI sequence and hardware on data quality metrics can vary spatially over the brain in a way that is difficult to understand without pilot data. 

Effect size estimation
^^^^^^^^^^^^^^^^^^^^^^

Effect size estimations based on statistical maps from functionally derived ROIs are common in fMRI analysis, but can lead to inflated estimates if the selection criteria are not independent from the effect statistic. Defining ROIs based on atlases, as is performed with fRAT, prevents a circular analysis and leading to more accurate effect size estimations.
