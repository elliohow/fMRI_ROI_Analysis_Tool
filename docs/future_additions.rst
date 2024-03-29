.. include:: links.rst

==============
Future updates
==============
.. contents:: :local:
    :depth: 2

.. note::
   Any update ideas can be given here: https://github.com/elliohow/fMRI_ROI_Analysis_Tool/discussions

General
=======
* Test fRAT with empty parameters for cases where a single metric is set to be reported and not a comparison between multiple parameters
  - May need to find references to config.parameter_dict/cfg.parameter_dict to see how it is handled when blank
* Summarise data across an ROI (time series analysis)
* Add support for custom statistical map functions
* Add option to only keep essential files when running statistical map functions (such as '_tSNR')
* To facilitate easier testing of new features during development, add ability to test only ROI analysis, only statistical map creation, or both.
* Dash barchart code needs an overhaul as it isn't currently viable to easily add new features
    - Long names currently get cutoff
* Give user ability to find fRAT version
* Print log of terminal output. This will allow the user to track where fRAT terminated in the case of a crash
* Save path of last analysed folder so its easier to string together statmap creation and ROI analysis
* Separation of noise volumes currently returns false positives
* Add option to use atlases with the 25 or 50 probability threshold
* (Potential) Add ability to analyse more than fMRI volumes (such as anatomical volumes)
* (Potential) Add easy way load analysis results data so jupyter notebook can be used to explore the data with graphs/statistics

GUI
===
* Separate fRAT into 3 components (main analysis, creation of statmaps such as tSNR, general fMRI tools)
* Ability to make saveable analysis profiles to quickly swap between settings
    - Need to also add ability to keep toml files up to date if any new options have been added/removed
* Increase linux compatibility

Analysis
========
* Add option to bootstrap confidence intervals
* Allow registration to more templates (such as http://ccraddock.github.io/cluster_roi/)
* Allow registration to custom templates and ROI atlases
* Add ability to skip running a file through analysis if file has already been ran through analysis
    - May be good for combining large datasets or adding new participants to the analysis
    - May only be possible with participant averaged data
    - Currently output folder is deleted when rerunning analysis due to mcf files
    - Need to find a way to deal with files which have now been set to ignore in paramValues.csv
    - Need to also force the analysis to rerun if parameter values have changed
* Increase clarity of output files
* fRAT should check paramValues.csv before running anatomical file alignment to see if any subjects can be skipped


Figures
=======
* 3d representation of the brain grid data
* Option to use scientific notation for brain figures: https://github.com/nilearn/nilearn/issues/2220
* Ability to add manual lines to graphs (for example, to represent minimum usable tSNR)
* Improve customisation options of figures (for example, currently barcharts look too big if fewer parameters selected)
* Add option to add number of voxels to histogram plots
* Give one IV barcharts colour
* Implement mixed-roi scaled brain figures
* Add option to underlay MNI template under each brain grid image
* (Potential) Interactive plots

Statistics
==========
* Add ability to compare if linear mixed model coefficients significantly vary between regions
* Add figures and testing for linear mixed model assumptions (linearity, normality and homogeneity)
* Add :math:`r^{2}` statistics to t-tests

Statmaps
========
* Handle the func_preprocessed folder more elegantly, currently it is replaced each time statmaps are created

Known issues
============
* Coefficient map creation during statistics step can fail with interaction effects if the the IV's contain an
independent variable.