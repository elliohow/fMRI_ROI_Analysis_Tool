.. include:: links.rst

.. image:: images/fRAT.gif
  :width: 500

=============================
fRAT - fMRI ROI Analysis Tool
=============================

.. image:: https://joss.theoj.org/papers/cc9c0cb3b12abaf30c8381728d3229d7/status.svg
  :target: https://joss.theoj.org/papers/cc9c0cb3b12abaf30c8381728d3229d7
  :alt: Paper status

.. image:: https://img.shields.io/github/v/release/elliohow/fmri_roi_analysis_tool?style=flat-square
    :target: https://github.com/elliohow/fmri_roi_analysis_tool/releases/latest
    :alt: Github release (latest by date)

.. image:: https://img.shields.io/hexpm/l/plug?style=flat-square
  :target: https://github.com/elliohow/fMRI_ROI_Analysis_Tool/blob/master/LICENSE
  :alt: License

.. image:: https://img.shields.io/github/issues/elliohow/fmri_roi_analysis_tool?style=flat-square
    :target: https://github.com/elliohow/fmri_roi_analysis_tool/issues
    :alt: Github issues
    
.. image:: https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square
  :target: http://makeapullrequest.com
  :alt: PRs welcome!

fRAT is an open-source python-based GUI application used to simplify the processing and analysis of fMRI data by
converting voxelwise maps into ROI-wise maps. An installation of FSL is required in order to run fRAT.

Project repository: https://github.com/elliohow/fMRI_ROI_Analysis_Tool

.. note::
    fRAT is written using Python version |python_version| for **MacOS, Linux and WSL2** and tested with FSL |fsl_version|.

.. figure:: images/ROI_example.png

    A region of interest map created using fRAT, showing the mean temporal Signal-to-Noise for each region.
    Data is displayed in MNI152 standard space and combines data from multiple subjects.

Using fRAT
----------
Installation instructions for `fRAT` can be found `here <https://fmri-roi-analysis-tool.readthedocs.io/en/latest/installation.html>`_.
Before running `fRAT`, it is also recommended that the `fRAT` and project dependency installation is tested. Information on how
to do this can also be found on the installation instructions page.

To learn how to use `fRAT`, see this :doc:`tutorial </tutorials/Basic-ROI-analysis>`.

Citation
--------

**When using fRAT, please include the following citation:**

Howley, E., Francis, S., & Schluppeck, D. (2023). fRAT: an interactive, Python-based tool for region-of-interest summaries of functional imaging data. Journal of Open Source Software, 8(85), 5200. https://doi.org/10.21105/joss.05200


Reporting bugs
--------------
To report a bug or suggest a new feature, please go to `fRAT's Issues <https://github.com/elliohow/fMRI_ROI_Analysis_Tool/issues/new/choose>`_.

For other questions, issues or discussion please go to `fRAT's Discussions <https://github.com/elliohow/fMRI_ROI_Analysis_Tool/discussions>`_.

Contributing with development
-----------------------------
If you'd like to contribute to the project please read our `contributing guidelines <https://github.com/elliohow/fMRI_ROI_Analysis_Tool/blob/master/CONTRIBUTING.md>`_. Please also read through our `code of conduct <https://github.com/elliohow/fMRI_ROI_Analysis_Tool/blob/master/CODE_OF_CONDUCT.md>`_.


GUI images
----------
.. image:: images/GUI.png
  :width: 700

HTML report images
------------------
.. image:: images/HTML_report.png
  :width: 900

Versioning
----------
We use `Semantic versioning <http://semver.org/>`_ for versioning. For the versions available, see the
`tag list <https://github.com/elliohow/fMRI_ROI_Analysis_Tool/tags>`_ for this project.

Licensing
---------
This project uses the Apache 2.0 license. For the text version of the license see
`here <https://github.com/elliohow/fMRI_ROI_Analysis_Tool/blob/master/LICENSE>`_.
Prior to version 1.0.0, this project used an MIT license.

.. toctree::
    :caption: Contents
    :maxdepth: 3

    Home <self>
    statement_of_need
    key_concepts_of_frat
    installation
    tutorials
    troubleshooting
    future_additions
