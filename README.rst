.. image:: images/fRAT.gif
  :width: 500

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
.. image:: images/voxel_and_roi_example.png

Documentation: https://fmri-roi-analysis-tool.readthedocs.io

Project repository: https://github.com/elliohow/fMRI_ROI_Analysis_Tool

Using fRAT
----------
The ``fRAT.py`` or ``fRAT_GUI.py`` files are used to run the non-GUI or GUI versions of fRAT respectively.
Configuration settings can be changed in the GUI, alternatively they can be changed directly in the config.toml files.
For shell scripting multiple analyses/plots, flags can be passed when running fRAT.py to specify the fMRI file locations
(for scriping multiple analyses), or the location of the JSON files outputted by the fRAT (for scripting
plotting/statistics), e.g. `fRAT.py --brain_loc BRAIN_LOC --json_loc JSON_LOC`, however using flags to script
statistical map creation is not currently possible. Help text for available flags can be
accessed with the command: `fRAT.py --help`. To learn the procedure of how to run an ROI analysis, follow this
:doc:`tutorial </tutorials/Basic-ROI-analysis>`.

.. note::
    It is recommended that the fRAT is first ran using the files in the ``example_data`` folder to test whether the project
    dependencies are correctly installed. This folder also gives demonstrates the input necessary for the fRAT and
    the output produced by the fRAT.


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
