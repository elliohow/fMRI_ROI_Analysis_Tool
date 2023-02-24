.. include:: links.rst

============
Installation
============
.. contents:: :local:
    :depth: 2

.. note::
    Make sure Python version |python_version| is used for installation. Download links can be found here:
    https://www.python.org/downloads/

    Installation as well as opening fRAT for the first time may take a while.

**1. Install brew**::

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

**2. Install pipx. This creates an isolated environment for the fRAT application and will allow you to run it from
anywhere**::

    brew install pipx
    pipx ensurepath

**3. Use pipx to install fRAT**::

    pipx install frat-brain --python $(which python3.10)

**4. fRAT can now be ran from the terminal from anywhere by using**::

    fRAT

fRAT can also be upgraded using::

    pipx upgrade frat-brain

External Dependencies
=====================
The FSL_ neuroimaging software tool is a core component of fRAT and is not able to be downloaded through Python’s
Package Index (PyPI), therefore FSL_ (version |fsl_version| or above)  should be downloaded and setup separately using FSL's
`installation instructions <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_.

While optiBET_ is not required for fRAT (`link <https://montilab.psych.ucla.edu/fmri-wiki/optibet/>`_), it is highly
recommended that this software tool be used for brain extraction
due to the necessity of accurate skull stripping for the fRAT to function optimally and the consistently better brain
extraction performance when using optiBET.

Checking installation
=====================
**1. Create the directory to save the example data folder**::

    mkdir ~/Documents/fRAT

This allows fRAT to find the example data. This folder will also be used to save configuration profiles in the future.

**2. Download the example dataset from** `here <https://osf.io/pbm3d/>`_.

The "subject_example_data" folder contains subject data and the tSNR maps calculated by fRAT, whereas the
"HarvardOxford-Cortical_ROI_report" folder contains the output of fRAT's ROI analysis.
If you only want to test fRAT runs correctly and do not want to check for missing files after testing the
installation, download only the "subject_example_data". If you wish to check for missing files after running the ROI
analysis, download both the "subject_example_data" and the "HarvardOxford-Cortical_ROI_report",

**3. Extract the chosen folder and place it in the newly created fRAT folder.**

If running the full comparison, also place the "HarvardOxford-Cortical_ROI_report" in the "subject_example_data" folder.


**4. Click the** :guilabel:`General` **button in the** ``fRAT`` **section of the GUI and then click the** :guilabel:`Run installation tests`
**button.**

This will first create a voxel-wise tSNR map for each subject in the ``example_data`` folder, before using these maps to
run an ROI analysis using these maps. ``fRAT`` will output progress to the terminal.
After the analysis has completed, if the full comparison option has been selected, the output will be compared to those
already present in the ``example_data`` folder and will warn if any files are missing.
