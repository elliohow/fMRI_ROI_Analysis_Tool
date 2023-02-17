.. include:: links.rst

============
Installation
============
.. contents:: :local:
    :depth: 2

.. note::
    Make sure Python version |python_version| is used for installation.

**1. Install pipx. This will create an isolated environment for the fRAT application and will allow you to run it from
anywhere**::

    brew install pipx
    pipx ensurepath

**2. Use pipx to install fRAT**::

    pipx install frat-brain --python /Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10

.. note::
    If you get a file not found error using the above, the link to the Python executable is not correct for your system.

    First, start a python session using the version of Python you want to use to install fRAT::

        python3.10

    And then print the path to the Python executable using::

        import sys
        print(sys.executable)

    This path should then be used inplace of the one above.

**3. fRAT can now be ran from the terminal from anywhere by using**::

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

**2. Download the example dataset from** `here <https://osf.io/pbm3d/>`_

**3. Extract the chosen folder and place it in the newly created fRAT folder.**
To check that fRAT and its dependencies are working correctly,

**4. Click the** :guilabel:`General` **button in the** ``fRAT`` **section of the GUI and then click the** :guilabel:`Run installation tests`
**button.**
This will first create a voxel-wise tSNR map for each subject in the ``example_data`` folder, before using these maps to
run an ROI analysis using these maps. ``fRAT`` will output progress to the terminal.
After the analysis has completed, the output from the ROI analysis will be compared to those already
present in the ``example_data`` folder and will warn if any files are missing.
