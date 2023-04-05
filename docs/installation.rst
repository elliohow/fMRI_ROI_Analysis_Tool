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

Install pipx
============

This creates an isolated environment for the fRAT application and will allow you to run it from
anywhere.

On MacOS
--------

First install brew::

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Then use brew to install pipx::

    brew install pipx
    pipx ensurepath

pipx can then be upgraded with::

    brew update && brew upgrade pipx

On Linux and WSL2
-----------------

You will need Python 3.10 on your path. One way to do this is by using the method described below, however this can
also be achieved by using `miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_ or
`mamba <https://mamba.readthedocs.io/en/latest/installation.html>`_.

.. warning::
    If using Conda or Anaconda to install Python, the Tkinter version downloaded alongside Python `may not have access
    to the Freetype library <https://github.com/ContinuumIO/anaconda-issues/issues/6833>`_. In this case the GUI will not render text properly, with text appearing as the wrong size
    or missing style settings such as bold. If the ``Run analysis``, ``Run statistics``, ``Run plotting`` settings on
    the `General` page are bold, then the GUI is rendering properly.

    If desired, although not recommended this can be fixed by removing Conda's Tkinter. This causes Python to use the
    system's Tkinter library::

    conda remove --force tk


Install python 3.10::

    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.10

Install necessary packages::

    sudo apt install bc python3-pip python3-venv python3.10-distutils python3.10-tk

Install pipx using pip::

    python3 -m pip install --user pipx
    python3 -m pipx ensurepath

pipx can then be upgraded with::

    python3 -m pip install --user -U pipx

.. note::
    For WSL2 users, if the example data files are downloaded to windows before being copied over to linux, redundant
    "zone identifier" files may have been created in the example data folders and you may not have write access to this
    folder. To change this::

        sudo find /home/<WSLUserName>/Documents/fRAT/ -exec chmod a+rwx {} ";"
        find /home/<WSLUserName>/Documents/fRAT/ -type f -name "*:Zone.Identifier" -exec rm -rf {} +

Install fRAT using pipx
=======================
::

    pipx install frat-brain --python $(which python3.10)

fRAT can now be ran from the terminal from anywhere by using::

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

Checking installation with a test analysis
============================================
**1. Create the directory to save the example data folder**::

    mkdir ~/Documents/fRAT

This allows fRAT to find the example data. This folder will also be used to save configuration profiles in the future.

**2. Download the example dataset from** `here <https://osf.io/pbm3d/>`_.

**3. Extract the folder and place it in the newly created fRAT folder.**

**4. Click the** :guilabel:`General` **button in the** ``fRAT`` **section of the GUI and then click the**
:guilabel:`Run installation tests` **button.**

The sample analysis will take roughly 15 minutes on a machine with atleast 4 cores. ``fRAT`` will output progress to the
terminal. After the analysis has completed, the output will be compared to those already present in the ``example_data``
folder and will warn if any files are missing. If you want to check the output of the sample analysis, set the
"delete_test_folder" option to "Never".

.. note::
    During the sample analysis, warnings may occur. This is expected and does not indicate that fRAT is not running
    correctly.
