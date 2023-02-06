.. include:: links.rst

============
Installation
============
.. contents:: :local:
    :depth: 2

.. note::
    Make sure Python is downloaded and is the same as the tested version (found on the :doc:`home page <index>`) or
    higher.

**1. Navigate to where you wish to install fRAT, and then run the following command in the terminal**::

    $ git clone https://github.com/elliohow/fMRI_ROI_Analysis_Tool.git

**2. Once the directory has been cloned, enter the newly created folder**::

    $ cd fMRI_ROI_Analysis_Tool


.. note::
    Installing the dependencies within a virtual environment keeps dependencies required by different projects separate,
    preventing a number of dependency issues. You can install all the dependencies required by the fRAT within a *virtual
    environment* using the `venv` module built-in to python, `conda`, or any other virtual environment package installed on
    your system.

**3. Create a virtual environment called venv**:

`conda` version::

    $ conda create --name venv python==3.8.0

`venv` version::

    $ python3 -m venv venv

.. note::
    Python version can also be specified when using `venv`::

        $ python3.8 -m venv venv

**4. Activate the virtual environment**:

`conda` version::

    $ conda activate fRAT

`venv` version::

    $ source fRAT/bin/activate

.. note::
    The virtual environment will need to be activated whenever running fRAT.

The ``requirements.txt`` file in the base folder lists the dependencies necessary for fRAT.

**5. Install all required dependencies from ``requirements.txt`` in the activated virtual environment**::

    $ pip3 install -r requirements.txt

.. warning::
    Dependencies will be installed globally instead of in the virtual environment if the virtual environment has not
    been activated before running this step. As installation of dependencies globally may cause dependency conflicts,
    it is suggested that a virtual environment is used.

You should see now ``(fRAT)`` at the start of your prompt.

**6. Change directory into the ``fRAT`` folder and open the GUI**::

    $ cd fRAT
    $ python3 fRAT_GUI.py

`venv` notes
~~~~~~~~~~~~~~~~~~~~~
* ``venv`` by default uses the currently installed version of Python to set up the virtual environment, however some other virtual environment packages (such as ``virtualenv`` and ``conda``) do not have this limitation.
* To deactive the ``venv`` environment you can use ``deactivate``.

`conda` notes
~~~~~~~~~~~~~~~~
* To check currently installed conda environments can use ``conda env list``.
* To revert back to another environment you can use ``conda deactivate``. This should set your prompt back to ``(base)``.

External Dependencies
=====================
The FSL_ neuroimaging software tool is a core component of fRAT and is not able to be downloaded through Python’s
Package Index (PyPI), therefore FSL_ (version 6.0.2 or above)  should be downloaded and setup separately using FSL's
`installation instructions <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_.

While optiBET_ is not required for fRAT (`link <https://montilab.psych.ucla.edu/fmri-wiki/optibet/>`_), it is highly
recommended that this software tool be used for brain extraction
due to the necessity of accurate skull stripping for the fRAT to function optimally and the consistently better brain
extraction performance when using optiBET.

Checking installation
=====================
To check that fRAT and its dependencies are working correctly, first download the example files `here
<https://osf.io/pbm3d/>`_, extract them, and place them in the ``fMRI_ROI_Analysis_Tool`` folder.
Next, click the :guilabel:`General` button in the ``fRAT`` section
of the GUI and then click the :guilabel:`Run installation tests` button. This will first create a voxel-wise tSNR map for each subject in the
``example_data`` folder, and will then run an ROI analysis using these maps. ``fRAT`` will output progress to the terminal.
After the analysis has completed, the output from the ROI analysis will be compared to those already
present in the ``example_data`` folder and will warn if any files are missing or different. The ``example_data`` folder
also gives demonstrates the input necessary for the fRAT and the output produced by the fRAT.

.. note::
    When running fRAT, you will be notified if there is an update available for Nipype; it is recommended that you update
    this library if possible.