.. include:: links.rst

============
Installation
============
To install all required modules use the following command in the project directory: ``pip install -r requirements.txt``.

The ``requirements.txt`` file in the base folder lists the dependencies necessary for fRAT. However, when running fRAT,
you will be notified if there is an update available for Nipype; it is recommended that you update this library if
possible.

Virtual environments
====================
Using virtual environments keeps dependencies required by different projects separate by creating an isolated virtual
environment for each project, preventing a number of dependency issues. You can install all the dependencies required by
the fRAT within a *virtual environment* using `conda`, the `venv` module built-in to python or any other virtual
environment package installed on your system.

`venv` and `conda` setup
------------------------
These instructions apply if you are running under `Linux` or `MacOS`. For `Windows` there may be similar solutions,
however be aware that fRAT is untested with Windows due to difficulties running software important to fRAT such as FSL.

.. note::
    Make sure Python is downloaded and is the same as the tested version or higher # TODO (found in README.rst)

First, create a virtual environment called fRAT: ::

    # conda version
    $ conda create --name fRAT python==3.8.0
    # venv version
    $ python3 -m venv fRAT

Next activate the virtual environment: ::

    # conda version
    $ conda activate fRAT
    # venv version
    $ source fRAT/bin/activate

After changing directory into the base fRAT folder, install dependencies using pip: ::

    $ pip install -r requirements.txt

You should see "(fRAT)" at start of your prompt. Now you can run the tool after changing directory into the fRAT folder: ::

    $ cd fRAT
    $ python fRAT_GUI.py

``venv`` python version
~~~~~~~~~~~~~~~~~~~~~~~
* ``venv`` by default uses the currently installed version of Python to set up the virtual environment, however some other
virtual environment packages (such as ``virtualenv`` and ``conda``) do not have this limitation.

``conda`` commands
~~~~~~~~~~~~~~~~~~
* To check currently installed conda environments can use ``conda env list``.
* To revert back to another environment you can use ``conda deactivate``. This should set your prompt back to ``(base)``.

External Dependencies
=====================
fRAT is written using Python 3.8.0 and is based on nipype.

As FSL_ is a core component of the fRAT and is not able to be downloaded through Python’s packaging system (Pypi),
FSL_ (version 6.0.2 or above) neuroimaging software tool should be downloaded and setup separately,