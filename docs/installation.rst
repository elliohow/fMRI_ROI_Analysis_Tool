.. include:: links.rst

============
Installation
============
.. contents:: :local:
    :depth: 2

Firstly navigate to where you wish to install fRAT, and then run the following command in the terminal::

    $ git clone https://github.com/elliohow/fMRI_ROI_Analysis_Tool.git

The ``requirements.txt`` file in the base folder lists the dependencies necessary for fRAT. To install all required
dependencies globally (as opposed to in a virtual environment), use the following command in the project directory::

    $ pip install -r requirements.txt

When running fRAT, you will be notified if there is an update available for Nipype; it is recommended that you update
this library if possible.

Virtual environments
====================
While it is possible to install the fRAT dependencies as shown above, installing the dependencies within a
virtual environment keeps dependencies required by different projects separate, preventing a number of dependency issues.
You can install all the dependencies required by the fRAT within a *virtual environment* using the `venv` module
built-in to python, `conda`, or any other virtual environment package installed on your system.

`venv` and `conda` setup
------------------------
These instructions apply if you are running under `Linux` or `MacOS`. For `Windows` there may be similar solutions,
however be aware that fRAT is untested with Windows due to difficulties running software necessary for fRAT such as FSL.

.. note::
    Make sure Python is downloaded and is the same as the tested version (found on the :doc:`home page <index>`) or
    higher.

First, create a virtual environment called fRAT:

`conda` version::

    $ conda create --name fRAT python==3.8.0

`venv` version::

    $ python3 -m venv fRAT

Next activate the virtual environment:

`conda` version::

    $ conda activate fRAT

`venv` version::

    $ source fRAT/bin/activate

After changing directory into the base fRAT folder, install dependencies using pip: ::

    $ pip install -r requirements.txt

You should see ``(fRAT)`` at the start of your prompt. Now you can run the tool after changing directory into the fRAT
folder: ::

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
As FSL_ is a core component of fRAT and is not able to be downloaded through Python’s Package Index (PyPI), therefore
FSL_ (version 6.0.2 or above) neuroimaging software tool should be downloaded and setup separately.

While optiBET_ is not required for fRAT, it is highly recommended that this software tool be used for brain extraction
due to the necessity of accurate skull stripping for the fRAT to function optimally and the consistently better brain
extraction performance when using optiBET.