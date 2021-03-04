# Installing depencies in virtual environments

## `venv` and `conda` setup
These instructions apply if you are running under `Linux` or `MacOS`. For Windows there may be similar solutions, however be aware that fRAT is untested with Windows due to difficulties running software important to fRAT such as FSL.

```bash
# Create an environment called fRAT

# conda version
# NOTE: Make sure python version is same as the tested version (found in README.md)
conda create --name fRAT python==3.7.5
# venv version
python3 -m venv fRAT

# Activate it

# conda version
conda activate fRAT
# venv version
source fRAT/bin/activate

# After changing directory into the base fRAT folder, install dependencies using pip
pip install -r requirements.txt

# You should see "(fRAT)" at start of your prompt
# Now you can run the tool after changing directory into the fRAT folder
cd fRAT
python fRAT_GUI.py
```

## Additional information
* The `requirements.txt` file in the fRAT base folder has details on exact versions of dependencies that worked when tested.

### `venv` python version
* `venv` by default uses the currently installed version of Python to set up the virtual environment, however some other virtual environment packages (such as `virtualenv` and `conda`) do not have this limitation.

### `conda` commands
* To check currently installed conda environments can use `conda env list`.
* To revert back to another environment you can use `conda deactivate`. This should set your prompt back to ``(base)``.
