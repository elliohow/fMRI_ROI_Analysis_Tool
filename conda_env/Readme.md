## Installing with `conda`

These instructions apply if you are running under `linux` or `macOS`. For Windows, there may be similar solutions:

- make sure you have [minconda / conda](https://docs.conda.io/en/latest/miniconda.html) installed.

- inspect the `conda_env` folder. The `yml` file in there contains the specification of the environment and dependencies

- the `req.txt` files has details on exact versions that work (when tested)

```bash
cd conda_env

# create a conda environment called fRAT 
conda-env create -n fRAT -f fRAT_env.yml

# activate it
conda activate fRAT

# you should see "(fRAT)" at start of your prompt
# now you can run the tool

cd ../fRAT
python --version # should be 3.7.5
python fRAT_GUI.py
```

- To revert back to other evironment (eg base) you can `conda deactivate` - this shold set your prompt back to ``(base)`` and standard setup.

