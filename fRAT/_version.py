import toml
import pathlib

pyproject_path = f"{pathlib.Path(__file__).parents[1].resolve()}/pyproject.toml"
pyproject_dict = toml.load(pyproject_path)

__version__ = pyproject_dict['tool']['poetry']['version']
