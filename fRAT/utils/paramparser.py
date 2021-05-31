import os
import re
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd

from .utils import Utils
from .analysis import Analysis


config = None


class ParamParser:
    @classmethod
    def run_parse(cls, cfg):
        global config
        config = cfg

        json_array = cls.json_search()
        combined_results_create = True

        # Double check if the user wants to skip parameter verification
        if config.verify_param_method == "manual":
            skip_verify = input(
                "Do you want to skip the verification of parameters and rely on file names instead? (y or n) ")
            skip_verify = skip_verify.lower()

            if skip_verify != "y":
                print("\nDo you want to verify the following MRI parameters during this step?")

                used_parameters = []
                for parameter in config.parameter_dict:
                    answer = input(parameter + "? (y or n) ")

                    if answer.lower() in ("y", "yes"):
                        used_parameters.append(1)
                    else:
                        used_parameters.append(0)

        else:
            skip_verify = True

        combined_dataframe = pd.DataFrame()

        if config.verify_param_method == 'table':
            table = cls.load_paramValues_file(config)

        for json in json_array:
            if json == "combined_results.json":
                continue

            if config.verify_param_method in ("manual", "name"):
                param_nums = cls.parse_params_from_file_name(json, config)
            else:
                param_nums = cls.parse_params_from_table_file(json, table)

            if not skip_verify:
                param_nums = cls.manually_verify_params(json, param_nums, used_parameters)

            if param_nums:
                combined_dataframe = cls.construct_combined_json(combined_dataframe, json, param_nums)

        # Save combined results
        if combined_results_create:
            combined_dataframe = combined_dataframe.reset_index()
            combined_dataframe.to_json("Summarised_results/combined_results.json", orient='records', indent=2)

    @classmethod
    def load_paramValues_file(cls, config):
        if os.path.isfile(f"{os.getcwd()}/paramValues.csv"):
            table = pd.read_csv("paramValues.csv")  # Load param table
        else:
            try:
                table = pd.read_csv(f"copy_paramValues.csv")  # Load param table
            except FileNotFoundError:
                raise Exception('Make sure a copy of paramValues.csv is in the chosen folder.')
        return table

    @classmethod
    def parse_params_from_table_file(cls, json_file_name, table):
        json_file_name = os.path.splitext(json_file_name)[0]  # Remove file extension

        table_row = table.loc[table["File name"] == json_file_name]
        table_row.columns = [x.lower() for x in table_row.columns]  # Convert to lower case for comparison to key later

        param_nums = []
        if table_row['ignore file? (y for yes, otherwise blank)'].to_string(index=False).strip().lower() in ('y', 'yes', 'true'):
            return param_nums

        else:
            for key in config.parameter_dict:
                try:
                    param_nums.append(float(table_row[key.lower()]))
                except ValueError:
                    param_nums.append(table_row[key.lower()].to_string(index=False).strip())
                except KeyError:
                    raise Exception(f'Key "{key}" not found in paramValues.csv. Check the Critical Parameters option '
                                    f'in the Parsing menu (parameter_dict1 if not using the GUI) correctly match the '
                                    f'paramValues.csv headers.')

        return param_nums

    @classmethod
    def parse_params_from_file_name(cls, json_file_name, cfg=config):
        """Search for MRI parameters in each json file name for use in table headers and created the combined json."""
        param_nums = []

        for key in cfg.parameter_dict:
            parameter = cfg.parameter_dict[key]  # Extract search term

            if parameter in cfg.binary_params:
                param = re.search("{}".format(parameter), json_file_name, flags=re.IGNORECASE)

                if param is not None:
                    param_nums.append('On')  # Save 'on' if parameter is found in file name
                else:
                    param_nums.append('Off')  # Save 'off' if parameter not found in file name

            else:
                # Float search
                param = re.search("{}[0-9]p[0-9]".format(parameter), json_file_name, flags=re.IGNORECASE)
                if param is not None:
                    param_nums.append(param[0][1] + "." + param[0][-1])
                    continue

                # If float search didnt work then Integer search
                param = re.search("{}[0-9]".format(parameter), json_file_name, flags=re.IGNORECASE)
                if param is not None:
                    param_nums.append(param[0][-1])  # Extract the number from the parameter

                else:
                    param_nums.append(str(param))  # Save None if parameter not found in file name

        return param_nums

    @classmethod
    def manually_verify_params(cls, json_file, param_nums, used_parameters):
        """Verify parsed parameters with user."""
        while True:
            print("\nFilename:\n" + json_file)

            print("\nParameters:")
            for counter, key in enumerate(config.parameter_dict):
                if used_parameters[counter] == 1:
                    print("{} = {}".format(key, param_nums[counter]))

            while True:  # Ask user if the parsed parameters are correct
                answer = input("Is this correct? (y or n) ")

                if answer.lower() not in ('y', 'yes', 'n', 'no'):
                    print("Not an appropriate choice.")
                else:
                    break

            if answer.lower() == "y" or answer.lower() == "yes":
                return param_nums

            else:  # If parameters are not correct, ask for the actual values
                print("Please input the correct values. Non-numeric input will not change the original values.")
                for counter, key in enumerate(config.parameter_dict):
                    if used_parameters[counter] == 1:
                        print("{} = {}".format(key, param_nums[counter]))
                        new_param_num = input("Actual value: ")

                        if re.match("[0-9]", new_param_num) and key not in config.binary_params:
                            param_nums[counter] = new_param_num
                        elif key in config.binary_params:
                            param_nums[counter] = new_param_num

                print("\nRe-checking...")  # Repeat the verification process

    @staticmethod
    def json_search():
        if config.run_analysis:
            from utils.analysis import Analysis
            json_directory = os.getcwd() + f"/{Analysis.save_location}"

            os.chdir(json_directory)

        elif config.output_folder_loc in ("", " "):
            print('Select the directory output by the fRAT.')
            json_directory = Utils.file_browser(title='Select the directory output by the fRAT', chdir=True)

        else:
            json_directory = config.output_folder_loc

            try:
                os.chdir(json_directory)
            except FileNotFoundError:
                raise FileNotFoundError('Output folder location (fRAT output folder location) in config.toml is not a valid directory.')

            if config.verbose:
                print(f'Output folder selection: {config.output_folder_loc}.')

        json_file_list = [os.path.basename(f) for f in glob(f"{json_directory}/Summarised_results/*.json")]

        if len(json_file_list) == 0:
            raise NameError('Folder selection error. Could not find json files in the "Summarised_results" directory.')
        else:
            return json_file_list

    @classmethod
    def construct_combined_json(cls, dataframe, json, parameters):
        if dataframe.empty:
            dataframe = pd.read_json(f"Summarised_results/{json}")
            dataframe = dataframe.transpose()

            for counter, parameter_name in enumerate(config.parameter_dict):
                dataframe[parameter_name] = parameters[counter]
                dataframe['File_name'] = os.path.splitext(json)[0]  # Save filename
        else:
            new_dataframe = pd.read_json(f"Summarised_results/{json}")
            new_dataframe = new_dataframe.transpose()

            for counter, parameter_name in enumerate(config.parameter_dict):
                new_dataframe[parameter_name] = parameters[counter]  # Add parameter columns

            new_dataframe['File_name'] = os.path.splitext(json)[0]  # Save filename

            dataframe = dataframe.append(new_dataframe, sort=True)

        return dataframe

    @classmethod
    def make_table(cls):
        config = Utils.load_config(Path(os.path.abspath(__file__)).parents[1], 'config.toml')  # Load config file

        print('--- Creating paramValues.csv ---')
        print('Select the NIFTI/ANALYZE file directory.')
        brain_directory = Utils.file_browser(title='Select the NIFTI/ANALYZE file directory', chdir=True)

        brain_file_list = Utils.find_files(brain_directory, "hdr", "nii.gz", "nii")
        brain_file_list = [os.path.splitext(brain)[0] for brain in brain_file_list]
        brain_file_list.sort()

        padding = np.empty((len(brain_file_list)))
        padding[:] = np.NaN

        params = []
        for file in brain_file_list:
            # Try to find parameters to prefill table
            params.append(ParamParser.parse_params_from_file_name(file, config))
        params = np.array(params).transpose()

        df = pd.DataFrame(data={'File name': brain_file_list})

        for counter, param in enumerate(config.parameter_dict):
            df[param] = params[counter]

        df['Ignore file? (y for yes, otherwise blank)'] = padding

        df.to_csv('paramValues.csv', index=False)

        print(f"\nparamValues.csv saved in {brain_directory}.\n\nInput parameter values in paramValues.csv and change "
              f"make_table_only to False in the config file to continue analysis. \nIf analysis has already been "
              f"conducted, move paramValues.csv into the ROI report folder. \nIf the csv file contains unexpected "
              f"parameters, update the parsing options in the GUI or parameter_dict2 in config.toml.")
