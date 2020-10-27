import os
import re
import sys
from glob import glob
import numpy as np
import pandas as pd

from roianalysis import config
from roianalysis.utils import Utils


class ParamParser:
    @classmethod
    def run_parse(cls):
        json_array = cls.json_search()
        combined_results_create = True

        while True and not config.always_replace_combined_json:
            if "combined_results.json" in json_array:
                answer = input("\nCombined results file already found in folder, "
                               "do you want to continue with this file or replace it? (continue or replace) ")

                if answer.lower() not in ('replace', 'continue'):
                    print("Not an appropriate choice.")
                else:
                    if answer.lower() == "replace":
                        os.remove("combined_results.json")
                        json_array.remove("combined_results.json")
                        print("File removed!")
                        combined_results_create = True
                    else:
                        combined_results_create = False
                    break
            else:
                break

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

        if config.verify_param_method == "table":
            if os.path.isfile(f"{os.getcwd()}/{config.param_table_name}"):
                table = pd.read_csv(config.param_table_name)  # Load param table
            else:
                try:
                    table = pd.read_csv(f"copy_{config.param_table_name}")  # Load param table
                except FileNotFoundError:
                    raise Exception('Make sure a copy of paramValues.csv is in the chosen folder.')

        for json in json_array:
            if json == "combined_results.json":
                continue

            if config.verify_param_method in ("manual", "name"):
                param_nums = cls.parse_params_from_file_name(json)
            else:
                param_nums = cls.parse_params_from_table_file(json, table)

            if not skip_verify:
                param_nums = cls.manually_verify_params(json, param_nums, used_parameters)

            if param_nums:
                combined_dataframe = cls.construct_combined_json(combined_dataframe, json, param_nums)

        # Save combined results
        if combined_results_create:
            combined_dataframe = combined_dataframe.reset_index()
            combined_dataframe.to_json("combined_results.json", orient='records')

    @classmethod
    def parse_params_from_table_file(cls, json_file_name, table):
        # Find atlas used to remove text from the start of the json file name
        with open('config_log.py', 'r') as config_log:
            for line in config_log:
                line = line.rstrip()  # remove '\n' at end of line

                atlas_number = re.match("atlas_number = [0-9]", line)  # Search for atlas used from config_log
                if atlas_number:
                    from roianalysis.analysis import Analysis
                    atlas_name = os.path.splitext(Analysis._atlas_label_list[int(atlas_number[0][-1])][1])[0] + "_"

                    break

        json_file_name = json_file_name[len(atlas_name):]  # Remove atlas name prefix from file name
        json_file_name = os.path.splitext(json_file_name)[0]  # Remove file extension

        table_row = table.loc[table["File name"] == json_file_name]

        param_nums = []
        if table_row['Ignore file? (y for yes, otherwise blank)'].to_string(index=False).strip().lower() == 'y':
            return param_nums
        else:
            for key in config.parameter_dict:
                param_nums.append(float(table_row[key]))

        return param_nums

    @classmethod
    def parse_params_from_file_name(cls, json_file_name):
        """Search for MRI parameters in each json file name for use in table headers and created the combined json."""
        param_nums = []

        for key in config.parameter_dict:
            parameter = config.parameter_dict[key]  # Extract search term

            if parameter in config.binary_params:
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
        if config.run_steps == "all":
            from roianalysis.analysis import Analysis
            json_directory = os.getcwd() + f"/{Analysis._save_location}"

            if config.verify_param_method == "table":  # Move excel file containing parameter file info
                Utils.move_file(config.param_table_name, os.getcwd(), json_directory, copy=True)

            os.chdir(json_directory)

        elif config.json_file_loc in ("", " "):
            print('Select the directory of json files.')
            json_directory = Utils.file_browser(chdir=True)

        else:
            json_directory = config.json_file_loc

            try:
                os.chdir(json_directory)
            except FileNotFoundError:
                raise FileNotFoundError('json_file_loc in config.py is not a valid directory.')

            if config.verbose:
                print(f'Gathering json files from {config.json_file_loc}.')

        json_file_list = [os.path.basename(f) for f in glob(json_directory + "/*.json")]

        if len(json_file_list) == 0:
            raise NameError('No json files found.')
        else:
            return json_file_list

    @classmethod
    def construct_combined_json(cls, dataframe, json, parameters):
        if dataframe.empty:
            dataframe = pd.read_json(json)
            dataframe = dataframe.transpose()

            for counter, parameter_name in enumerate(config.parameter_dict):
                dataframe[parameter_name] = parameters[counter]
                dataframe['File_name'] = os.path.splitext(json)[0]  # Save filename
        else:
            new_dataframe = pd.read_json(json)
            new_dataframe = new_dataframe.transpose()

            for counter, parameter_name in enumerate(config.parameter_dict):
                new_dataframe[parameter_name] = parameters[counter]  # Add parameter columns

            new_dataframe['File_name'] = os.path.splitext(json)[0]  # Save filename

            dataframe = dataframe.append(new_dataframe, sort=True)

        return dataframe

    @classmethod
    def make_table(cls):
        print('Select the nifti/analyse file directory.')
        brain_directory = Utils.file_browser(chdir=True)

        brain_file_list = Utils.find_files(brain_directory, "hdr", "nii.gz", "nii")
        brain_file_list = [os.path.splitext(brain)[0] for brain in brain_file_list]
        brain_file_list.sort()

        padding = np.empty((len(brain_file_list)))
        padding[:] = np.NaN

        params = []
        for file in brain_file_list:
            # Try to find parameters to prefill table
            params.append(ParamParser.parse_params_from_file_name(file))
        params = np.array(params).transpose()

        df = pd.DataFrame(data={'File name': brain_file_list})

        for counter, param in enumerate(config.parameter_dict):
            df[param] = params[counter]

        df['Ignore file? (y for yes, otherwise blank)'] = padding

        df.to_csv('paramValues.csv', index=False)

        print(f"\nparamValues.csv saved in {brain_directory}.\n\nInput parameter values in paramValues.csv and change "
              f"make_table_only to False in the config file to continue analysis. \nIf analysis has already been "
              f"conducted, move paramValues.csv into the ROI report folder. \nIf the csv file contains unexpected "
              f"parameters, update config.parameter_dict.")

        sys.exit()