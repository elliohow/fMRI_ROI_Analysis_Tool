import re
import simplejson as json

from utils import *

def printResults():
    print('--- Printing results ---')
    print('Select the location of the ROI_report folder.')
    result_loc = Utils.file_browser()

    with open(f"{result_loc}/combined_results.json", "r") as results, \
            open(f"{result_loc}/config_log.py", "r") as config:
        config = config.read()
        results = json.load(results)

    params = parse_config(config)
    blacklist = ['index', 'File_name', *params]
    rois = sorted({result['index'] for result in results})  # Using set returns only unique values

    chosen_rois = user_input(rois)

    for roi in chosen_rois:  # For each chosen roi
        print(f"\n----------------------Chosen ROI {roi}: {rois[roi]}----------------------------")
        for result in results:  # For each entry_create in the combined_df
            if result['index'] == rois[roi]:  # If the entry_create is the correct roi, print the result
                print(f"\n -- File name: {result['File_name']}")
                for param in params:
                    print(f" -- {param}: {result[param]}")

                [print(f"{key}: {value}") for key, value in result.items() if key not in blacklist]
        print(f"----------------------Chosen ROI {roi}: {rois[roi]} end--------------------------")


def user_input(rois):
    print(f"--- Analysed ROIs ---")

    for counter, roi in enumerate(rois):
        print(f"{counter}: {roi}")

    while True:
        roi_ans = input(
        "Type a comma-separated list of the ROIs (listed above) you want to produce a figure for, "
        "'e.g. 2, 15, 7, 23' or 'all' for all rois. \nAlternatively press enter to skip this step: ")

        if roi_ans.lower() == "all":
            chosen_rois = list(range(0, len(rois)))

        elif len(roi_ans) > 0:
            chosen_rois = [x.strip() for x in roi_ans.split(',')]  # Split by comma and whitespace

            try:
                chosen_rois = list(map(int, chosen_rois))  # Convert each list item to integers
            except ValueError:
                print('Comma-separated list contains non integers.\n')
                chosen_rois = []

            # Error checking for list indices out of range
            for roi in chosen_rois:
                if roi > len(rois) - 1:
                    print('List contains non-valid selections.\n')
                    chosen_rois = False
                    break

            if not chosen_rois:
                continue

        else:  # Else statement for blank input, this skips printing results
            chosen_rois = []
            break

        if isinstance(chosen_rois, int):
            chosen_rois = [chosen_rois]

        return chosen_rois


def parse_config(config):
    # TODO: need to change this function as wont work with new version of config file, implement config_load function?
    param_dict_start = config.find('{', config.find('parameter_dict')) + 1
    param_dict_end = config.find('}', config.find('parameter_dict'))

    params = re.findall('["\'][A-Za-z0-9]*["\']:', config[param_dict_start: param_dict_end])
    params = [re.sub(r'[^A-Za-z0-9]', '', param) for param in params]

    return params


if __name__ == '__main__':
    printResults()
