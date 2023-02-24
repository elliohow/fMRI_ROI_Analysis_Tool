import json

from .utils import *


def printResults(config_name):
    print('--- Printing results ---')
    print('Select the results directory created by fRAT.')
    result_loc = Utils.file_browser(title='Select the directory output by the fRAT')

    config = Utils.load_config(f'{Path(os.path.abspath(__file__)).parents[1]}/configuration_profiles/roi_analysis',
                               config_name)

    if config.averaging_type == 'Session averaged':
        subfolder = 'Session_averaged_results'
    else:
        subfolder = 'Participant_averaged_results'

    print(f'Showing {config.averaging_type} results.\n')

    with open(f"{result_loc}/Overall/Summarised_results/{subfolder}/combined_results.json", "r") as results:
        results = json.load(results)
        rois = sorted({result['index'] for result in results})  # Using set returns only unique values

    blacklist = ['index', 'File_name', *config.parameter_dict1]

    chosen_rois = user_input(rois)

    for roi in chosen_rois:  # For each chosen roi
        print(f"\n----------------------Chosen ROI {roi}: {rois[roi]}----------------------------")
        for result in results:  # For each entry_create in the combined_df
            if result['index'] == rois[roi]:  # If the entry_create is the correct roi, print the result
                print(f"\n -- File name: {result['File_name']}")
                for param in config.parameter_dict1:
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


if __name__ == '__main__':
    printResults()
