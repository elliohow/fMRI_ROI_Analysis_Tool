import itertools
import os
import importlib
from pathlib import Path

from ..utils import Utils

config = None
UTILITIES = None


def import_utilities():
    global UTILITIES
    from . import __all__

    UTILITIES = [importlib.import_module(f".{utility}", package=__package__) for utility in __all__]


def file_setup(utility):
    file_location = config.input_folder_name

    if config.output_folder_name != 'DEFAULT':
        output_folder = config.output_folder_name
    else:
        output_folder = utility.FOLDER_NAME

    if config.base_folder in ("", " "):
        print('Select the directory which contains the subject folders.')
        base_sub_location = Utils.file_browser(title='Select the directory which contains the subject folders')

    else:
        base_sub_location = config.base_folder

    # Create dictionary from each participant directory
    participant_dir, _ = Utils.find_participant_dirs(base_sub_location)
    participants = {participant: None for participant in participant_dir}

    for participant in participants:
        # Find all nifti and analyze files
        participants[participant] = Utils.find_files(f"{participant}/{file_location}", "hdr", "nii.gz", "nii")

        # Make directory to save results
        if isinstance(output_folder, list):
            for folder in output_folder:
                Utils.check_and_make_dir(f"{participant}/{folder}", delete_old=True)
        else:
            Utils.check_and_make_dir(f"{participant}/{output_folder}", delete_old=True)

    return participants, output_folder, file_location


def utility_handler(participants, output_folder, file_location, utility):
    if config.multicore_processing:
        pool = Utils.start_processing_pool()
    else:
        pool = None

    if config.verbose and isinstance(output_folder, list):
        direcs = ', '.join(output_folder)
        print(f'\nSaving output in directories: {direcs}')
    elif config.verbose and len(output_folder) == 1:
        print(f'\nSaving output in directory: {output_folder}')

    for participant_dir, files in participants.items():
        if config.verbose:
            print(f"\nRunning '{utility.UTILITY_NAME}' utility on participant: {participant_dir.split('/')[-1]}"
                  f"\n      Running utility on {len(files)} files")

        iterable = zip(files,
                       itertools.repeat(participant_dir),
                       itertools.repeat(file_location),
                       itertools.repeat(output_folder),
                       itertools.repeat(utility),
                       itertools.repeat(config))

        if config.multicore_processing:
            return_val = list(pool.starmap(run_utility, iterable))

        else:
            return_val = list(itertools.starmap(run_utility, iterable))

        if return_val[0] is not None:
                iterable = zip(files,
                               itertools.repeat(participant_dir),
                               itertools.repeat(file_location),
                               itertools.repeat(output_folder),
                               itertools.repeat(utility),
                               itertools.repeat(config),
                               itertools.repeat(return_val))

                if config.multicore_processing:
                    list(pool.starmap(run_utility, iterable))

                else:
                    list(itertools.starmap(run_utility, iterable))

    if config.multicore_processing:
        Utils.join_processing_pool(pool, restart=False)


def run_utility(file, participant_dir, file_location, output_folder, utility, config, *args):
    no_ext_file = Utils.strip_ext(file)
    file = f"{participant_dir}/{file_location}/{file}"
    base_sub_location = Path(participant_dir).parents[0]
    participant_name = participant_dir.split('/')[-1]

    if config.verbose and not args:
        print(f'        Running utility on file: {no_ext_file}')

    return_val = utility.run(config, file, no_ext_file, base_sub_location, participant_name, participant_dir, output_folder, return_val=args)

    return return_val


def find_current_utility(func):
    for utility in UTILITIES:
        if func == utility.UTILITY_NAME:
            return utility


def HOUSE(cfg, func):
    global config
    config = cfg

    import_utilities()
    utility = find_current_utility(func)
    participants, output_folder, file_location = file_setup(utility)
    utility_handler(participants, output_folder, file_location, utility)

