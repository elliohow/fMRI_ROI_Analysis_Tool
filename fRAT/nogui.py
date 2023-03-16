import datetime
import itertools
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

from fRAT.utils import *
from fRAT._version import __version__


def fRAT(config_filename, path=None):
    start_time = time.perf_counter()
    Utils.checkversion(__version__)

    config_path = f'{Path(os.path.abspath(__file__)).parents[0]}/configuration_profiles/roi_analysis/'
    config, orig_path = load_config(config_filename, path=path)
    config = argparser(config)

    # CompareOutputs.run(config)  # TODO THIS IS TEST CODE
    # sys.exit()

    if config.verbose and config.run_analysis and config.run_plotting and config.run_statistics:
        print(f"\n--- Running analysis, plotting and statistics steps ---")

    # Run the ROI analysis
    if config.run_analysis:
        analysis(config, config_path, config_filename)

    # Plot the results
    if config.run_plotting:
        plotting(config, config_path, config_filename, orig_path)

    # Run the statistical analysis
    if config.run_statistics:
        if config.verbose:
            print('\n------------------\n--- Statistics ---\n------------------')

        Utils.chdir_to_output_directory('Statistics', config)
        statistics_main(config, config_path, config_filename)

    os.chdir(orig_path)  # Reset path

    if config.verbose:
        elapsed = datetime.timedelta(seconds=time.perf_counter() - start_time)
        elapsed_min = elapsed.seconds // 60
        elapsed_sec = elapsed.seconds % 60
        print(f"--- Completed in {elapsed_min} minutes {elapsed_sec} seconds ---\n\n")


def load_config(config_filename, path=None):
    orig_path = Path(os.path.abspath(__file__)).parents[0]

    # Reload config file incase GUI has changed it
    config = Utils.load_config(f'{Path(os.path.abspath(__file__)).parents[0]}/configuration_profiles/roi_analysis',
                               config_filename, path=path)
    config_check(config)

    return config, orig_path


def argparser(config):
    # Check arguments passed over command line
    args = Utils.argparser()

    if args.make_table == 'true':
        from __main__ import make_table
        make_table()
        sys.exit()
    else:
        args.__dict__.pop('make_table')

    for arg in args.__dict__:
        if args.__dict__[arg] is not None:
            user_arg = Utils.convert_toml_input_to_python_object(args.__dict__[arg])
            config.__dict__[arg] = user_arg

    config = Utils.clean_config_options(config)

    return config


def plotting(config, config_path, config_filename, orig_path):
    if config.verbose:
        print('\n----------------\n--- Plotting ---\n----------------')

    # Plotting
    Utils.chdir_to_output_directory('Plotting', config)
    Figures.make_figures(config, config_path, config_filename)

    # Create html report
    create_html_report(str(orig_path))
    if config.verbose:
        print('Created html report.')


def analysis(config, config_path, config_filename):
    if config.verbose:
        print('\n----------------\n--- Analysis ---\n----------------')

    if config.multicore_processing:
        pool = Utils.start_processing_pool()
    else:
        pool = None

    # Run class setup
    participant_list, matched_brains = Environment_Setup.setup_analysis(config, config_path, config_filename, pool)

    Utils.move_file(config.parameter_file, os.getcwd(), os.getcwd() + f"/{Environment_Setup.save_location}",
                    copy=True, parameter_file=True)

    if config.verbose:
        print('\n--- Running individual analysis ---')

    brain_list = []
    for participant in participant_list:
        brain_list.extend(participant.run_analysis(pool))

        if config.file_cleanup == 'move' and not participant.all_files_ignored:
            shutil.move(f"{participant.save_location}/motion_correction_files",
                        f"{participant.save_location}Intermediate_files/motion_correction_files")

        elif config.file_cleanup == 'delete' and not participant.all_files_ignored:
            shutil.rmtree(f"{participant.save_location}/motion_correction_files")

    matched_brains = run_pooled_analysis(brain_list, matched_brains, config, pool)

    calculate_cost_function_and_displacement_values(participant_list, brain_list, config, pool)
    atlas_scale(matched_brains, config, pool)

    if config.multicore_processing:
        Utils.join_processing_pool(pool, restart=False)


def run_pooled_analysis(brain_list, matched_brains, config, pool):
    if config.verbose:
        print('\n--- Running pooled analysis ---')

    for parameter_comb in matched_brains:
        for brain in brain_list:
            try:
                if brain.no_ext_brain in parameter_comb.brains[brain.participant_name]:
                    parameter_comb.ungrouped_summarised_results.append(brain.roi_results)
                    parameter_comb.ungrouped_raw_results.append(brain.roi_temp_store)
                    parameter_comb.participant_grouped_summarised_results[brain.participant_name].append(brain.roi_results)

            except KeyError:
                pass

    # Save each raw and overall results for each parameter combination
    iterable = zip(matched_brains,
                   itertools.repeat("compile_results"),
                   itertools.repeat(os.getcwd()),
                   itertools.repeat(config))

    if config.multicore_processing:
        matched_brains = pool.starmap(Utils.instance_method_handler, iterable)
    else:
        matched_brains = list(itertools.starmap(Utils.instance_method_handler, iterable))

    # Compile the overall results for every parameter combination
    construct_combined_results(MatchedBrain.save_location, subfolder='session averaged')
    construct_combined_results(MatchedBrain.save_location, subfolder='participant averaged')

    return matched_brains


def calculate_cost_function_and_displacement_values(participant_list, brain_list, config, pool):
    if config.verbose:
        print('\n--- Calculating cost function and displacement values ---')

    vals = []

    for counter, participant in enumerate(participant_list):
        if config.verbose:
            print(f'Calculating cost function value for anatomical file: {participant.anat_brain_no_ext}')

        if participant.all_files_ignored:
            continue

        df = participant.calculate_anat_flirt_cost_function()
        vals.append(df)

    # Set arguments to pass to fmri_get_additional_info
    iterable = zip(brain_list, itertools.repeat("fmri_get_additional_info"),
                   range(len(brain_list)),
                   itertools.repeat(len(brain_list)),
                   itertools.repeat(config))

    if config.multicore_processing:
        results = pool.starmap(Utils.instance_method_handler, iterable)
    else:
        results = list(itertools.starmap(Utils.instance_method_handler, iterable))

    vals.extend(results)

    if config.verbose:
        print(f'\nSaving dataframe as additional_info.csv')

    df = pd.concat(vals).replace([np.nan], [None]).sort_values(['Participant', 'File']).reset_index(drop=True)

    with open(f"{Environment_Setup.save_location}additional_info.csv", 'w') as file:
        df.to_csv(path_or_buf=file)


def run_atlas_scale_function_for_each_statistic(first_combination, matched_brains, statistic_number,
                                                pool, config, data):
    if data == 'Session averaged':
        roi_stats = deepcopy(first_combination.session_averaged_results[statistic_number, :])

    else:
        roi_stats = deepcopy(first_combination.participant_averaged_results[statistic_number, :])

    for parameter_combination in matched_brains:
        roi_stats = find_highest_value_from_overall_results(parameter_combination, roi_stats, statistic_number,
                                                            data=data)

    # Set arguments to pass to atlas_scale function
    iterable = zip(matched_brains, itertools.repeat("atlas_scale"), itertools.repeat(roi_stats),
                   range(len(matched_brains)), itertools.repeat(len(matched_brains)),
                   itertools.repeat(statistic_number), itertools.repeat(Environment_Setup.atlas_path),
                   itertools.repeat(data), itertools.repeat(config))

    # Run atlas_scale function and pass in max roi stats for between brain scaling
    if config.multicore_processing:
        pool.starmap(Utils.instance_method_handler, iterable)

    else:
        list(itertools.starmap(Utils.instance_method_handler, iterable))


def find_highest_value_from_overall_results(parameter_combination, roi_stats, statistic_number, data):
    if data == 'Session averaged':
        results = parameter_combination.session_averaged_results[statistic_number, :]
    else:
        results = parameter_combination.participant_averaged_results[statistic_number, :]

    for counter, roi_stat in enumerate(results):
        if roi_stat > roi_stats[counter]:
            roi_stats[counter] = roi_stat

    return roi_stats


def atlas_scale(matched_brains, config, pool):
    """Save a copy of each statistic for each ROI from the first brain. Then using sequential comparison
       find the largest statistic values for each ROI out of all the brains analyzed."""
    if config.verbose:
        print('\n--- Atlas scaling ---')

    # Make directory to store scaled brains
    Utils.check_and_make_dir(f"{os.getcwd()}/{MatchedBrain.save_location}NIFTI_ROI/")
    Utils.check_and_make_dir(f"{os.getcwd()}/{MatchedBrain.save_location}NIFTI_ROI/Session_averaged_results/")
    Utils.check_and_make_dir(f"{os.getcwd()}/{MatchedBrain.save_location}NIFTI_ROI/Participant_averaged_results/")

    first_combination = next(iter(matched_brains))

    if config.verbose:
        print('Creating NIFTI ROI images using session averaged data.\n')

    for statistic_number in range(len(first_combination.session_averaged_results)):
        run_atlas_scale_function_for_each_statistic(first_combination, matched_brains, statistic_number,
                                                    pool, config, data='Session averaged')
    if config.verbose:
        print('\nCreating NIFTI ROI images using participant data.\n')

    for statistic_number in range(len(first_combination.participant_averaged_results)):
        run_atlas_scale_function_for_each_statistic(first_combination, matched_brains, statistic_number,
                                                    pool, config, data='Participant averaged')


def config_check(config):
    pass


if __name__ == '__main__':
    fRAT()
