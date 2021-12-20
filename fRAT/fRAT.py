import datetime
import itertools
import sys
import time
import os
from copy import deepcopy
from pathlib import Path

from utils import *


def fRAT():
    start_time = time.perf_counter()
    Utils.checkversion()

    config, orig_path = load_config()
    config = argparser(config)

    # CompareOutputs.run(config)  # TODO THIS IS TEST CODE
    # sys.exit()

    if config.verbose and config.run_analysis and config.run_plotting and config.run_statistics == 'all':
        print(f"\n--- Running all steps ---")

    # Run the analysis
    if config.run_analysis:
        analysis(config)

    # Plot the results
    if config.run_plotting:
        plotting(config, orig_path)

    if config.run_statistics:
        statistics(config)

    os.chdir(orig_path)  # Reset path

    if config.verbose:
        elapsed = datetime.timedelta(seconds=time.perf_counter() - start_time)
        elapsed_min = elapsed.seconds // 60
        elapsed_sec = elapsed.seconds % 60
        print(f"--- Completed in {elapsed_min} minutes {elapsed_sec} seconds ---\n\n")


def load_config():
    orig_path = Path(os.path.abspath(__file__)).parents[0]
    config = Utils.load_config(orig_path, 'fRAT_config.toml')  # Reload config file incase GUI has changed it
    config_check(config)
    return config, orig_path


def argparser(config):
    # Check arguments passed over command line
    args = Utils.argparser()

    if args.make_table == 'true':
        from fRAT_GUI import make_table
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


def plotting(config, orig_path):
    if config.verbose:
        print('\n----------------\n--- Plotting ---\n----------------')

    # Plotting
    Figures.setup_environment(Environment_Setup.save_location, config)
    Figures.make_figures()

    # Create html report
    html_report.main(str(orig_path))
    if config.verbose:
        print('Created html report.')


def analysis(config):
    if config.verbose:
        print('\n----------------\n--- Analysis ---\n----------------')

    if config.multicore_processing:
        pool = Utils.start_processing_pool()
    else:
        pool = None

    # Run class setup
    participant_list, matched_brains = Environment_Setup.setup_analysis(config, pool)

    Utils.move_file("paramValues.csv", os.getcwd(), os.getcwd() + f"/{Environment_Setup.save_location}", copy=True)

    if config.verbose:
        print('\n--- Running individual analysis ---')

    brain_list = []
    for participant in participant_list:
        brain_list.extend(participant.run_analysis(pool))

        if config.file_cleanup == 'move':
            shutil.move(f"{participant.save_location}/motion_correction_files",
                        f"{participant.save_location}Intermediate_files/motion_correction_files")
        elif config.file_cleanup == 'delete':
            shutil.rmtree(f"{participant.save_location}/motion_correction_files")

    if config.multicore_processing:
        pool = Utils.join_processing_pool(pool, restart=True)

    matched_brains = run_pooled_analysis(brain_list, matched_brains, config, pool)

    if config.outlier_detection_method == 'pooled':
        brain_list = assign_pooled_thresholds_to_brains(brain_list, matched_brains)

    calculate_cost_function_and_displacement_values(participant_list, brain_list, config, pool)
    atlas_scale(matched_brains, config, pool)

    if config.multicore_processing:
        Utils.join_processing_pool(pool, restart=False)


def assign_pooled_thresholds_to_brains(brain_list, matched_brains):
    for parameter_comb in matched_brains:
        for brain in brain_list:
            if parameter_comb.brains[brain.participant_name] == brain.no_ext_brain:
                brain.noise_threshold = parameter_comb.noise_threshold
                brain.lower_gaussian_threshold = parameter_comb.lower_gaussian_threshold
                brain.upper_gaussian_threshold = parameter_comb.upper_gaussian_threshold

    return brain_list


def run_pooled_analysis(brain_list, matched_brains, config, pool):
    if config.verbose:
        print('\n--- Running pooled analysis ---')

    for parameter_comb in matched_brains:
        for brain in brain_list:
            if parameter_comb.brains[brain.participant_name] == brain.no_ext_brain:
                parameter_comb.overall_results.append(brain.roi_results)
                parameter_comb.raw_results.append(brain.roi_temp_store)

    # Save each raw and overall results for each parameter combination
    iterable = zip(matched_brains, itertools.repeat("compile_results"), itertools.repeat(config))

    if config.multicore_processing:
        matched_brains = pool.starmap(Utils.instance_method_handler, iterable)
    else:
        matched_brains = list(itertools.starmap(Utils.instance_method_handler, iterable))

    # Compile the overall results for every parameter combination
    construct_combined_results(MatchedBrain.save_location)

    return matched_brains


def calculate_cost_function_and_displacement_values(participant_list, brain_list, config, pool):
    if config.verbose:
        print('\n--- Calculating cost function and displacement values ---')

    vals = []

    if config.anat_align:
        for counter, participant in enumerate(participant_list):
            if config.verbose:
                print(f'Calculating cost function value for anatomical file: {participant.anat_brain_no_ext}')

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
        print(f'\nSaving dataframe as additional_info.json')

    df = pd.concat(vals).reset_index(drop=True).replace([np.nan], [None])

    with open(f"{Environment_Setup.save_location}additional_info.txt", 'w') as file:
        df.to_markdown(buf=file, floatfmt=".2f")


def atlas_scale(matched_brains, config, pool):
    """Save a copy of each statistic for each ROI from the first brain. Then using sequential comparison
       find the largest statistic values for each ROI out of all the brains analyzed."""
    if config.verbose:
        print('\n--- Atlas scaling ---')

    # Make directory to store scaled brains
    Utils.check_and_make_dir(f"{os.getcwd()}/{MatchedBrain.save_location}NIFTI_ROI")

    first_combination = next(iter(matched_brains))
    for statistic_number in range(len(first_combination.overall_results)):
        roi_stats = deepcopy(first_combination.overall_results[statistic_number, :])

        for parameter_combination in matched_brains:
            for counter, roi_stat in enumerate(parameter_combination.overall_results[statistic_number, :]):
                if roi_stat > roi_stats[counter]:
                    roi_stats[counter] = roi_stat

        # Set arguments to pass to atlas_scale function
        iterable = zip(matched_brains, itertools.repeat("atlas_scale"), itertools.repeat(roi_stats),
                       range(len(matched_brains)), itertools.repeat(len(matched_brains)),
                       itertools.repeat(statistic_number), itertools.repeat(Environment_Setup.atlas_path),
                       itertools.repeat(config))

        # Run atlas_scale function and pass in max roi stats for between brain scaling
        if config.multicore_processing:
            pool.starmap(Utils.instance_method_handler, iterable)

        else:
            list(itertools.starmap(Utils.instance_method_handler, iterable))


def config_check(config):
    if config.grey_matter_segment and not config.anat_align:
        raise ImportError(f'grey_matter_segment is True but anat_align is set to False. '
                          f'grey_matter_segment requires anat_align to be true to function.')


if __name__ == '__main__':
    fRAT()
