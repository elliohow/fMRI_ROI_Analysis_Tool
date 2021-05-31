import itertools
import sys
import time
import os
from copy import deepcopy
from pathlib import Path

from utils import *


def fRAT():
    start_time = time.time()
    checkversion()

    config, orig_path = load_config()
    argparser(config)

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
        print(f"--- Completed in {round((time.time() - start_time), 2)} seconds ---\n\n")


def load_config():
    orig_path = Path(os.path.abspath(__file__)).parents[0]
    config = Utils.load_config(orig_path, 'config.toml')  # Reload config file incase GUI has changed it
    config_check(config)
    return config, orig_path


def argparser(config):
    # Check arguments passed over command line
    args = Utils.argparser()

    if args.make_table:
        ParamParser.make_table()
        sys.exit()
    if args.brain_loc is not None:
        config.brain_file_loc = args.brain_loc
    if args.output_loc is not None:
        config.output_folder_loc = args.output_loc


def plotting(config, orig_path):
    if config.verbose:
        print('\n----------------\n--- Plotting ---\n----------------')

    # Parameter Parsing
    ParamParser.run_parse(config)

    # Plotting
    Figures.figures(config)

    # Create html report
    html_report.main(str(orig_path))
    if config.verbose:
        print('\nCreated html report')


def analysis(config):
    if config.verbose:
        print('\n----------------\n--- Analysis ---\n----------------')

    # Run class setup
    brain_list = Analysis.setup_analysis(config)

    if config.verbose:
        print('\n--- Running analysis ---')

    if config.anat_align:
        Analysis.anat_setup()

        for brain in brain_list:
            brain.save_class_variables()

    if config.multicore_processing:
        pool = Utils.start_processing_pool()
    else:
        pool = None

    brain_list = run_analysis(brain_list, config, pool)
    atlas_scale(brain_list, config, pool)

    if config.verify_param_method == 'table':
        Utils.move_file("paramValues.csv", os.getcwd(), os.getcwd() + f"/{Analysis.save_location}", copy=True)

    return


def run_analysis(brain_list, config, pool):
    # Set arguments to pass to run_analysis function
    iterable = zip(brain_list, itertools.repeat("run_analysis"), range(len(brain_list)),
                   itertools.repeat(len(brain_list)), itertools.repeat(config))

    if config.multicore_processing:
        brain_list = pool.starmap(Utils.instance_method_handler, iterable)
    else:
        brain_list = list(itertools.starmap(Utils.instance_method_handler, iterable))

    if config.anat_align:
        Analysis.file_cleanup(Analysis)

    return brain_list


def atlas_scale(brain_list, config, pool):
    """Save a copy of each statistic for each ROI from the first brain. Then using sequential comparison
       find the largest statistic values for each ROI out of all the brains analyzed."""
    if config.verbose:
        print('\n--- Atlas scaling ---')

    # Make directory to store scaled brains
    Utils.check_and_make_dir(f"{os.getcwd()}/{Analysis.save_location}NIFTI_ROI")

    for statistic_number in range(len(brain_list[0].roiResults)):
        roi_stats = deepcopy(brain_list[0].roiResults[statistic_number, :])

        for brain in brain_list:
            for counter, roi_stat in enumerate(brain.roiResults[statistic_number, :]):
                if roi_stat > roi_stats[counter]:
                    roi_stats[counter] = roi_stat

        # Set arguments to pass to atlas_scale function
        iterable = zip(brain_list, itertools.repeat("atlas_scale"), itertools.repeat(roi_stats), range(len(brain_list)),
                       itertools.repeat(len(brain_list)), itertools.repeat(statistic_number), itertools.repeat(config))

        # Run atlas_scale function and pass in max roi stats for between brain scaling
        if config.multicore_processing:
            pool.starmap(Utils.instance_method_handler, iterable)

        else:
            list(itertools.starmap(Utils.instance_method_handler, iterable))

    if config.multicore_processing:
        pool.close()
        pool.join()


def config_check(config):
    if config.grey_matter_segment and not config.anat_align:
        raise ImportError(f'grey_matter_segment is True but anat_align is set to False. '
                          f'grey_matter_segment requires anat_align to be true to function.')


def checkversion():
    # Check Python version:
    expect_major = 3
    expect_minor = 7
    expect_rev = 5

    print(f"\nfRAT is developed and tested with Python {str(expect_major)}.{str(expect_minor)}.{str(expect_rev)}")
    if sys.version_info[:3] < (expect_major, expect_minor, expect_rev):
        current_version = f"{str(sys.version_info[0])}.{str(sys.version_info[1])}.{str(sys.version_info[2])}"
        print(f"INFO: Python version {current_version} is untested. Consider upgrading to version "
              f"{str(expect_major)}.{str(expect_minor)}.{str(expect_rev)} if there are errors running the fRAT.")


if __name__ == '__main__':
    fRAT()
