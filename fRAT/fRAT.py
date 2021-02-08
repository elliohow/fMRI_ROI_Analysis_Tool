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

    orig_path = Path(os.path.abspath(__file__)).parents[0]
    config = Utils.load_config(orig_path, 'config.toml')  # Reload config file incase GUI has changed it
    config_check(config)

    args = Utils.argparser()

    combined_results_created = False

    # Check arguments passed over command line
    if args.make_table:
        ParamParser.make_table()
        sys.exit()

    if args.brain_loc is not None:
        config.brain_file_loc = args.brain_loc

    if args.output_loc is not None:
        config.output_folder_loc = args.output_loc

    if config.verbose and config.run_steps == 'all':
        print(f"\n--- Running all steps ---")

    # Run the analysis
    if config.run_steps in ("analyse", "all"):
        if config.verbose:
            print('\n----------------\n--- Analysis ---\n----------------')

        # Run class setup
        brain_list = Analysis.setup_analysis(config)

        if config.anat_align:
            Analysis.anat_setup()

            for brain in brain_list:
                brain.save_class_variables()

        if config.grey_matter_segment == 'freesurfer':
            Analysis.freesurfer_to_anat()

        # Set arguments to pass to run_analysis function
        iterable = zip(brain_list, itertools.repeat("run_analysis"), range(len(brain_list)),
                       itertools.repeat(len(brain_list)), itertools.repeat(config))

        if config.multicore_processing:
            pool = Utils.start_processing_pool()

            # Run analysis
            brain_list = pool.starmap(Utils.instance_method_handler, iterable)
        else:
            # Run analysis
            brain_list = list(itertools.starmap(Utils.instance_method_handler, iterable))

        if config.anat_align:
            Analysis.file_cleanup(Analysis.file_list, Analysis._save_location)
            Analysis.file_list = []

        # Atlas scaling
        '''Save a copy of the stats (default mean) for each ROI from the first brain. Then using sequential comparison
        to find the largest ROI stat out of all the brains analyzed.'''
        roi_stats = deepcopy(brain_list[0].roiResults[config.roi_stat_number, :])
        for brain in brain_list:
            for counter, roi_stat in enumerate(brain.roiResults[config.roi_stat_number, :]):
                if roi_stat > roi_stats[counter]:
                    roi_stats[counter] = roi_stat

        # Set arguments to pass to atlas_scale function
        iterable = zip(brain_list, itertools.repeat("atlas_scale"), itertools.repeat(roi_stats),
                       range(len(brain_list)), itertools.repeat(len(brain_list)), itertools.repeat(config))

        # Make directory to store scaled brains
        Utils.check_and_make_dir(f"{os.getcwd()}/{Analysis._save_location}NIFTI_ROI")

        # Run atlas_scale function and pass in max roi stats for between brain scaling
        if config.multicore_processing:
            pool.starmap(Utils.instance_method_handler, iterable)

            pool.close()
            pool.join()
        else:
            list(itertools.starmap(Utils.instance_method_handler, iterable))

        # Move csv file containing parameter info
        try:
            Utils.move_file("paramValues.csv", os.getcwd(), os.getcwd() + f"/{Analysis._save_location}", copy=True)

        except FileNotFoundError:
            if config.verify_param_method == 'table' and config.run_steps == 'all':
                raise

        else:
            if config.verify_param_method == 'table':
                # Create combined_results.json
                curr_dir = os.getcwd()
                ParamParser.run_parse(config)
                os.chdir(curr_dir)
                combined_results_created = True

    # Plot the results
    if config.run_steps in ("plot", "all"):
        if config.verbose:
            print('\n----------------\n--- Plotting ---\n----------------')

        if not combined_results_created:
            # Parameter Parsing
            ParamParser.run_parse(config)

        # Plotting
        Figures.Make_figures(config)

        html_report.main(str(orig_path))

    os.chdir(orig_path)  # Reset path

    if config.verbose:
        print(f"--- Completed in {round((time.time() - start_time), 2)} seconds ---\n\n")


def config_check(config):
    if config.grey_matter_segment is not None and not config.anat_align:
        raise ImportError(f'grey_matter_segment is not None but anat_align is set to False. '
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
