import itertools
import sys
import time
import os
from copy import deepcopy
from pathlib import Path

from utils import *


def fRAT():
    start_time = time.time()

    orig_path = Path(os.path.abspath(__file__)).parents[0]
    config = Utils.load_config(orig_path, 'config.toml')  # Reload config file incase GUI has changed it

    args = Utils.argparser()

    # Check arguments passed over command line
    if args.print_info:
        Analysis.print_info()

    if args.make_table:
        ParamParser.make_table()
        sys.exit()

    if args.brain_loc is not None:
        config.brain_file_loc = args.brain_loc

    if args.json_loc is not None:
        config.json_file_loc = args.json_loc

    # Run the analysis
    if config.run_steps in ("analyse", "all"):
        # Run class setup
        brain_list = Analysis.setup_analysis(config)

        if config.verbose:
            print('\n--- Analysis ---')

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
                       range(len(brain_list)), itertools.repeat(len(brain_list)))

        # Run atlas_scale function and pass in max roi stats for between brain scaling
        if config.multicore_processing:
            pool.starmap(Utils.instance_method_handler, iterable)

            pool.close()
            pool.join()
        else:
            list(itertools.starmap(Utils.instance_method_handler, iterable))

    # Plot the results
    if config.run_steps in ("plot", "all"):
        # Parameter Parsing
        ParamParser.run_parse(config)

        # Plotting
        Figures.Make_figures(config)

    os.chdir(orig_path)  # Reset path

    if config.verbose:
        print(f"--- Completed in {round((time.time() - start_time), 2)} seconds ---\n\n")


if __name__ == '__main__':
    fRAT()
