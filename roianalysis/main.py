import itertools
import time
from copy import deepcopy

import config
from roianalysis.paramparser import ParamParser
from roianalysis.utils import Utils
from roianalysis.analysis import Analysis
from roianalysis.figures import Figures

if __name__ == '__main__':
    start_time = time.time()

    args = Utils.argparser()

    # Check arguments passed over command line
    if args.print_info:
        Analysis.print_info()

    if config.make_table_only or args.make_table:
        ParamParser.make_table()

    if args.brain_loc is not None:
        config.brain_file_loc = args.brain_loc

    if args.json_loc is not None:
        config.json_file_loc = args.json_loc

    # Run the analysis
    if config.run_steps in ("analyse", "all"):
        # Run class setup
        brain_list = Analysis.setup_analysis()

        if config.verbose:
            print('\n--- Analysis ---')

        if config.use_freesurf_file:
            csf_or_wm_voxels = Analysis.freesurfer_space_to_native_space()
        else:
            csf_or_wm_voxels = None

        # Set arguments to pass to run_analysis function
        iterable = zip(brain_list, itertools.repeat("run_analysis"), range(len(brain_list)),
                       itertools.repeat(len(brain_list)), itertools.repeat(csf_or_wm_voxels))

        if config.multicore_processing:
            pool = Utils.start_processing_pool()

            # Run analysis
            brain_list = pool.starmap(Utils.instance_method_handler, iterable)
        else:
            # Run analysis
            brain_list = list(itertools.starmap(Utils.instance_method_handler, iterable))

        # TODO: Fix bootstrapping crashing computer. I think bootstrapping starts at 0, fix this too.

        # TODO Make a copy of paramValues and search for that too

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
        ParamParser.run_parse()

        # Plotting
        Figures.construct_plots()

    if config.verbose:
        print(f"--- Completed in {round((time.time() - start_time), 2)} seconds ---")