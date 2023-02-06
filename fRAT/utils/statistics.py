import copy
import itertools
import os
import random
import re
import shutil
import warnings
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.linalg
import pandas as pd
import plotnine as pltn
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tools.sm_exceptions
from nilearn import plotting
from scipy.stats import zscore
from statsmodels.stats.weightstats import ttest_ind
from scipy.stats import ttest_rel
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from .analysis import Environment_Setup, MatchedBrain
from .figures import Histogram, BrainGrid
from .utils import Utils

config = None
STATISTICS_PATH = ''
STATISTICS_LOGFILE = None


class Coefficient_map(MatchedBrain, Environment_Setup):
    @classmethod
    def label_list_setup(cls):
        cls.atlas_path = f'{os.environ["FSLDIR"]}/data/atlases/' \
                         f'{cls.atlas_label_list[int(config.atlas_number)][0]}'
        cls.atlas_label_path = f'{os.environ["FSLDIR"]}/data/atlases/' \
                               f'{cls.atlas_label_list[int(config.atlas_number)][1]}'
        cls.roi_label_list()

    @classmethod
    def to_braingrid(cls, df, subfolder, standardise_cbar, subsubfolder='/'):
        cls.label_list_setup()

        Utils.check_and_make_dir(f"{STATISTICS_PATH}/{subfolder}/")
        Utils.check_and_make_dir(f"{STATISTICS_PATH}/{subfolder}/{subsubfolder}/")
        Utils.check_and_make_dir(f"{STATISTICS_PATH}/{subfolder}/{subsubfolder}/NIFTI_ROI")
        Utils.check_and_make_dir(f"{STATISTICS_PATH}/{subfolder}/{subsubfolder}/Images")

        subfolder = f"{subfolder}/{subsubfolder}"

        cls.create_images(df, subfolder, standardise_cbar)

    @classmethod
    def save_brain_imgs(cls, file_name, subfolder, vmax):
        # Save brain image using nilearn
        plot = plotting.plot_stat_map(f"{STATISTICS_PATH}/{subfolder}/NIFTI_ROI/{file_name}.nii.gz",
                                      draw_cross=False, annotate=False, colorbar=True,
                                      vmax=vmax, symmetric_cbar=True, cbar_tick_format="%.2f",
                                      display_mode='xz', cut_coords=(0, 18),
                                      cmap='seismic')
        plot.savefig(f"{STATISTICS_PATH}/{subfolder}/images/{file_name}.png")
        plot.close()

    @classmethod
    def create_images(cls, df, subfolder, standardise_cbar):
        if standardise_cbar:
            # Find max value seen in df so the same colour bar can be used
            vmax = max(abs(np.min(df.iloc[:, 0])), abs(np.max(df.iloc[:, 0])))
        else:
            vmax = None

        for variable in df.index.unique():
            curr_df = df.loc[variable]

            try:
                data = [curr_df.iloc[:, 0].values]
                columns = curr_df['ROI'].values
            except pd.core.indexing.IndexingError:
                data = [curr_df.values[0]]
                columns = [curr_df[1]]

            reorganised_df = pd.concat([pd.DataFrame(columns=cls.label_array),
                                        pd.DataFrame(data=data,
                                                     columns=columns)])

            if variable[0:2] == 'C(':
                # Reformat filename if variable is categorical as it would be in the wrong format, e.g. C(MB)[T.On]
                variable = re.sub('\(|\)|\[|\]', '', variable)
                variable = re.sub('(T\.)', '_', variable)[1:]

            cls.scale_and_save_atlas_images(cls.atlas_path, None, reorganised_df.to_numpy(),
                                            0, f"{STATISTICS_PATH}/{subfolder}/NIFTI_ROI", variable)
            cls.save_brain_imgs(variable, subfolder, vmax)


def main(cfg, config_path, config_filename):
    global config

    config = cfg

    Coefficient_map.label_list_setup()

    stats_setup(config_path, config_filename)
    param_queries, critical_params = find_baseline_parameters()

    # Only use selected parameters for statistics
    critical_params = [param for param in critical_params if
                       param.lower() in [x.lower() for x in config.include_as_variable]]

    statistics(param_queries, critical_params)

    STATISTICS_LOGFILE.close()


def stats_setup(config_path, config_filename):
    global STATISTICS_PATH, STATISTICS_LOGFILE

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    STATISTICS_PATH = f'{os.getcwd()}/Statistics/{config.statistics_subfolder_name}/'
    Utils.check_and_make_dir(f"{os.getcwd()}/Statistics")
    Utils.check_and_make_dir(STATISTICS_PATH)

    # Save copy of statistics options used
    shutil.copyfile(f"{os.getcwd()}/../statisticsOptions.csv", f"{STATISTICS_PATH}/copy_statisticsOptions.csv")

    # Save log of only the statistics settings from the config file
    Utils.save_config(STATISTICS_PATH, config_path, config_filename,
                      relevant_sections=['Statistics'], new_config_name='statistics_log',
                      additional_info=[
                          f'data_used_for_statistics = "{config.averaging_type.replace(" ", "_").lower()}"'])

    STATISTICS_LOGFILE = open(f'{os.getcwd()}/Statistics/{config.statistics_subfolder_name}/'
                              f'statistics_terminal_output.txt', 'w')

    Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                         f'\nCalculating statistics using {config.averaging_type} data.')


def clean_results(individual_roi_results, excluded_parameters, raw_data,
                  averaged_roi_results, combined_roi_results, voxel_cutoff):
    """For each ROI, remove each session that has a voxel count below the minimum voxel cutoff and make a log of which
    ROIs this applies to."""
    rois_below_max_r2_thresh = []

    Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                         f"\nNOTE: Within-subjects t-tests are paired Student's t-tests, however between-subjects "
                         f"t-tests are Welch's t-tests with degrees of freedom calculated using Satterthwaite Approximation."
                         f'\nMinimum voxel count set to {voxel_cutoff}.\n\n'
                         f'Sessions removed:')

    # Remove No ROI and overall data to be used with percentage change statistics
    combined_roi_results = combined_roi_results[(combined_roi_results['index'] != 'No ROI')
                                                & (combined_roi_results['index'] != 'Overall')]
    combined_roi_results = combined_roi_results[combined_roi_results['Average voxels per session'] > voxel_cutoff]

    for key in excluded_parameters:
        combined_roi_results = combined_roi_results.loc[(~combined_roi_results[key].isin(excluded_parameters[key]))]

    for roi in individual_roi_results:
        for key in excluded_parameters:
            individual_roi_results[roi] = individual_roi_results[roi].loc[
                (~individual_roi_results[roi][key].isin(excluded_parameters[key]))]

            raw_data[roi] = raw_data[roi].loc[(~raw_data[roi][key].isin(excluded_parameters[key]))]

            averaged_roi_results[roi] = averaged_roi_results[roi].loc[
                (~averaged_roi_results[roi][key].isin(excluded_parameters[key]))]

        session_count = len(individual_roi_results[roi])
        below_thresh_number = np.count_nonzero(individual_roi_results[roi]['voxel_amount'] < voxel_cutoff)

        individual_roi_results[roi] = individual_roi_results[roi][individual_roi_results[roi]['voxel_amount'] > voxel_cutoff]

        averaged_roi_results[roi] = averaged_roi_results[roi][averaged_roi_results[roi]['voxel_amount'] > voxel_cutoff]

        if below_thresh_number:
            percent_removed = (below_thresh_number / session_count) * 100

            if percent_removed > config.max_below_thresh:
                rois_below_max_r2_thresh.append(roi)

            Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                                 f'{roi}: {below_thresh_number}/{session_count} ({"{:.2f}".format(percent_removed)}%)')

    return rois_below_max_r2_thresh, individual_roi_results, averaged_roi_results, raw_data, combined_roi_results


def extract_statistics_options_info():
    statistics_options_table = pd.read_csv(f"{os.getcwd()}/../statisticsOptions.csv", header=None)
    statistics_options_table = np.split(statistics_options_table,
                                        statistics_options_table[statistics_options_table.isnull().all(1)].index)

    # Drop any blank rows
    statistics_options_table = [df.dropna(how='all') for df in statistics_options_table if not df.dropna(how='all').empty]

    main_effect_parameters = {}
    simple_effect_parameters = {}
    excluded_parameters = {}

    for counter, parameter in enumerate(statistics_options_table):
        if counter % 2 == 0:
            main_effect_temp_list = parameter[0].loc[parameter[1].isin(['y', 'Y', 'yes', 'Yes'])]
            main_effect_parameters[parameter.values[0][0]] = list(main_effect_temp_list.values)

        else:
            excluded_values_temp_list = parameter[0].loc[parameter[2].isin(['y', 'Y', 'yes', 'Yes'])]
            excluded_parameters[parameter.values[0][0]] = list(excluded_values_temp_list.values)

            simple_effects_temp_list = parameter[0].loc[parameter[1].isin(['y', 'Y', 'yes', 'Yes'])]
            simple_effect_parameters[parameter.values[0][0]] = list(simple_effects_temp_list.values)

    return main_effect_parameters, simple_effect_parameters, excluded_parameters


def find_baseline_parameters():
    baseline_param_query = ''
    non_baseline_params_query = ''

    table, _ = Utils.load_paramValues_file()
    columns = table.columns
    _, critical_column_locs, baseline_column_loc = Utils.find_column_locs(table)

    baseline_params = []  # Make a list in case user has selected multiple rows
    for index, row in table.iterrows():
        if str(row[baseline_column_loc]).strip().lower() in ('yes', 'y', 'true'):
            baseline_params.append(dict(row[critical_column_locs]))

    # Check number of unique param combinations given, need to convert from dict to str as dict is not hashable
    if len(set([str(x) for x in baseline_params])) > 1:
        raise ValueError('Multiple parameter combinations have been selected as the baseline to use for statistics.\n'
                         'Only one row should be selected, the other brain volumes with the same parameter combination '
                         'will also be used for the baseline.')

    elif len(set([str(x) for x in baseline_params])) == 0:
        Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                             f'\nNo baseline set in parameter values file. Skipping calculation of percent change '
                             f'versus baseline.\n')

    else:
        for param, value in baseline_params[0].items():
            if param in config.categorical_variables:
                baseline_params[0][param] = f'"{value}"'

        baseline_param_query = " & ".join(" == ".join((str(k), str(v))) for k, v in baseline_params[0].items())
        non_baseline_params_query = " | ".join(" != ".join((str(k), str(v))) for k, v in baseline_params[0].items())

    table.columns = columns
    return [baseline_param_query, non_baseline_params_query], table.columns[list(critical_column_locs)]


def structure_data(config, participant_paths):
    combined_results = Utils.read_combined_results(os.getcwd(), config.averaging_type)

    all_subject_result_jsons = []  # Used for lmm
    averaged_subject_result_jsons = []  # Used for t-tests

    for path in participant_paths:
        all_subject_result_jsons.extend(
            [f"{path}/Summarised_results/{jsn}" for jsn in Utils.find_files(f"{path}/Summarised_results/", "json") if
             'combined_results' not in jsn])

        averaged_subject_result_jsons.extend(
            [f"{path}/Summarised_results/Averaged_results/{jsn}" for jsn in
             Utils.find_files(f"{path}/Summarised_results/Averaged_results/", "json")
             if 'combined_results' not in jsn])

    individual_roi_results = Histogram.load_and_restructure_jsons(config, all_subject_result_jsons, combined_results[0],
                                                                  data_type='statistics')

    averaged_roi_results = Histogram.load_and_restructure_jsons(config, averaged_subject_result_jsons,
                                                                combined_results[0],
                                                                data_type='statistics')

    return individual_roi_results, averaged_roi_results, combined_results


def split_dict_by_roi(result_dict):
    """Split combined dataframe into a dict entry for each ROI."""
    ROI_dict = {}
    grouped = result_dict.groupby(result_dict.ROI)
    for ROI in result_dict.ROI.unique():
        ROI_dict[f'{ROI}'] = grouped.get_group(ROI).dropna()
        ROI_dict[f'{ROI}'].columns = [x.replace(' ', '_') for x in ROI_dict[f'{ROI}'].columns]

    return ROI_dict


def statistics(param_queries, critical_params):
    individual_roi_results, averaged_roi_results, raw_data, combined_roi_results, combined_roi_results_path = load_data()
    main_effect_parameters, simple_effect_parameters, excluded_parameters = extract_statistics_options_info()

    # Get list of rois before cleaning up the dataframe
    list_rois = list(combined_roi_results['index'].unique())

    rois_below_max_r2_thresh, individual_roi_results, averaged_roi_results, raw_data, combined_roi_results \
        = clean_results(individual_roi_results, excluded_parameters, raw_data,
                        averaged_roi_results, combined_roi_results,
                        voxel_cutoff=config.minimum_voxels)

    if '' not in param_queries:  # If no baseline parameter combination set, skip this
        calculate_percent_change_versus_baseline(param_queries, combined_roi_results, combined_roi_results_path)

    if config.run_t_tests or config.run_linear_mixed_models:
        if not [parameter for parameter in [*main_effect_parameters.values(), *simple_effect_parameters.values()] if parameter] \
                and config.run_t_tests:
            warnings.warn('No main or simple effects chosen in statisticsOptions.csv.'
                          '\nUpdate this file to run t-tests.')

        chosen_rois = Utils.find_chosen_rois(list_rois, func_name="Statistics",
                                             config_region_var=config.regional_stats_rois)

        roi_statistics(critical_params, rois_below_max_r2_thresh,
                       individual_roi_results, averaged_roi_results,
                       chosen_rois, raw_data,
                       main_effect_parameters, simple_effect_parameters)


def load_data():
    participants, _ = Utils.find_participant_dirs(os.getcwd())

    individual_roi_results, averaged_roi_results, combined_results = structure_data(config, participants)

    individual_roi_results = split_dict_by_roi(individual_roi_results)
    averaged_roi_results = split_dict_by_roi(averaged_roi_results)

    raw_data = copy.deepcopy(individual_roi_results)

    return individual_roi_results, averaged_roi_results, raw_data, combined_results[0], combined_results[1]


def roi_statistics(critical_params, rois_below_max_r2_thresh,
                   individual_roi_results_dict, averaged_roi_results,
                   chosen_rois, raw_data, main_effect_parameters, simple_effect_parameters):
    glm_formula_types = []

    if config.main_effects:
        glm_formula_types.append('main_effects')
    if config.main_and_interaction_effects:
        glm_formula_types.append('main_and_interaction_effects')
    if config.interaction_effects:
        glm_formula_types.append('interaction_effects')

    for glm_formula_type in glm_formula_types:
        standardised_coeffs_df = pd.DataFrame()
        unstandardised_coeffs_df = pd.DataFrame()

        Utils.check_and_make_dir(f"{STATISTICS_PATH}/{glm_formula_type}")
        formula = construct_glm_formula(critical_params, glm_formula_type)

        Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                             f'\n==============================================================================\n'
                             f'\nUsing formula: "{formula}" for "{glm_formula_type.replace("_", " ")}" calculations.\n'
                             f'\n==============================================================================\n')

        adj_marginal_r_square_data = []
        adj_conditional_r_square_data = []
        voxel_data = []

        for ROI in chosen_rois:
            Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                                 f'\n==============================================================================\n'
                                 f'\nCalculating {ROI} statistics ({glm_formula_type.replace("_", " ")}).\n')

            if not individual_roi_results_dict[ROI].empty:
                if config.run_linear_mixed_models:
                    individual_roi_results_dict[ROI].columns = individual_roi_results_dict[ROI].columns.str.lower()

                    adj_marginal_r_square, \
                    adj_conditional_r_square, \
                    converged, \
                    standardised_coeffs, \
                    unstandardised_coeffs, \
                    pvalues = run_glm(individual_roi_results_dict[ROI], ROI=ROI, formula=formula,
                                      glm_formula_type=glm_formula_type)

                    if adj_marginal_r_square is not None \
                            and ROI not in ['Overall', 'No ROI'] \
                            and ROI not in rois_below_max_r2_thresh \
                            and converged:
                        adj_marginal_r_square_data.append(adj_marginal_r_square)
                        adj_conditional_r_square_data.append(adj_conditional_r_square)
                        voxel_data.append(individual_roi_results_dict[ROI].mean()['voxel_amount'])

                        # Save list of coefficients to make brain images
                        unstandardised_coeffs = unstandardised_coeffs.loc[pvalues <= config.brain_map_p_thresh]
                        standardised_coeffs = standardised_coeffs.loc[pvalues <= config.brain_map_p_thresh]

                        if not standardised_coeffs.empty:
                            standardised_coeffs['ROI'] = ROI
                            standardised_coeffs_df = pd.concat(
                                [standardised_coeffs_df, standardised_coeffs[['standardised coef', 'ROI']]])

                        if not unstandardised_coeffs.empty:
                            unstandardised_coeffs = pd.DataFrame(unstandardised_coeffs, columns=['unstandardised coef'])
                            unstandardised_coeffs['ROI'] = ROI
                            unstandardised_coeffs_df = pd.concat([unstandardised_coeffs_df, unstandardised_coeffs])

                if config.run_t_tests:
                    averaged_roi_results[ROI].columns = averaged_roi_results[ROI].columns.str.lower()
                    raw_data[ROI].columns = raw_data[ROI].columns.str.lower()

                    run_t_tests(averaged_roi_results[ROI], critical_params, raw_data[ROI],
                                main_effect_parameters, simple_effect_parameters, ROI)

            else:
                Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                                     'All sessions below minimum voxel count.\n'
                                     'Skipping statistics for this ROI.\n')

        if voxel_data:
            r_square_dict = {'adj_marginal_rsquare': adj_marginal_r_square_data,
                             'adj_conditional_rsquare': adj_conditional_r_square_data}

            for rsquare_type in r_square_dict:
                compute_rsquare_regression(voxel_data, r_square_dict[rsquare_type], rsquare_type, glm_formula_type)

            if config.verbose:
                print("\n=============================================================================="
                      "\nCreating standardised and unstandardised coefficient brain maps"
                      "\n==============================================================================\n")
            Coefficient_map.to_braingrid(standardised_coeffs_df, "standardised_coeffs",
                                         standardise_cbar=True, subsubfolder=glm_formula_type)
            Coefficient_map.to_braingrid(unstandardised_coeffs_df, "unstandardised_coeffs",
                                         standardise_cbar=False, subsubfolder=glm_formula_type)


def compute_rsquare_regression(voxel_data, r_square_data, r_square_type, glm_formula_type):
    # GLM
    voxel_data = sm.add_constant(voxel_data)
    voxel_df = pd.DataFrame(columns=['Intercept', 'Voxels'], data=voxel_data)
    r_square_df = pd.DataFrame(columns=['R_Square'], data=r_square_data)

    model = sm.OLS(r_square_df, voxel_df)
    result = model.fit()

    formula = construct_glm_formula(critical_params=['Voxels'], glm_formula_type=glm_formula_type)
    standardised_coeffs = calculate_glm_standardised_coeffs(pd.concat([voxel_df[['Voxels']], r_square_df], axis=1),
                                                            'R2', formula, glm_formula_type, variation=r_square_type)

    Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                         f'\n==============================================================================\n'
                         f'====================== {r_square_type.replace("_", " ")} results =======================\n'
                         f'Calculating r2 vs mean voxel amount statistics for {glm_formula_type.replace("_", " ")}.\n'
                         f'Note: ROIs with more than {config.max_below_thresh}% of sessions excluded (due to them\n'
                         f'being under the minimum voxel count), or ROIs '
                         f'where the model failed to converge, are excluded from this analysis.\n\n'
                         f'{result.summary()}\n\n\n'
                         f'Standardised coefficients:\n'
                         f'{standardised_coeffs}')

    with open(f"{STATISTICS_PATH}/{glm_formula_type}/Overall/{r_square_type}_vs_voxels_GLM.csv", "w") as f:
        f.write(result.summary().as_csv())

    # Scatter plot
    df = pd.concat([voxel_df, r_square_df], axis=1)
    figure = (
            pltn.ggplot(df, pltn.aes(x="Voxels", y="R_Square"))
            + pltn.theme_538()
            + pltn.geom_point()
            + pltn.geom_smooth(method='lm')
            + pltn.labs(x='Mean voxel count', y='$r^2$')
            + pltn.theme(
        panel_grid_minor_x=pltn.themes.element_line(alpha=0),
        panel_grid_major_x=pltn.themes.element_line(alpha=1),
        panel_grid_major_y=pltn.element_line(alpha=0),
        plot_background=pltn.element_rect(fill="white"),
        panel_background=pltn.element_rect(fill="gray", alpha=0.1),
        axis_title_x=pltn.element_text(weight='bold', color='black', size=20),
        axis_title_y=pltn.element_text(weight='bold', color='black', size=20),
        strip_text_x=pltn.element_text(weight='bold', size=10, color='black'),
        strip_text_y=pltn.element_text(weight='bold', size=10, color='black'),
        axis_text_x=pltn.element_text(size=10, color='black'),
        axis_text_y=pltn.element_text(size=10, color='black'),
        dpi=config.plot_dpi
    )
    )

    figure.save(f"{STATISTICS_PATH}/{glm_formula_type}/Overall/{r_square_type}_vs_voxels.png",
                height=config.plot_scale,
                width=config.plot_scale * 2,
                verbose=False, limitsize=False)

    figure.save(f"{STATISTICS_PATH}/{glm_formula_type}/Overall/{r_square_type}_vs_voxels.svg",
                height=config.plot_scale,
                width=config.plot_scale * 2,
                verbose=False, limitsize=False)


def calculate_percent_change_versus_baseline(param_queries, combined_results, combined_results_path):
    if config.verbose:
        print(f'\nCalculating percentage change statistics.\n')

    combined_results['Percentage change from baseline'] = np.nan
    combined_results['Baseline'] = ""
    combined_results = combined_results.sort_values('index')

    columns = combined_results.columns  # Save old column names before converting to lower case

    # Need to convert to lowercase as the df query is easier to make when lowercase
    combined_results.columns = combined_results.columns.str.lower()

    # Add percentage change statistics to original combined_results df
    combined_results = calculate_percentage_change_per_roi(combined_results, param_queries)

    bootstrap_df = bootstrap_overall_percentage_change(combined_results)
    save_dfs(bootstrap_df, columns, combined_results, combined_results_path)


def run_t_tests(dataset, critical_params, raw_data, main_effect_parameters, simple_effect_parameters, ROI):
    Utils.check_and_make_dir(f"{STATISTICS_PATH}/Overall")

    number_of_simple_effect_params = len(
        [parameter for parameter in simple_effect_parameters if simple_effect_parameters[parameter]])
    number_of_main_effect_params = len(
        [parameter for parameter in main_effect_parameters if main_effect_parameters[parameter]])

    if number_of_main_effect_params > 0:
        calculate_main_effects(ROI, critical_params, dataset, main_effect_parameters, raw_data)

    if number_of_simple_effect_params > 1:
        calculate_simple_effects(ROI, critical_params, dataset, simple_effect_parameters)

    elif number_of_simple_effect_params == 1:
        Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                             f"\nOnly one variable setup for simple effect t-tests. Skipping calculation.\n")


def calculate_main_effects(ROI, critical_params, dataset, main_effect_parameters, raw_data):
    t_test_results = []
    for counter, param in enumerate(critical_params):
        t_test_type = config.IV_type[counter]

        for main_effect_contrast in main_effect_parameters[param]:
            data1 = dataset.loc[dataset[param.lower()].astype(str) == main_effect_contrast.split(' v ')[0]].copy()
            data2 = dataset.loc[dataset[param.lower()].astype(str) == main_effect_contrast.split(' v ')[1]].copy()

            data1, data2, current_result = balance_main_effect_data(critical_params, [data1, data2], param, raw_data,
                                                                    t_test_type)

            if not current_result:
                data1 = data1.groupby('subject').mean()['voxel_value'].sort_index()
                data2 = data2.groupby('subject').mean()['voxel_value'].sort_index()

            if not current_result and t_test_type == 'Within-subjects':
                current_result = list(ttest_rel(a=data1, b=data2))

                if np.isnan(current_result[0]):
                    current_result[0] = 'One participant not sufficient to run statistical test, analysis not ran'

                # Degrees of freedom calculation
                current_result.append(len(data1) - 1)

            elif not current_result and t_test_type == 'Between-subjects':
                current_result = list(ttest_ind(x1=data1, x2=data2, usevar='unequal'))

            # Insert label of parameter combination being tested
            current_result.insert(0, f"({t_test_type}) {param}: {main_effect_contrast}")

            if not np.isnan(current_result[-1]):
                current_result.append(data1.mean() - data2.mean())
                current_result.append(calculate_cohens_d(x1=data1, x2=data2))

            t_test_results.append(current_result)

    if t_test_results:
        t_test_df = pd.DataFrame(data=t_test_results, columns=['Variable contrast', 'tstat', 'pvalue', 'df',
                                                               'Mean difference (x1 - x2)', "Cohen's d"])
        Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                             f"Main effect t-test results\n"
                             f'{t_test_df}\n')

        t_test_df.to_csv(f"{STATISTICS_PATH}/Overall/{ROI}_main_effect_t_tests.csv", index=False)


def calculate_simple_effects(ROI, critical_params, dataset, simple_effect_parameters):
    t_test_results = []

    for counter, param in enumerate(simple_effect_parameters):
        if not simple_effect_parameters[param] or len(simple_effect_parameters[param]) == 1:
            continue

        t_test_type = config.IV_type[counter]

        simple_effect_combinations = list(itertools.combinations(simple_effect_parameters[param], 2))

        for simple_effect_contrast in simple_effect_combinations:
            other_params = {other_param.lower(): simple_effect_parameters[other_param] for other_param in
                            simple_effect_parameters if other_param != param}

            keys, values = zip(*other_params.items())
            for combination in itertools.product(*values):
                current_combination = dict(zip(keys, combination))

                contrast_1, contrast_2 = fix_parameter_trailing_zero_issue_if_float(
                    dataset,
                    param,
                    simple_effect_contrast[0],
                    simple_effect_contrast[1])

                data1 = dataset.loc[dataset[param.lower()].astype(str) == contrast_1].copy()
                data2 = dataset.loc[dataset[param.lower()].astype(str) == contrast_2].copy()

                # Control for each of the other parameters
                for other_param in other_params:
                    controlled_param, = fix_parameter_trailing_zero_issue_if_float(
                        dataset,
                        other_param,
                        current_combination[other_param])

                    data1 = data1.loc[data1[other_param.lower()].astype(str) == controlled_param].copy()
                    data2 = data2.loc[data2[other_param.lower()].astype(str) == controlled_param].copy()

                data1, data2, current_result = balance_simple_effect_data([data1, data2], t_test_type)

                if not current_result:
                    data1, data2 = data1['voxel_value'].sort_index(), data2['voxel_value'].sort_index()

                if not current_result and t_test_type == 'Within-subjects':
                    current_result = list(ttest_rel(a=data1, b=data2))

                    if np.isnan(current_result[0]):
                        current_result[0] = 'One participant not sufficient to run statistical test, analysis not ran'

                    # Degrees of freedom calculation
                    current_result.append(len(data1) - 1)

                elif not current_result and t_test_type == 'Between-subjects':
                    current_result = list(ttest_ind(x1=data1, x2=data2, usevar='unequal'))

                # Insert label of parameter combination being tested
                current_result.insert(0, f"({t_test_type}) {param}: {' v '.join(simple_effect_contrast)}")
                # Insert which variables have been controlled
                current_result.insert(0, str(current_combination))

                if not np.isnan(current_result[-1]):
                    current_result.append(data1.mean() - data2.mean())
                    current_result.append(calculate_cohens_d(x1=data1, x2=data2))

                t_test_results.append(current_result)

    if t_test_results:
        t_test_df = pd.DataFrame(data=t_test_results, columns=['Controlled variables', 'Variable contrast',
                                                               'tstat', 'pvalue', 'df',
                                                               'Mean difference (x1 - x2)', "Cohen's d"])

        Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                             f"Simple effect t-test results\n"
                             f'{t_test_df}\n')

        t_test_df.to_csv(f"{STATISTICS_PATH}/Overall/{ROI}_simple_effect_t_tests.csv", index=False)


def fix_parameter_trailing_zero_issue_if_float(dataset, parameter_name, *argv):
    if dataset[parameter_name.lower()].dtype == 'float64':
        parameters = []

        for arg in argv:
            parameters.append(str(float(arg)))

        return parameters
    else:
        return list(argv)


def balance_simple_effect_data(data_subsets, t_test_type):
    if t_test_type == 'Within-subjects':
        # If data is repeated measures data, remove any participants not in both data subsets
        participants_in_each_df = pd.merge(data_subsets[0]['subject'].drop_duplicates(),
                                           data_subsets[1]['subject'].drop_duplicates(),
                                           how='outer', indicator=True)
        participants_to_remove = participants_in_each_df.loc[(participants_in_each_df['_merge'] == 'left_only')
                                                             | (participants_in_each_df['_merge'] == 'right_only')][
            'subject']

        data_subsets[0] = data_subsets[0][~data_subsets[0]['subject'].isin(participants_to_remove)]
        data_subsets[1] = data_subsets[1][~data_subsets[1]['subject'].isin(participants_to_remove)]

    if data_subsets[0].empty or data_subsets[1].empty:
        current_result = ['No participants remaining after balancing data, analysis not ran',
                          np.NaN, np.NaN, np.NaN, np.NaN]

        return data_subsets[0], data_subsets[1], current_result

    else:
        current_result = []

        return data_subsets[0], data_subsets[1], current_result


def balance_main_effect_data(critical_params, data_subsets, param, raw_data, t_test_type):
    # Compare parameters not currently being compared to see if they are the same between both datasets
    # If not, don't run the analysis
    other_params = {other_param.lower(): config.IV_type[counter] for counter, other_param in enumerate(critical_params)
                    if other_param != param}
    within_subject_params = [other_param for other_param in other_params if
                             other_params[other_param] == 'Within-subjects']

    for other_param in other_params.keys():
        if set(data_subsets[0][other_param]) != set(data_subsets[1][other_param]):
            current_result = ['Parameters not equal, analysis not ran', np.NaN, np.NaN, np.NaN, np.NaN]

            return data_subsets[0], data_subsets[1], current_result

    if within_subject_params:
        for data in data_subsets:
            for subject in data['subject'].unique():
                # Check each subject to see if they contain all within-subject parameter combinations seen in full
                # dataset, if not then remove the participant
                if 'left_only' in pd.merge(raw_data[within_subject_params].drop_duplicates(),
                                           data.loc[data['subject'] == subject][within_subject_params],
                                           how='outer', indicator=True)['_merge'].values:
                    rows_to_remove = data.loc[(data['subject'] == subject)].index
                    data.drop(rows_to_remove, inplace=True)

    if t_test_type == 'Within-subjects':
        # If data is repeated measures data, remove any participants not in both data subsets
        participants_in_each_df = pd.merge(data_subsets[0]['subject'].drop_duplicates(),
                                           data_subsets[1]['subject'].drop_duplicates(),
                                           how='outer', indicator=True)
        participants_to_remove = participants_in_each_df.loc[(participants_in_each_df['_merge'] == 'left_only')
                                                             | (participants_in_each_df['_merge'] == 'right_only')]['subject']

        data_subsets[0] = data_subsets[0][~data_subsets[0]['subject'].isin(participants_to_remove)]
        data_subsets[1] = data_subsets[1][~data_subsets[1]['subject'].isin(participants_to_remove)]

    if data_subsets[0].empty or data_subsets[1].empty:
        current_result = ['No participants remaining after balancing data, analysis not ran',
                          np.NaN, np.NaN, np.NaN, np.NaN]

        return data_subsets[0], data_subsets[1], current_result

    else:
        current_result = []

        return data_subsets[0], data_subsets[1], current_result


def calculate_cohens_d(x1, x2):
    """Calculation of cohens d using pooled standard deviation. Used for both within and between subjects designs as
    with equal sample sizes for each group, the calculation is the same as for calculation average standard deviation."""
    # Calculate the size of samples
    n1, n2 = len(x1), len(x2)

    # calculate the variance of the samples
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)

    # calculate the pooled standard deviation
    dof = n1 + n2 - 2
    sigma = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / dof)

    # calculate the means of the samples
    u1, u2 = np.mean(x1), np.mean(x2)

    # return the effect size
    return (u1 - u2) / sigma


def calculate_r2_measurements(current_df, result, path_to_folder, ROI):
    var_resid = result.scale
    var_random_effect = float(result.cov_re.iloc[0])
    var_fixed_effect = result.predict(current_df).var()

    total_var = var_fixed_effect + var_random_effect + var_resid

    marginal_r2 = var_fixed_effect / total_var
    conditional_r2 = (var_fixed_effect + var_random_effect) / total_var

    intercept_correction_factor = 0
    if 'Intercept' in result.params.index:
        intercept_correction_factor = 1

    adj_marginal_r2 = 1 - ((1 - marginal_r2) * (result.nobs - 1) / (
            result.nobs - (len(result.params) - intercept_correction_factor) - 1))

    adj_conditional_r2 = 1 - ((1 - conditional_r2) * (result.nobs - 1) / (
            result.nobs - (len(result.params) - intercept_correction_factor) - 1))

    headers = ['Marginal R2',
               'Conditional R2',
               'Adj. Marginal R2',
               'Adj. Conditional R2']

    r2 = pd.DataFrame(index=headers,
                      data=[marginal_r2,
                            conditional_r2,
                            adj_marginal_r2,
                            adj_conditional_r2])

    Utils.print_and_save(STATISTICS_LOGFILE, config.print_result, f'R\u00b2 values:\n{r2.to_string(header=False)}',
                         '\n')

    with open(f"{path_to_folder}/{ROI}_r2.csv", "w") as f:
        f.write(r2.to_csv(header=False))

    return adj_marginal_r2, adj_conditional_r2


def calculate_information_criteria(formula, current_df, path_to_folder, ROI):
    model = smf.mixedlm(formula=formula, data=current_df, groups=current_df["subject"])

    try:
        result = model.fit(reml=False)

    except numpy.linalg.LinAlgError as exception:
        information_criteria = pd.DataFrame(index=["Error: Singular matrix"],
                                            data=["Information criteria could not be calculated"])
        Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                             f'{exception} error: skipping ROI.')
    else:
        negative_two_log_likelihood = -2 * result.llf
        number_of_parameters = result.df_modelwc + 1
        aic = result.aic
        bic = result.bic

        headers = ['Converged',
                   '-2 Log Likelihood',
                   'Number of parameters',
                   "Akaike's Information Criterion (AIC)",
                   "Schwarz's Bayesian Criterion (BIC)"]

        information_criteria = pd.DataFrame(index=headers,
                                            data=[result.converged,
                                                  negative_two_log_likelihood,
                                                  number_of_parameters,
                                                  aic,
                                                  bic])

    Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                         f"Information criteria:\n{information_criteria.to_string(header=False)}\n\n")

    with open(f"{path_to_folder}/{ROI}_information_criteria.csv", "w") as f:
        f.write(information_criteria.to_csv(header=False))


def run_glm(dataset, ROI, formula, glm_formula_type):
    adj_marginal_r2, adj_conditional_r2, standardised_coeffs, unstandardised_coeffs, pvalues = None, None, None, None, None
    converged = False

    Utils.check_and_make_dir(f"{STATISTICS_PATH}/{glm_formula_type}/{ROI}")

    current_df = dataset.copy()
    model = smf.mixedlm(formula=formula, data=current_df, groups=current_df["subject"])

    try:
        result = model.fit()

    except numpy.linalg.LinAlgError as exception:
        Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                             f'{exception} error: skipping ROI.')
    else:
        path_to_folder = f"{STATISTICS_PATH}/{glm_formula_type}/{ROI}"

        with open(f"{path_to_folder}/{ROI}_LMM.csv", "w") as f:
            f.write(result.summary().as_text())

        standardised_coeffs = calculate_glm_standardised_coeffs(current_df, ROI, formula, glm_formula_type)

        Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                             f'{result.summary()}\n'
                             f'Standardised coefficients:\n'
                             f'{standardised_coeffs}\n')

        converged = result.converged
        unstandardised_coeffs = result.fe_params

        # Get p-values and only keep rows that are in the unstandardised coeffs dataframe (excludes group variance)
        pvalues = result.pvalues.loc[result.pvalues.index.isin(unstandardised_coeffs.index)]

        adj_marginal_r2, adj_conditional_r2 = calculate_r2_measurements(current_df, result, path_to_folder, ROI)
        calculate_information_criteria(formula, current_df, path_to_folder, ROI)

    return adj_marginal_r2, adj_conditional_r2, converged, standardised_coeffs, unstandardised_coeffs, pvalues


def calculate_glm_standardised_coeffs(current_df, folder, formula, glm_formula_type, variation=None):
    # Convert values in numeric columns into zscored values, this will be used to find standardised zscores
    zscored_df = current_df.select_dtypes(include=[np.number]).apply(zscore, ddof=1)

    if folder == 'R2':
        # Least squares regression
        zscored_model = smf.ols(formula=formula, data=zscored_df)
    else:
        if config.categorical_variables != ['']:
            categorical_columns = current_df[[x.lower() for x in config.categorical_variables]]
            zscored_df = pd.concat([categorical_columns, zscored_df], axis=1)

        # Linear mixed model
        zscored_model = smf.mixedlm(formula=formula, data=zscored_df, groups=current_df["subject"])

    zscored_result = zscored_model.fit()

    if folder == 'R2':
        path = f"{STATISTICS_PATH}/{glm_formula_type}/Overall/{variation}_vs_voxels_GLM_standardised_coeffs.csv"
        standardised_coeffs = pd.DataFrame(
            data=pd.concat([zscored_result.params[1:], zscored_result.conf_int()[1:]], axis=1))
    else:
        path = f"{STATISTICS_PATH}/{glm_formula_type}/{folder}/{folder}_GLM_standardised_coeffs.csv"
        standardised_coeffs = pd.DataFrame(
            data=pd.concat([zscored_result.params[1:-1], zscored_result.conf_int()[1:-1]], axis=1))

    standardised_coeffs.columns = ['standardised coef', '[0.025   ', '   0.975]']

    with open(path, "w") as f:
        f.write(standardised_coeffs.to_csv())

    return standardised_coeffs


def construct_glm_formula(critical_params, glm_formula_type):
    if glm_formula_type == 'main_and_interaction_effects':
        symbol = '*'
    elif glm_formula_type == 'main_effects':
        symbol = '+'
    elif glm_formula_type == 'interaction_effects':
        symbol = ':'

    if critical_params[0] == 'Voxels':
        formula = f'R_Square ~ Voxels'
    else:
        formula = f'voxel_value ~ {f" {symbol} ".join(critical_params).lower()}'

        for param in critical_params:
            if param in config.categorical_variables:
                formula = formula.replace(param.lower(), f"C({param.lower()})")

    return formula


def save_dfs(bootstrap_df, columns, combined_results, combined_results_path):
    Utils.check_and_make_dir(f"{STATISTICS_PATH}/Overall")

    bootstrap_df = bootstrap_df.sort_values('Parameter_combination')

    bootstrap_df.to_json(f"{STATISTICS_PATH}/Overall/parameter_percentage_change_vs_baseline.json",
                         orient='records', indent=2)

    combined_results.columns = columns  # Convert columns names back to old versions
    combined_results.to_json(f"{combined_results_path.rsplit('.', 1)[0]}_with_pctchange.json", orient='records',
                             indent=2)


def bootstrap_each_param_comb(combined_results, param_combination, bootstrap_info, current_critical_params):
    columns = {'mean':
                   {'mean': None,
                    f'{config.bootstrap_confidence_interval}_CI': None},
               'percentage change from baseline':
                   {'mean': None,
                    f'{config.bootstrap_confidence_interval}_CI': None}
               }

    percentiles = [(100 - config.bootstrap_confidence_interval) / 2,
                   100 - (100 - config.bootstrap_confidence_interval) / 2]

    # Bootstrap CIs for mean and percentage change
    for column in columns.keys():
        current_df = combined_results.loc[combined_results['file_name'] == param_combination][column].copy()

        # Replace infinite updated data with nan
        current_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Drop rows with NaN
        current_df.dropna(inplace=True)

        bootstrap_sample = resample(current_df, n_samples=config.bootstrap_samples)

        bs_sample_mean = []
        for sample in range(len(bootstrap_sample)):
            bs_sample_mean.append(np.mean(bootstrap_sample[sample]))

        columns[column]['mean'] = current_df.mean()
        columns[column][f'{config.bootstrap_confidence_interval}_CI'] = np.percentile(bs_sample_mean,
                                                                                      [percentiles[0],
                                                                                       percentiles[1]])

    bootstrap_info.append([param_combination,
                           *current_critical_params.values(),
                           columns['mean']['mean'], columns['mean'][f'{config.bootstrap_confidence_interval}_CI'],
                           columns['percentage change from baseline']['mean'],
                           columns['percentage change from baseline'][f'{config.bootstrap_confidence_interval}_CI']])

    return bootstrap_info


def bootstrap_overall_percentage_change(combined_results):
    bootstrap_info = []
    for param_combination in combined_results['file_name'].unique():
        critical_params_dict = dict.fromkeys(config.parameter_dict1, None)

        for critical_param in config.parameter_dict1:
            # Find critical parameter values
            critical_params_dict[critical_param] = \
                combined_results.loc[combined_results['file_name'] == param_combination][critical_param.lower()].iloc[0]

        bootstrap_info = bootstrap_each_param_comb(combined_results, param_combination,
                                                   bootstrap_info, critical_params_dict)

    bootstrap_df = pd.DataFrame(data=bootstrap_info,
                                columns=['Parameter_combination', *critical_params_dict.keys(),
                                         'Mean',
                                         f'Bootstrapped_mean_{config.bootstrap_confidence_interval}_CI',
                                         'Percentage_change',
                                         f'Bootstrapped_percentage_change_{config.bootstrap_confidence_interval}_CI'])

    return bootstrap_df


def calculate_percentage_change_per_roi(combined_results, param_queries):
    for ROI in combined_results['index'].unique():
        current_df = combined_results.loc[combined_results['index'] == ROI].copy()

        try:
            baseline_mean = current_df.query(param_queries[0])['mean'].values[0]
        except IndexError:
            continue

        combined_results.loc[current_df.query(param_queries[0]).index, 'percentage change from baseline'] = 0
        combined_results.loc[current_df.query(param_queries[0]).index, 'baseline'] = "y"

        results = current_df.query(param_queries[1])['mean'].apply(
            lambda x: ((x - baseline_mean) / baseline_mean) * 100)
        combined_results.loc[current_df.query(param_queries[1]).index, 'percentage change from baseline'] = results

    return combined_results


def resample(dataset, n_samples):
    resampled = [None] * n_samples  # Create list of size n_samples and fill with None
    dataset = np.asarray(dataset)  # Make sure dataset is in correct format

    for i in range(n_samples):
        resampled[i] = random.choices(dataset, k=len(dataset))  # Resample dataset n_sample number of times

    return np.asarray(resampled)
