import itertools
import os
import random
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as pltn
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import zscore
from statsmodels.stats.weightstats import ttest_ind as ttest

from .figures import Histogram
from .utils import Utils

config = None
STATISTICS_PATH = ''
STATISTICS_LOGFILE = None


def main(cfg):
    global config

    config = cfg

    stats_setup()
    param_queries, critical_params = find_baseline_parameters()
    statistics(param_queries, critical_params)

    STATISTICS_LOGFILE.close()


def stats_setup():
    global STATISTICS_PATH
    global STATISTICS_LOGFILE
    STATISTICS_PATH = f'{os.getcwd()}/Statistics/{config.statistics_subfolder_name}/'
    Utils.check_and_make_dir(f"{os.getcwd()}/Statistics")
    Utils.check_and_make_dir(STATISTICS_PATH)

    STATISTICS_LOGFILE = open(f'{os.getcwd()}/Statistics/{config.statistics_subfolder_name}/'
                              f'statistics_terminal_output.txt', 'w')

    # Save log of only the statistics info from the config file
    Utils.save_config(STATISTICS_PATH,
                      f"{Path(os.path.abspath(__file__)).parents[1]}/fRAT_config",
                      relevant_sections=['Statistics'], config_name='statistics_log',
                      additional_info=[
                          f'data_used_for_statistics = "{config.averaging_type.replace(" ", "_").lower()}"'])

    Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                         f'\nCalculating statistics using {config.averaging_type} data.')


def clean_data(combined_results, voxel_cutoff):
    # Remove No ROI and overall data
    combined_results = combined_results[
        (combined_results['index'] != 'No ROI') & (combined_results['index'] != 'Overall')]
    try:
        below_thresh = combined_results[combined_results['voxels'] < voxel_cutoff]
        combined_results = combined_results[combined_results['voxels'] > voxel_cutoff]
    except KeyError:
        below_thresh = combined_results[combined_results['total voxels'] < voxel_cutoff]
        combined_results = combined_results[combined_results['total voxels'] > voxel_cutoff]

    return combined_results, below_thresh['index'].unique()


def find_baseline_parameters():
    baseline_param_query = ''
    non_baseline_params_query = ''

    table = Utils.load_paramValues_file()
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
    else:
        try:
            baseline_param_query = " & ".join(" == ".join((str(k), str(v))) for k, v in baseline_params[0].items())
            non_baseline_params_query = " | ".join(" != ".join((str(k), str(v))) for k, v in baseline_params[0].items())
        except IndexError:
            Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                                 '\nNo baseline set in paramValues.csv. Change versus baseline statistics will not be '
                                 'calculated.\n')

        table.columns = columns
        return [baseline_param_query, non_baseline_params_query], table.columns[list(critical_column_locs)]


def structure_data(config, participant_paths):
    combined_results = Utils.read_combined_results(os.getcwd(), config.averaging_type)

    jsons = []
    for path in participant_paths:
        jsons.extend([f"{path}/Summarised_results/{jsn}" for jsn in Utils.find_files(f"{path}/Summarised_results/", "json") if 'combined_results' not in jsn])

    individual_roi_results = Histogram.load_and_restructure_jsons(config, jsons, combined_results[0], data_type='statistics')

    return individual_roi_results, combined_results


def split_individual_subject_df(individual_subject_results):
    """Split combined dataframe into a dict entry for each ROI"""
    ROI_dict = {}
    grouped = individual_subject_results.groupby(individual_subject_results.ROI)
    for ROI in individual_subject_results.ROI.unique():
        ROI_dict[f'{ROI}'] = grouped.get_group(ROI).dropna()
        ROI_dict[f'{ROI}'].columns = [x.replace(' ', '_') for x in ROI_dict[f'{ROI}'].columns]

    return ROI_dict


def statistics(param_queries, critical_params):
    individual_roi_results, combined_results = load_data()

    rois_below_thresh = []
    if '' not in param_queries:  # If no baseline parameter combination set, skip this
        rois_below_thresh = calculate_percent_change_versus_baseline(critical_params, param_queries, combined_results)

    roi_statistics(critical_params, rois_below_thresh, individual_roi_results, combined_results[0])


def load_data():
    participants, _ = Utils.find_participant_dirs(os.getcwd())

    individual_roi_results, combined_results = structure_data(config, participants)
    individual_roi_results = split_individual_subject_df(individual_roi_results)

    return individual_roi_results, combined_results


def roi_statistics(critical_params, rois_below_thresh, individual_roi_results_dict, combined_results):
    glm_formula_types = ['main_effects', 'main_and_interaction_effects', 'interaction_effects']

    for glm_formula_type in glm_formula_types:
        Utils.check_and_make_dir(f"{STATISTICS_PATH}/{glm_formula_type}")

        r_square_data = []
        voxel_data = []

        for ROI in individual_roi_results_dict:
            Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                                 f'\n==============================================================================\n'
                                 f'\nCalculating {ROI} statistics ({glm_formula_type.replace("_", " ")}).\n')

            individual_roi_results_dict[ROI].columns = individual_roi_results_dict[ROI].columns.str.lower()

            r_square, voxels = run_glm(individual_roi_results_dict[ROI], critical_params, combined_results,
                                       ROI=ROI, glm_formula_type=glm_formula_type)

            if ROI not in rois_below_thresh and ROI not in ['Overall', 'No ROI']:
                r_square_data.append(r_square)
                voxel_data.append(voxels)

        # compute_rsquare_regression(voxel_data, r_square_data, glm_formula_type) ## TODO finish this

    # for ROI in individual_roi_results_dict:
    #     t_test(individual_roi_results_dict[ROI], critical_params)  # TODO do I change ROI dict


def compute_rsquare_regression(voxel_data, r_square_data, glm_formula_type):
    # GLM
    voxel_data = sm.add_constant(voxel_data)
    voxel_df = pd.DataFrame(columns=['Intercept', 'Voxels'], data=voxel_data)
    r_square_df = pd.DataFrame(columns=['R_Square'], data=r_square_data)
    model = sm.OLS(r_square_df, voxel_df)
    result = model.fit()

    formula = construct_glm_formula(critical_params=['Voxels'], glm_formula_type=glm_formula_type)
    standardised_coeffs = calculate_glm_standardised_coeffs(pd.concat([voxel_df[['Voxels']], r_square_df], axis=1),
                                                            'R2', formula, glm_formula_type)

    Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                         f'\n==============================================================================\n'
                         f'Calculating r2 vs voxel amount statistics for {glm_formula_type.replace("_", " ")}.\n'
                         f'{result.summary()}\n\n\n'
                         f'Standardised coefficients:\n'
                         f'{standardised_coeffs}')

    with open(f"{STATISTICS_PATH}/{glm_formula_type}/Overall/r_square_vs_voxels_GLM.csv", "w") as f:
        f.write(result.summary().as_csv())

    # Scatter plot
    df = pd.concat([voxel_df, r_square_df], axis=1)
    figure = (
            pltn.ggplot(df, pltn.aes(x="Voxels", y="R_Square"))
            + pltn.theme_538()
            + pltn.geom_point()
            + pltn.geom_smooth(method='lm')
            + pltn.labs(x='Voxel count', y='$r^2$')
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

    figure.save(f"{STATISTICS_PATH}/{glm_formula_type}/Overall/r_square_vs_voxels.png",
                height=config.plot_scale,
                width=config.plot_scale * 2,
                verbose=False, limitsize=False)

    figure.save(f"{STATISTICS_PATH}/{glm_formula_type}/Overall/r_square_vs_voxels.svg",
                height=config.plot_scale,
                width=config.plot_scale * 2,
                verbose=False, limitsize=False)


def calculate_percent_change_versus_baseline(critical_params, param_queries, combined_results):
    if config.verbose:
        print(f'\nCalculating percentage change statistics.\n')

    path = combined_results[1]
    combined_results = combined_results[0]

    combined_results['Percentage change from baseline'] = np.nan
    combined_results['Baseline'] = ""
    combined_results = combined_results.sort_values('index')

    columns = combined_results.columns  # Save old column names before converting to lower case
    # Need to convert to lowercase as the df query is easier to make when lowercase
    combined_results.columns = combined_results.columns.str.lower()

    # Add percentage change statistics to original combined_results df
    combined_results = calculate_percentage_change_per_roi(combined_results, param_queries)

    cleaned_combined_results, rois_below_thresh = clean_data(combined_results, voxel_cutoff=config.minimum_voxels)
    bootstrap_df = bootstrap_overall_percentage_change(cleaned_combined_results, critical_params)
    save_dfs(bootstrap_df, columns, combined_results, path)

    return rois_below_thresh


def t_test(dataset, critical_params):
    t_test_results = []

    for param in critical_params:
        param = param.lower()
        combinations = list(itertools.combinations(sorted(dataset[param].unique()), 2))

        for i, combination in enumerate(combinations):
            x1 = dataset.loc[dataset[param] == combination[0]]['mean']
            x2 = dataset.loc[dataset[param] == combination[1]]['mean']
            current_result = list(ttest(x1=x1, x2=x2,
                                        usevar='unequal'))

            # Insert label of parameter combination being tested
            current_result.insert(0, f"{param}: {' v '.join(str(v) for v in combinations[i])}")

            current_result.append(x2.mean() - x1.mean())

            current_result.append(calculate_cohens_d(x1=x1, x2=x2))
            # todo cohens d, save t-test as welch t-test
            t_test_results.append(current_result)

    t_test_df = pd.DataFrame(data=t_test_results, columns=['Parameter combination', 'tstat', 'pvalue', 'df',
                                                           'Mean difference', "Cohen's d"])


def calculate_cohens_d(x1, x2):
    # Calculate the size of samples
    n1, n2 = len(x1), len(x2)

    # calculate the variance of the samples
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)

    # calculate the pooled standard deviation
    dof = n1 + n2 - 2
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / dof)

    # calculate the means of the samples
    u1, u2 = np.mean(x1), np.mean(x2)

    # return the effect size
    return (u1 - u2) / s


def linear_mixed_model(formula, current_df, glm_formula_type, ROI):
    model = smf.mixedlm(formula=formula, data=current_df, groups=current_df["subject"])
    result = model.fit()

    path_to_folder = f"{STATISTICS_PATH}/{glm_formula_type}/{ROI}"

    with open(f"{path_to_folder}/{ROI}_LMM.csv", "w") as f:
        f.write(result.summary().as_text())

    Utils.print_and_save(STATISTICS_LOGFILE, config.print_result, f'{result.summary()}')

    adj_conditional_r2, adj_marginal_r2 = calculate_r2_measurements(current_df, result, path_to_folder, ROI)
    calculate_information_criteria(formula, current_df, path_to_folder,
                                   ROI)

    return result, [adj_conditional_r2, adj_marginal_r2]  # TODO: what to do for plotting the r2 values


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

    Utils.print_and_save(STATISTICS_LOGFILE, config.print_result, f'R\u00b2 values:\n{r2.to_string(header=False)}', '\n')

    with open(f"{path_to_folder}/{ROI}_r2.csv", "w") as f:
        f.write(r2.to_csv(header=False))

    return adj_conditional_r2, adj_marginal_r2


def calculate_information_criteria(formula, current_df, path_to_folder, ROI):
    model = smf.mixedlm(formula=formula, data=current_df, groups=current_df["subject"])
    result = model.fit(reml=False)

    negative_two_log_likelihood = -2 * result.llf
    number_of_parameters = result.df_modelwc + 1
    aic = result.aic
    bic = result.bic

    headers = ['-2 Log Likelihood',
               'Number of parameters',
               "Akaike's Information Criterion (AIC)",
               "Schwarz's Bayesian Criterion (BIC)"]

    information_criteria = pd.DataFrame(index=headers,
                                        data=[negative_two_log_likelihood,
                                              number_of_parameters,
                                              aic,
                                              bic])

    Utils.print_and_save(STATISTICS_LOGFILE, config.print_result,
                         f"Information criteria:\n{information_criteria.to_string(header=False)}\n\n")

    with open(f"{path_to_folder}/{ROI}_information_criteria.csv", "w") as f:
        f.write(information_criteria.to_csv(header=False))


def run_glm(dataset, critical_params, combined_results, ROI, glm_formula_type):
    Utils.check_and_make_dir(f"{STATISTICS_PATH}/{glm_formula_type}/{ROI}")

    current_df = dataset.copy()
    formula = construct_glm_formula(critical_params, glm_formula_type)

    result, rsquared = linear_mixed_model(formula, current_df, glm_formula_type, ROI)

    try:
        nobs = combined_results.loc[combined_results['index'] == ROI].sum()['Total voxels']
    except KeyError:
        nobs = combined_results.loc[combined_results['index'] == ROI].sum()['Voxels']

    return rsquared, nobs


def calculate_glm_standardised_coeffs(current_df, folder, formula, glm_formula_type):
    # Convert values in numeric columns into zscored values, this will be used to find standardised zscores
    zscored_df = current_df.select_dtypes(include=[np.number]).apply(zscore, ddof=1)
    zscored_model = smf.ols(formula=formula, data=zscored_df)
    zscored_result = zscored_model.fit()

    standardised_coeffs = pd.DataFrame(data=pd.concat([zscored_result.params, zscored_result.conf_int()], axis=1))
    standardised_coeffs.columns = ['standardised coef', '[0.025   ', '   0.975]']

    if folder == 'R2':
        path = f"{STATISTICS_PATH}/{glm_formula_type}/Overall/r_square_vs_voxels_GLM_standardised_coeffs.csv"
    else:
        path = f"{STATISTICS_PATH}/{glm_formula_type}/{folder}/{folder}_GLM_standardised_coeffs.csv"

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
    bootstrap_df = bootstrap_df.sort_values('Parameter_combination')

    Utils.check_and_make_dir(f"{STATISTICS_PATH}/Overall")
    bootstrap_df.to_json(f"{STATISTICS_PATH}/Overall/parameter_percentage_change_vs_baseline.json",
                         orient='records', indent=2)

    combined_results.columns = columns  # Convert columns names back to old versions
    combined_results.to_json(combined_results_path, orient='records', indent=2)


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


def bootstrap_overall_percentage_change(combined_results, critical_params):
    bootstrap_info = []
    for param_combination in combined_results['file_name'].unique():
        current_critical_params = dict.fromkeys(critical_params, None)

        for critical_param in critical_params:
            # Find critical parameter values
            current_critical_params[critical_param] = \
                combined_results.loc[combined_results['file_name'] == param_combination][critical_param.lower()].iloc[0]

        bootstrap_info = bootstrap_each_param_comb(combined_results, param_combination,
                                                   bootstrap_info, current_critical_params)

    bootstrap_df = pd.DataFrame(data=bootstrap_info,
                                columns=['Parameter_combination', *current_critical_params.keys(),
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
