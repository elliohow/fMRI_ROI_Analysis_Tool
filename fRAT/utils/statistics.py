import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .figures import Histogram
from .utils import Utils

config = None


def main(cfg):
    global config
    config = cfg
    stats_setup()
    param_queries, critical_params = find_baseline_parameters()
    statistics(param_queries, critical_params)


def stats_setup():
    Utils.check_and_make_dir(f"{os.getcwd()}/Overall/Statistics")

    # Save log of only the statistics info from the config file
    Utils.save_config(f"{os.getcwd()}/Overall/Statistics/",
                      f"{Path(os.path.abspath(__file__)).parents[1]}/fRAT_config",
                      relevant_section='Statistics', config_name='statistics_log')


def clean_data(combined_results, voxel_cutoff):
    # Remove No ROI and overall data
    combined_results = combined_results[(combined_results['index'] != 'No ROI') & (combined_results['index'] != 'Overall')]
    combined_results = combined_results[combined_results['voxels'] > voxel_cutoff]

    return combined_results


def find_baseline_parameters():
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
        baseline_param_query = " & ".join(" == ".join((str(k), str(v))) for k, v in baseline_params[0].items())
        non_baseline_params_query = " | ".join(" != ".join((str(k), str(v))) for k, v in baseline_params[0].items())

        table.columns = columns
        return [baseline_param_query, non_baseline_params_query], table.columns[list(critical_column_locs)]


def load_raw_data(config):
    combined_results = pd.read_json("Overall/Summarised_results/combined_results.json")
    raw_jsons = Utils.find_files("Overall/Raw_results", "json")
    combined_raw_df = Histogram.make_raw_df(config, raw_jsons, combined_results)

    return combined_raw_df


def split_raw_df(raw_results):
    """Split combined dataframe into a dict entry for each ROI"""
    ROI_dict = {}
    grouped = raw_results.groupby(raw_results.ROI)
    for ROI in raw_results.ROI.unique():
        ROI_dict[f'{ROI}'] = grouped.get_group(ROI).dropna()
        ROI_dict[f'{ROI}'].columns = [x.replace(' ', '_') for x in ROI_dict[f'{ROI}'].columns]

    return ROI_dict


def statistics(param_queries, critical_params):
    overall_statistics(critical_params, param_queries)
    roi_statistics(critical_params)


def roi_statistics(critical_params):
    raw_results = load_raw_data(config)
    ROI_dict = split_raw_df(raw_results)

    for ROI in ROI_dict:
        if config.verbose:
            print(f'\nCalculating {ROI} statistics\n')

        ROI_dict[ROI].columns = ROI_dict[ROI].columns.str.lower()
        glm(ROI_dict[ROI], critical_params, folder=ROI)


def overall_statistics(critical_params, param_queries):
    if config.verbose:
        print(f'\nCalculating overall statistics\n')

    combined_results = pd.read_json("Overall/Summarised_results/combined_results.json")
    combined_results['Percentage change from baseline'] = np.nan
    combined_results['Baseline'] = ""
    combined_results = combined_results.sort_values('index')

    columns = combined_results.columns  # Save old column names before converting to lower case
    # Need to convert to lowercase as the df query is easier to make when lowercase
    combined_results.columns = combined_results.columns.str.lower()

    # Add percentage change statistics to original combined_results df
    combined_results = calculate_percentage_change_per_roi(combined_results, param_queries)

    cleaned_combined_results = clean_data(combined_results, voxel_cutoff=config.minimum_voxels)
    bootstrap_df = bootstrap_cis(cleaned_combined_results, critical_params)
    glm(cleaned_combined_results, critical_params, folder='Overall')

    save_dfs(bootstrap_df, columns, combined_results)


def glm(dataset, critical_params, folder):
    current_df = dataset.copy()

    Utils.check_and_make_dir(f"{os.getcwd()}/Overall/Statistics/{folder}")

    formula = construct_glm_formula(critical_params, df_type=folder)

    if config.bootstrap_statistics:
        bootstrap_confidence_intervals(current_df, formula, folder=folder)

    model = smf.glm(formula=formula, data=current_df)
    result = model.fit()

    if config.print_result:
        print(result.summary())

    with open(f"{os.getcwd()}/Overall/Statistics/{folder}/{folder}_GLM.csv", "w") as f:
        f.write(result.summary().as_csv())


def bootstrap_confidence_intervals(current_df, formula, folder):
    coefficients = []
    percentiles = [(100 - config.bootstrap_confidence_interval) / 2,
                   100 - (100 - config.bootstrap_confidence_interval) / 2]
    for i in range(config.bootstrap_samples):
        current_sample = current_df.sample(n=current_df[current_df.columns[0]].count(), replace=True)
        model = smf.glm(formula=formula, data=current_sample)
        result = model.fit()
        coefficients.append(np.array(result.params))

    coefficients = np.asarray(coefficients)
    coefficients = np.percentile(coefficients, [percentiles[0], percentiles[1]], axis=0).transpose()
    coefficients = pd.DataFrame(data=coefficients, columns=[f'[{percentiles[0] / 100}',
                                                            f'{percentiles[1] / 100}]'])
    coefficients.insert(loc=0, column='x', value=result.params.keys())

    if config.print_result:
        print(coefficients)

    with open(f"{os.getcwd()}/Overall/Statistics/{folder}/Overall_GLM_bootstrap_confidence_intervals.csv", "w") as f:
        f.write(coefficients.to_csv(index=False))



def construct_glm_formula(critical_params, df_type):
    if config.glm_formula == 'Main + interaction effects':
        symbol = '*'
    elif config.glm_formula == 'Main effects only':
        symbol = '+'
    elif config.glm_formula == 'Interaction effects only':
        symbol = ':'

    if df_type == 'Overall':
        y_var = config.glm_statistic.lower()
    else:
        y_var = 'voxel_value'

    formula = f'{y_var} ~ {f" {symbol} ".join(critical_params).lower()}'

    for param in critical_params:
        if param in config.categorical_variables:
            formula = formula.replace(param.lower(), f"C({param.lower()})")

    if config.remove_intercept:
        formula += " -1"

    return formula


def save_dfs(bootstrap_df, columns, combined_results):
    bootstrap_df = bootstrap_df.sort_values('Parameter_combination')
    bootstrap_df.to_json(f"{os.getcwd()}/Overall/Statistics/Overall/parameter_percentage_change.json", orient='records', indent=2)

    combined_results.columns = columns  # Convert columns names back to old versions
    combined_results.to_json(f"{os.getcwd()}/Overall/Summarised_results/combined_results.json", orient='records',
                             indent=2)


def bootstrap_each_param_comb(combined_results, param_combination, bootstrap_info, current_critical_params):
    columns = {'mean':
                   {'mean': None,
                    f'{config.bootstrap_confidence_interval}_CI': None},
               'percentage change from baseline':
                   {'mean': None,
                    f'{config.bootstrap_confidence_interval}_CI': None}
               }

    percentiles = [(100 - config.bootstrap_confidence_interval)/2, 100 - (100-config.bootstrap_confidence_interval)/2]

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


def bootstrap_cis(combined_results, critical_params):
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

        results = current_df.query(param_queries[1])['mean'].apply(lambda x: ((x - baseline_mean) / baseline_mean) * 100)
        combined_results.loc[current_df.query(param_queries[1]).index, 'percentage change from baseline'] = results

    return combined_results


def resample(dataset, n_samples):
    resampled = [None] * n_samples  # Create list of size n_samples and fill with None
    dataset = np.asarray(dataset)  # Make sure dataset is in correct format

    for i in range(n_samples):
        resampled[i] = random.choices(dataset, k=len(dataset))  # Resample dataset n_sample number of times

    return np.asarray(resampled)
