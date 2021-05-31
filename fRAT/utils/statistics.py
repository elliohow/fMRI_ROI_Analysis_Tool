import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.formula.api import ols

from .utils import Utils
from .figures import Histogram
from .paramparser import ParamParser

def main(config):
    if config.verbose:
        print('\n------------------\n--- Statistics ---\n------------------')

    ParamParser.chdir_to_output_directory('Statistics', config)
    raw_results = load_raw_data(config)
    ROI_dict = split_raw_df(raw_results)
    statistics(ROI_dict)


def load_raw_data(config):
    combined_results = pd.read_json("Summarised_results/combined_results.json")
    jsons = Utils.find_files("Raw_results", "json")
    combined_raw_df = Histogram.make_raw_df(config, jsons, combined_results)

    return combined_raw_df


def split_raw_df(raw_results):
    ROI_dict = {}
    grouped = raw_results.groupby(raw_results.ROI)
    for ROI in raw_results.ROI.unique():
        ROI_dict[f'{ROI}'] = grouped.get_group(ROI).dropna()
        ROI_dict[f'{ROI}'].columns = [x.replace(' ', '_') for x in ROI_dict[f'{ROI}'].columns]
        
    return ROI_dict


def statistics(ROI_dict):
    for ROI in ROI_dict:
        model = ols('voxel_value ~ C(hyperband) + C(inplane_acceleration) + C(hyperband):C(inplane_acceleration)', data=ROI_dict[ROI]).fit()
        results = sm.stats.anova_lm(model, typ=2)
        # TODO: Testing interaction effect creates error, find out why
        if np.isnan(results['PR(>F)'][1]) or results['PR(>F)'][1] != 0.0:
            print(ROI, np.format_float_positional(results['PR(>F)'][1]))
    # TODO: consider adding ROI as a variable


if __name__ == '__main__':
    main()