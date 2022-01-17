import numpy as np
import pandas as pd
import statsmodels.api as sm

from .figures import Histogram
from .utils import Utils


def main(config):
    raw_results = load_raw_data(config)
    ROI_dict = split_raw_df(raw_results)
    statistics(ROI_dict)


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


def statistics(ROI_dict):
    res = []
    for ROI in ROI_dict:
        x = ROI_dict[ROI][['Hyperband', 'Inplane_acceleration']]
        x = sm.add_constant(x)
        print(x.shape)

        y = ROI_dict[ROI]['voxel_value']
        model = sm.OLS(y, x)
        results = model.fit()
        res.append(results.rsquared)

    print('Linear:', res, np.max(res))

    res = []
    for ROI in ROI_dict:
        x = ROI_dict[ROI][['Hyperband', 'Inplane_acceleration']]
        y = ROI_dict[ROI]['voxel_value']
        x = np.hstack([x**(i+1) for i in range(2)])

        x = sm.add_constant(x)

        model = sm.OLS(y, x)
        results = model.fit()
        res.append(results.rsquared)

    print('Poly 2:', res, np.max(res))

    res = []
    for ROI in ROI_dict:
        x = ROI_dict[ROI][['Hyperband', 'Inplane_acceleration']]
        y = ROI_dict[ROI]['voxel_value']
        x = np.hstack([x**(i+1) for i in range(3)])

        x = sm.add_constant(x)

        model = sm.OLS(y, x)
        results = model.fit()
        res.append(results.rsquared)

    print('Poly 3:', res, np.max(res))

    res = []
    for ROI in ROI_dict:
        x = ROI_dict[ROI][['Hyperband', 'Inplane_acceleration']]
        y = ROI_dict[ROI]['voxel_value']
        x = np.hstack([x**(i+1) for i in range(9)])

        x = sm.add_constant(x)

        model = sm.OLS(y, x)
        results = model.fit()
        res.append(results.rsquared)

    print('Poly 3:', res, np.max(res))

# model = sm.ols(formula='voxel_value ~ C(Hyperband) + C(Inplane_acceleration) + C(Hyperband):C(Inplane_acceleration)', data=ROI_dict[ROI]).fit()
        # results = sm.stats.anova_lm(model, typ=2)
        #
        # if np.isnan(results['PR(>F)'][1]) or results['PR(>F)'][1] != 0.0:
        #     print(ROI, np.format_float_positional(results['PR(>F)'][1]))
    # TODO: consider adding ROI as a variable
    # TODO: Remove and raise warning if interaction effect calculation not possible
    # Drop rows this way: df_new = raw_results.drop(raw_results[(raw_results['inplane acceleration'] == 1) & (raw_results['Hyperband'] == 1)].index)
    # TODO: move from r style formula to more readable style
    # TODO: C means categorical

def k_means():
    pass

if __name__ == '__main__':
    main()