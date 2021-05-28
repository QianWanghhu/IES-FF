#!/usr/bin/env ffexplore
import numpy as np
import pandas as pd
import pyapprox as pya
import json
import os
from pyapprox.random_variable_algebra import product_of_independent_random_variables_pdf

from basic.boots_pya import fun
from basic.partial_rank import partial_rank
from basic.read_data import read_specify, file_settings
from basic.utils import sa_df_format, dist_return

# # import the original parameter sets
def pya_boot_sensitivity(outpath, nboot, seed, product_uniform, filename):

    if not product_uniform:
        samples, values = read_specify('model', 'full', product_uniform=False, num_vars=22)
        # import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
        variable, param_all = read_specify('parameter', 'full', product_uniform=False, num_vars=22)
        len_params = variable.num_vars()
    else:
        variable, _ = read_specify('parameter', 'reduced', product_uniform, num_vars=11)
        samples, values = read_specify('model', 'reduced', product_uniform, num_vars=11)
        len_params = variable.num_vars()
    # Adaptively increase the size of training dataset and conduct the bootstrap based partial ranking

    ##==========test random sampling=============##
    random_samples = np.random.randint(0, 550, size=(20, 550))
    # for rd in range(random_samples.shape[0]):
    n_strat, n_end, n_step = [50, 200, 100]
    n_list = [*np.arange(n_strat, n_end+1, n_step)]
    errors_cv_all = {}
    partial_results = {}
    total_effects_all = {}
    approx_list_all = {}
    error_best = values.flatten().std()
    for i in range(n_strat, n_end+1, n_step):
    # for i in n_list:
        if (n_end - i)  < n_step:
            i = n_end
        np.random.seed(seed)                                                    
        errors_cv, _, total_effects, approx_list = fun(
            variable, samples[:, :i], values[:i], product_uniform, nboot=nboot)# random_samples[rd][:i]

        # partial ranking
        total_effects = np.array(total_effects)
        sa_shape = list(total_effects.shape)[0:2]
        total_effects = total_effects.reshape((sa_shape))
        rankings = partial_rank(total_effects,len_params, conf_level=0.95)
        partial_results[f'nsample_{i}'] = rankings
        errors_cv_all[f'nsample_{i}'] = errors_cv
        total_effects_all[f'nsample_{i}'] = total_effects
        approx_list_all[f'nsample_{i}'] = approx_list 
        print(f'error_cv: {errors_cv.mean()}')
    # End for

    if not os.path.exists(f'{outpath}'): os.mkdir(f'{outpath}')
    np.savez(f'{outpath}\{filename}',errors_cv=errors_cv_all, sensitivity_indices=partial_results, total_effects=total_effects_all)
    import pickle
    pickle.dump(approx_list_all, open(f'{outpath}\{filename[:-4]}-approx-list.pkl', "wb"))
# END pya_boot_sensitivity()

def run_pya(outpath, nboot, seed, product_uniform):
    dist_type = dist_return(product_uniform)
    filename = f'adaptive-reduce-{dist_type}_552.npz'

    print(f'{outpath}{filename}')
    if not os.path.exists(f'{outpath}{filename}'):
        pya_boot_sensitivity(
            outpath, nboot, seed, product_uniform, filename)

    # for ii in range(20):

    fileread = np.load(f'{outpath}\{filename}', allow_pickle=True)
    errors_cv = fileread[fileread.files[0]][()]
    sensitivity_indices = fileread[fileread.files[1]][()]
    # Look the error change with the increase of sample size
    errors_cv = pd.DataFrame.from_dict(errors_cv)
    error_stats = pd.DataFrame()
    error_stats['mean'] = np.round(errors_cv.mean(axis=0), 4)
    error_stats['error_lower'] = np.round(np.quantile(errors_cv, 0.025, axis=0), 4)
    error_stats['error_upper'] = np.round(np.quantile(errors_cv, 0.975, axis=0), 4)
    error_stats.index.name = 'index'

    error_stats.to_csv(f'{outpath}\error_cv_{dist_type}_552.csv')
    with open(f'{outpath}\partial_reduce_{dist_type}_552.json', 'w') as fp:
        json.dump(sensitivity_indices, fp, indent=2)


def pce_22(nboot, seed, ntrain_samples):
    fpath_save = file_settings()[0]
    if os.path.exists(f'{fpath_save}sa_pce_22.csv'):
        return
    samples, values = read_specify('model', 'full', product_uniform=False, num_vars=22)
    # import parameter inputs and generate the dataframe of analytical ratios between sensitivity indices
    variable, param_all = read_specify('parameter', 'full', product_uniform=False, num_vars=22)
    # Used for PCE fitting
    np.random.seed(seed) 
    error_cv, _, total_effects, _ = fun(
        variable, samples[:,:ntrain_samples], 
        values[:ntrain_samples], product_uniform=False,
        nboot=nboot)                                          
    sa_df = sa_df_format(total_effects, variable, 
                    param_all, conf_level=0.95)
    sa_df.to_csv(f'{fpath_save}sa_pce_22.csv', index=True)
    error_cv_df = pd.DataFrame(data = error_cv, columns = [f'22_{ntrain_samples}_uniform'],
                    index = np.arange(len(error_cv)))
    error_cv_mean = error_cv_df.apply(np.mean, axis=0)
    error_cv_lower = np.round(error_cv_df.quantile(0.025, axis=0), 4)
    error_cv_upper= np.round(error_cv_df.quantile(0.975, axis=0), 4)
    error_stats_df = pd.DataFrame(data=[error_cv_mean, error_cv_lower, error_cv_upper], 
                    index=['mean', 'lower', 'upper']).T
    error_stats_df.to_csv(f'{fpath_save}error_cv_compare.csv', index=True)
