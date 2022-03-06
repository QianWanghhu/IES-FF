#!/usr/bin/env python
import numpy as np
import pandas as pd
import pickle
import json

from scipy import stats
# from scipy.optimize import root
import pyapprox as pya
from scipy import stats

from funcs.read_data import file_settings, variables_prep
from funcs.utils import partial_rank
from adaptive_gp_model import *

def sa_gp(fsave, gp, ind_vars, variables, param_names, 
    cal_type='sampling', save_values=True, norm_y=False):
    """
    Sampling-based and analytic sensitivity analysis with use of GP.

    Parameters:
    ===========
    fsave: str, the path to save results
    gp: Gaussian Process object
    ind_vars: int, the number of variables in the model for analysis
    variables: random variables
    param_names: list of str, the parameter names and the length should be ind_vars
    cal_type: str, "sampling" or "analytic"
    save_values: bool, if True, save the sensitivty values
    notm_y: bool, if True, normalize the GP outputs as the y for sensitivity analysis

    Returns:
    =========
    index_sort: dict, the partial rankings of parameters
    """
    if not os.path.exists(fsave):
        os.mkdir(fsave)

    filename = f'{fsave}/rankings.json'
    if not os.path.exists(filename):
        order = 2
        interaction_terms = pya.compute_hyperbolic_indices(len(ind_vars), order)
        interaction_terms = interaction_terms[:, np.where(
        interaction_terms.max(axis=0) == 1)[0]]
        if cal_type == 'sampling':
            if norm_y:
                sa = sa_norm_f(gp, ind_vars, variables)
            else:
                sa = pyapprox.sampling_based_sobol_indices_from_gaussian_process(gp, 
                    variables, interaction_terms=interaction_terms, nsamples=600, 
                        ngp_realizations=100, ninterpolation_samples = 500, nsobol_realizations = 10,
                        stat_functions=(np.mean, np.std))

            np.savez('sa', sa)
            ST_values = sa['total_effects']['values']
            ST = np.zeros(shape=(ST_values.shape[0] * ST_values.shape[2], ST_values.shape[1]))
            for ii in range(ST_values.shape[2]):
                ST[ii*10:(ii+1)*10, :] = ST_values[:, :, ii].\
                    reshape(ST_values.shape[0], ST_values.shape[1])
        else:
            sa = pyapprox.analytic_sobol_indices_from_gaussian_process(gp, variables, 
            interaction_terms, ngp_realizations=100, ninterpolation_samples=500, 
                use_cholesky=True, ncandidate_samples=10000, nvalidation_samples=200,
                stat_functions=(np.mean, np.std))
            np.savez(filename, total_effects=sa['total_effects']['values'])
            ST = sa['total_effects']['values']

        index_sort = partial_rank(ST, ST.shape[1], conf_level=0.95)
        print(index_sort)
        if save_values:
            ST_mean = sa['total_effects']['mean']
            ST_mean = pd.DataFrame(data = ST_mean, index = param_names, columns=['ST'])
            ST_mean['std'] = sa['total_effects']['std']
            ST_mean.to_csv(f'{fsave}/ST.csv')

            Si_mean = sa['sobol_indices']['mean']           
            Si_mean = pd.DataFrame(data = Si_mean, index = np.arange(Si_mean.shape[0]), columns=['Si'])
            Si_mean['std'] = sa['sobol_indices']['std']
            for ii in range(Si_mean.shape[0]):
                val_pars = param_names[interaction_terms[:, ii] == 1]
                if len(val_pars) == 2:
                    Si_mean.loc[ii, ['par1', 'par2']] = val_pars
                else:
                    Si_mean.loc[ii, ['par1', 'par2']] = [val_pars[0], val_pars[0]]

            Si_mean.to_csv(f'{fsave}/SI.csv')

        with open(filename, 'w') as fp:
            json.dump(index_sort, fp, indent=2)

    else:
        with open(filename, 'r') as fp:
            index_sort_load = json.load(fp)
        index_sort = {}
        for k, v in index_sort_load.items():
            index_sort[int(k)] = index_sort_load[k]
        
        ST = pd.read_csv(f'{fsave}/ST.csv')

    return index_sort, ST


def sa_norm_f(gp, ind_vars, variables):
    """
    Sampling-based approach to calculate sensitivity analysis.
    Parameters:
    ===========
    fsave: str, the path to save results
    fun: Gaussian Process object
    ind_vars: int, the number of variables in the model for analysis
    variables: random variables

    Returns:
    =========
    index_sort: dict, the partial rankings of parameters
    """
    # Define problems

    from pyapprox.gaussian_process import generate_gp_realizations
    from pyapprox.sensitivity_analysis import repeat_sampling_based_sobol_indices

    def fun_wrapper(fun):
        """
        Function wrapper for calculating the normalized objective function values.
        """
        def cal_fun(x):
            y = 1 / (2 - fun(x))
            return y
        return cal_fun

    order = 2
    interaction_terms = pya.compute_hyperbolic_indices(len(ind_vars), order)
    interaction_terms = interaction_terms[:, np.where(
    interaction_terms.max(axis=0) == 1)[0]]
    nsamples=800; ngp_realizations=100; ninterpolation_samples = 500;
    nsobol_realizations = 10; stat_functions=(np.mean, np.std)
    ncandidate_samples=10000; nvalidation_samples=100
    sampling_method='sobol'

    assert nsobol_realizations > 0

    if ngp_realizations > 0:
        assert ncandidate_samples > ninterpolation_samples
        gp_realizations = generate_gp_realizations(
            gp, ngp_realizations, ninterpolation_samples, nvalidation_samples,
            ncandidate_samples, variables, True, 0)
        fun = gp_realizations
    else:
        fun = gp

    sobol_values, total_values, variances, means = \
        repeat_sampling_based_sobol_indices(
            fun_wrapper(fun), variables, interaction_terms, nsamples,
            sampling_method, nsobol_realizations)

    result = dict()
    data = [sobol_values, total_values, variances, means]
    data_names = ['sobol_indices', 'total_effects', 'variance', 'mean']
    for item, name in zip(data, data_names):
        subdict = dict()
        for ii, sfun in enumerate(stat_functions):
            # have to deal with averaging over axis = (0, 1) and axis = (0, 2)
            # for mean, variance and sobol_indices, total_effects respectively
            subdict[sfun.__name__] = sfun(item, axis=(0, -1))
        subdict['values'] = item
        result[name] = subdict
    return result
