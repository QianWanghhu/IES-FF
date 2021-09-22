#!/usr/bin/env python
from numba.core.typing.templates import BaseRegistryLoader
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import var
import pandas as pd
import pickle
import json
import scipy
import seaborn as sns

from scipy import stats
# from scipy.optimize import root
from scipy.optimize import bisect
import pyapprox as pya
from pyapprox import generate_independent_random_samples
import matplotlib as mpl
import spotpy as sp
from scipy import stats

mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['text.usetex'] = False  # use latex for all text handling
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.format'] = 'png'  # gives best resolution plots
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 16
# print mpl.rcParams.keys()
mpl.rcParams['text.latex.preamble'] = \
    r'\usepackage{siunitx}\usepackage{amsmath}\usepackage{amssymb}'

from funcs.read_data import file_settings, variables_prep
from funcs.utils import partial_rank, return_sa
from gp_pce_model import *

    # import GP

fpath = '../output/gp_run_0816/'
gp = pickle.load(open(f'{fpath}gp_1.pkl', "rb"))
x_training = gp.X_train_
y_training = gp.y_train_

# Resample in the ranges where the objective values are above 0
x_select = x_training[np.where(y_training>0)[0], :]
x_range = x_select.max(axis=0)
univariable_temp = [stats.uniform(0, x_range[ii]) for ii in range(0, x_range.shape[0])]
variable_temp = pyapprox.IndependentMultivariateRandomVariable(univariable_temp)

# visualization the effects of factor fixing
# define the variables for PCE
param_file = file_settings()[-1]
ind_vars, variables = variables_prep(param_file, product_uniform='uniform', dummy=False)
var_trans = AffineRandomVariableTransformation(variables, enforce_bounds=True)
param_names = pd.read_csv(param_file, usecols=[2]).values.flatten()

def sa_gp(fsave, gp, ind_vars, variables, cal_type='sampling', save_values=True):
    filename = f'{fsave}/rankings-sample.json'
    if not os.path.exists(filename):
        order = 2
        interaction_terms = pya.compute_hyperbolic_indices(len(ind_vars), order)
        interaction_terms = interaction_terms[:, np.where(
        interaction_terms.max(axis=0) == 1)[0]]
        if cal_type == 'sampling':
            sa = pyapprox.sampling_based_sobol_indices_from_gaussian_process(gp, 
                variables, interaction_terms=interaction_terms, nsamples=600, 
                    ngp_realizations=100, ninterpolation_samples = 500, nsobol_realizations = 10,
                    stat_functions=(np.mean, np.std))

            np.savez(filename, total_effects=sa['total_effects']['values'])
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
        if save_values:
            ST_mean = sa['total_effects']['mean']
            ST_mean = pd.DataFrame(data = ST_mean, index = param_names)
            ST_mean['std'] = sa['total_effects']['std']
            ST_mean.to_csv(f'{fsave}/ST.csv')

        with open(filename, 'w') as fp:
            json.dump(index_sort, fp, indent=2)

    else:
        with open(filename, 'r') as fp:
            index_sort_load = json.load(fp)
        index_sort = {}
        for k, v in index_sort_load.items():
            index_sort[int(k)] = index_sort_load[k]

    return index_sort


sa_gp(fpath+'sampling-sa/', gp, ind_vars, variables, cal_type='sampling', save_values=True)