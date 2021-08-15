#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pandas.io import json
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
from funcs.utils import partial_rank
from gp_pce_model import *
fpath = '../output/gp_run_0813/'
gp = pickle.load(open(f'{fpath}gp_1.pkl', "rb"))
x_training = gp.X_train_
y_training = gp.y_train_


gp1 = pickle.load(open(f'{fpath}gp_0.pkl', "rb"))
x1_training = gp1.X_train_
y1_training = gp1.y_train_

# histplot of y training values
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
y_plot = 1 - y_training
axes[0].hist(y_training.flatten())
axes[0].set_xlabel('Viney F')
axes[0].set_ylabel('Counts')
axes[0].set_title('(a)') 
axes[1].hist(y_training[y_training>-10].flatten())
axes[1].set_xlabel('Viney F')
axes[1].set_title('(b)') 
plt.savefig(f'{fpath}figs/training_vals', dpi=300)

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
filename = f'{file_settings()[0]}sa_gp.npz'
if not os.path.exists(filename):
    order = 2
    interaction_terms = pya.compute_hyperbolic_indices(len(ind_vars), order)
    interaction_terms = interaction_terms[:, np.where(
    interaction_terms.max(axis=0) == 1)[0]]
    sa = pyapprox.sampling_based_sobol_indices_from_gaussian_process(gp, 
        variable_temp, interaction_terms=interaction_terms, nsamples=1000, 
            ngp_realizations=10, ninterpolation_samples = 100, nsobol_realizations = 100)
    np.savez(filename, total_effects=sa['total_effects']['values'])
else:
    data = np.load(filename, allow_pickle=True)
    sa = data

ST_values = sa['total_effects']['values']
# reshape the matrix as N*P
ST = np.zeros(shape=(ST_values.shape[0] * ST_values.shape[2], ST_values.shape[1]))
for ii in range(ST_values.shape[2]):
        ST[ii*100:(ii+1)*100, :] = ST_values[:, :, ii].\
            reshape(ST_values.shape[0], ST_values.shape[1])
index_sort = partial_rank(ST, ST.shape[1], conf_level=0.95)

# define the order to fix parameters
from funcs.utils import dotty_plot, define_constants, fix_sample_set
dot_samples = generate_independent_random_samples(variable_temp, 100000)
dot_vals = gp.predict(dot_samples.T)
dot_samples = dot_samples[:, np.where(dot_vals > 0)[0]]
dot_vals = dot_vals[dot_vals>0]
samples_opt = dot_samples[:, np.where(dot_vals>0.382)[0]]
x_default = define_constants(samples_opt, stats = 'median')
param_names = pd.read_csv(param_file, usecols=[2]).values.flatten()
num_opt = []
vals_dict = {}
index_fix = np.array([], dtype=int)

for ii in range(max(index_sort.keys()), 0, -1):
    index_fix = np.append(index_fix, index_sort[ii])
    print(f'Fix {index_fix.shape[0]} parameters')
    print(f'index: {index_fix}')
    samples_fix = fix_sample_set(index_fix, dot_samples, x_default)
    vals_fix = np.zeros_like(dot_vals)
    # calculate with PCE 
    vals_fix = gp.predict(samples_fix.T)
        
    vals_dict[f'fix_{len(index_fix)}'] = vals_fix
    # select points statisfying the optima
    index_opt_fix = np.where(vals_fix.flatten() >= 0.382)
    num_opt.append(index_opt_fix[0].shape[0])
    samples_opt_fix = samples_fix[:, index_opt_fix[0]]
    vals_opt_fix = vals_fix[index_opt_fix]
                                                                                                                                                                                                                                                                                                                                                   
    # plot     
    fig = dotty_plot(samples_fix, vals_fix, samples_opt_fix, vals_opt_fix, param_names, 'Viney F', orig_x_opt=dot_samples, orig_y_opt=dot_vals);
    plt.savefig(f'{fpath}/{len(index_fix)}.png', dpi=300)
