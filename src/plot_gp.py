#!/usr/bin/env python
from numba.core.typing.templates import BaseRegistryLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json
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


def plot(gp, fpath, plot_range='full', save_vali=False):
    gp1 = pickle.load(open(f'../output/gp_run_0819/gp_0.pkl', "rb"))
    x1_training = gp1.X_train_
    y1_training = gp1.y_train_
    if plot_range == 'full':
        y_hat = gp.predict(x1_training)[50:150]
        y_eval = y1_training[50:150]
    else:
        y_hat = gp.predict(x1_training)
        x_hat = x1_training[np.where(y_hat>0)[0][0:100], :]
        y_eval = y1_training[y_hat>0][0:100]
        y_hat = y_hat[y_hat>0][0:100]

    np.linalg.norm(y_hat.flatten() - y_eval.flatten()) / np.linalg.norm(y_eval.flatten())
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(y_eval.flatten(), y_hat.flatten(), linestyle='', marker='o', ms=8)
    ax.set_xlabel('Modelled F')
    ax.set_ylabel('GPR simulation')
    y_eval_opt = y_eval[y_eval>0.382]
    y_hat_opt = y_hat[y_eval>0.382]
    ax.plot(y_eval_opt.flatten(), y_hat_opt.flatten(), linestyle='', 
        marker='o', color='darkorange', alpha=0.7, ms=8)
    ax.plot(np.linspace(y_eval.min(), 0.8, 100), np.linspace(y_eval.min(), 0.8, 100), 
        linestyle='--', color='slategrey', alpha=0.5)
    plt.savefig(f'{fpath}figs/gpr_validation_full_range', dpi=300)

    # save validation samples
    if save_vali:
        vali_samples = np.append(x_hat, y_eval.reshape(y_eval.shape[0], 1), axis=1)
        np.savetxt('validation_samples.txt', vali_samples)
# END plot()     

def vali_samples_subreg(gp, variable, num_candidate_samples=2000):
    candidates_samples = generate_gp_candidate_samples(gp, gp.X_train_.shape[1], 
        num_candidate_samples = num_candidate_samples, 
            generate_random_samples=None, variables=variable)
    
    y_pred = gp.predict(candidates_samples)
    samples_vali_subreg = candidates_samples[y_pred[y_pred>0][0:100], :]
    samples_vali_full = candidates_samples[0:100, :]
    vali_samples = np.zeros(shape=(200, 14))
    vali_samples[0:100, 0:13] = samples_vali_subreg
    vali_samples[100:200, 0:13] = samples_vali_full
    vali_samples[0:100, 13] = y_pred[y_pred>0][0:100]
    vali_samples[100:200, 13] = y_pred[0:100]
    return vali_samples
# END vali_samples_subreg()

# Calculate the ratio of samples in the subregion
def ratio_subreg(gp):
    y_training = gp.y_train_
    num_new_samples = np.asarray([0]+[20]+[8]*10+[16]*20+[24]*16+[40]*14)
    num_samples = np.cumsum(num_new_samples)
    ratio_samples = np.zeros(shape=(num_new_samples.shape[0]-2, 2))
    ratio_sum = 0
    for ii in range(num_new_samples.shape[0] - 2):
        num_subreg = np.where(y_training[num_samples[ii]: num_samples[ii+1]]>0)[0].shape[0]
        ratio_sum = ratio_sum + num_subreg
        ratio_samples[ii, 0] = num_subreg / num_new_samples[ii+1]
        ratio_samples[ii, 1] = ratio_sum / num_samples[ii+1]

    ratio_df = pd.DataFrame(data=ratio_samples, 
        index=np.arange(ratio_samples.shape[0]), columns=['Subregion', 'FullSpcace'])
    ratio_df['num_samples'] = num_samples[1:-1]
    return ratio_df
# END ratio_subreg()

# define the order to fix parameters
def fix_plot(gp, variable_temp, plot_range='full'):
    from funcs.utils import dotty_plot, define_constants, fix_sample_set
    dot_fn = f'{file_settings()[0]}gp_run_0816/dotty_samples.txt'
    if not os.path.exists(dot_fn):
        dot_samples = generate_independent_random_samples(variable_temp, 1000000)
        np.savetxt(dot_fn, dot_samples)
    else:
        dot_samples = np.loadtxt(dot_fn)
    dot_vals = np.zeros(shape=(dot_samples.shape[1], 1))
    for ii in range(100):
        dot_vals[10000*ii:(ii+1)*10000] = gp.predict(dot_samples[:, 10000*ii:(ii+1)*10000].T)
    # dot_samples = dot_samples[:, np.where(dot_vals >= 0.382)[0]]
    # dot_vals = dot_vals[dot_vals>=0.382]
    
    samples_opt = dot_samples[:, np.where(dot_vals>0.382)[0]]
    vals_opt = dot_vals[dot_vals>0.382]
    print(f'Number of values beyond the threshold: {samples_opt.shape[0]}')
    if plot_range == 'full_median':
        x_default = define_constants(dot_samples, 13, stats = np.median)
        fig_path = 'fix_median_full'
    elif plot_range == 'sub_median':
        x_default = define_constants(samples_opt, 13, stats = np.median)
        fig_path = 'fix_median_subreg'
    elif plot_range == 'sub_mean':
        x_default = define_constants(samples_opt, 13, stats = np.mean)
        fig_path = 'fix_mean_subreg'
    elif plot_range == 'sub_rand':
        x_default = dot_samples[:, np.where(dot_vals>0.382)[0]][:, 10]
        fig_path = 'fix_rand_subreg'
    else:
        AssertionError

    y_default = gp.predict(x_default.reshape(x_default.shape[0], 1).T)[0]
    print(f'F of the point with default values: {y_default}')
    x_default = np.append(x_default, y_default)
    np.savetxt(f'{fpath}/{fig_path}/fixed_values.txt', x_default)

    num_opt = []
    vals_dict = {}
    index_fix = np.array([], dtype=int)
    pct_optimal = {}

    for ii in range(max(index_sort.keys()), 0, -1):
        index_fix = np.append(index_fix, index_sort[ii])
        print(f'Fix {index_fix.shape[0]} parameters')
        print(f'index: {index_fix}')
        samples_fix = fix_sample_set(index_fix, samples_opt, x_default)
        vals_fix = np.zeros_like(vals_opt)
        # calculate with surrogate 
        vals_fix = gp.predict(samples_fix.T)
        # for ii in range(100):
        #     vals_fix[10000*ii:(ii+1)*10000] = gp.predict(samples_fix[:, 10000*ii:(ii+1)*10000].T)
                    
        # select points statisfying the optima
        index_opt_fix = np.where(vals_fix.flatten() >= 0.382)[0]
        samples_opt_fix = samples_fix[:, index_opt_fix]
        vals_opt_fix = vals_fix[index_opt_fix]
        vals_dict[f'fix_{len(index_fix)}'] = vals_fix.flatten()
                                                                                                                                                                                                                                                                                                                                                    
        # plot     
        fig = dotty_plot(samples_opt, vals_opt.flatten(), samples_opt_fix, vals_opt_fix.flatten(), 
            param_names, 'F', orig_x_opt=samples_fix, orig_y_opt=vals_fix);
        plt.savefig(f'{fpath}/{fig_path}/{len(index_fix)}.png', dpi=300)
        # calculate the ratio of optimal values
        pct_optimal[f'fix_{len(index_fix)}'] = vals_opt_fix.shape[0] / dot_vals.shape[0]
    # # END For

    pct_optimal = pd.DataFrame.from_dict(pct_optimal, orient='index', columns=['Proportion'])
    pct_optimal.to_csv(f'{file_settings()[0]}gp_run_0816/{fig_path}/Proportion_optimal.csv')
    # PDF plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)
    sns.distplot(vals_opt.flatten(), hist=False,  ax=axes[0])
    k = 0
    df_stats = pd.DataFrame(columns=['mean', 'std', 'qlow','qup'])
    df_stats.loc['full_set', ['mean', 'std']] = [vals_opt.mean(), vals_opt.std()]
    df_stats.loc['full_set', 'qlow':'qup'] = np.quantile(vals_opt, [0.025, 0.957])
    for key, value in vals_dict.items():
        sns.distplot(value.flatten(), hist=False, ax=axes[k//4]);
        df_stats.loc[key, 'mean'] = value.mean()
        df_stats.loc[key, 'std'] = value.std()
        df_stats.loc[key, 'qlow':'qup'] = np.quantile(value, [0.025, 0.975])
        k += 1

    axes[0].legend(['full_set', *list(vals_dict.keys())[0:4]])
    axes[1].set_xlabel('F')
    axes[1].set_ylabel('')
    axes[1].legend(list(vals_dict.keys())[4:8])
    axes[2].legend(list(vals_dict.keys())[8:])
    axes[2].set_ylabel('')
    for ii in range(3):
        axes[ii].axvline(0.382,  color='grey', linestyle='--', alpha=0.7)
    plt.savefig(f'{fpath}/{fig_path}/objective_dist.png', dpi=300)

    # Box plot
    # breakpoint()
    fig2 = plt.figure(figsize=(8, 6))
    df = pd.DataFrame.from_dict(vals_dict)
    df['fix_0'] = vals_opt.flatten()
    df.columns = [*np.arange(1, 13), 0]
    df = df[np.arange(13)]
    ax = sns.boxplot(data=df, saturation=0.5, linewidth=1, whis=0.5)
    ax.axhline(0.382,  color='orange', linestyle='--', alpha=1 , linewidth=1)
    ax.set_xlabel('Number of fixed parameters')
    ax.set_ylabel('F')
    plt.savefig(f'{fpath}{fig_path}/boxplot.png')
    df_stats.to_csv(f'{fpath}{fig_path}/F_stats.csv')
# END fix_plot()


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
filename = f'{fpath}rankings.json'
param_names = pd.read_csv(param_file, usecols=[2]).values.flatten()
if not os.path.exists(filename):
    order = 2
    interaction_terms = pya.compute_hyperbolic_indices(len(ind_vars), order)
    interaction_terms = interaction_terms[:, np.where(
    interaction_terms.max(axis=0) == 1)[0]]
    sa = pyapprox.analytic_sobol_indices_from_gaussian_process(gp, variables, 
        interaction_terms, ngp_realizations=100, ninterpolation_samples=500, 
            use_cholesky=True, ncandidate_samples=10000, nvalidation_samples=200,
            stat_functions=(np.mean, np.std))
    np.savez(filename, total_effects=sa['total_effects']['values'])
    ST = sa['total_effects']['values']
    ST_mean = sa['total_effects']['mean']
    ST_mean = pd.DataFrame(data = ST_mean, index = param_names)
    ST_mean['std'] = sa['total_effects']['std']
    ST_mean.to_csv(f'{fpath}ST.csv')
    index_sort = partial_rank(ST, ST.shape[1], conf_level=0.95)
    with open(filename, 'w') as fp:
        json.dump(index_sort, fp, indent=2)

else:
    with open(filename, 'r') as fp:
        index_sort_load = json.load(fp)
    index_sort = {}
    for k, v in index_sort_load.items():
        index_sort[int(k)] = index_sort_load[k]

# Calculate the ratio of calibrating samples in the sub-region
if not os.path.exists(f'{fpath}ratio_cali_subreg.csv'):
    df = ratio_subreg(gp)
    df.to_csv(f'{fpath}ratio_cali_subreg.csv')

# Scatter plot and Pdf plot VS fixing parameters
fix_plot(gp, variable_temp, plot_range='sub_rand')