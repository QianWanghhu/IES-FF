#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns

from scipy import stats
# from scipy.optimize import root
from pyapprox import generate_independent_random_samples
import matplotlib as mpl
from scipy import stats
from scipy.stats import spearmanr

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
    """
    Function used to plot the figures of GP validation.
    Parameters:
    ===========
    gp: Gaussian Process object
    fpath: str, path to save figures
    plot_range: str, defining the set of validation samples to use.
                Use global samples if "full", else local. Default is "full".
    save_vali: Bool, save figures if true. Default is False.

    """
    gp1 = pickle.load(open(f'../output/gp_run_0816/gp_0.pkl', "rb"))
    plot_range = 'sub'
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

    l2 = np.linalg.norm(y_hat.flatten() - y_eval.flatten()) / np.linalg.norm(y_eval.flatten())
    r2 = 1 - l2 ** 2
    rmse = np.linalg.norm(y_eval - y_eval.mean(), ord=2) / y_eval.shape[0] ** 0.5
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(y_eval.flatten(), y_hat.flatten(), linestyle='', marker='o', ms=8)
    ax.set_xlabel('Modeled F')
    ax.set_ylabel('GP simulation')
    y_eval_opt = y_eval[y_eval>0.382]
    y_hat_opt = y_hat[y_eval>0.382]
    ax.plot(y_eval_opt.flatten(), y_hat_opt.flatten(), linestyle='', 
        marker='o', color='darkorange', alpha=0.7, ms=8)
    ax.plot(np.linspace(y_eval.min(), 0.8, 100), np.linspace(y_eval.min(), 0.8, 100), 
        linestyle='--', color='slategrey', alpha=0.5)
    # ax.text(-950, -100, r'$R^2 = %.3f$'%r2)
    # ax.text(-950, -200, r'$RMSE = %.3f$'%rmse)
    ax.text(0.05, 0.75, r'$R^2 = %.3f$'%r2)
    ax.text(0.05, 0.65, r'$RMSE = %.3f$'%rmse)
    # plt.savefig(f'{fpath}figs/gpr_validation_{plot_range}_range_text.png', dpi=300)

    # save validation samples
    if save_vali:
        vali_samples = np.append(x_hat, y_eval.reshape(y_eval.shape[0], 1), axis=1)
        np.savetxt('validation_samples.txt', vali_samples)
# END plot()     

def vali_samples_subreg(gp, variable, num_candidate_samples=20000):
    """
    Function used to generate validation samples.
    """
    candidates_samples = generate_gp_candidate_samples(gp.X_train_.shape[1], 
        num_candidate_samples = num_candidate_samples, 
            generate_random_samples=None, variables=variable)
    
    y_pred = gp.predict(candidates_samples.T)
    samples_vali_subreg1 = candidates_samples[:, np.where(y_pred>0.382)[0][0:20]]
    samples_vali_subreg2 = candidates_samples[:, np.where(y_pred>0)[0]]
    y_sub1 = gp.predict(samples_vali_subreg2.T)
    samples_vali_subreg2 = samples_vali_subreg2[:, np.where(y_sub1<=0.382)[0][0:80]]
    samples_vali_full = candidates_samples[:, 0:100]
    vali_samples = np.zeros(shape=(14, 200))
    vali_samples[0:13, 0:20] = samples_vali_subreg1
    vali_samples[0:13, 20:100] = samples_vali_subreg2
    vali_samples[0:13, 100:200] = samples_vali_full
    vali_samples[13, 0:20] = gp.predict(samples_vali_subreg1.T).flatten()
    vali_samples[13, 20:100] = gp.predict(samples_vali_subreg2.T).flatten()
    vali_samples[13, 100:200] = y_pred[0:100].flatten()
    return vali_samples
# END vali_samples_subreg()


# Calculate the ratio of samples in the subregion
def ratio_subreg(gp):
    """
    Function to calculate the ratio of samples in the subregion in the adaptive procedure.
    Parameters:
    ===========
    gp: Gaussian Process object

    Return:
    =======
    ration_df: pd.DataFrame, dataframe of the ratios at each iteration.
    """
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

from funcs.utils import define_constants
def choose_fixed_point(plot_range, dot_samples, samples_opt, dot_vals):
    """
    Function used to set the nomial point for fixing parameters at.
    Parameters:
    ===========
    plot_range: str, decide which type of nomial values to use.
    dot_samples: np.ndarray, of shape D*N where D is the number of parameters, 
                the initial parameter samples for calculation objective functions
    samples_opt: np.ndarray, of shape D*M where D is the number of parameters,
                parameter samples resulting in objective functions above the threshold
    dot_vals: np.ndarray, objective function values from dot_samples

    Return:
    ===========
    x_default: list, the nominal values for all D parameters
    fig_path: str, the dir defined by the type of nominal values for results to save
    """
    if plot_range == 'full_mean':
        x_default = define_constants(dot_samples, 13, stats = np.mean)
        fig_path = 'fix_median_full'
    elif plot_range == 'sub_median':
        samples_opt = dot_samples[:, np.where(dot_vals>0.382)[0]]
        x_default = define_constants(samples_opt, 13, stats = np.median)
        fig_path = 'fix_median_subreg'
    elif plot_range == 'sub_mean':
        samples_opt = dot_samples[:, np.where(dot_vals>0.382)[0]]
        x_default = define_constants(samples_opt, 13, stats = np.mean)
        fig_path = 'fix_mean_subreg'
    elif plot_range == 'sub_rand':
        x_default = dot_samples[:, np.where(dot_vals>0.382)[0]][:, 10]
        fig_path = 'fix_rand_subreg'
    elif plot_range == 'sub_max':
        x_default = dot_samples[:, np.where(dot_vals>=dot_vals.max())[0]]
        fig_path = 'fix_max_subreg'
    else:
        AssertionError
    return x_default, fig_path

def cal_stats(vals_opt, vals_dict, re_eval):
    """
    Function used to calculate the statstics of the objective values VS parameter fixing.
    
    Parameters:
    ===========
    vals_dict: dict, containing the objective function values with parameters being fixed
    vals_opt: np.ndarray, objective function values used to calculate the statistics
    re_eval: Bool, re-evaluate the OBJ using the whole samples if True, 
                    else using the optimal set only for parameter fixing

    Return:
    ===========
    df_stats: pd.DataFrame, of statistics
    """
    # PDF plot
    df_stats = pd.DataFrame(columns=['mean', 'std', 'qlow','qup'])
    if re_eval:
        df_stats.loc['full_set', ['mean', 'std']] = [vals_opt[vals_opt>0.382].mean(), vals_opt[vals_opt>0.382].std()]
        df_stats.loc['full_set', 'qlow':'qup'] = np.quantile(vals_opt[vals_opt>0.382], [0.025, 0.957])
    else:
        df_stats.loc['full_set', ['mean', 'std']] = [vals_opt.mean(), vals_opt.std()]
        df_stats.loc['full_set', 'qlow':'qup'] = np.quantile(vals_opt, [0.025, 0.957])

    for key, value in vals_dict.items():
        if key != 'fix_13':
            if re_eval:
                value = value[value>0.382]

            df_stats.loc[key, 'mean'] = value.mean()
            df_stats.loc[key, 'std'] = value.std()
            df_stats.loc[key, 'qlow':'qup'] = np.quantile(value, [0.025, 0.975])

    return df_stats

def cal_prop_optimal(vals_dict, dot_vals, fig_path):
    """
    Used to calculate the ratio of optimal values.
    Parameters:
    ===========
    fig_path: str, dir to save the result formed into a pd.DataFrame
    """
    pct_optimal = {}
    for key, value in vals_dict.items():
        pct_optimal[key] = value[value>0.382].shape[0] / dot_vals.shape[0]
    pct_optimal = pd.DataFrame.from_dict(pct_optimal, orient='index', columns=['Proportion'])
    pct_optimal.to_csv(f'{fig_path}/Proportion_optimal.csv')
# END cal_prop_optimal()

def plot_pdf(vals_opt, vals_dict, re_eval, fig_path):
    """
    Used to generate the plot of probability distribution function.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)
    sns.distplot(vals_opt.flatten(), hist=False,  ax=axes[0])

    k = 0
    for key, value in vals_dict.items():    
        if key != 'fix_13':
            if re_eval:
                value = value[value>0.382]
            sns.distplot(value.flatten(), hist=False, ax=axes[k//4]);
            k += 1

    axes[0].legend(['full_set', *list(vals_dict.keys())[0:4]])
    axes[1].set_xlabel('F')
    axes[1].set_ylabel('')
    axes[1].legend(list(vals_dict.keys())[4:8])
    axes[2].legend(list(vals_dict.keys())[8:])
    axes[2].set_ylabel('')
    for ii in range(3):
        axes[ii].axvline(0.382,  color='grey', linestyle='--', alpha=0.7)
    plt.savefig(f'{fig_path}/objective_dist.png', dpi=300)

def box_plot(vals_dict, vals_opt, num_fix, fig_path):
    """
    Used to generate the boxplot of objective values.
    """
    fig2 = plt.figure(figsize=(8, 6))
    df = pd.DataFrame.from_dict(vals_dict)
    df['fix_0'] = vals_opt.flatten()
    df.columns = [*num_fix, 0]
    df = df[[0, *num_fix]]
    # if re_eval:
    #     df_filter = df.where(df>0.382)
    # else:
    df_filter = df

    ax = sns.boxplot(data=df_filter, saturation=0.5, linewidth=1, whis=0.5)
    ax.axhline(1/(2 - 0.382),  color='orange', linestyle='--', alpha=1 , linewidth=1)
    ax.set_xlabel('Number of fixed parameters')
    ax.set_ylabel('1/(2-F)')
    # ax.set_ylim(0.2, 1.4)
    plt.savefig(f'{fig_path}/boxplot_full_norm.png', dpi=300)

def spr_coef(dot_samples, dot_vals, fsave):
    """
    Calculate the spearman-rank correlation.
    """
    samples_opt = dot_samples[:, np.where(dot_vals>0.382)[0]]
    coef_dict = pd.DataFrame(index=np.arange(0, 13), columns=np.arange(0, 13))
    p_dict = pd.DataFrame(index=np.arange(0, 13), columns=np.arange(0, 13))
    for ii in range(13):
        for jj in range(ii+1, 13):
            coef_dict.loc[ii, jj], p_dict.loc[ii, jj] = spearmanr(samples_opt[ii], samples_opt[jj])
    coef_dict.to_csv(fsave+'spearman_coeff.csv')
    p_dict.to_csv(fsave+'spearman_p.csv')

# define the order to fix parameters
def fix_plot(gp, variables, fsave, param_names, ind_vars, sa_cal_type, 
    plot_range='full', re_eval=False, norm_y=False):
    """
    Used to fix parameter sequentially and obtaining unconditional outputs,
    as well as boxplot and scatterplots.
    Parameters:
    ===========
    gp: Gaussian Process object
    variables: variable
    fsave: the outer dir for saving results of, for example, spearman correlation
    param_names: list, parameter names
    ind_vars: individual parameter variable
    sa_cal_type: str, the type of SA to conduct. Should be from ['analytic', 'sampling']
    plot_range: str, defining the set of validation samples to use.
                Use global samples if "full", else local. Default is "full".
    re_eval: Bool
    norm_y: Bool, whether to normalize objective functions when sensitive analysis
    
    Return:
    ========
    dot_vals: np.ndarray, objective function values from dot_samples
    vals_dict: dict, containing the objective function values with parameters being fixed
    index_fix: list, the ordered index of fixed parameters
    """
    from funcs.utils import fix_sample_set, dotty_plot

    dot_fn = f'{file_settings()[0]}gp_run_0816/dotty_samples2.txt'
    if not os.path.exists(dot_fn):
        dot_samples = generate_independent_random_samples(variables, 200000)
        np.savetxt(dot_fn, dot_samples)
    else:
        dot_samples = np.loadtxt(dot_fn)
    dot_vals = np.zeros(shape=(dot_samples.shape[1], 1))
    for ii in range(20):
        dot_vals[10000*ii:(ii+1)*10000] = gp.predict(dot_samples[:, 10000*ii:(ii+1)*10000].T)
    
    # Whether to re-evaluate the optimal values.
    if re_eval:
        samples_opt = dot_samples 
        vals_opt = dot_vals 
    else:
        samples_opt = dot_samples[:, np.where(dot_vals>0.382)[0]]
        vals_opt = dot_vals[dot_vals>0.382]

    # Calculate the Spearman correlation between parameters
    spr_coef(dot_samples, dot_vals, fsave)
    
    # Choose the fixed values
    print(f'Number of values beyond the threshold: {samples_opt.shape[0]}')
    x_default, fig_path = choose_fixed_point(plot_range, dot_samples, samples_opt, dot_vals)
    fig_path = fsave + fig_path
    y_default = gp.predict(x_default.reshape(x_default.shape[0], 1).T)[0]
    print(f'F of the point with default values: {y_default}')
    x_default = np.append(x_default, y_default)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    np.savetxt(f'{fig_path}/fixed_values.txt', x_default)

    # calculate / import parameter rankings
    from sensitivity import sa_gp
    if sa_cal_type == 'analytic':
        vars = variables_full
    else:
        vars = variable_temp
    _, ST = sa_gp(fsave, gp, ind_vars, vars, param_names, 
        cal_type=sa_cal_type, save_values=True, norm_y=norm_y)
    par_rank = np.argsort(ST['ST'].values)
    index_sort = {ii:par_rank[12-ii] for ii in range(13)}

    num_fix = []
    vals_dict = {}
    index_fix = np.array([], dtype=int)
    for ii in range(max(index_sort.keys()), -1, -1):
        index_fix = np.append(index_fix, index_sort[ii])
        num_fix.append(index_fix.shape[0])
        print(f'Fix {index_fix.shape[0]} parameters')
        print(f'index: {index_fix}')
        samples_fix = fix_sample_set(index_fix, samples_opt, x_default)
        vals_fix = np.zeros_like(vals_opt)
    
        # calculate with surrogate 
        if re_eval == True:
            for ii in range(20):
                vals_fix[10000*ii:(ii+1)*10000] = gp.predict(samples_fix[:, 10000*ii:(ii+1)*10000].T)
        else:
            vals_fix = gp.predict(samples_fix.T)

        # select points statisfying the optima
        index_opt_fix = np.where(vals_fix.flatten() >= 0.382)[0]
        samples_opt_fix = samples_fix[:, index_opt_fix]
        vals_opt_fix = vals_fix[index_opt_fix]
        vals_dict[f'fix_{len(index_fix)}'] = vals_fix.flatten()                                                                                                                                                                                                                                                                                                                                         
        # plot     
        index_opt = np.where(vals_opt.flatten() >= 0.382)[0]
        samples_opt_no_fix = samples_opt[:, index_opt]
        vals_opt_no_fix = vals_opt[index_opt]
        # fig = dotty_plot(samples_opt_no_fix, vals_opt_no_fix.flatten(), samples_opt_fix, vals_opt_fix.flatten(), 
        #     param_names, 'F'); #, orig_x_opt=samples_fix, orig_y_opt=vals_fix
        # plt.savefig(f'{fig_path}/{len(index_fix)}.png', dpi=300)

    # cal_prop_optimal(vals_dict, dot_vals, fig_path)
    # Calculate the stats of objectives vs. Parameter Fixing
    df_stats = cal_stats(vals_opt, vals_dict, re_eval)
    df_stats.to_csv(f'{fig_path}/F_stats.csv')
    # pdf plot
    # plot_pdf(vals_opt, vals_dict, re_eval, fig_path)

    # Box plot
    # normalize the vals in vals_dict so as to well distinguish the feasible F.
    vals_dict_norm = {}
    for key, v in vals_dict.items():
        vals_dict_norm[key] = 1 / (2 - v)
    vals_opt_norm = 1 / (2 - vals_opt)
    box_plot(vals_dict_norm, vals_opt_norm, num_fix, fig_path, re_eval)
    return dot_vals, vals_dict, index_fix
 # END fix_plot()

# import GP
if __name__ == '__main__':
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
    ind_vars, variables_full = variables_prep(param_file, product_uniform='uniform', dummy=False)
    var_trans = AffineRandomVariableTransformation(variables_full, enforce_bounds=True)
    param_names = pd.read_csv(param_file, usecols=[2]).values.flatten()

    # Calculate the ratio of calibrating samples in the sub-region
    # if not os.path.exists(f'{fpath}ratio_cali_subreg.csv'):
    #     df = ratio_subreg(gp)
    #     df.to_csv(f'{fpath}ratio_cali_subreg.csv')

    # Calculate results with and create plots VS fixing parameters
    fsave = fpath + 'analytic-sa/'
    norm_y = False
    vals_fix_dict = {}
    # dot_vals, vals_fix_dict['full_mean'] = fix_plot(gp, variable_temp, fsave, param_names, ind_vars, 'sampling', 
    #     plot_range='full_mean', re_eval=True, norm_y = True)
    dot_vals, vals_fix_dict['sub_mean'], index_fix = fix_plot(gp, variable_temp, fsave, param_names, 
        ind_vars, 'analytic', plot_range='sub_mean', re_eval=True, norm_y = norm_y)
    _, vals_fix_dict['sub_rand'], _  = fix_plot(gp, variable_temp, fsave, param_names, 
        ind_vars, 'analytic', plot_range='sub_rand', re_eval=True, norm_y = norm_y)
    _, vals_fix_dict['sub_max'], _  = fix_plot(gp, variable_temp, fsave, param_names, 
        ind_vars, 'analytic', plot_range='sub_max', re_eval=True, norm_y = norm_y)

    # validation plot
    # vali_samples = vali_samples_subreg(gp, variable_temp, 20000)
    # np.savetxt(f'{fpath}vali_samples.txt', vali_samples)
    # plot(gp, fpath, plot_range='sub', save_vali=False)
