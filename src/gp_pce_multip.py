#!/usr/bin/env python
from multiprocessing import Pool
from sys import breakpointhook
import numpy as np
import os
import matplotlib.pyplot as plt
from functools import partial
import time
import copy
import pandas as pd
import pickle
import pyapprox

from scipy import stats
# from scipy.optimize import root
from scipy.optimize import bisect

from sklearn.gaussian_process.kernels import RBF, \
    Matern

from pyapprox.density import tensor_product_pdf
from pyapprox.gaussian_process import CholeskySampler, AdaptiveGaussianProcess, generate_candidate_samples
from pyapprox.low_discrepancy_sequences import transformed_halton_sequence
from pyapprox.utilities import compute_f_divergence, \
    get_tensor_product_quadrature_rule
from pyapprox.probability_measure_sampling import generate_independent_random_samples_deprecated, rejection_sampling
from pyapprox.visualization import get_meshgrid_function_data
from pyapprox import generate_independent_random_samples
from pyapprox.variables import IndependentMultivariateRandomVariable
from pyapprox.variable_transformations import AffineRandomVariableTransformation

import matplotlib as mpl
from matplotlib import rc
import spotpy as sp

from funcs.read_data import variables_prep, file_settings
from funcs.modeling_funcs import vs_settings, \
        modeling_settings, paralell_vs, obtain_initials, change_param_values


mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['text.usetex'] = False  # use latex for all text handling
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.format'] = 'pdf'  # gives best resolution plots
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 16
# print mpl.rcParams.keys()
mpl.rcParams['text.latex.preamble'] = \
    r'\usepackage{siunitx}\usepackage{amsmath}\usepackage{amssymb}'

# read pces and use them for prediction
vs_list = []
def run_source_annual(vars, vs_list):
    pass

pce_names = [f'../output/old_pce/pce-{i}-level4.pkl' for i in range(2009, 2018)]
pce_list = []
for fn in pce_names:
    pce_list.append(pickle.load(open(fn, "rb")))


def run_source_lsq(vars, pce_list=pce_list):
    """
    Script used to run_source and return the output file.
    The function is called by AdaptiveLejaPCE.
    """
    
    # Use annual or monthly loads
    def timeseries_sum(df, temp_scale = 'annual'):
        """
        Obtain the sum of timeseries of different temporal scale.
        temp_scale: str, default is 'Y', monthly using 'M'
        """
        assert temp_scale in ['monthly', 'annual'], 'The temporal scale given is not supported.'
        if temp_scale == 'monthly':
            sum_126001A = df.resample('M').sum()
        else:
            month_126001A = df.resample('M').sum()
            sum_126001A = pd.DataFrame(index = np.arange(df.index[0].year, df.index[-1].year), 
                columns=df.columns)
            for i in range(sum_126001A.shape[0]):
                sum_126001A.iloc[i, :] = month_126001A.iloc[i*12: (i+1)*12, :].sum()

        return sum_126001A
    # End timeseries_sum()

    def viney_F(evaluation, simulation):
        pb = sp.objectivefunctions.pbias(evaluation, simulation) / 100
        nse = sp.objectivefunctions.nashsutcliffe(evaluation, simulation)
        F = nse - 5 *( np.abs(np.log(1 + pb)))**2.5
        return F

    # Define functions for the objective functions
    def cal_obj(x_obs, x_mod, obj_type = 'nse'):
        obj_map = {'nse': sp.objectivefunctions.nashsutcliffe,
                    'rmse': sp.objectivefunctions.rmse,
                    'pbias': sp.objectivefunctions.pbias,
                    'viney': viney_F
                }
        obj = []
        for  k in range(x_mod.shape[1]):
            obj.append(obj_map[obj_type](x_obs, x_mod[:, k].reshape(x_mod.shape[0], 1)))
        # if obj[0] == 0: obj[0] = 1e-8
        obj = np.array(obj)
        if obj_type =='pbias':
            obj = obj / 100
        obj = obj.reshape(obj.shape[0], 1)
        print(obj)
        return obj
    # End cal_obj
    
    # rescale samples to the absolute range
    vars_copy = copy.deepcopy(vars)
    vars_copy[0, :] = vars_copy[0, :] * 100
    vars_copy[1, :] = vars_copy[1, :] * 100
    # import observation if the output.txt requires the use of obs.
    date_range = pd.to_datetime(['2009/07/01', '2018/06/30'])
    observed_din = pd.read_csv(f'{file_settings()[1]}126001A.csv', index_col='Date')
    observed_din.index = pd.to_datetime(observed_din.index)
    observed_din = observed_din.loc[date_range[0]:date_range[1], :].filter(items=[observed_din.columns[0]]).apply(lambda x: 1000 * x)
    
    # obtain the sum at a given temporal scale
    obs_din = timeseries_sum(observed_din, temp_scale = 'annual')
    obs_din = pd.DataFrame(obs_din,dtype='float').values
    
    din_126001A = np.zeros((len(pce_list), vars_copy.shape[1]))
    for ii in range(len(pce_list)):
        din_126001A[ii, :] = pce_list[ii](vars_copy).flatten()

    obj = cal_obj(obs_din, din_126001A, obj_type = 'viney')
    print(f'Finish {obj.shape[0]} run')

    # calculate the objective NSE and PBIAS
    obj_nse = cal_obj(obs_din, din_126001A, obj_type = 'nse')
    obj_pbias = cal_obj(obs_din, din_126001A, obj_type = 'pbias')
    train_iteration = np.append(vars, obj_nse.T, axis=0)
    train_iteration = np.append(train_iteration, obj_pbias.T, axis=0)
    # save sampling results of NSE and PBIAS
    train_file = 'training_samples.txt'
    if os.path.exists(train_file):
        train_samples = np.loadtxt(train_file)
        train_samples = np.append(train_samples, train_iteration, axis=1)
        np.savetxt(train_file, train_samples)
    else:
        np.savetxt(train_file, train_iteration) 
    # END if-else

    return obj
# END run_source_lsq()

def resample_candidate(gp, sampler, thsd, num_samples, gp_ob1=None, gp_ob2=None):

    def filter_samples(gp_surrogate, num_candidates, samples, threshold, num_samples, return_std=True, obj = 'nse'):
        """
        This is used to filter the samples that satisfy the constraint, 
            e.g., objective values > a certain threshold.
        """
        # check whether to return the standard deviation for the uncertainty
        if return_std: 
            breakpoint()
            y_hat, y_std = gp_surrogate.predict(samples.T, return_std)
        else:
            y_hat = gp_surrogate.predict(samples.T, return_std)
            y_std = np.zeros_like(y_hat)
        # End if-else
        
        # if the objective function is PBIAS, use both the upper bound and lower bound to constrain
        if obj != 'pbias':
            # breakpoint()
            y_upper = y_hat.flatten() + 1.96 * y_std
            index_temp = np.where(y_upper > threshold)[0]
            threshold_temp = np.sort(y_upper, axis=0)[-(num_candidates - gp_surrogate.y_train_.shape[0] + 1)]
            if ((index_temp.shape[0] < num_samples[-1]) | (threshold_temp < 0)):
                index_temp = np.where(y_upper > threshold_temp)[0]
            else:
                # threshold = 0
                index_temp = np.where(y_upper > threshold_temp)[0]
        else: 
            cond1 = ((y_hat.flatten() + 1.96 * y_std) < threshold)
            cond2 = ((y_hat.flatten() - 1.96 * y_std) > -threshold)
            index_temp = np.where(cond1 & cond2)[0]
        
        return index_temp, threshold_temp
        # END filter_samples  
    
    y_temp, y_temp_std = gp.predict(sampler.candidate_samples.T, return_std=True)
    y_temp_upper =  y_temp.flatten() + 1.96 * y_temp_std
    # if sampler.candidate_samples.shape[1] > 1000:
    if np.sort(y_temp_upper, axis=0)[-200] > 0:
        index_temp = np.where(y_temp_upper > 0)[0]
    else:
        index_temp = np.where(y_temp_upper > np.sort(y_temp_upper, axis=0)[-100])[0]

    x_select = sampler.candidate_samples[:, index_temp]
    
    x_max = x_select.max(axis=1)
    print(f'------------Narrowing parameter ranges to {x_max}------------')

    assert len(thsd) == 3, "The first dimension of thsd should be 3."
    univariable_temp = [stats.uniform(0, x_max[ii]) for ii in range(0, x_max.shape[0])]
    variable_temp = pyapprox.IndependentMultivariateRandomVariable(univariable_temp)
    new_candidates = generate_candidate_samples(sampler.candidate_samples.shape[0],
        num_candidate_samples = 40000, 
            generate_random_samples=None, 
                variables=variable_temp)
    
    obj_list = ['viney', 'nse', 'pbias'][0:1]
    gp_all = [gp, gp_ob1, gp_ob2 ][0:1]
    thsd = thsd[0:1]
    index_satis = {}
    thsd_dict = {}
    # find the sample index that satisfy each criteria
    for gp_model, threshold, objective in zip(gp_all, thsd, obj_list):
        index_satis[objective], thsd_dict[objective] = filter_samples(gp_model, 
            sampler.candidate_samples.shape[1], new_candidates, 
                threshold, num_samples, return_std=True, obj = objective)
    # breakpoint()
    # find the common index filtered by the three constraints
    # index_intersect = np.intersect1d(index_satis[obj_list[0]], index_satis[obj_list[1]])
    # index_intersect = np.intersect1d(index_intersect, index_satis[obj_list[2]])
    all_pivots = np.arange(sampler.candidate_samples.shape[1])
    new_candidates_select = new_candidates[:, index_satis[obj_list[0]]]
    # breakpoint()        
    sampler.candidate_samples[:, np.delete(all_pivots, sampler.init_pivots)] = new_candidates_select

    return sampler.candidate_samples, thsd_dict[objective]

def convergence_study(kernel, function, sampler,
                      num_vars, generate_samples, num_new_samples,
                      update_kernel_scale_num_samples,
                      noise_level=0, return_samples=False,
                      norm=np.linalg.norm, callback=None, gp_kernel=None,
                      gp_kernel_ob1=None, gp_kernel_ob2=None):

    # dirty hack to include two GP kernel types (for IVAR)
    if hasattr(kernel, "__len__"):
        # in this case, kernel is an array and we assume to have received
        # two kernels
        sampler_kernel = kernel[1]
        kernel = kernel[0]
    else:
        sampler_kernel = kernel

    # Instantiate a Gaussian Process model
    if gp_kernel is None:
        gp_kernel = kernel
    
    if gp_kernel_ob1 is not None:
        gp_ob1 =  AdaptiveGaussianProcess(
        gp_kernel_ob1, n_restarts_optimizer=2, alpha=1e-12)
        gp_ob1.optimizer = "fmin_l_bfgs_b"
    
    if gp_kernel_ob2 is not None:
        gp_ob2 =  AdaptiveGaussianProcess(
        gp_kernel_ob2, n_restarts_optimizer=2, alpha=1e-12)
        gp_ob2.optimizer = "fmin_l_bfgs_b"

    gp = AdaptiveGaussianProcess(
        gp_kernel, n_restarts_optimizer=2, alpha=1e-12)
    gp.setup(function, sampler)
    if hasattr(sampler, "set_gaussian_process"):
        sampler.set_gaussian_process(gp)
    
    print('sampler kernel', kernel, 'gp kernel', gp_kernel)

    # Mesh the input space for evaluations of the real function,
    # the prediction and its MSE

    num_samples = np.cumsum(num_new_samples)
    num_steps = num_new_samples.shape[0]
    errors = np.empty(num_steps, dtype=float)
    nsamples = np.empty(num_steps, dtype=int)
    sample_step = 0
    optimizer_step = 0
    # breakpoint()
    while sample_step < num_steps:
        if hasattr(gp, 'kernel_'):
            # if using const * rbf + noise kernel
            # kernel.theta = gp.kernel_.k1.k2.theta
            # if using const * rbf
            # kernel.theta = gp.kernel_.k2.theta
            # if using rbf
            kernel.theta = gp.kernel_.theta

        # Fit to data using Maximum Likelihood Estimation of the parameters
        # if True:
        if ((optimizer_step >= update_kernel_scale_num_samples.shape[0]) or
            (sampler.ntraining_samples <
             update_kernel_scale_num_samples[optimizer_step])):
            gp.optimizer = None
        else:
            gp.optimizer = "fmin_l_bfgs_b"
            optimizer_step += 1
        
        flag = gp.refine(np.sum(num_new_samples[:sample_step+1]))

        # load the taining samples for NSE and PBIAS
        # train_samples_other = np.loadtxt('training_samples.txt')
        # gp_ob1.fit(train_samples_other[0:num_vars, :], train_samples_other[num_vars:(num_vars + 1), :].T)
        # gp_ob2.fit(train_samples_other[0:num_vars, :], train_samples_other[(num_vars + 1):(num_vars + 2), :].T)
        
        # allow points to be added to gp more often than gp is evaluated for
        # validation
        if sampler.ntraining_samples >= num_samples[sample_step]:
            if callback is not None:
                callback(gp)

            print(gp.kernel_)
             
            if sample_step >=1:
                # Compute error
                gp_load = pickle.load(open(f'gp_{np.mod(sample_step - 1, 2)}.pkl', "rb"))
                validation_sub = sampler.training_samples[:, num_samples[sample_step - 1]:num_samples[sample_step]]
                pred_values = gp_load(validation_sub, return_cov=False).squeeze()
                values_sub = gp(validation_sub, return_cov=False).squeeze()
                error_gp_comp = norm(pred_values-values_sub)/norm(values_sub)
                print('-----------error_gp_comp---------', error_gp_comp)

                print('N', sampler.ntraining_samples, 'Error', error_gp_comp)

            if sample_step >= 1:
                errors[sample_step -1] = error_gp_comp
                nsamples[sample_step - 1] = num_samples[sample_step -1]

            pickle.dump(gp, open(f'gp_{np.mod(sample_step, 2)}.pkl', "wb"))
            sample_step += 1            

        if flag > 0:
            errors, nsamples = errors[:sample_step], nsamples[:sample_step]
            print('Terminating study. Points are becoming ill conditioned')
            break

        # check whether the performance of gp is satisfactory
        if sample_step >= 3:
            if (errors[sample_step - 2] <= 0.05) & (errors[sample_step-3] <= 0.05):
                print('----case 1-----')
                thsd = [0.382, 0.0, 1]
                resample_flag = True
            elif (errors[sample_step - 2] > 0.5) | (errors[sample_step-3] > 0.5):
                print('----case 2-----')
                resample_flag = False
            else:
                print('----case 3-----')
                thsd = [-10, 0.0, 1]
                resample_flag = True

            print(f'--------Error of step {sample_step - 2}: {errors[sample_step - 2]}')
            print(f'--------Error of step {sample_step - 3}: {errors[sample_step - 3]}')

            y_training = gp.y_train_
            num_y_optimize = y_training[y_training >= 0.382].shape[0]

            if resample_flag & (num_y_optimize <= 20):
                new_candidates, thsd_viney = resample_candidate(gp, sampler, thsd, 
                    num_samples, gp_ob1=gp_ob1, gp_ob2=gp_ob2)

                sampler.candidate_samples = new_candidates
                # sampler.init_pivots = None

                print(f'---------Threhsolds used to constrain parameter ranges:')
                print(f' viney_F: {thsd_viney}')
                print(f'---------The size of new candidate samples is {new_candidates.shape[1]}')

    # save GP for NSE and PBIAS
    # pickle.dump(gp_ob1, open('gp_ob1.pkl', "wb"))
    # pickle.dump(gp_ob2, open('gp_ob2.pkl', "wb"))

    if return_samples:
        return errors, nsamples, sampler.training_samples[:, 0:num_samples[sample_step - 1]]

    return errors, nsamples


def unnormalized_posterior(gp, prior_pdf, samples, temper_param=1):
    prior_vals = prior_pdf(samples).squeeze()
    gp_vals = gp.predict(samples.T).squeeze()
    vals_max = max(gp_vals.max(), 0.1)
    # breakpoint()
    # unnormalized_posterior_vals = prior_vals*(1 / (1 - gp_vals))**temper_param
    unnormalized_posterior_vals = prior_vals*np.exp(-(1 - gp_vals / vals_max))**temper_param
    return unnormalized_posterior_vals


class BayesianInferenceCholeskySampler(CholeskySampler):
    def __init__(self, prior_pdf, num_vars,
                 num_candidate_samples, variables,
                 max_num_samples=None, generate_random_samples=None,
                 temper=True, true_nll=None):
        self.prior_pdf = prior_pdf
        if not temper:
            self.temper_param = 1
        else:
            self.temper_param = 0
        self.true_nll = true_nll
        self.gp = None

        super().__init__(num_vars, num_candidate_samples, variables,
                         None, generate_random_samples)

    def set_gaussian_process(self, gp):
        self.gp = gp

    def increment_temper_param(self, num_training_samples):

        samples = generate_independent_random_samples(self.variables, 1000)
        density_vals_prev = self.weight_function(samples)

        def objective(beta):
            new_weight_function = partial(
                unnormalized_posterior, self.gp, self.prior_pdf,
                temper_param=beta)
            density_vals = new_weight_function(samples)

            # breakpoint()
            II = np.where(density_vals_prev > 1e-15)[0]
            JJ = np.where(density_vals_prev < 1e-15)[0]
            assert len(np.where(density_vals[JJ] > 1e-15)[0]) == 0
            ratio = np.zeros(samples.shape[1])
            ratio[II] = density_vals[II]/density_vals_prev[II]
            obj = ratio.std()/ratio.mean()
            return obj
        print('temper parameter', self.temper_param)
        x0 = self.temper_param+1e-3
        # result = root(lambda b: objective(b)-1, x0)
        # x_opt = result.x
        
        x_opt = bisect(lambda b: objective(b)-1, x0, 1)
        # if not optimize temper_param
        # x_opt = self.temper_param + 1e-2
        self.temper_param = x_opt

    def __call__(self, num_samples):
        if self.gp is None:
            raise ValueError("must call self.set_gaussian_process()")
        
        if self.ntraining_samples > 0 and self.temper_param < 1:
            self.increment_temper_param(self.training_samples)
        assert self.temper_param <= 1
        if self.ntraining_samples == 0:
            weight_function = self.prior_pdf
        else:
            if self.true_nll is not None:
                def weight_function(x): return self.prior_pdf(x)*np.exp(
                    -self.true_nll(x)[:, 0])**self.temper_param
            else:
                weight_function = partial(
                    unnormalized_posterior, self.gp, self.prior_pdf,
                    temper_param=self.temper_param)

        self.set_weight_function(weight_function)

        samples, flag = super().__call__(num_samples)
        return samples, flag


def get_prior_samples(num_vars, variables, nsamples):
    rosenbrock_samples = generate_independent_random_samples(variables, nsamples)

    return rosenbrock_samples

def bayesian_inference_example():
    # read parameter distributions
    datapath = file_settings()[1]
    para_info = pd.read_csv(datapath + 'Parameters-PCE.csv')

    # define the variables for PCE
    param_file = file_settings()[-1]
    
    # Must set variables if not using uniform prior on [0,1]^D
    # variables = None
    ind_vars, variables = variables_prep(param_file, product_uniform='uniform', dummy=False)
    var_trans = AffineRandomVariableTransformation(variables, enforce_bounds=True)
    init_scale = 0.2# used to define length_scale for the kernel
    num_vars = variables.nvars
    num_candidate_samples = 20000
    num_new_samples = np.asarray([20]+[10]*6+[25]*6+[20]*10)

    nvalidation_samples = 10000

    from scipy import stats 
    # breakpoint()
    prior_pdf = partial(tensor_product_pdf, 
        univariate_pdfs=[partial(stats.beta.pdf, a=1, b=1, scale=ind_vars[ii].args[1]) for ii in range(num_vars)])

    # Get validation samples from prior
    rosenbrock_samples = get_prior_samples(num_vars, variables, nvalidation_samples + num_candidate_samples)

    def generate_random_samples(nsamples, idx=0):
        assert idx+nsamples <= rosenbrock_samples.shape[1]
        return rosenbrock_samples[:, idx:idx+nsamples]

    generate_validation_samples = partial(
        generate_random_samples, nvalidation_samples,
        idx=num_candidate_samples)

    def get_filename(method, fixed_scale):
        filename = 'bayes-example-%s-d-%d-n-%d.npz' % (
            method, num_vars, num_candidate_samples)
        if not fixed_scale:
            filename = filename[:-4]+'-opt.npz'
        return filename

    # defining kernel
    length_scale = [init_scale, init_scale, *(3*np.ones(num_vars -2, dtype=float))]
    kernel = RBF(length_scale, [(5e-2, 1), (5e-2, 1), (5e-2, 20), (5e-2, 10),
        (5e-2, 20), (5e-2, 10), (5e-2, 20), (5e-2, 10), (5e-2, 20), 
        (5e-2, 10), (5e-2, 20), (5e-2, 10), (5e-2, 20)])

    # define gp_kernel_ob1/ob2 for objective functions individually
    gp_kernel_ob1 = RBF(length_scale, [(5e-2, 1), (5e-2, 1), (5e-2, 20), (5e-2, 10),
        (5e-2, 20), (5e-2, 10), (5e-2, 20), (5e-2, 10), (5e-2, 20), 
        (5e-2, 10), (5e-2, 20), (5e-2, 10), (5e-2, 20)])

    gp_kernel_ob2 = RBF(length_scale, [(5e-2, 1), (5e-2, 1), (5e-2, 20), (5e-2, 10),
        (5e-2, 20), (5e-2, 10), (5e-2, 20), (5e-2, 10), (5e-2, 20), 
        (5e-2, 10), (5e-2, 20), (5e-2, 10), (5e-2, 20)])

    # this is the one Qian should use. The others are for comparision only
    adaptive_cholesky_sampler = BayesianInferenceCholeskySampler(
        prior_pdf, num_vars, num_candidate_samples, variables,
        max_num_samples=num_new_samples.sum(),
        generate_random_samples=None)
    adaptive_cholesky_sampler.set_kernel(copy.deepcopy(kernel))

    samplers = [adaptive_cholesky_sampler]
    methods = ['Learning-Weighted-Cholesky-b']
    labels = [r'$\mathrm{Adapted\;Weighted\;Cholesky}$']
    fixed_scales = [False]

    for sampler, method, fixed_scale in zip(samplers, methods, fixed_scales):
        filename = get_filename(method, fixed_scale)
        print(filename)
        if os.path.exists(filename):
            continue

        if fixed_scale:
            update_kernel_scale_num_samples = np.empty(0)
        else:
            update_kernel_scale_num_samples = np.cumsum(num_new_samples)

        cond_nums = []
        temper_params = []

        def callback(gp):
            cond_nums.append(np.linalg.cond(gp.L_.dot(gp.L_.T)))
            if hasattr(sampler, 'temper_param'):
                temper_params.append(sampler.temper_param)
                print(temper_params)

        errors, nsamples, samples = convergence_study(
            kernel, run_source_lsq, sampler, num_vars,
            generate_validation_samples, num_new_samples,
            update_kernel_scale_num_samples, callback=callback,
            return_samples=True, 
            gp_kernel_ob1=gp_kernel_ob1, gp_kernel_ob2=gp_kernel_ob2)

        np.savez(filename, nsamples=nsamples, errors=errors,
                 cond_nums=np.asarray(cond_nums), samples=samples,
                 temper_params=np.asarray(temper_params))

    fig, axs = plt.subplots(1, 3, figsize=(3*8, 6), sharey=False)
    styles = ['-']
    # styles = ['k-','r-.','b--','g:']
    for method, label, ls, fixed_scale in zip(
            methods, labels, styles, fixed_scales):
        filename = get_filename(method, fixed_scale)
        data = np.load(filename)
        nsamples, errors = data['nsamples'][:-1], data['errors'][:-1]
        temper_params, cond_nums = data['temper_params'][1:-1], data['cond_nums'][:-1]
        axs[0].loglog(nsamples, errors, ls=ls, label=label)
        axs[1].loglog(nsamples, cond_nums, ls=ls, label=label)
        axs[2].semilogy(np.arange(1, nsamples.shape[0]),
                    temper_params, 'k-o')
        axs[2].set_xlabel(r'$\mathrm{Iteration}$ $j$')
        axs[2].set_ylabel(r'$\beta_j$')

    for ii in range(2):
        axs[ii].set_xlabel(r'$m$')
        axs[ii].set_xlim(10, 1000)
    axs[0].set_ylabel(r'$\tilde{\epsilon}_{\omega,2}$', rotation=90)
    ylim0 = axs[0].get_ylim()
    ylim1 = axs[1].get_ylim()
    ylim = [min(ylim0[0], ylim1[0]), max(ylim0[1], ylim1[1])]
    axs[0].set_ylim(ylim)
    axs[1].set_ylim(ylim)
    axs[1].set_ylabel(r'$\kappa$', rotation=90)

    figname = 'bayes_example_comparison_%d.pdf' % num_vars
    axs[0].legend()
    plt.savefig(figname) 

if __name__ == '__main__':
    try:
        import sklearn
    except:
        msg = 'Install sklearn using pip install sklearn'
        raise Exception(msg)

    bayesian_inference_example()

# gp_load = pickle.load(open(f'gp_0.pkl', "rb"))
# x_training = gp_load.X_train_
# y_training = gp_load.y_train_