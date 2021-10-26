#!/usr/bin/env python
import numpy as np
import pandas as pd

# Note: do save the results while running the original model.
from multiprocessing import Pool
import numpy as np
import os
import matplotlib.pyplot as plt
from functools import partial
import time
import copy
import pandas as pd
import pickle
from scipy import stats
# from scipy.optimize import root
from scipy.optimize import bisect
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
mpl.rcParams['savefig.format'] = 'jpg'  # gives best resolution plots
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 16
# print mpl.rcParams.keys()
mpl.rcParams['text.latex.preamble'] = \
    r'\usepackage{siunitx}\usepackage{amsmath}\usepackage{amssymb}'

# Create the copy of models and veneer list
project_name = 'MW_BASE_RC10.rsproj'
veneer_name = 'vcmd45\\FlowMatters.Source.VeneerCmd.exe'   
first_port=15000; num_copies = 1
_, things_to_record, _, _, _ = modeling_settings()
processes, ports = paralell_vs(first_port, num_copies, project_name, veneer_name)

vs_list = vs_settings(ports, things_to_record)
# obtain the initial values of parameters 
initial_values = obtain_initials(vs_list[0])
def run_source_lsq(vars, vs_list=vs_list):
    """
    Script used to run_source and return the output file.
    The function is called by AdaptiveLejaPCE.
    """
    from funcs.modeling_funcs import modeling_settings, generate_observation_ensemble
    import spotpy as sp
    print('Read Parameters')
    parameters = pd.read_csv('../data/Parameters-PCE2.csv', index_col='Index')

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
    # vars_copy[0, :] = vars_copy[0, :] * 100
    # vars_copy[1, :] = vars_copy[1, :] * 100

    # import observation if the output.txt requires the use of obs.
    date_range = pd.to_datetime(['2009/07/01', '2018/06/30'])
    observed_din = pd.read_csv(f'{file_settings()[1]}126001A.csv', index_col='Date')
    observed_din.index = pd.to_datetime(observed_din.index)
    observed_din = observed_din.loc[date_range[0]:date_range[1], :].filter(items=[observed_din.columns[0]]).apply(lambda x: 1000 * x)
    
    # loop over the vars and try to use parallel     
    parameter_df = pd.DataFrame(index=np.arange(vars.shape[1]), columns=parameters.Name_short)
    for i in range(vars.shape[1]):
        parameter_df.iloc[i] = vars[:, i]

    # set the time period of the results
    retrieve_time = [pd.Timestamp('2009-07-01'), pd.Timestamp('2018-06-30')]

    # define the modeling period and the recording variables
    _, _, criteria, start_date, end_date = modeling_settings()
    initial_values = obtain_initials(vs_list[0])
    din = generate_observation_ensemble(vs_list, 
        criteria, start_date, end_date, parameter_df, retrieve_time, initial_values)

    # obtain the sum at a given temporal scale
    din_126001A = timeseries_sum(din, temp_scale = 'annual')
    obs_din = timeseries_sum(observed_din, temp_scale = 'annual')
    din_126001A = pd.DataFrame(din_126001A,dtype='float').values
    obs_din = pd.DataFrame(obs_din,dtype='float').values


    obj = cal_obj(obs_din, din_126001A, obj_type = 'viney')
    print(f'Finish {obj.shape[0]} run')

    # calculate the objective NSE and PBIAS
    obj_nse = cal_obj(obs_din, din_126001A, obj_type = 'nse')
    obj_pbias = cal_obj(obs_din, din_126001A, obj_type = 'pbias')
    train_iteration = np.append(vars, obj_nse.T, axis=0)
    train_iteration = np.append(train_iteration, obj_pbias.T, axis=0)
    train_iteration = np.append(train_iteration, obj.T, axis=0)
    # save sampling results of NSE and PBIAS
    train_file = 'outlier_samples.txt'
    if os.path.exists(train_file):
        train_samples = np.loadtxt(train_file)
        train_samples = np.append(train_samples, train_iteration, axis=1)
        np.savetxt(train_file, train_samples)
    else:
        np.savetxt(train_file, train_iteration) 
    # END if-else

    return obj
# END run_source_lsq()

# Obtain samples satisfying the criteria
# Call the function to run DIN model with selected samples
fdir = '../output/gp_run_0816/sampling-sa/fix_mean_subreg/'
samples = np.loadtxt(f'{fdir}samples_fix_2.txt')
values = np.loadtxt(f'{fdir}values_fix_2.txt')

# Plot the results comparing GP and the original model outputs
if os.path.exists(f'{fdir}values_fix_2_filter.txt') and \
    (os.path.exists(f'{fdir}outlier_samples.txt')):
    y_outlier_gp = np.loadtxt(f'{fdir}values_fix_2_filter.txt')
    y_outlier_true = np.loadtxt(f'{fdir}outlier_samples.txt')[-1, :]
else:
    index_filter = np.where((values>0.382) & (values<1))[0]
    samples_filter = samples[:, index_filter]
    values_filter = values[index_filter]
    np.savetxt(f'{fdir}samples_fix_2_filter.txt', samples_filter)
    np.savetxt(f'{fdir}values_fix_2_filter.txt', values_filter)
    # run the main model
    obj_model = run_source_lsq(samples_filter, vs_list=vs_list)
    np.savetxt('outlier_samples.txt', obj_model)

y_residuals = y_outlier_true.flatten() - y_outlier_gp[0:100].flatten()
ax = plt.scatter(y_outlier_true, y_residuals)
# plt.plot(np.arange(0, 10, 1)/10, np.arange(0, 10, 1)/10, linestyle='--')
plt.xlabel('Model output')
plt.ylabel('GP output')
