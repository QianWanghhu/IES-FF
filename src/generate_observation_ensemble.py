# import packages
"""
RUN SOURCE to generate observation_ensemble.csv
"""
import pandas as pd
from veneer.pest_runtime import *
from veneer.manage import start,kill_all_now
import os

from funcs.modeling_funcs import vs_settings, \
    change_param_values, generate_observation_ensemble, generate_parameter_ensemble,\
        modeling_settings, paralell_vs, obtain_initials

project_name = 'MW_BASE_RC10.rsproj'
veneer_name = 'vcmd45\\FlowMatters.Source.VeneerCmd.exe'   
first_port=15000; num_copies = 1
NODEs, things_to_record, criteria, start_date, end_date = modeling_settings()
processes, ports = paralell_vs(first_port, num_copies, project_name, veneer_name)
vs_list = vs_settings(ports, things_to_record)

# generate parameter emsenble
datapath = '../data/'
nsample = 1000
param_ensemble = 'samples.csv'
generate_parameter_ensemble(nsample, param_ensemble, datapath, seed=88)

# obtain the initial values of parameters 
initial_values = obtain_initials(vs_list[0])


# run to generate observation with default parameter values in the model
print('------------------Generate observation with default parameter values-----------------')
retrieve_time = [pd.Timestamp('2008-07-01'), pd.Timestamp('2018-06-30')]

# run to generate observation ensemble with parameter ensemble
print('------------------Generate observation ensemble-----------------')
obs_ensemble_name = 'DIN_126001A'   
parameters = pd.read_csv('samples.csv', index_col='real_name').iloc[0:2]

# generate the observation ensemble
def run_obs_ensemble(vs, criteria, start_date, end_date, parameters, 
    obs_ensemble_name, retrieve_time, datapath):
    if not os.path.exists(f'{datapath}{obs_ensemble_name}.csv'):
        load = generate_observation_ensemble(vs_list, criteria, 
            start_date, end_date, parameters, retrieve_time)
        load.to_csv(f'{datapath}{obs_ensemble_name}.csv')
    else:
        print(f'{obs_ensemble_name}.csv exists.')

fromList=True
run_obs_ensemble(vs_list, criteria, start_date, end_date, parameters, 
    obs_ensemble_name, retrieve_time, datapath)

# set parameter to the initial values
for vs in vs_list:
    change_param_values(vs, initial_values, 
        fromList=fromList)

kill_all_now(processes)


