
import numpy as np
import pandas as pd
from veneer.pest_runtime import *

import pyapprox as pya
from functools import partial 
from pyapprox.adaptive_sparse_grid import max_level_admissibility_function
from pyapprox.adaptive_polynomial_chaos import variance_pce_refinement_indicator
from pyapprox.univariate_quadrature import clenshaw_curtis_rule_growth
from pyapprox.variable_transformations import AffineRandomVariableTransformation

from funcs.read_data import variables_prep, file_settings
from funcs.modeling_funcs import vs_settings, \
        modeling_settings, paralell_vs, obtain_initials, change_param_values

# Create the copy of models and veneer list
project_name = 'MW_BASE_RC10.rsproj'
veneer_name = 'vcmd45\\FlowMatters.Source.VeneerCmd.exe'   
first_port=15000; num_copies = 2
_, things_to_record, _, _, _ = modeling_settings()
processes, ports = paralell_vs(first_port, num_copies, project_name, veneer_name)

vs_list = vs_settings(ports, things_to_record)
# obtain the initial values of parameters 
initial_values = obtain_initials(vs_list[0])

def run_source(vars, vs_list=vs_list):
    """
    Script used to run_source and return the output file.
    The function is called by AdaptiveLejaPCE.
    """
    from funcs.modeling_funcs import modeling_settings, generate_observation_ensemble
    print('Read Parameters')
    parameters = pd.read_csv('../data/Parameters-PCE.csv', index_col='Index')

    # import observation if the output.txt requires the use of obs.
    date_range = pd.to_datetime(['2017/07/01', '2018/06/30'])
    observed_din = pd.read_csv(f'{file_settings()[1]}126001A.csv', index_col='Date')
    observed_din.index = pd.to_datetime(observed_din.index)
    observed_din = observed_din.loc[date_range[0]:date_range[1], :].filter(items=[observed_din.columns[0]]).apply(lambda x: 1000 * x)
    
    # loop over the vars and try to use parallel     
    parameter_df = pd.DataFrame(index=np.arange(vars.shape[1]), columns=parameters.Name_short)
    for i in range(vars.shape[1]):
        parameter_df.iloc[i] = vars[:, i]

    # set the time period of the results
    retrieve_time = [pd.Timestamp('2017-07-01'), pd.Timestamp('2018-06-30')]

    # define the modeling period and the recording variables
    NODEs, things_to_record, criteria, start_date, end_date = modeling_settings()
    din = generate_observation_ensemble(vs_list, 
        criteria, start_date, end_date, parameter_df, retrieve_time)
    breakpoint()
    if din.mean(axis=0)[0] == 0:
        cv = np.array([0])
    else:
        breakpoint()
        cv = np.array([din.std(axis = 0) / din.mean(axis = 0)].values)

    if vars.shape[1] > 1:
        breakpoint()
    cv = cv.reshape(cv.shape[0], 1)
    print(f'Finish {cv.shape[0]} run')
    return cv
# END run_source()

# read parameter distributions
datapath = file_settings()[1]
para_info = pd.read_csv(datapath + 'Parameters-PCE.csv')

# define the variables for PCE
param_file = file_settings()[-1]
ind_vars, variable = variables_prep(param_file, product_uniform='uniform', dummy=False)
var_trans = AffineRandomVariableTransformation(variable, enforce_bounds=True)

# Create PyApprox model
n_candidate_samples = 10000
candidate_samples = -np.cos(np.pi*pya.sobol_sequence(var_trans.num_vars(),
                        n_candidate_samples))
pce = pya.AdaptiveLejaPCE(var_trans.num_vars(), candidate_samples=candidate_samples)

# Define criteria
max_level = 2
err_tol = 1e-4
max_num_samples = 100
max_level_1d = [max_level]*(pce.num_vars)

admissibility_function = partial(
    max_level_admissibility_function, max_level, max_level_1d,
    max_num_samples, err_tol)

refinement_indicator = variance_pce_refinement_indicator
pce.set_function(run_source, var_trans)

pce.set_refinement_functions(
    refinement_indicator, 
    admissibility_function,
    clenshaw_curtis_rule_growth
)

# Generate emulator
pce.build()

# store PCE
import pickle
pickle.dump(pce, open(f'{file_settings[0]}\pce-cv.pkl', "wb"))

# set the parameter values to initial values
for vs in vs_list:
    vs = change_param_values(vs, initial_values, fromList=True)