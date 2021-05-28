"""
The script is used to build adaptive pce for signatures and RMSE
"""

import numpy as np
import pandas as pd
from veneer.pest_runtime import *
import pyapprox as pya
from functools import partial 
from pyapprox.adaptive_sparse_grid import max_level_admissibility_function
from pyapprox.adaptive_polynomial_chaos import variance_pce_refinement_indicator
from pyapprox.univariate_quadrature import clenshaw_curtis_rule_growth

from funcs.read_data import variables_prep, file_settings
from funcs.boots_pya import fun


def call_source():
    pass
    # initialize the vs

    # change parameter values

    # run the model, save and return results


def adaptive_pce(variable):
    pass
    num_vars = variable.num_vars()
    # Create PyApprox model
    pce = pya.AdaptiveInducedPCE(num_vars, cond_tol=1e2)
    # Define criteria
    max_level = 4
    err_tol = 0.0
    max_num_samples = 1000

    max_level_1d = [max_level]*(pce.num_vars)
    admissibility_function = partial(
        max_level_admissibility_function, max_level, max_level_1d,
        max_num_samples, err_tol)
    refinement_indicator = variance_pce_refinement_indicator

    pce.set_function(call_source, variable)
    pce.set_refinement_functions(
        refinement_indicator, 
        admissibility_function,
        clenshaw_curtis_rule_growth
    )

    # Generate emulator
    pce.build()

    # fit the PCE

# define the variables for PCE
param_file = file_settings()[-1]
variable = variables_prep(param_file, product_uniform='uniform', dummy=False)

"""
RUN SOURCE to generate observation_ensemble.csv
"""
from veneer.pest_runtime import *
from veneer.manage import start,kill_all_now
import os

import source_runner as sr 
from source_runner import parameter_funcs

from modeling_funcs import vs_settings, \
    change_param_values, generate_observation_ensemble, \
        modeling_settings, paralell_vs

project_name = 'MW_BASE_RC10.rsproj'
veneer_name = 'vcmd45\\FlowMatters.Source.VeneerCmd.exe'   
first_port=15000; num_copies = 1
NODEs, things_to_record, criteria, start_date, end_date = modeling_settings()
processes, ports = paralell_vs(first_port, num_copies, project_name, veneer_name)
vs_list = vs_settings(ports, things_to_record)

# generate parameter emsenble
datapath = file_settings()[1]
para_info = parameter_funcs.load_parameter_file(datapath + 'parameters.csv')

# obtain the initial values of parameters 
param_names, param_vename_dic, param_vename, param_types = sr.group_parameters(para_info)
initial_values = parameter_funcs.get_initial_param_vals(vs_list[0], param_names, param_vename, param_vename_dic)
# initial_values = obtain_initials(vs_list[0])
# set the time period of the results
retrieve_time = [pd.Timestamp('2000-07-01'), pd.Timestamp('2018-06-30')]
