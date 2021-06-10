
# import packages
import sklearn
from scipy import stats

from pyapprox.approximate import approximate_polynomial_chaos
import pyapprox_interface as pyi
import os
import numpy as np
import pandas as pd
from veneer.pest_runtime import *
import pyapprox as pya

from funcs.read_data import variables_prep, file_settings

# define the variables for PCE
param_file = file_settings()[-1]
ind_vars, variable = variables_prep(param_file, product_uniform='uniform', dummy=False)

# from funcs.modeling_funcs import vs_settings, \
#         modeling_settings, paralell_vs, obtain_initials, change_param_values

# project_name = 'MW_BASE_RC10.rsproj'
# veneer_name = 'vcmd45\\FlowMatters.Source.VeneerCmd.exe'   
# first_port=15000; num_copies = 1
# NODEs, things_to_record, criteria, start_date, end_date = modeling_settings()
# processes, ports = paralell_vs(first_port, num_copies, project_name, veneer_name)
# vs_list = vs_settings(ports, things_to_record)
# vs = vs_list[0]
def run_source(x):
    print(x.shape)
    return x.sum()
# generate parameter emsenble
datapath = file_settings()[1]

para_info = pd.read_csv(datapath + 'Parameters-PCE.csv')
# obtain the initial values of parameters 
# initial_values = obtain_initials(vs_list[0])
# set the time period of the results
retrieve_time = [pd.Timestamp('2000-07-01'), pd.Timestamp('2018-06-30')]
num_vars = variable.num_vars()
# Create PyApprox model

# Note that for adaptive approaches, it is not necessary to sample from the model beforehand.
emulator = (pyi.PyaModel(target_model=run_source,
                         variables=ind_vars
                        )
               .define_pce(
                    approach=pya.AdaptiveLejaPCE,
                    transform_approach=pya.AffineRandomVariableTransformation,
                    enforce_bounds = True)
               .sample_validation(100) 
               .build(admissibility={'max_num_samples': 100, 'error_tol':1e-4, 'max_level': 4}, 
                        track_error_decay=True)
)

x = emulator.sample_validation(600).validation_samples
y = emulator.target_model(x)
y_hat = emulator.model(x)

# pyi.plot_fit(y, y_hat, 
#              metric=sklearn.metrics.mean_squared_error,
#              metric_name="RMSE",
#              squared=False)

print(emulator.sensitivities())