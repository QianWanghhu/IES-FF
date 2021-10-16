
import numpy as np
import pandas as pd
from veneer.pest_runtime import *
from veneer.manage import start,kill_all_now

import pyapprox as pya
from pyapprox.utilities import total_degree_space_dimension

from funcs.read_data import file_settings, variables_prep
from funcs.modeling_funcs import vs_settings, \
        modeling_settings, paralell_vs, obtain_initials, change_param_values

# Create the copy of models and veneer list
project_name = 'MW_BASE_RC10.rsproj'
veneer_name = 'vcmd45\\FlowMatters.Source.VeneerCmd.exe'   
first_port=15003; num_copies = 2
_, things_to_record, _, _, _ = modeling_settings()
processes, ports = paralell_vs(first_port, num_copies, project_name, veneer_name)

vs_list = vs_settings(ports, things_to_record)
from run_source import run_source_fix_samples, run_source_lsq
# obtain the initial values of parameters 
initial_values = obtain_initials(vs_list[0])

# select 10 samples from the optimal sample set
fpath = f'{file_settings()[0]}adaptive/rmse_iter3/'
num_iter = int(fpath[-2])
samples_import = np.loadtxt(f'{fpath}samples_selected.txt')
rand_index = np.random.randint(0, samples_import.shape[1], 10)
samples_opt = samples_import[0:13, rand_index]
vals_opt = run_source_fix_samples(vs_list, samples_opt)

# import PCE and update the surrogate with added training samples
import pickle
from pyapprox.approximate import approximate
pce_load = pickle.load(open(f'{file_settings()[0]}adaptive/rmse_iter{num_iter - 1}/pce-rmse-iter{num_iter - 1}.pkl', "rb"))

# update training samples
# train_samples = pce_load.variable_transformation.map_from_canonical_space(pce_load.samples)
# train_vals = pce_load.values
train_sets = np.loadtxt(f'{file_settings()[0]}adaptive/rmse_iter{num_iter - 1}/train_samples.txt')
train_samples = train_sets[:13, :]
train_vals = train_sets[-1, :].reshape(train_samples.shape[1], 1)

train_samples = np.append(train_samples, samples_opt, axis=1)
train_vals = np.append(train_vals, vals_opt, axis=0)
train_set = np.append(train_samples, train_vals.T, axis=0)
np.savetxt(f'{fpath}train_samples.txt', train_set)

param_file = file_settings()[-1]
ind_vars, variable = variables_prep(param_file, product_uniform='uniform', dummy=False)
nfolds = min(10, train_samples.shape[1])
solver_options = {'cv': nfolds}
nterms = total_degree_space_dimension(train_samples.shape[0], 3)
options = {'basis_type': 'expanding_basis', 'variable': ind_vars,
            'verbosity': 0, 
            'options': {'max_num_init_terms': nterms,
            'linear_solver_options': solver_options}}

approx_update = approximate(train_samples, train_vals, method='polynomial_chaos', options=options).approx
pickle.dump(approx_update, open(f'{fpath}pce-rmse-iter{num_iter}.pkl', "wb"))
# set the parameter values to initial values
for vs in vs_list:
    vs = change_param_values(vs, initial_values, fromList=True)

kill_all_now(processes)
