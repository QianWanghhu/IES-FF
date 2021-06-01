import numpy as np
from numpy.random import uniform
import pandas as pd
from pyapprox import variables
from veneer.pest_runtime import *
import pyapprox as pya
from scipy.stats import beta,uniform
from functools import partial 
from pyapprox.adaptive_sparse_grid import max_level_admissibility_function
from pyapprox.adaptive_polynomial_chaos import variance_pce_refinement_indicator
from pyapprox.univariate_quadrature import clenshaw_curtis_rule_growth
from pyapprox.variable_transformations import AffineRandomVariableTransformation
from pyapprox.variables import IndependentMultivariateRandomVariable

num_vars = 2
alph = 5
bet = 5.
err_tol = 1e-7
a = np.random.uniform(0, 100, (num_vars, 1))
# variable =  IndependentMultivariateRandomVariable(
#                 [beta(alph, bet, 0, 1)], [np.arange(num_vars)])
# var_trans = AffineRandomVariableTransformation(
#             IndependentMultivariateRandomVariable(
#                 [beta(alph, bet, 0, 1)], [np.arange(num_vars)]))


variable =  IndependentMultivariateRandomVariable(
                [uniform(0, 1)], [np.arange(num_vars)])
var_trans = AffineRandomVariableTransformation(
            IndependentMultivariateRandomVariable(
                [uniform(0, 1)], [np.arange(num_vars)]))                

def function(x):
    vals = [np.cos(np.pi*a[ii]*x[ii, :]) for ii in range(x.shape[0])]
    vals = np.array(vals).sum(axis=0)[:, np.newaxis]
    breakpoint()
    return vals

# def run_source(x):
#     """
#     A test function for adaptive PCE.
#     """
#     y = np.array(x[0:10].sum() + x[10]**2 + x[11] * 4 + 0.1)
#     # breakpoint()
#     print(y.shape)
#     return y.reshape(y.shape[0], 1)


# num_vars = variable.num_vars()
# Create PyApprox model
pce = pya.AdaptiveInducedPCE(num_vars, cond_tol=1e2)
# Define criteria
max_level = 4
# err_tol = 0.0
max_num_samples = 1000

max_level_1d = [max_level]*(pce.num_vars)
admissibility_function = partial(
    max_level_admissibility_function, max_level, max_level_1d,
    max_num_samples, err_tol)
refinement_indicator = variance_pce_refinement_indicator

pce.set_function(function, var_trans)
pce.set_refinement_functions(
    refinement_indicator, 
    admissibility_function,
    clenshaw_curtis_rule_growth
)

# Generate emulator
pce.build()

# fit the PCE

validation_samples = pya.generate_independent_random_samples(variable, 1000)
validation_vals = function(validation_samples)
hat_vals = pce(validation_samples)
np.std(validation_vals - hat_vals)
