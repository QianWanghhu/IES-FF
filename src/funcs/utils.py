# import packages
import numpy as np
import pandas as pd
import pyapprox as pya
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from .read_data import file_settings

def return_sa(year, parameters):
    """
    The function is used to calculate the sensitivity indices by PCE.
    Parameters：
    ===========
    year: int, determines which PCE to load.
    parameters: np.ndarray or list, of parameter names.

    Returns:
    sa: pd.DataFrame, contains both the main and total effects
    """

    # import pces
    pce_load = pickle.load(open(f'{file_settings()[0]}pce-{year}-level4.pkl', "rb"))
    sobol_analysis = pya.sensitivity_analysis.analyze_sensitivity_polynomial_chaos
    res = sobol_analysis(pce_load.pce, 2)
    total_effects = res.total_effects
    main_effects = res.main_effects
    # export the sensitivity of parameters in terms of SSE    
    sa = pd.DataFrame(index = parameters, columns=['main_effects', 'total_effects'])
    sa['main_effects'] = main_effects
    sa['total_effects'] = total_effects
    return sa


def dotty_plot(x_samples, y_vals, x_opt, y_opt, param_names, orig_x_opt=None, orig_y_opt=None):
    """
    Create dotty plots for the model inputs and outputs.
    Parameteres:
    ============
    x_samples: np.ndarray, input sample set of the shape D * N where D is the number of parameters and N is the sample size;
    y_vals: np.ndarray, outputs corresponding to the x_samples and of the shape N * 1
    x_opt: np.ndarray, parameter data points resulting in the selected optima
    y_opt: np.ndarray, output values of the selected optima corresponding to x_opt
    param_names: list, parameter names
    orig_x_opt: np.ndarray, parameter data points resulting in the selected optima 
                and the selection is based on outputs without factor fixing.
    orig_y_opt: np.ndarray, output values of the selected optima corresponding to x_opt
    
    Returns:
    ========
    fig
    """
    fig, axes = plt.subplots(4, 4, figsize = (18, 18))
    for ii in range(x_samples.shape[0]): 
        if orig_x_opt is not None:
            ax = sns.scatterplot(x=orig_x_opt[ii, :], y=orig_y_opt.flatten(), ax=axes[ii // 4, ii % 4], color='g', s=20, alpha=0.3)
            
        ax = sns.scatterplot(x=x_samples[ii, :], y=y_vals.flatten(), ax=axes[ii // 4, ii % 4], s=20, alpha=0.7)
        ax = sns.scatterplot(x=x_opt[ii, :], y=y_opt.flatten(), ax=axes[ii // 4, ii % 4], color='r', s=20)
                   
        ax.set_title(param_names[ii])
        if ii % 4 == 0: ax.set_ylabel('SSE')
            
    return fig


def define_constants(samples_opt, stats = 'median'):
    """Return default values of the parameters to fix
    """
    if stats == 'median':
        x_default = np.median(samples_opt, axis=1)
    elif stats == 'mean':
        x_defau;t = samples_opt.mean()
    else:
        AssertionError
    return x_default

def fix_sample_set(index_fix, dot_samples, x_default):
    import copy
    """ 
    Return the samples by fixing certain parameters.
    Parameters:
    ===========
    index_fix = 
    
    Returns:
    ========
    
    """
    samples_fix = copy.deepcopy(dot_samples)
    for ii in range(index_fix.shape[0]):
        samples_fix[index_fix[ii], :] = x_default[index_fix[ii]]    
    return samples_fix