# import packages
import numpy as np
import pandas as pd
import pyapprox as pya
import pickle
from read_data import file_settings

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