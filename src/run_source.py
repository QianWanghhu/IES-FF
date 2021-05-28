"""
Script used to run_source and return the output file.
"""
import veneer
import numpy as np
from veneer.pest_runtime import *
import pandas as pd
import time
import os
import spotpy as sp

from funcs.read_data import file_settings

def run_source(vars, vs):

    from modeling_funcs import change_param_values, modeling_settings

    print('Read Parameters')
    parameters = pd.read_csv('parameters.csv')

    # Define objective functions
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

    # import observation if the output.txt requires the use of obs.
    # Here the observation can be monitoring data or synthetic data.
    date_range = pd.to_datetime(['2009/07/01', '2018/06/30'])
    observed_din = pd.read_csv('126001A.csv', index_col='Date')
    observed_din.index = pd.to_datetime(observed_din.index)
    observed_din = observed_din.loc[date_range[0]:date_range[1], :].filter(items=[observed_din.columns[0]]).apply(lambda x: 1000 * x)

    parameter_dict = {}
    for i,j in parameters.iterrows():
        scaled_value = (j.upper - j.lower) * j.value/100 + j.lower 
        parameter_dict[j.parameter] = scaled_value
        
    # define the modeling period and the recording variables
    NODEs, things_to_record, criteria, start_date, end_date = modeling_settings()
    vs.configure_recording(disable=[{}], enable=things_to_record)

    # subcatchments = [114, 106, 113, 107, 112, 108, 109, 157, 110, 111, 105, 161, 104, 103]
    vs = change_param_values(vs, parameter_dict) # , subcatchment=subcatchments
    vs.drop_all_runs()
    vs.run_model(params={'NoHardCopyResults':True}, start = start_date, end = end_date) 

    # define the criteria for retrieve multiple_time_series
    column_names = ['din']
    get_din = vs.retrieve_multiple_time_series(criteria=criteria)
    get_din.columns = column_names
    din = get_din.loc[date_range[0]:date_range[1], :]

    # obtain the sum at a given temporal scale
    # din_pbias = sp.objectivefunctions.pbias(observed_din[observed_din.columns[0]], din[column_names[0]])
    # din_126001A = timeseries_sum(din, temp_scale = 'annual')

    output_file = 'output.txt'
    output_path = file_settings()[0]
    if os.path.exists(output_path + output_file):
        output =  np.loadtxt(f'{output_path}{output_file}')
        output = np.append(output, din.values, axis=1)
    
    np.savetxt(f'{output_path}{output_file}', output)

    cv = din.std()[0] / din.mean()[0]

    return cv
        
    

