#!/usr/bin/env python
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['text.usetex'] = False  # use latex for all text handling
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.format'] = 'png'  # gives best resolution plots
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 16
# print mpl.rcParams.keys()
mpl.rcParams['text.latex.preamble'] = \
    r'\usepackage{siunitx}\usepackage{amsmath}\usepackage{amssymb}'


def calculate_weight(prior_vals, gp_vals, gp_max, temper_parameter):
    weight = prior_vals*np.exp(-(1 - gp_vals / gp_max))**temper_parameter
    # weight = prior_vals*((2 - gp_max) / (2 - gp_vals))**temper_parameter
    return weight

prior_vals = 6.4e-10
vals_max = [0.1, 0.5, 1]
temper_parameter = [0.01, 0.1, 0.5]
legends = [f'GP max value: {v}' for v in vals_max]

fig, axes = plt.subplots(1, 3, figsize=(22, 6))

for ii in range(len(temper_parameter)):
    for jj in range(len(vals_max)):
        gp_vals = np.linspace(-100, vals_max[jj], 10000)
        weights = calculate_weight(prior_vals, gp_vals, vals_max[jj], temper_parameter[ii])
        axes[ii].semilogx((1 - gp_vals), weights, alpha=0.7)
    axes[ii].set_xlabel('1 - (GP outputs)')
    axes[ii].set_title(r'$\beta: {%0.2f}$'%(temper_parameter[ii]))

axes[0].legend(legends)
axes[0].set_ylabel(r'${w_{approx}}$')
plt.savefig('../output/gp_weights_map.png', dpi=300)
