#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl

from scipy import stats

from funcs.read_data import file_settings

mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 2
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

step = 0.001
pbias = np.arange(-0.3, 0.3, step)
nse = np.arange(0.3, 0.6, step)
x, y = np.meshgrid(nse, pbias)
F = x - 5 * np.abs(np.log(1 + y))**2.5

plt.figure(figsize=(8, 6))
contour = plt.contour(x, y, F, colors='gray', linewidth=1)
plt.hlines(-0.20, 0.5, nse[-1], color='cyan', alpha=0.7, linewidth=2)
plt.hlines(0.20, 0.5, nse[-1], color='cyan', alpha=0.7, linewidth=2)
plt.vlines(0.50, -0.2, 0.2, color='cyan', alpha=0.7, linewidth=2)
plt.contour(x, y, F, [0.382], colors='orange', linewidth=2, alpha=0.5)
plt.clabel(contour, colors='k')
plt.text(0.45, 0.22, 'F=0.382', rotation=20)
plt.xlabel('NSE')
plt.ylabel('B')
plt.savefig(f'{file_settings()[0]}gp_run_0816/contour_F', dpi=300)
