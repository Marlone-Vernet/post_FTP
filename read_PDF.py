# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:21:50 2024

@author: VERNET MARLONE 

"""

import numpy as np
import matplotlib.pyplot as plt 

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')
plt.rcParams['font.size']=20

#%%

folder = "E:/DATA_FTP/310724/"
#data2 = dict( np.load(folder2+'spectre_vs_t_k2D_20Hz.npy', allow_pickle=True).item() )
data = dict( np.load(folder+'data_vs_t_27Hz_30Hz.npy', allow_pickle=True).item() )

gg = 25/(7.5) * 1e3 # facteur, en mm

bin_ = data['bin_t']
pdf_ = data['pdf_t']
std_ = data['std_t']

bin_m = np.mean(bin_, axis=1)*gg
pdf_m = np.mean(pdf_, axis=1)
std_m = np.mean(std_)*gg

#%%

def gauss_fit(x,sig):
    sortie = np.exp(-x**2/(2*sig**2))
    masse = np.sqrt(2*np.pi*sig**2)
    return sortie/masse

x_ = (bin_m/std_m)
fit_ = gauss_fit(x_, std_m)
plt.figure()
plt.semilogy(x_, pdf_m/(max(pdf_m)), 'ok')
plt.semilogy(x_, fit_/max(fit_), '--k')


plt.grid(which='both')
plt.tight_layout()
plt.show()
