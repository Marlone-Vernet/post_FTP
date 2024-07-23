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

folder = '/home/tanu/data1/DATA_post/180724/' 
# psd = np.load(folder+'spectre_25Hz_hann.npy')
psd_v = np.load(folder+'spectre_vitesse_vent_20Hz.npy')
psd = np.load(folder+'spectre_vent_20Hz.npy')

k_ = np.load(folder+'fft_k_vent_20Hz/'+'array_k.npy')
k_ = k_-k_[0]

Nk,Nt = psd.shape
fech = 120


#%%

df = fech

lamb = k_[1:]/(2*np.pi)
psd_k = np.mean(psd, axis=1) * df
psd_v_k = np.mean(psd_v, axis=1) * df


plt.figure()

plt.loglog(k_[1:]/(2*np.pi), psd_k[1:], '--.')

plt.ylabel(r'$S_\eta~[a.u]$')
plt.xlabel(r'$1/\lambda~[m^{-1}$')
plt.grid(which='both')
plt.tight_layout()
plt.show()

plt.figure()
A,exp = 1e2, -3
plt.loglog(k_[1:]/(2*np.pi), psd_v_k[1:], '--.')
plt.loglog(lamb, A*lamb**exp,'--k',label=rf'${exp}$')
plt.ylabel(r'$S_v~[a.u]$')
plt.xlabel(r'$1/\lambda~[m^{-1}$')
plt.grid(which='both')
plt.legend()
plt.tight_layout()
plt.show()
