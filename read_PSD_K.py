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

name_gen = '50Hz_60Hz'
folder = "E:/DATA_FTP/100425/"
# psd = np.load(folder+'spectre_25Hz_hann.npy')
psd_v = np.load(folder+f'spectre_vitesse_{name_gen}_pad2.npy')
psd = np.load(folder+f'spectre_{name_gen}_pad2.npy')

k_ = np.load(folder+f'fft_k_{name_gen}/'+'array_k.npy')
k_ = k_-k_[0]

Nk,Nt = psd.shape
fech = 180


#%%

k_end = 60
df = fech

lamb = k_[1:]/(2*np.pi)
# psd_v[0,:] = 0
# psd_v[1,Nt//2+566:] = 0
# psd_v[1,:Nt//2-566] = 0
# psd_v[2,Nt//2+566:] = 0
# psd_v[2,:Nt//2-566] = 0

psd_k = np.mean(psd[:k_end,:], axis=1) * df
psd_v_k = np.mean(psd_v[:k_end,:], axis=1) * df


plt.figure()
A,exp = 0.6e-10, -1
plt.loglog(lamb, A*lamb**exp,'--k',label=rf'${exp}$')
#plt.loglog(lamb, 1000*A*lamb**(-3),':k',label=rf'${-3}$')

plt.loglog(k_[1:k_end]/(2*np.pi), psd_k[1:], 'ok',mfc='w')
plt.ylabel(r'$S_\eta~[a.u]$')
plt.xlabel(r'$1/\lambda~[m^{-1}$')
plt.grid(which='both')
plt.tight_layout()
plt.legend()
plt.show()

plt.figure()
A,exp = 2e-11, 2
plt.loglog(k_[1:k_end]/(2*np.pi), psd_v_k[1:], '--.')
plt.loglog(lamb, A*lamb**exp,'--k',label=rf'${exp}$')
plt.ylabel(r'$S_v~[a.u]$')
plt.xlabel(r'$1/\lambda~[m^{-1}$')
plt.grid(which='both')
plt.legend()
plt.tight_layout()
plt.show()


#%%
f_array = (np.arange(0,Nt,1)-Nt//2)*fech/(Nt)
f_array2 = (np.arange(0,Nt-1,1)-(Nt-1)//2)*fech/(Nt-1)

pix = 1e-2/37
dk = 2*np.pi/(2*len(k_)*pix)
psd_f = np.mean(psd, axis=0) * dk
psd_v_f = np.mean(psd_v, axis=0) * dk

f_array_ = f_array[9700:11700]

A,exp = 1e-9,-4
plt.figure()

plt.loglog(f_array, psd_f)
plt.loglog(f_array_, A*f_array_**exp, '--k')

plt.xlabel(r'$f~[Hz]$')
plt.ylabel(r'$S_\eta~[..]$')
plt.tight_layout()
plt.show()


A2,exp2 = 1e-7, -2
plt.figure()
plt.loglog(f_array2, psd_v_f)
plt.loglog(f_array_, A2*f_array_**exp2, '--k')

plt.xlabel(r'$f~[Hz]$')
plt.ylabel(r'$S_\eta~[..]$')
plt.tight_layout()
plt.show()
