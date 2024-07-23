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
# psd = np.load(folder+'spectre_vitesse_vent_20Hz.npy')
psd = np.load(folder+'spectre_vent_20Hz.npy')

k_ = np.load(folder+'fft_k_vent_20Hz/'+'array_k.npy')
k_ = k_-k_[0]

Nk,Nt = psd.shape


fech = 120


#%%

H = 0.1
h = 1e-3
E = 70e3
nu = 0
g = 9.81
rho = 1e3
T_ = 2.0#np.array([0,0.5,1,2,4,10,20])
B = h**3*E/(12*(1-nu**2))
# k_c = np.sqrt(T_/B)
# k_g = (rho*g/B)**0.25

lambda_ = np.arange(5e-3, 0.6, 1e-3)
k_th = 2*np.pi/lambda_


def relation(T,k,h,g,B):
    return np.sqrt( g*k + T*k**3/rho + B*k**5/rho)/(2*np.pi)
def relation_th(T,k,h,g,B):
    return np.sqrt( (g*k + T*k**3/rho + B*k**5/rho) * np.tanh(k*H))/(2*np.pi)

f_th = relation(T_, k_th, h, g, B)
f_th3 = relation(T_, k_th/3, h, g, B)
f_th2 = relation(T_, k_th/2, h, g, B)

f_th0 = relation(0, k_th, h, g, B)
f_th_tan = relation_th(T_, k_th, h, g, B)

#%%

""" compute relation dispersion """

faq = 120
if Nt%2==0:
    f = np.arange(-int(Nt/2),int(Nt/2),1)*(faq/(Nt))
else:
    f = np.arange(-int(Nt/2),int(Nt/2)+1,1)*(faq/(Nt))


relation_dispersion = np.log(psd)

mid_k = Nk//2
mid_t = 0 
x_k = 60

""" PLOTsss """


K_,F_ = np.meshgrid(k_, f, indexing='ij')

k_end = 30

plt.figure(figsize=(7,6))
plt.pcolormesh(K_[:k_end,:]/(2*np.pi), 
                F_[:k_end,:], 
                relation_dispersion[:k_end,:], 
                shading='gouraud',
                vmin=-11,
                vmax=-6,
                cmap='turbo')
plt.colorbar()
plt.plot(k_th/(2*np.pi), f_th, '-w')
plt.plot(-k_th/(2*np.pi), f_th, '-w')
plt.xlim((0,600/(2*np.pi)))
plt.ylim((0,60))
plt.xlabel(r'$1/\lambda~[m^{-1}]$')
plt.ylabel(r'$f~[Hz]$')
plt.title(rf'T={T_}N/m')
plt.tight_layout()
plt.show()




#%%


time_ = np.arange(0,Nt,1)*1/fech

Kspec, TIME = np.meshgrid(k_, time_, indexing='ij')

plt.figure()
plt.pcolormesh(TIME[:,:], 
               Kspec[:,:], 
               relation_dispersion[:,:],
               cmap='turbo')

plt.colorbar()
plt.xlabel(r'$t~[s]$')
plt.ylabel(r'$k_x~[m^{-1}]$')
plt.tight_layout()
plt.show()

