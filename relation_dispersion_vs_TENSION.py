#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:02:06 2024

@author: VERNET MALRONE

"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks
from scipy import signal
from tqdm import tqdm 
from matplotlib import cm
import scipy.signal as scs
import h5py

import pickle as pk

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')
plt.rcParams['font.size']=20


#%%

""" open spectrum """

folder1 = '/home/tanu/data1/DATA_post/080724/'
data1 = dict( np.load(folder1+'spectre_vs_t_sweep120s_amp3.npy', allow_pickle=True).item() )

folder2 = '/home/tanu/data1/DATA_post/090724/'
data2 = dict( np.load(folder2+'spectre_vs_t_sweep120s_amp3.npy', allow_pickle=True).item() )

k1 = data1['k']
psd_t1 = data1['psd_t']

k2 = data2['k']
psd_t2 = data2['psd_t']

k_index = dict()
psd_t = dict()

k_index.update({"1":k1})
k_index.update({"2":k2})
psd_t.update({"1":psd_t1})
psd_t.update({"2":psd_t2})

p = 1e-2/(40) # conversion en m/pixel
lmin = p



#%%


""" Relation dispersion theorique """

H = 0.1
h = 1e-3
E = 70e3
nu = 0
g = 9.81
rho = 1e3
T_ = 1.3 #np.array([0,0.5,1,2,4,10,20])
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


x_k = 60

k_v = dict()
f_v = dict()

N_item = 2
list_item = ["1","2"]

for item in range(N_item):
    
    psd_eta = psd_t[list_item[item]]
    psd_v = np.diff(psd_eta, axis=1) # compute velocity, time is last coordinate
    
    Nk,Nt = psd_v.shape
    window = np.hanning(Nt)
    psd_v_window = psd_v * window
    fft_temps = np.fft.fft(psd_v_window, axis=1)
    relation_dispersion = np.log( np.fft.fftshift( fft_temps ) )

    k_ = k_index[ list_item[item] ]
    dk = 2*np.pi/(lmin * 2*Nk ) # extra factor 2 due to average over theta, lost of the real size ! 
    km = k_ * dk
    fm = np.arange(-int(Nt/2)+1,int(Nt/2)+1,1)*(faq/(Nt))
    mid_t = Nt//2
    k_shift = km[Nk//2]
    
    k_LDR,f_LDR = np.zeros((mid_t,)), np.zeros((mid_t,))
    for j in tqdm(range(mid_t)):
        max_value = max(relation_dispersion[:,mid_t+j])
        idx = np.where( relation_dispersion[:,mid_t+j] == max_value )
        
        k_LDR[j] = km[idx[0][0]]-k_shift
        f_LDR[j] = fm[mid_t+j]
        
    k_v.update({list_item[item]:k_LDR})
    f_v.update({list_item[item]:f_LDR})


#%%

""" plot scatter relation dispersion """
symbol = ['o','v','s','^']


plt.figure()

for k in range(N_item):
    plt.scatter(k_v[list_item[k]],f_v[list_item[k]], marker=symbol[k] )


plt.xlabel(r'$k~[m^{-1}]$')
plt.ylabel(r'$f~[Hz]$')
plt.grid()
plt.tight_layout()
plt.show()