#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:00:08 2024

@author: VERNET MARLONE
"""


import numpy as np
from tqdm import tqdm 
import multiprocessing
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline, interp2d
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','qt5')


folder = "/Users/vernet/Desktop/hydroelastic_project/test_2/h_map/"
j = 50

with h5py.File(folder+f'h_map_{j}.jld2', 'r') as file:
    # Access the data
    h_init = file['h_map'][:].T

h_init = h_init[20:-20, 20:-20]

#%%

pix = 1e-2/(37) * 1e3 # MAP tout en mm

Nx,Ny = np.shape(h_init) # if test height has been checked

X,Y = np.arange(0,Nx,1)*pix, np.arange(0,Ny,1)*pix
X_,Y_ = np.meshgrid(Y,X)

#%%

plt.figure()

plt.plot(-h_init[:,400])

plt.show()
#%%

gain = 25/(7.5) * 1e3 # map en mm

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X_,Y_,-h_init*gain,rstride=10, 
                cstride=10,
                cmap=cm.Spectral, 
                antialiased=False)

plt.show()






