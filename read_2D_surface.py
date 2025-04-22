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


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.sans-serif": "Helvetica",
})

plt.rcParams['font.size'] = 14

folder = "E:/DATA_FTP/150425/h_map_20Hz/"
j = 250

with h5py.File(folder+f'h_map_{j}.jld2', 'r') as file:
    # Access the data
    h_init = file['h_map'][:].T

h_init = h_init[20:-20, 20:-20]

#%%

pix = 1e-2/(33) * 1e3 # MAP tout en mm

Nx,Ny = np.shape(h_init) # if test height has been checked

X,Y = np.arange(0,Nx,1)*pix, np.arange(0,Ny,1)*pix
X_,Y_ = np.meshgrid(X,Y,indexing='ij')

#%%

plt.figure()
plt.plot(-h_init[676,:])
plt.show()

#%%

mean_y = np.mean(-h_init,axis=0)

plt.figure()
plt.plot(mean_y)
plt.show()


#%%
# from matplotlib.colors import LightSource
# light = LightSource(azdeg=315,altdeg=45)

Vmin = -2
Vmax = 2

h_zero = np.repeat(mean_y[np.newaxis,:], Ny, axis=0 )
gain = 1.6 * 1e3 # map en mm

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')
ZZ = (-h_init-h_zero)*gain
# rgb = light.shade(ZZ, cmap=cm.Grays)
# [250:750,250:750]
cc = ax.plot_surface(X_,Y_,ZZ,rstride=6, 
                cstride=6,
                cmap='Blues',
                linewidth=0,
                vmin=Vmin,
                vmax=Vmax,
                antialiased=False) # [250:750]


#ax.set_zlim((-0.6,0.6))
ax.view_init(40,126)
ax.xaxis.pane.fill=False
ax.yaxis.pane.fill=False
ax.zaxis.pane.fill=False
ax.xaxis.pane.set_edgecolor('grey')
ax.yaxis.pane.set_edgecolor('grey')
ax.zaxis.pane.set_edgecolor('grey')

ax.grid(False)
ax.set_xlabel(r'$x~{\normalfont [mm]}$')
ax.set_ylabel(r'$y~{\normalfont [mm]}$')
ax.set_zlabel(r'$z~{\normalfont [mm]}$')

#plt.colorbar(cc, ax=ax)

plt.tight_layout()
#plt.savefig(f"C:/Users/turbulence/Desktop/fig_map/map_j{j}_051124_s")
plt.show()



plt.figure()
plt.pcolormesh(X_, Y_, h_init, cmap='turbo')
plt.show()


#%%
import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.graph_objects as go
from plotly.offline import plot
gain = 1.6 * 1e3 # map en mm



fig = go.Figure(go.Surface(x=X_, y=Y_, z=-h_init*gain))
pio.show(fig)

#%%

fig = go.Figure(go.Surface(x=X_, y=Y_, z=-h_init*gain))
#fig.update_layout(title='3d plot')
frames = []
for j in range(1,10):
    with h5py.File(folder+f'h_map_{j}.jld2', 'r') as file:
        # Access the data
        h_init = file['h_map'][:].T
    surf = -h_init*gain
    frames.append( go.Frame(data=[{'type':'surface','z': surf}]) )

fig.frames = frames
#fig.layout.updatemenus = [{'args': [None, {'frames': {'duration':100, 'redraw': False},
#                                           'fromcurrent':True, 'transition': {'duration':50}}],
#                           'label': 'Play',
#                           'method': 'animate'}
                          
#]

plot(fig)

#%%


