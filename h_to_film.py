# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:08:47 2024

@author: VERNET MARLONE 

"""


import numpy as np
import matplotlib.pyplot as plt 
from scipy import signal
from tqdm import tqdm 
from matplotlib.animation import FuncAnimation
plt.rcParams['animation.ffmpeg_path']='C:/ffmpeg-2024/bin/ffmpeg.exe'
from matplotlib import cm
import h5py

from scipy.interpolate import RectBivariateSpline #interp2d


from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
plt.rcParams['font.size']=20
from mpl_toolkits.mplot3d import axes3d

from matplotlib import pyplot as plt
plt.ion()
plt.close('all')



""" Moovie """


folder = '/home/tanu/data1/DATA_post/150724/h_30Hz_33Hz/' 
folder_save = '/home/tanu/data1/DATA_post/150724/' 
name_movie = 'h_fps15_30Hz_33Hz.mp4'

# h_0 = np.load(folder+f'h_total/h_map_0.npy')

with h5py.File(folder+'h_map_1.jld2', 'r') as file:
    # Access the data
    h_0 = file['h_map'][:].T

N_stop = 12
fft2k = 0
Nx,Ny = np.shape(h_0) # if test height has been checked

X,Y = np.arange(0,Nx,1), np.arange(0,Ny,1)
X_,Y_ = np.meshgrid(Y,X)

n_ = 1000
with h5py.File(folder+f'h_map_{n_}.jld2', 'r') as file:
    # Access the data
    h_0 = file['h_map'][:].T
plt.figure()
plt.pcolormesh(X_,Y_,h_0,vmin=-0.002,vmax=0.002, cmap=cm.Spectral_r)
plt.show()

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111)
def animate(n):
    ax.cla()
    
    # h_k = np.load(folder+f'h_total/h_map_{n}.npy')
    
    with h5py.File(folder+f'h_map_{n}.jld2', 'r') as file:
        # Access the data
        h_k = file['h_map'][:].T
    # ax.pcolormesh(X_, Y_, h_k, vmin=-0.002,vmax=0.002, cmap=cm.Spectral_r)
    ax.plot_surface(X_,Y_,h_k,rstride=10, 
                    cstride=10,
                    vmin=-0.002,
                    vmax=0.002,
                    cmap=cm.Spectral, 
                    antialiased=False)
    ax.set_zlim((-0.002,0.002))

    ax.set_title(f'{n}')

    print(f'{100*n/N_stop} %')
    
    return fig,
  
anim = FuncAnimation(fig = fig, func = animate, frames = range(2,N_stop), interval = 1, repeat = False)
anim.save(folder_save+name_movie, fps=15, writer='ffmpeg',dpi=200)
plt.show()

plt.close('all')