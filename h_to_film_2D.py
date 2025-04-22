# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 17:12:43 2025

@author: VERNET
"""



import numpy as np
import matplotlib.pyplot as plt 
from scipy import signal
from tqdm import tqdm 
from matplotlib.animation import FuncAnimation
plt.rcParams['animation.ffmpeg_path']='C:/ProgramData/anaconda3/pkgs/ffmpeg-6.1.0-gpl_h1627b0f_100/Library/bin/ffmpeg.exe' # C:\ProgramData\anaconda3\pkgs\ffmpeg-6.1.0-gpl_h1627b0f_100\Library\bin

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.sans-serif": "Helvetica",
})

plt.rcParams['font.size'] = 14

from matplotlib import cm
import h5py

from scipy.interpolate import RectBivariateSpline #interp2d


from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
from mpl_toolkits.mplot3d import axes3d

from matplotlib import pyplot as plt
plt.ion()
plt.close('all')



""" Moovie """
stride = 5
N_stop = 1000
fft2k = 0
col_map = 'turbo'

Vmin = -2
Vmax = 2
img_start = 250

name_gen = '20Hz'
folder = f"E:/DATA_FTP/150425/h_map_{name_gen}/"
#folder_save = "E:/DATA_FTP/111024/"
folder_save = "C:/Users/turbulence/Desktop/fig_map/"
name_movie = f'2D_h_fps10_{name_gen}_150525_N{N_stop}_cut_final.mp4'

# h_0 = np.load(folder+f'h_total/h_map_0.npy')

with h5py.File(folder+'h_map_1.jld2', 'r') as file:
    # Access the data
    h_0 = file['h_map'][:].T

gain = 1.6
xc = 30
pix = 1e-2/(33)


Nx,Ny = np.shape(h_0[xc:-xc,xc:-xc]) # if test height has been checked

X,Y = np.arange(0,Nx,1), np.arange(0,Ny,1)
X_,Y_ = np.meshgrid(X*pix*1e3,Y*pix*1e3, indexing='ij')

n_ = 1
with h5py.File(folder+f'h_map_{n_}.jld2', 'r') as file:
    # Access the data
    h_0 = file['h_map'][:].T
    
mean_y = np.mean(-h_0[xc:-xc,xc:-xc],axis=0)


# plt.figure()
# plt.pcolormesh(X_,Y_,(h_0[xc:-xc,xc:-xc]-mean_y)*gain,vmin=-0.002,vmax=0.002, cmap=cm.Spectral_r)
# plt.show()

fig = plt.figure(figsize=(6,6), dpi=300)
ax = fig.add_subplot(111)
with h5py.File(folder+f'h_map_{img_start}.jld2', 'r') as file:
    # Access the data
    h_k = file['h_map'][:].T
# ax.pcolormesh(X_, Y_, h_k[xc:-xc,xc:-xc]*gain, vmin=-0.002,vmax=0.002, cmap=cm.Spectral_r)
mean_y =  np.mean(-h_k[xc:-xc,xc:-xc],axis=1)
h_zero = np.repeat(mean_y[:,np.newaxis], Ny, axis=1 )
hh = (h_k[xc:-xc,xc:-xc]+h_zero)*gain

cc = ax.pcolormesh(X_, Y_, (hh-np.mean(hh))*1e3, 
                   cmap=col_map,
                   vmin=Vmin, 
                   vmax=Vmax)

cbar = fig.colorbar(cc, ax=ax, label = r"$\eta$~$[\mathrm{mm}]$")


def animate(n):
    ax.cla()
    
    # h_k = np.load(folder+f'h_total/h_map_{n}.npy')
    
    with h5py.File(folder+f'h_map_{img_start+n}.jld2', 'r') as file:
        # Access the data
        h_k = file['h_map'][:].T
    # ax.pcolormesh(X_, Y_, h_k[xc:-xc,xc:-xc]*gain, vmin=-0.002,vmax=0.002, cmap=cm.Spectral_r)
    mean_y =  np.mean(-h_k[xc:-xc,xc:-xc],axis=1)
    h_zero = np.repeat(mean_y[:,np.newaxis], Ny, axis=1 )
    hh = (h_k[xc:-xc,xc:-xc]+h_zero)*gain

    plt.pcolormesh(X_, Y_, (hh-np.mean(hh))*1e3, 
                   cmap=col_map, 
                   vmin=Vmin, 
                   vmax=Vmax)

    cbar.update_ticks()

    
    ax.grid(False)
    ax.set_xlabel(r'$x~[\mathrm{mm}]$')
    ax.set_ylabel(r'$y~[\mathrm{mm}]$')


    #ax.set_title(f'{n}')

    print(f'{100*n/N_stop} %')
    
    return fig,
  
anim = FuncAnimation(fig = fig, func = animate, frames = range(2,N_stop), interval = 1, repeat = False)
anim.save(folder_save+name_movie, fps=10
          , writer='ffmpeg',dpi=300)
plt.show()

plt.close('all')


