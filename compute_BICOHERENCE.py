# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:59:07 2024

@author: VERNET MARLONE


"""


import numpy as np
from tqdm import tqdm 
import multiprocessing
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline, interp2d
import h5py
import scipy.special as scp
import scipy.fftpack as scf 
import matplotlib.pyplot as plt


# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib','qt5')

""" MASK + PADDING """
xa,xb = 30,30
xc,xd = 30,30

def bicoherence_func(args):
    folder, j, pix_, reduction = args
    
    with h5py.File(folder+f'h_map_{j}.jld2', 'r') as file:
        # Access the data
        h_map = file['h_map'][:].T
    
    nx,ny = h_map.shape
    nx_sub,ny_sub = nx//2, ny//2
    
    h_map[:xa,:] = 0
    h_map[nx-xa:,:] = 0
    h_map[:,:xa] = 0
    h_map[:,ny-xa:] = 0

    """ Hanning in space - ~bof """
    window_x = np.hanning(nx)
    window_y = np.hanning(ny)
    h_map_window_x = h_map * window_x[:,np.newaxis]
    h_map_window_xy = h_map_window_x * window_y
    
    """ 2D fft """
    fft2 = scf.fft2( h_map_window_xy ) * pix_**2 # pour avoir la bonne unite dans la TF 2D
    fft2_shift = scf.fftshift( fft2 )    
    fft2_crop = fft2_shift[nx_sub-reduction:nx_sub+reduction, ny_sub-reduction:ny_sub+reduction]
    fft2_conj = np.conjugate(fft2_crop)
    
    m,n = fft2_crop.shape
    km = np.fft.fftfreq(m).reshape(-1,1)
    kn = np.fft.fftfreq(n).reshape(-1,1)

    bispectrum = np.zeros((m,n), dtype=np.complex64)
    normalization = np.zeros((m,n))

    for i_ in range(m):
        for j_ in range(n):
            
            produit = fft2_crop[i_,j_] * fft2_crop
            bispectrum[i_,j_] = np.sum( produit * fft2_conj[(i_+km[:,0].astype(int))%m, 
                                                            j_+kn.astype(int)%n] )
            normalization[i_,j_] = np.sum(np.abs(produit))
            
            
    return j, normalization, np.abs(fft2_crop), bispectrum



def main(folder,folder_save, N_images, pix_, reduction_):


    with h5py.File(folder+'h_map_1.jld2', 'r') as file:
        # Access the data
        h_map = file['h_map'][:].T

    Nx,Ny = h_map.shape
    
    # declare arrays
    
    norme = np.zeros((2*reduction_,2*reduction_))
    fft2_abs = np.zeros((2*reduction_,2*reduction_))
    bispectre = np.zeros((2*reduction_,2*reduction_), dtype=np.complex64)
    
    # prepare multiprocessing
    num_processes = 18 # multiprocessing.cpu_count() - 1  # Number of CPU cores - 1
    pool = multiprocessing.Pool(processes=num_processes)
    

    items = [(folder, item_k, pix_, reduction_) for item_k in range(1, N_images+1)]
    

    with tqdm(total=N_images, desc="Recording Arrays") as pbar:
        # Process arrays in parallel
        for result in pool.imap(bicoherence_func, items):
            j_, normalization_, fft2_, bispectrum_  = result
            norme+=normalization_
            fft2_abs+=fft2_
            bispectre+=bispectrum_ # average over time !!!
            
            pbar.update(1)

    pool.close()
    pool.join()
    
    norme_bi = (fft2_abs/N_images) * (norme/N_images)
    bicoherence = np.abs(bispectrum_/N_images)/norme_bi
    
    
    return bicoherence


if __name__ == '__main__':
    
    pix = 1e-2/37
    N_images = 18868
    folder = "E:/DATA_FTP/310724/"
    name_gen = '40Hz_50Hz'
    folder_h = folder+f'h_map_{name_gen}/'
    folder_fft = folder+f'fft_k_{name_gen}/'
    dict_ = dict()
    reduction = 30

    bicoh = main(folder_h, 
                 folder, 
                 N_images, 
                 pix, 
                 reduction)

    np.save(folder+f'bicoherence_k1k2_{name_gen}', bicoh)


    with h5py.File(folder_h+'h_map_1.jld2', 'r') as file:
        # Access the data
        h_map = file['h_map'][:].T
    
    nx_,ny_ = h_map.shape
    # check code:
    # j = 3000
    
    # reduction = 30
    # args = folder_h, j, pix, reduction 
    # j_, bicoh = bicoherence_func(args)
    
    nx,ny = bicoh.shape
    dkx,dky = 2*np.pi/(nx_*pix), 2*np.pi/(ny_*pix)
    kx,ky = (np.arange(nx)-nx//2)*dkx , (np.arange(ny)-ny//2)*dky
    KX,KY = np.meshgrid(kx,ky, indexing='ij')
    
    plt.figure()
    plt.pcolormesh(KX/(2*np.pi),
                   KY/(2*np.pi),
                   np.log(bicoh), 
                   cmap='magma')
    plt.colorbar()
    plt.show()
    
