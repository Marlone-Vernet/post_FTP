# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:59:07 2024

@author: VERNET MARLONE


"""


import numpy as np
from tqdm import tqdm 
import h5py


if __name__ == '__main__':

    faq = 180
    pix = 1e-2/33
    N_images = 3999
    folder = "E:/DATA_FTP/150425/"
    name_gen = '50Hz_100Hz'
    
    # folder_h = folder+f'h_map_{name_gen}/'
    folder_fft = folder+f'fft_k_{name_gen}/'
    k_ = np.load(folder_fft+'array_k.npy')    
    Nk = len(k_)
    min_ = min(k_)
    if min_ != 0:
        L_box = 2*np.pi/min_
    else:
        L_box = 2*np.pi/k_[1]
    
    spectre_vs_t = np.zeros((Nk,N_images), dtype = np.complex64)
    for j in tqdm(range(N_images)):
        spectre_vs_t[:,j] = np.load(folder_fft+f'fft_k_{j}.npy',allow_pickle=True)
    
    dt = 1/faq
    spectre_v_t = np.diff(spectre_vs_t,axis=1) / dt # pour avoir unite de vitesse
    
    window = np.blackman(N_images)
    window_v = np.blackman(N_images-1)
    
    spectre_window = spectre_vs_t*window
    spectre_window_v = spectre_v_t*window_v
    spectre = np.fft.fftshift( np.fft.fft( spectre_window, axis=1 ), axes=1) /  faq # unite TF en temps
    spectre_v = np.fft.fftshift( np.fft.fft( spectre_window_v, axis=1 ), axes=1) / faq # unite TF en temps
    
    Nk,Nt = spectre.shape
    fenetre_temps = Nt * dt
    
    psd = abs(spectre)**2 / (L_box**2 * fenetre_temps) # normalization
    psd_v = abs(spectre_v)**2 /(L_box**2 * fenetre_temps) # normalization
    np.save(folder+f'spectre_{name_gen}_pad2',psd)
    np.save(folder+f'spectre_vitesse_{name_gen}_pad2',psd_v)

    
    
