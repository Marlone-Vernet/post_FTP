# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:59:07 2024

@author: VERNET MARLONE


"""


import numpy as np
from tqdm import tqdm 
import h5py


if __name__ == '__main__':

    faq = 120
    pix = 1e-2/40
    N_images = 14999
    folder = "/home/tanu/data1/DATA_post/180724/"
    folder_h = folder+'h_vent_20Hz/'
    folder_fft = folder+'fft_k_vent_20Hz/'
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
    
    window = np.hanning(N_images)
    window_v = np.hanning(N_images-1)
    spectre_window = spectre_vs_t*window
    spectre_window_v = spectre_v_t*window_v
    spectre = np.fft.fftshift( np.fft.fft( spectre_window, axis=1 ), axes=1) /  faq # unite TF en temps
    spectre_v = np.fft.fftshift( np.fft.fft( spectre_window_v, axis=1 ), axes=1) / faq # unite TF en temps
    
    Nk,Nt = spectre.shape
    fenetre_temps = Nt * dt
    
    psd = abs(spectre)**2 / (L_box * fenetre_temps)
    psd_v = abs(spectre_v)**2 /(L_box * fenetre_temps)
    np.save(folder+'spectre_vent_20Hz',psd)
    np.save(folder+'spectre_vitesse_vent_20Hz',psd_v)

    
    
