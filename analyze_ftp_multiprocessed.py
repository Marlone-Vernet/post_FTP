# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:34:38 2024

@author: Marlone VERNET

Multiprocessed calcul of h from the Cobelli Method

"""
import numpy.ma as ma
import numpy as np
from PIL import Image

from scipy.signal import find_peaks
from scipy import signal
from tqdm import tqdm 
import skimage.restoration as skr
import matplotlib.pyplot as plt

import multiprocessing
from numba import jit, objmode

from pathlib import Path


#%%


## SIMPLE CODE TO OBTAIN THE FREE SURFACE DEFORMATION FIELD
## 
## Requires:
##     Images to analyze:
##         - gray
##         - reference
##         - deformed
##     Parameters:
##         - L  : Distance between the water surface and the videoprojector/camera
##         - D  : Distance between the camera and the videoprojector
##         - p  : pixel size
##         - n  : parameter for the filter (typically 1)
##         - th : parameter for the filter TO BE DETERMINED (between 0.1 and 0.9)
## 


def calculate_phase_diff_map_1D(dY, dY0, th, ns):

    """
    # % Basic FTP treatment.
    # % This function takes a deformed and a reference image and calculates the phase difference map between the two.
    # %
    # % INPUTS:
    # % dY	= deformed image
    # % dY0	= reference image
    # % ns	= size of gaussian filter if =1, then Hann windows is used in Tukey
    # %
    # % OUTPUT:
    # % dphase 	= phase difference map between images
    """

    nx, ny = np.shape(dY)
    phase0 = np.zeros_like(dY0)
    phase  = np.zeros_like(dY)

    fY0 = np.fft.fft(dY0, axis=1)
    fY = np.fft.fft(dY, axis=1)
    
    imax=np.argmax(np.abs(fY0[int(np.floor(nx/2)),9:int(np.floor(ny/2))]))
    ifmax=imax+9

    HW=np.round(ifmax*th)
    W=2*HW+1
    win=signal.windows.tukey(int(W),ns)
    
    win2D = np.tile(win, (nx,1))

    gaussfilt1D= np.zeros((nx,ny))
    gaussfilt1D[:,int(ifmax-HW):int(ifmax-HW+W)] = win2D

    # Multiplication by the filter
    Nfy0 = fY0 * gaussfilt1D
    Nfy = fY * gaussfilt1D        

    Ny0=np.fft.ifft(Nfy0, axis=1)
    Ny=np.fft.ifft(Nfy, axis=1)    

    phase0 =  np.unwrap( np.unwrap( np.angle(Ny0), axis=1 ), axis=0 )
    phase = np.unwrap( np.unwrap( np.angle(Ny), axis=1  ), axis=0 )
    
    
    # Definition of the phase difference map
    dphase_ = phase-phase0

    return dphase_ - np.mean(dphase_)


@jit(nopython=True)
def height_map_from_phase_map(dphase, L, D, p):
    """
    Converts a phase difference map to a height map using the phase to height
    relation.
    
    INPUTS:
         dphase    = phase difference map (already unwrapped)
         L         = distance between the reference surface and the plane of the entrance  pupils
         D         = distance between centers of entrance pupils
         p         = wavelength of the projected pattern (onto the reference surface)
         spp       = physical size of the projected pixel (as seen onto the reference  surface)
        
         OUTPUT:
            h         = height map of the surface under study
"""
    return -L*dphase/(2*np.pi/p*D-dphase)



""" compute & record h(x,y) - MULTIPROCESSED """



def compute_h(args):
    """
    Parameters
    ----------
    k           : Iteration index over which the multiprocessing is done
    name_def    : Name of the deformed image
    folder      : Working path 
    folder_der  : Folder of the deformed images
    resfactor   : Factor computed to remove the mean of the reference image
    gray        : Gray image to remove 
    ref_m_gray  : Reference minus the gray image
    th          : Parameter for the filter TO BE DETERMINED (between 0.1 and 0.9)
    L           : Distance between the water surface and the videoprojector/camera
    D           : Distance between the camera and the videoprojector
    pspp        : wavelength of the pattern size in m
    n           : parameter for the filter (typically 1)
    
    Returns
    -------
    Save npy file of the 2D h(x,y) height field
     
    """
    k, files_def, folder, folder_def, resfactor, gray, ref_m_gray, th, n, L, D, pspp = args
    
    name_deformed_k = files_def[k]
    deformed_k = np.array(Image.open(folder_def + name_deformed_k)).T

    def_m_gray_k = deformed_k - resfactor * gray
    DELTA_PHI = calculate_phase_diff_map_1D(def_m_gray_k, ref_m_gray, th, n) 
    
    dphase_k = np.arctan2( np.sin(DELTA_PHI), np.cos(DELTA_PHI) )
    height_k = height_map_from_phase_map(dphase_k, L, D, pspp)
    
    np.save(folder + f'h_total/h_map_{k-1}.npy', height_k)
    
    return k



print('So far so good')


def main():
    num_processes = multiprocessing.cpu_count() - 1  # Number of CPU cores
    pool = multiprocessing.Pool(processes=num_processes)
        
    items = [(k, files_def, folder, folder_def, resfactor, GRAY, ref_m_gray, th, n, L, D, pspp) for k in range(1, N_images)]

    with tqdm(total=N_images, desc="Recording Arrays") as pbar:
        # Process arrays in parallel
        results = []
        for result in pool.imap(compute_h, items):
            results.append(result)
            pbar.update(1)

    pool.close()
    pool.join()


if __name__ == '__main__':
    
    """ ATTENTION files NOT SORTED CODE CANNOT BE USED AS IT is """
    
    #### TO BE CHECK !!! ####
    N_images = 250
    N_ref = 250
    
    folder = 'D:/Marlone/Profilometrie/100624/'
    folder_def = folder+'deformed/'
    folder_ref = folder+'ref/'
    folder_gray = folder+'gray/'
    
    directory_def = Path(folder)/folder_def
    directory_ref = Path(folder)/folder_ref
    directory_gray = Path(folder)/folder_gray

    files_def = [f.name for f in directory_def.iterdir()]
    files_ref = [f.name for f in directory_ref.iterdir()]
    files_gray = [f.name for f in directory_gray.iterdir()]

    name_gray = files_gray[0]
    name_ref = files_ref[0]
    
    gray = np.array( Image.open(folder_gray+name_gray) ).T
    reference = np.array( Image.open(folder_ref+name_ref) ).T
    
    
    plt.figure()
    plt.plot(reference[250,:],'--o')
    plt.show()
    
    L = 1.40
    D = 0.68
    p = 1e-2/(40) # conversion en m/pixel
    n = 1
    th = 0.5 # taille du filtre 
    lin_idx_grating = 500
    
    
    resfactor_test = np.mean(reference)/np.mean(gray)
    ref_m_gray_test = reference - resfactor_test*gray

    # Calculate wavelength of the projected pattern
    line_ref = np.average(ref_m_gray_test[:,:],axis=0)
    peaks, _ = find_peaks(line_ref, height=0)

    wavelength_pix = np.mean(np.diff(peaks))
    pspp = p*wavelength_pix
    
    """ compute reference & back ground """
    
    REFERENCE = 0
    GRAY = 0
    
    
    for i in tqdm(range(N_ref)):
        
        name_ref_i = files_ref[i]
        ref_i = np.array( Image.open(folder_ref+name_ref_i) ).T
        
        name_gray_i = files_gray[i]
        gray_i = np.array( Image.open(folder_gray + name_gray_i) ).T
        
        REFERENCE+= ref_i/N_ref
        GRAY+= gray_i/N_ref
    
    
    resfactor = np.mean(REFERENCE)/np.mean(GRAY)
    ref_m_gray = REFERENCE - resfactor*GRAY
        
    
    main()
    



