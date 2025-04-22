import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scs 
import h5py
import scipy.fftpack as scf 
from tqdm import tqdm 
from scipy.signal import spectrogram



def compute_triple_correlation_spectrogram(s, fs, nperseg, max_freq, f_test):
    """
    Computes the triple correlation C_3(f1, f2, f3) using the spectrogram of the signal over all frequency combinations.
    
    Parameters:
        s (array): The input time-domain signal s(t).
        fs (float): The sampling frequency of the signal.
        nperseg (int): The number of samples per segment for the spectrogram.
    
    Returns:
        C3 (3D numpy array): The triple correlation C_3(f1, f2, f3) for all frequency combinations.
    """
    # Compute the spectrogram (returns f, t, and Sxx)
    f, t, Sxx = spectrogram(s, fs, nperseg=nperseg, nfft=nperseg, mode='complex')
    
    # Get the number of frequency bins
    f_filtre__ = np.where(f < max_freq)[0]
    num_freqs = len(f_filtre__)
    
    # Initialize a 3D array for the triple correlation
    C3 = np.zeros((num_freqs, num_freqs), dtype=complex)
    
    idx_f2 = np.argmin(np.abs(f - f_test))
    
    # Loop through all combinations of frequency bins f1, f2, and f3
    for idx_f1 in tqdm(range(num_freqs)):
        for idx_f3 in range(num_freqs):
            # Compute the product of the spectrogram components at f1, f2, f3
            product = np.conj(Sxx[idx_f1, :]) * Sxx[idx_f2, :] * Sxx[idx_f3, :]
            norm__ = np.sqrt( np.sum(np.abs(Sxx[idx_f2, :] * Sxx[idx_f3, :])**2)/len(product) * np.sum(np.abs(Sxx[idx_f1, :])**2)/len(product) )
                # Compute the temporal average (mean over time bins)
            C3[idx_f1, idx_f3] = np.sum(product)/len(product)/norm__
    
    
    
    print(f"### working frequency : {round(f[idx_f2],1)} ###")
    return f[:num_freqs], C3


def compute_Bicoherence_spectrogram(s, fs, nperseg, max_freq):
    """
    Computes the triple correlation C_3(f1, f2, f3) using the spectrogram of the signal over all frequency combinations.
    
    Parameters:
        s (array): The input time-domain signal s(t).
        fs (float): The sampling frequency of the signal.
        nperseg (int): The number of samples per segment for the spectrogram.
    
    Returns:
        C3 (3D numpy array): The triple correlation C_3(f1, f2, f3) for all frequency combinations.
    """
    # Compute the spectrogram (returns f, t, and Sxx)
    f, t, Sxx = spectrogram(s, fs, nperseg=nperseg, nfft=nperseg, mode='complex')
    
    # Get the number of frequency bins
    f_filtre__ = np.where(f < max_freq)[0]
    num_freqs = len(f_filtre__)
    
    # Initialize a 3D array for the triple correlation
    B3 = np.zeros((num_freqs, num_freqs), dtype=complex)
    
    
    # Loop through all combinations of frequency bins f1, f2, and f3
    for idx_f2 in tqdm(range(num_freqs)):
        for idx_f3 in range(num_freqs):
            f2 = f_filtre__[idx_f2]
            f3 = f_filtre__[idx_f3]
            f1 = f2+f3
            idx_f1 = np.argmin(np.abs(f_filtre__ - f1))
            # Compute the product of the spectrogram components at f1, f2, f3
            product = np.conj(Sxx[idx_f1, :]) * Sxx[idx_f2, :] * Sxx[idx_f3, :]
            norm__ = np.sqrt( np.sum(np.abs(Sxx[idx_f2,:] * Sxx[idx_f3,:])**2)/len(product) * np.sum(np.abs(Sxx[idx_f1,:])**2)/len(product) )
            # Compute the temporal average (mean over time bins)
            B3[idx_f2, idx_f3] = np.sum(product)/len(product)/norm__
    
    
    
    print(f"### working frequency : {round(f[idx_f2],1)} ###")
    return f[:num_freqs], B3


