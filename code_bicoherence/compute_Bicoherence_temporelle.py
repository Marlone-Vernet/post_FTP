import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scs 
import h5py
import scipy.fftpack as scf 
from tqdm import tqdm 

def Bicoherence(S1, S2, faq, dt_choose):

    if len(S1)!=len(S2):
        raise AttributeError("S1 and S2 must have the same length")

    f1, t1, spectre_1 = scs.spectrogram(S1, fs = faq, nperseg=dt_choose, nfft=dt_choose, mode='complex')
    f2, t2, spectre_2 = scs.spectrogram(S2, fs = faq, nperseg=dt_choose, nfft=dt_choose, mode='complex')

    spectre_1 = np.transpose(spectre_1, [1, 0])
    spectre_2 = np.transpose(spectre_2, [1, 0])

    arg = np.arange(f1.size//2, dtype=np.int64)
    sumarg = arg[:, None] + arg[None, :]
    Cross_product = spectre_1[:,arg,None]*spectre_1[:,None,arg]*np.conjugate(spectre_2[:,sumarg])
    bispectrum = np.abs(np.mean(Cross_product, axis=0))**2

    norm1 = np.mean(np.abs(spectre_1[:,arg,None]*spectre_1[:,None,arg])**2, axis=0)
    norm2 = np.mean(np.abs(np.conjugate(spectre_2[:,sumarg]))**2, axis=0)
    normalisation = norm1*norm2

    return f1, f2, bispectrum/normalisation


# def Bicoherence_2D(args):
#     folder, j, pix_, xa = args
    
#     with h5py.File(folder+f'h_map_{j}.jld2', 'r') as file:
#         # Access the data
#         h_map = file['h_map'][:].T
    
#     nx,ny = h_map.shape
    
#     h_map[:xa,:] = 0
#     h_map[nx-xa:,:] = 0
#     h_map[:,:xa] = 0
#     h_map[:,ny-xa:] = 0

#     """ Hanning in space - ~bof """
#     window_x = np.hanning(nx)
#     window_y = np.hanning(ny)
#     h_map_window_x = h_map * window_x[:,np.newaxis]
#     h_map_window_xy = h_map_window_x * window_y
    
#     """ 2D fft """
    
#     fft2 = scf.fft2( h_map_window_xy ) * pix_**2 # pour avoir la bonne unite dans la TF 2D
#     fft2_shift = scf.fftshift( fft2 )
    
#     nx_sub, ny_sub = nx//2, ny//2
#     conjugate_ = np.zeros((nx_sub, ny_sub))
#     bispectrum_2d = np.zeros((nx_sub,ny_sub,nx_sub,ny_sub))
#     for i in tqdm(range(nx_sub)):
#         for j in range(ny_sub):
#             conjugate_[i,j] = fft2_shift[i+j,i+j]
    
#     part_mul = fft2_shift[:nx_sub,:ny_sub]*np.conjugate(conjugate_)
#     bispectrum_2d = np.tensordot(fft2_shift[:nx_sub,:ny_sub], part_mul, axes=0)
    
#     norm1 = np.abs(fft2_shift)
#     norm_conj = np.abs(conjugate_)
#     return j, norm1, norm_conj, bispectrum_2d


# def compute_2d_bicoherence(args):

#     folder, j, pix_, xa, reduction = args
    
#     with h5py.File(folder+f'h_map_{j}.jld2', 'r') as file:
#         # Access the data
#         h_map = file['h_map'][:].T
    
#     nx,ny = h_map.shape
#     nx_sub = nx//2
#     ny_sub = ny//2

#     h_map[:xa,:] = 0
#     h_map[nx-xa:,:] = 0
#     h_map[:,:xa] = 0
#     h_map[:,ny-xa:] = 0

#     # Compute the 2D Fourier Transform
#     F_ = scf.fft2(h_map)
#     F = scf.fftshift(F_)

#     Fe = F[nx_sub-reduction:nx_sub+reduction,
#            ny_sub-reduction:ny_sub+reduction]
#     Fe_conj = np.conj(Fe)

#     # Reduction in k
#     M,N = Fe.shape

#     # Create a grid of frequency indices
#     k1 = np.fft.fftfreq(M).reshape(-1, 1)
#     k2 = np.fft.fftfreq(N).reshape(1, -1)
    
#     # Initialize bispectrum and normalization arrays
#     bispectrum = np.zeros((M, N), dtype=complex)
#     normalization = np.zeros((M, N))
    
#     # Compute the bispectrum and normalization
#     for f1 in tqdm(range(M)):
#         for f2 in range(N):
#             prod = Fe[f1, f2] * Fe
#             bispectrum[f1, f2] = np.sum(prod * Fe_conj[(f1 + k1[:, 0].astype(int)) % M, (f2 + k2[0, :].astype(int)) % N])
#             normalization[f1, f2] = np.sum(np.abs(prod)**2)
    
#     # Compute the bicoherence
#     bicoherence = np.abs(bispectrum)**2 / (normalization * np.abs(Fe)**2)
#     # Handle division by zero
#     bicoherence = np.nan_to_num(bicoherence)
    
#     return bicoherence
