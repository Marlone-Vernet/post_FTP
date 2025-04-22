# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:34:36 2025

@author: turbulence
"""

from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline, interp2d
import numpy as np


def polar_average(array_2D,pix_):
    
    nx,ny = array_2D.shape
    
    midx, midy = nx//2, ny//2
    
    x = np.arange(0,nx,1) - midx
    y = np.arange(0,ny,1) - midy
    
    t = np.linspace(0, 2 * np.pi, 129)
    
    if nx<ny:
        mid_i = midx
        r_i = x
        r_ = r_i[mid_i:]
        r, theta = np.meshgrid(r_i[mid_i:], t[:128], indexing = 'ij')

    elif nx>ny:
        mid_i = midy
        r_i = y
        r_ = r_i[mid_i:]
        r, theta = np.meshgrid(r_i[mid_i:], t[:128], indexing = 'ij')
        
    else:
        mid_i = midy
        r_i = y
        r_ = r_i[mid_i:]
        r_ = r_ + (r_[1]-r_[0])/2        
        #k, theta = np.meshgrid(k_i[mid_i:], t[:128], indexing = 'ij')
        r, theta = np.meshgrid(r_, t[:128], indexing = 'ij')
    
    xp = r * np.cos(theta)
    yp = r * np.sin(theta)   

    dtheta = t[1] - t[0]
    
    interpolator_real = RegularGridInterpolator((x, y), array_2D, bounds_error=False, fill_value=0)

    # Perform the interpolation
    points = np.vstack([xp.ravel(), yp.ravel()]).T
    Interp_array = interpolator_real(points)
    
    result_2D = np.reshape(Interp_array, (r_.shape[0], 128))
    
    Nr = len(r_)
    dr = pix_ / ( 2 * Nr)
    result_averaged = np.sum(result_2D, axis=1) * dtheta * r_ * dr # dk donne la bonne dimension a integration en theta 

    return result_averaged
