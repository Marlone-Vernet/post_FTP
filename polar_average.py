#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:17:30 2024

@author: tanu
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','qt5')

def polar_average(M):
    """
    Compute the polar average of a 2D map M, weighting each value by the circumference of the circle at that radius.

    Parameters:
    M (numpy.ndarray): 2D array representing the map.

    Returns:
    r_values (numpy.ndarray): 1D array of radial distances.
    polar_avg (numpy.ndarray): 1D array of the average values of M at each radial distance.
    """
    # Get the shape of the map
    ny, nx = M.shape
    
    # Create coordinate grids
    y, x = np.indices((ny, nx))
    
    # Calculate the center of the map
    center_x = (nx - 1) / 2
    center_y = (ny - 1) / 2
    
    # Calculate the radial distance from the center for each point
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Flatten the arrays
    r_flat = r.flatten()
    M_flat = M.flatten()
    
    # Define radial bins
    r_max = np.max(r_flat)
    bin_edges = np.linspace(0, r_max, min(ny, nx)//2)
    
    # Digitize the radial distances into bins
    bin_indices = np.digitize(r_flat, bin_edges)
    
    # Calculate the weighted average value of M for each bin
    polar_avg = []
    for i in range(1, len(bin_edges)):
        bin_mask = bin_indices == i
        r_bin = r_flat[bin_mask]
        M_bin = M_flat[bin_mask]
        
        if np.any(bin_mask):
            weights = 2 * np.pi * r_bin  # Weight by circumference
            weighted_avg = np.average(M_bin, weights=weights)
            polar_avg.append(weighted_avg)
        else:
            polar_avg.append(0)
    
    # Calculate the radial value for each bin
    r_values = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return r_values, np.array(polar_avg)

# Example usage
M = np.random.random((100, 100))  # Example 2D map
r_values, polar_avg = polar_average(M)

plt.figure()
plt.plot(r_values, polar_avg)

plt.show()