# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:25:10 2024

@author: turbulence
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','qt5')


folder = "E:/DATA_FTP/180225/"
name_file = 'h_y600_sweep120s.jld2'

with h5py.File(folder+name_file, 'r') as file:
    # Access the data
    h_init = file['h_profile'][:].T
    
#%%
gg = 25/(7.5) * 1e3

# plot point vs time
plt.figure()
plt.plot(h_init[:,600]*gg)
plt.show()

pix = 1e-2/37
extrait = h_init[8761,:]*gg
extrait2 = h_init[8746,:]*gg
extrait3 = h_init[8776,:]*gg

nx = len(extrait)
x_ = np.arange(0,nx,1.0)*pix*1e3
plt.figure()
plt.plot(x_,extrait,'-')
plt.plot(x_,extrait2,'-')
plt.plot(x_,extrait3,'-')

plt.show()
