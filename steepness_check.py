# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 09:50:44 2025

@author: VERNET 
"""


import numpy as np
import matplotlib.pyplot as plt 

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')
plt.rcParams['font.size']=20

#%%

faq = 120
folder = "E:/DATA_FTP/071124/"
name_gen = 'T8'


data = np.load(folder+f'data_vs_t_{name_gen}_pad2_n1000.npy', allow_pickle=True)
dict_ = data.item()

steepness = dict_["steepness_t"]


plt.figure()
plt.plot(steepness)
plt.tight_layout()
plt.show()

print(np.mean(steepness))