# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:21:50 2024

@author: VERNET MARLONE 

"""

import numpy as np
import matplotlib.pyplot as plt 

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')
plt.rcParams['font.size']=20

#%%

faq = 180
folder = "E:/DATA_FTP/100425/"
#data2 = dict( np.load(folder2+'spectre_vs_t_k2D_20Hz.npy', allow_pickle=True).item() )
data = dict( np.load(folder+'data_vs_t_50Hz_65Hz_pad2.npy', allow_pickle=True).item() )


psd_x = data['psd_x']
psd_y = data['psd_y']

spectre_xy = data['spectre_kxky']


p = 1e-2/(33) # conversion en m/pixel

l_interfrange = 16 * p
k_interfrange = 2 * np.pi / l_interfrange

lmin = p


n_kx, n_ = np.shape(psd_x)
n_ky, n_ = np.shape(psd_y)

midx, midy = int(n_kx/2), int(n_ky/2)
kx,ky = np.linspace(-midx,midx,n_kx), np.linspace(-midy,midy,n_ky)
dkx = 2*np.pi/(lmin*n_kx )
dky = 2*np.pi/(lmin*n_ky )

k_x = kx*dkx
k_y = ky*dky

Nk, Nt = psd_x[:,:].shape


#%%

H = 0.1
h = 1e-3
E = 70e3
nu = 0
g = 9.81
rho = 1e3
T_ = 2.0 #np.array([0,0.5,1,2,4,10,20])
B = h**3*E/(12*(1-nu**2))
# k_c = np.sqrt(T_/B)
# k_g = (rho*g/B)**0.25

lambda_ = np.arange(5e-3, 0.6, 1e-3)
k_th = 2*np.pi/lambda_


def relation(T,k,h,g,B):
    return np.sqrt( g*k + T*k**3/rho + B*k**5/rho)/(2*np.pi)
def relation_th(T,k,h,g,B):
    return np.sqrt( (g*k + T*k**3/rho + B*k**5/rho) * np.tanh(k*H))/(2*np.pi)

f_th = relation(T_, k_th, h, g, B)
f_th3 = relation(T_, k_th/3, h, g, B)
f_th2 = relation(T_, k_th/2, h, g, B)

f_th0 = relation(0, k_th, h, g, B)
f_th_tan = relation_th(T_, k_th, h, g, B)



""" compute relation dispersion """

f = np.arange(-int(Nt/2),int(Nt/2),1)*(faq/(Nt))
beta = 12
# window = np.hanning(Nt)
# window_v = np.hanning(Nt-1)
# window = np.blackman(Nt)
# window_v = np.blackman(Nt-1)
# window = np.kaiser(Nt,beta)
# window_v = np.kaiser(Nt-1,beta)

# fft_temps_x = np.fft.fft( (psd_x[:,:]*window) , axis=1)
# fft_temps_y = np.fft.fft( (psd_y[:,:]*window) , axis=1)
fft_temps_x = np.fft.fft( (psd_x[:,:]) , axis=1)
fft_temps_y = np.fft.fft( (psd_y[:,:]) , axis=1)

relation_dispersion_x = np.log(abs(np.fft.fftshift(fft_temps_x, axes=1))**2)
relation_dispersion_y = np.log(abs(np.fft.fftshift(fft_temps_y,axes=1))**2)

mid_k = int(Nk/2)
mid_t = 0#int(Nt/2)
x_k = 60

""" PLOTsss """

ix = int(len(kx)/2)
iy = int(len(ky)/2)


Kx = np.arange(0,np.shape(relation_dispersion_x)[0],1)
Ky = np.arange(0,np.shape(relation_dispersion_y)[0],1)

KX,FX = np.meshgrid(Kx, f, indexing='ij')
KY,FY = np.meshgrid(Ky, f, indexing='ij')

KX_ = (KX-Kx[ix])*dkx
KY_ = (KY-Ky[iy])*dky

kp = 170
f_th_bounded = 2*relation(T_, k_th/2, h, g, B)
f_th_bounded3 = 3*relation(T_, k_th/3, h, g, B)

di = 33



plt.figure(figsize=(7,6))
plt.pcolormesh(KX_[ix-di:ix+di,:]/(2*np.pi), 
                FX[ix-di:ix+di,:], 
                relation_dispersion_x[ix-di:ix+di,:], 
                shading='gouraud',
#                vmin=-15,
#                vmax=-9,
                cmap='turbo')
plt.colorbar()
plt.plot(k_th/(2*np.pi), f_th, '-w')
plt.plot(-k_th/(2*np.pi), f_th, '-w')
plt.xlim((-700/(2*np.pi),700/(2*np.pi)))
plt.ylim((0,60))
plt.xlabel(r'$1/\lambda_x~[m^{-1}]$')
plt.ylabel(r'$f~[Hz]$')
plt.title(rf'$T\sim {T_}~[N/m]$')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,6))
plt.pcolormesh(KY_[iy-di:iy+di,:]/(2*np.pi), 
                FY[iy-di:iy+di,:], 
                relation_dispersion_y[iy-di:iy+di,:], 
                shading='gouraud',
#                vmin=-15,
#                vmax=-9,
                cmap='turbo')
plt.plot(k_th/(2*np.pi), f_th, '-w')
plt.plot(-k_th/(2*np.pi), f_th, '-w')
plt.xlim((-600/(2*np.pi),600/(2*np.pi)))
plt.ylim((0,60))
plt.colorbar()
plt.xlabel(r'$1/\lambda_y~[m^{-1}]$')
plt.ylabel(r'$f~[Hz]$')
plt.title(rf'$T\sim {T_}~[N/m]$')
plt.tight_layout()
plt.show()



#%%
""" spectrum mean kx ky """


XX,YY = np.meshgrid(Kx,Ky, indexing='ij')
XX_ = (XX-Kx[ix])*dkx
YY_ = (YY-Ky[iy])*dky

spectre_plot = np.log10(np.abs(spectre_xy))

circle2 = plt.Circle((0, 0), 42, color='b', fill=False)
circle3 = plt.Circle((0, 0), 55, color='r', fill=False)
circle4 = plt.Circle((0, 0), 22, color='w', fill=False)

fig,ax = plt.subplots()

plt.pcolormesh(XX_[ix-di:ix+di,iy-di:iy+di]/(2*np.pi), 
               YY_[ix-di:ix+di,iy-di:iy+di]/(2*np.pi),
               spectre_plot[ix-di:ix+di,iy-di:iy+di],
               shading='gouraud',
               cmap='magma_r',
               vmin=-11.6,
               vmax=-10.6)

x,y=56,-3
xf,yf=20,12.8

plt.arrow(0, 0, x, y,color='b')
# plt.arrow(0, 0, -40, 12,color='b')
# plt.arrow(40, 19, -12.5-40, 15-19,color='b')
plt.arrow(x, y, xf-x, yf-y,color='r')
plt.arrow(0, 0, xf-x, yf-y,color='r')

plt.arrow(0, 0, xf, yf,color='w')

ax.add_patch(circle2) 
ax.add_patch(circle3)
ax.add_patch(circle4)

plt.colorbar()
plt.xlabel(r'$1/\lambda_x~[m^{-1}]$')
plt.ylabel(r'$1/\lambda_y~[m^{-1}]$')
plt.tight_layout()
plt.show()


k1ns = x**2+y**2
k2ns = (xf-x)**2+(yf-y)**2
k3ns = xf**2+yf**2

angle = np.arccos((k3ns-k2ns-k1ns)/(2*np.sqrt(k1ns*k2ns)))

#%%

psd_abs = np.log(abs(psd_x)**2)

time_ = np.arange(0,Nt,1)*1/fech

Kspec, TIME = np.meshgrid(Kx, time_, indexing='ij')
Kspec_ = (Kspec-Kx[ix])*dkx

plt.figure()
plt.pcolormesh(TIME[ix-di:ix+di,:], 
               Kspec_[ix-di:ix+di,:], 
               psd_abs[ix-di:ix+di,:],
               cmap='turbo')

plt.colorbar()
plt.xlabel(r'$t~[s]$')
plt.ylabel(r'$k_x~[m^{-1}]$')
plt.tight_layout()
plt.show()

