# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 14:55:15 2025

@author: turbulence
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')
plt.rcParams['font.size']=20


# Remplacez 'chemin/vers/votre/image.jpg' par le chemin de votre image
chemin_image = 'C:/Users/turbulence/Desktop/Data_images/130325/position2.tiff'

# Ouvrir l'image
image = Image.open(chemin_image)

pix_m = 3.37e-4 # en m/pixel s

# Afficher l'image
plt.figure()
plt.imshow(image)
plt.show()

nx_,ny_ = np.shape(image)
nx,ny = nx_*pix_m, ny_*pix_m
xc,yc = 1074*pix_m, ny - 800*pix_m

xb,yb = 1765*pix_m,920*pix_m
xb2,yb2 = 1762*pix_m,544*pix_m
xb3,yb3 = 1760*pix_m,1176*pix_m

# data du 130325
xc0,yc0 = 950*pix_m, 59*pix_m
xp1,yp1 = 1023*pix_m, 186*pix_m
xp2,yp2 = 1033*pix_m, 114*pix_m

# data du 060325
x1,y1 = 905*pix_m,475*pix_m
x2,y2 = 1164*pix_m,460*pix_m
x3,y3 = 1359*pix_m,447*pix_m
x4,y4 = 1296*pix_m, 728*pix_m
x5,y5 = 1251*pix_m, 125*pix_m
x6,y6 = 912*pix_m, 120*pix_m
x7,y7 = 920*pix_m, 505*pix_m
x8,y8 = 923*pix_m, 508*pix_m
x9,y9 = 1527*pix_m, 373*pix_m
x10,y10 = 1245*pix_m, 412*pix_m
x11,y11 = 911*pix_m, 414*pix_m
x12,y12 = 921*pix_m, 424*pix_m
x13,y13 = 916*pix_m, 111*pix_m


#%% 060325

# Créer une figure et un axe
fig, ax = plt.subplots(figsize=(6,6))
# Définir les paramètres du cercle
centre = (0, 0)  # Coordonnées du centre (x, y)
rayon = 0.3          # Rayon du cercle

# Créer un objet Cercle
cercle = patches.Circle(centre, rayon, linewidth=2, edgecolor='k', facecolor='none')
cercle_batteur = patches.Circle((xb-xc,(ny-yb)-yc), 0.025, linewidth=2, color='grey')
cercle_batteur2 = patches.Circle((xb2-xc,(ny-yb2)-yc), 0.025, linewidth=2, color='black')
cercle_batteur3 = patches.Circle((xb3-xc,(ny-yb3)-yc), 0.025, linewidth=2, color='red')

# Ajouter le cercle à l'axe
ax.add_patch(cercle)
ax.add_patch(cercle_batteur)
ax.add_patch(cercle_batteur2)
ax.add_patch(cercle_batteur3)

plt.scatter(x1-xc,(ny-y1)-yc, marker='o', color='grey')
plt.scatter(x2-xc,(ny-y2)-yc, marker='o', color='grey')
plt.scatter(x3-xc,(ny-y3)-yc, marker='o', color='grey')
plt.scatter(x4-xc,(ny-y4)-yc, marker='o', color='grey')
plt.scatter(x5-xc,(ny-y5)-yc, marker='o', color='grey')
plt.scatter(x6-xc,(ny-y6)-yc, marker='o', color='grey')
plt.scatter(x7-xc,(ny-y7)-yc, marker='o', color='grey')
plt.scatter(x8-xc,(ny-y8)-yc, marker='o', color='black')
plt.scatter(x9-xc,(ny-y9)-yc, marker='o', color='black')
plt.scatter(x10-xc,(ny-y10)-yc, marker='o', color='black')
plt.scatter(x11-xc,(ny-y11)-yc, marker='o', color='black')
plt.scatter(x12-xc,(ny-y12)-yc, marker='o', color='red')
plt.scatter(x13-xc,(ny-y13)-yc, marker='o', color='red')

# Définir les limites de l'axe
ax.set_xlim((-0.32, 0.32))
ax.set_ylim((-0.32,0.32))

# Afficher le graphique
plt.show()


#%% 130325


# Créer une figure et un axe
fig, ax = plt.subplots(figsize=(6,6))
# Définir les paramètres du cercle
centre = (0, 0)  # Coordonnées du centre (x, y)
rayon = 0.3          # Rayon du cercle

# Créer un objet Cercle
cercle = patches.Circle(centre, rayon, linewidth=2, edgecolor='k', facecolor='none')
cercle_batteur = patches.Circle((xc0-xc,(ny-yc0)-yc), 0.025, linewidth=2, color='grey')
# cercle_batteur2 = patches.Circle((xb2-xc,(ny-yb2)-yc), 0.025, linewidth=2, color='black')
# cercle_batteur3 = patches.Circle((xb3-xc,(ny-yb3)-yc), 0.025, linewidth=2, color='red')

# Ajouter le cercle à l'axe
ax.add_patch(cercle)
ax.add_patch(cercle_batteur)
# ax.add_patch(cercle_batteur2)
# ax.add_patch(cercle_batteur3)

plt.scatter(xp1-xc,(ny-yp1)-yc, marker='o', color='grey')
plt.scatter(xp2-xc,(ny-yp2)-yc, marker='o', color='grey')
# plt.scatter(x3-xc,(ny-y3)-yc, marker='o', color='grey')
# plt.scatter(x4-xc,(ny-y4)-yc, marker='o', color='grey')
# plt.scatter(x5-xc,(ny-y5)-yc, marker='o', color='grey')
# plt.scatter(x6-xc,(ny-y6)-yc, marker='o', color='grey')
# plt.scatter(x7-xc,(ny-y7)-yc, marker='o', color='grey')
# plt.scatter(x8-xc,(ny-y8)-yc, marker='o', color='black')
# plt.scatter(x9-xc,(ny-y9)-yc, marker='o', color='black')
# plt.scatter(x10-xc,(ny-y10)-yc, marker='o', color='black')
# plt.scatter(x11-xc,(ny-y11)-yc, marker='o', color='black')
# plt.scatter(x12-xc,(ny-y12)-yc, marker='o', color='red')
# plt.scatter(x13-xc,(ny-y13)-yc, marker='o', color='red')

# Définir les limites de l'axe
ax.set_xlim((-0.32, 0.32))
ax.set_ylim((-0.32,0.32))

# Afficher le graphique
plt.show()
