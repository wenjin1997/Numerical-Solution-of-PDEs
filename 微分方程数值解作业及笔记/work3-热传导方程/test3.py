# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:17:13 2015

@author: Eddy_zheng
"""

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
m = 320
n = 64
X = np.arange(0, 0.03, 0.03/m)
Y = np.arange(0, 1, 1.0/n)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
Z1 = np.exp(- (4*np.pi)**2*X)*np.sin(4*np.pi*Y)
print(Z1.shape)

# 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap='rainbow')

plt.show()