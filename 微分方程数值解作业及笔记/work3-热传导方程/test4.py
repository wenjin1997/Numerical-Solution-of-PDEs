# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:17:13 2015

@author: Eddy_zheng
"""

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math

fig = plt.figure()
ax = Axes3D(fig)
t0 = 0.03
x0 = 1.0
m = 320
n = 64
dt = t0 / m
dx = x0 / n

def generate_u(m,n):
    u = np.zeros([m,n])
    # 边界条件
    for j in range(n):
        u[0,j] = math.sin(4 * math.pi * j * dx)
    for i in range(m):
        u[i,0] = 0
        u[i,-1] = 0

    for i in range(m - 1):
        for j in range(1,n - 1):
            u[i+1, j] = dt * (u[i, j + 1] + u[i, j - 1] - 2 * u[i, j]) / dx ** 2 + u[i, j]
    return u

# u = np.zeros([m,n])
# for j in range(n):
#     u[0,j] = math.sin(4*math.pi*j*dx)

# for i in range(m):
#     u[i,0] = 0
#     u[i,-1] = 0

# for i in range(1,m):
#     for j in range(1,n - 1):
#         u[i, j] = dt * (u[i-1,j+1]-2*u[i-1,j]+u[i-1,j-1])/dt**2 + u[i-1,j]
# u1 = np.reshape(u, (n,m))
u2 = generate_u(m,n)
u2 = np.reshape(u2, (n,m))
print(u2.shape)
print(u2)
# print(u1.shape)
# print(u1)
# print(u.shape)

X = np.arange(0, t0, dt)
Y = np.arange(0, x0, dx)
X, Y = np.meshgrid(X, Y)
Z1 = np.exp(- (4*np.pi)**2*X)*np.sin(4*np.pi*Y)
print(Z1.shape)
# print(u1)
# print(u)
print(Z1)

# 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap='rainbow')

plt.show()