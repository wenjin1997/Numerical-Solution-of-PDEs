# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:16:13 2021

@author: XieWenjin
"""

import math
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

t = 0.03     # 时间范围
x = 1.0      # 空间范围
m = input("请输入m：")
m = int(m)
n = input("请输入n：")
n = int(n)
# m = 320        # 时间方向分为320个格子
# n = 64        # 空间方向的格子数
dt = t / (m - 1)  # 时间步长
dx = x / (n - 1)  # 空间步长
 
def generate_u(m,n):
    u = np.zeros([m,n])
    # 边界条件
    for j in range(n):
        u[0,j] = math.sin(4 * math.pi * j * dx)
    for i in range(m):
        u[i,0] = 0
        u[i,-1] = 0

    # 差分法
    for i in range(m - 1):
        for j in range(1,n - 1):
            u[i+1, j] = dt * (u[i, j + 1] + u[i, j - 1] - 2 * u[i, j]) / dx ** 2 + u[i, j]
    return u

def drawing(X,Y,Z):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()

def error(u,u_exact):
    err = abs(u - u_exact)
    return max(map(max, err))

X = np.arange(0, t + dt, dt) # remark:t+dt,not t
Y = np.arange(0, x + dx, dx)
X, Y = np.meshgrid(X, Y)
u_exact = np.exp(- (4*np.pi)**2*X)*np.sin(4*np.pi*Y)

u = generate_u(m,n)
u = np.transpose(u) # 注意这里是转置，而不是np.reshape(u,(n,m))

# print(m,n)
print(error(u,u_exact))

drawing(X,Y,u) # 数值解
# drawing(X,Y,u_exact) # 精确解
# drawing(X,Y,abs(u-u_exact)) # 数值解与精确解之差的绝对值