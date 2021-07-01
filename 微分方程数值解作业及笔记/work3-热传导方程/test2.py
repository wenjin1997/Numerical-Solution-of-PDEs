# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 21:57:52 2019

@author: qwy
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig)

t0 = 1
x0 = 1
m = 3  # 时间方向为320个
n = 4   # 空间方向分为64个格子
dt = t0 / (m - 1)  # 时间步长
dx = x0 / (n - 1)  # 空间步长



delta = 0.125
# 生成代表X轴数据的列表
x = np.arange(0, t0, dt)
# 生成代表Y轴数据的列表
y = np.arange(0, x0, dx)
# 对x、y数据执行网格化
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# 计算Z轴数据（高度数据）
Z = (Z1 - Z2) * 2
U = np.exp(-(4 * np.pi) ** 2 * Y) * np.sin(4 * np.pi * X)
u_exact = np.zeros([m-1, n-1])
print(u_exact)
for i in range(m-1):
    for j in range(n-1):
        u_exact[i, j] = math.exp(-(4 * math.pi) ** 2) * math.sin(4 * math.pi * j)
print(Z)
# 绘制3D图形
surf = ax.plot_surface(x, y, U,
rstride=1,  # rstride（row）指定行的跨度
cstride=1,  # cstride(column)指定列的跨度
cmap=plt.get_cmap('rainbow'))  # 设置颜色映射
# 设置Z轴范围
ax.set_zlim(-2, 2)
# 设置标题
plt.title("3D")
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()