import numpy as np
import matplotlib.pyplot as plt
import math
N = 3
M = 4
dx = 1./(N+1)
dy = 1./(M+1)
x = np.linspace(0, 1, N+2)
y = np.linspace(0, 1, M+2)
u_exact = np.zeros([N,M])
X, Y = np.meshgrid(x[1:N+1], y[1:M+1])      
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')   
ax.plot_surface(X, Y, u_exact, cmap='rainbow')
plt.show()
