import numpy as np
import math
from scipy import linalg # 求解方程组需要引用的包
import matplotlib.pyplot as plt # 画图

# 求解微分方程 -u'' = 1, u(0) = 0, u'(1) = 0
# 精确解为 u(x) = -0.5 * x^2 + x
x0 = 0.0 
xN = 1.0
N = 10  # 节点x0, x1, ..., xN
h = (xN - x0) / N 

# 生成矩阵A和右端项RHS
A = np.zeros([N+1, N+1])
RHS = np.zeros([N+1,1])
for i in range(1, N+1):
    A[i,i] += 1/h
    A[i-1, i-1] += 1/h
    A[i-1,i] += -1/h
    A[i,i-1] += -1/h
    RHS[i] += (h/2) * 1  # f=1时积分结果为1，f不同时这里需要修改
    RHS[i-1] += (h/2) * 1 # f=1时积分结果为1，f不同时这里需要修改


# 处理边界条件 u(0) = 0

# 1. 先更新右端项
RHS[0] = 0
    
# 2. 更改矩阵A
for i in range(N+1):
    A[0,i] = 0
    A[i,0] = 0
A[0,0] = 1

# 数值解 求解方程组 Ax=RHS
x = linalg.solve(A, RHS)
# print(x)

# 精确解
u = np.zeros([N+1,1])
for i in range(N+1):
    u[i] = (-1) * 0.5 *(i * h) * (i * h) + (i * h)
# print(u)

# 输出误差
err = max(abs(u - x))
print(N,err)

# 画图比较精确解和数值解
t = np.arange(x0, xN + h, h)
plt.title('Result with h=0.1')
plt.plot(t, x, color='green', label='numerical solution')
plt.plot(t, u, color='blue', label='exact solution')
plt.legend() # show the legend

plt.xlabel('t')
plt.ylabel('u')
plt.show()