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

# 精确解 u(x) = -0.5 * x^2 + x
def exactf(x):
    return (-1) * 0.5 * x * x + x

# f(F(\xi))N1(\xi)在[-1,1]上的积分值
def integrate_f1():
    return 1

# f(F(\xi))N2(\xi)在[-1,1]上的积分值
def integrate_f2():
    return 1


# 生成矩阵A和右端项RHS
def generate(N, f1, f2):
    A = np.zeros([N+1, N+1])
    RHS = np.zeros([N+1,1])
    for i in range(1, N+1):
        A[i,i] += 1/h
        A[i-1, i-1] += 1/h
        A[i-1,i] += -1/h
        A[i,i-1] += -1/h
        RHS[i] += (h/2) * f1 
        RHS[i-1] += (h/2) * f2 
    return A,RHS


# 处理边界条件 u(0) = 0
def modified_matrix(A, RHS, u0, uN):
    # 1. 先更新右端项
    # 处理边界条件u0
    for i in range(1,N):
        RHS[i] = RHS[i] - A[i,0] * u0
    RHS[0] = u0

    # # 处理边界条件uN，当给出u(xN)的值时，取消注释
    # for i in range(1,N):
    #     RHS[i] = RHS[i] - A[i,N] * uN
    # RHS[N] = uN
    
    # 2. 更改矩阵A
    for i in range(N+1):
        A[0,i] = 0
        A[i,0] = 0
        # A[N,i] = 0 # 当给出u(xN)的值时，取消注释
        # A[i,N] = 0 # 当给出u(xN)的值时，取消注释
    A[0,0] = 1
    # A[N,N] = 1 # 当给出u(xN)的值时，取消注释
    return A,RHS

A,RHS = generate(N,integrate_f1(),integrate_f2())
A,RHS = modified_matrix(A, RHS, 0, 0)

# 数值解 求解方程组 Ax=RHS
x = linalg.solve(A, RHS)
# print(x)

# 精确解
u = np.zeros([N+1,1])
for i in range(N+1):
    u[i] = exactf(i * h)
# print(u)

# 输出误差
err = max(abs(u - x))
print(N,err)

# 画图比较精确解和数值解
t = np.arange(x0, xN + h, h)
plt.title('Result')
plt.plot(t, x, color='green', label='numerical solution')
plt.plot(t, u, color='blue', label='exact solution')
plt.legend() # show the legend

plt.xlabel('t')
plt.ylabel('u')
plt.show()