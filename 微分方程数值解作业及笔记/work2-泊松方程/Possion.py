import numpy as np
import matplotlib.pyplot as plt
import math

# 生成矩阵T、D，为生成矩阵A做准备
def generate_TD(N, dx, dy):
    T = np.zeros([N,N])
    D = np.zeros([N,N])
    a = (dy/dx)**2
    for i in range(N):
        T[i,i] = -2*(1+a)
        D[i,i] = 1
        if (i < N-1):
            T[i,i+1] = a
        if (i > 0):
            T[i,i-1] = a
    return T, D

# 生成矩阵A
def assemble_A(N, M, dx, dy):
    T, D = generate_TD(N, dx, dy)
    A = np.zeros([N*M, N*M])
    for j in range(M):
        A[j*N:(j+1)*N, j*N:(j+1)*N] = T
        if (j < M-1):
            A[j*N:(j+1)*N, (j+1)*N:(j+2)*N] = D
        if (j > 0):
            A[j*N:(j+1)*N, (j-1)*N:(j)*N] = D
    return A


def f(x, y):
    return 5 * math.exp(x + 2 * y)

# 精确解
def exact_f(x, y):
    return math.exp(x + 2 * y)

def gL(y):
    return math.exp(2 * y)

def gR(y):
    return math.exp(1 + 2 * y)

def gB(x):
    return math.exp(x)

def gT(x):
    return math.exp(x + 2)

def assemble_F(x, y, dx, dy, N, M, gL, gR, gB, gT):
    F = np.zeros(N*M)
    
    a = (dy/dx)**2

    # dy^2 * f(i,j)
    for j in range(M):
        for i in range(N):
            F[j * N + i] += ((dy) ** 2) * f(x[i + 1], y[j + 1])

    # left BCs
    for j in range(M):
        F[j*N] += -a*gL(y[j+1])
        
    # right BCs
    for j in range(M):
        F[(j+1)*N - 1] += -a*gR(y[j+1])
    
    # top BCs
    for i in range(N):
        F[N * (M - 1) + i] += -gT(x[i+1])
    
    # bottom BCs
    for i in range(N):
        F[i] += -gB(x[i + 1])
    
    return F 

def exact_solution(N, M, x, y):
    U_exact = np.zeros(N * M)
    for j in range(M):
        for i in range(N):
            U_exact[j * N + i] = exact_f(x[i + 1], y[j + 1])
    return U_exact

def Possion_solver(N, M, gL, gR, gB, gT):
    dx = 1./(N+1)
    dy = 1./(M+1)
    x = np.linspace(0, 1, N+2)
    y = np.linspace(0, 1, M+2)
    
    A = assemble_A(N, M, dx, dy)
    
    F = assemble_F(x, y, dx, dy, N, M, gL, gR, gB, gT)
    
    U = np.linalg.solve(A, F)
    U_exact = exact_solution(N, M, x, y)
    error = max(abs(U-U_exact))
    
    u = np.reshape(U, (N,M))
    u_exact = np.reshape(U_exact, (N,M))
    
    
    X, Y = np.meshgrid(x[1:N+1], y[1:M+1])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    ax.plot_surface(X, Y, u, cmap='rainbow')
    
    # print (u)
    # print(error)
    plt.show()
    print(0.0013997692884775148/0.005519939625335368)
    
Possion_solver(9, 9, gL, gR, gB, gT)