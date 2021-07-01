# implict euler method
import numpy as np
import matplotlib.pyplot as plt

# the right term of the ODE
def f(t, u):
    f = -u/t
    return f

# the exact solution of ODE 
def fexact(t):
    fexact = 1/t
    return fexact

N = 64
t_n = 2.0
dt = (t_n - 1.0) / N
t = np.arange(1.0, t_n + dt, dt)
u_euler = np.arange(1.0, t_n + dt, dt)
u = np.arange(1.0, t_n + dt, dt)
u_true = np.arange(1.0, t_n + dt, dt)

i = 0
while i < N:
    t[i+1] = t[i] + dt
    u_euler[i+1] = u_euler[i] + dt * f(t[i], u_euler[i])
    u[i+1] = (u[i] * t[i+1])/(t[i+1] + dt)
    u_true[i+1] = fexact(t[i+1])
    i = i + 1

err_euler = max(abs(u_euler - u_true))
err_implict_euler = max(abs(u - u_true))
print(N)
print("The error of euler method: ",err_euler)
print(err_euler * N)
print("The error of implict euler method: ",err_implict_euler)
print(err_implict_euler * N * N)

# begin drawing
plt.title('Result')
plt.plot(t, u_euler, color='green', label='euler')
plt.plot(t, u, color='blue', label='implict euler')
plt.plot(t, u_true, color='red', label='exact')
plt.legend() # show the legend

plt.xlabel('t')
plt.ylabel('u')
plt.show()