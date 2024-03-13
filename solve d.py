import numpy as np
from scipy.optimize import fsolve


m = 9.11e-31
q = 1.6e-19
pi= 3.1415926
h = 6.626e-34

cc1= 1.5

RA = 7.796e-5
def transcendental_eq(x):
    # 例：x * np.exp(x) - 1 = 0
    return (1/x)*(cc1*q**2/h**2)*(2*m*q)**(0.5)*x**0.5*np.exp((4*pi*(2*m*q)**(0.5)*x**(0.5)/h))-RA
    # return x * np.exp(x) - 1

# 选择一个初始猜测值
initial_guess = 0.5

# 使用fsolve求解
solution = fsolve(transcendental_eq, 5e-9)

print('The solution is:', solution)
