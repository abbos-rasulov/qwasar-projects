# Importing libraries

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.optimize import minimize_scalar
from scipy.optimize import linprog

# Finding Root of Function

def find_root_bisection(f,a,b):
    c = a
    while( (b - a) >= 0.001):
        c = (a + b) / 2
        if (f(c) == 0):
            break
        
        if (f(a) * f(c) < 0):
            b = c
        else:
            a = c
    return c


def find_root_newton_raphson(f, deriv_f, n):
    return optimize.newton(f, n, deriv_f)


def gradient_descent(f, deriv_f, start, learning_rate):

    x_0 = start
    x_1 = x_0 - learning_rate*deriv_f(x_0)
    precision = 0.0001
    iter = 0
    while abs(x_1 - x_0) >= precision:
        x_0 = x_1
        x_1 = x_0 - learning_rate*deriv_f(x_0)
    return x_1


def print_a_function():
    f = lambda x : (x - 1)**4 + x**2
    f_prime = lambda x : 4*((x-1)**3) + 2*x
    res = minimize_scalar(f, method='brent')
    print('Brent\'s method\t\tx_min: %.02f, f(x_min): %.02f' % (res.x, res.fun))

    # plot curve
    x = np.linspace(res.x - 1, res.x + 1, 100)
    y = [f(val) for val in x]
    plt.plot(x, y, color='blue', label='f')

    # plot optima
    plt.scatter(res.x, res.fun, color='red', marker='x', label='Minimum')

    plt.grid()
    plt.legend(loc = 1)
    plt.savefig('img1.png')

    start = -1
    x_min = gradient_descent(f, f_prime, start, 0.01)
    f_min = f(x_min)

    print("Gradient Descent Methods\txmin: %0.2f, f(x_min): %0.2f" % (x_min, f_min))
print_a_function()


def solve_linear_problem(A, b, c):
    # return linprog(c, A_ub=A, b_ub=b, bounds=(0, None))
    x0_bounds = (0, None)
    x1_bounds = (0, None)
    res = linprog(c, A_ub=A, b_ub=b,  bounds=(x0_bounds, x1_bounds), method='simplex', options={"disp": True})
    return res.fun,res.x