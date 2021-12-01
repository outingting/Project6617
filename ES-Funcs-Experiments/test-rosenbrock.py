# Main file to test the 4 functions. 

import numpy as np
from scipy.stats import multivariate_normal
from methods import ES_Hessian, ES_vanilla_gradient, Hess_Aware, LP_Gradient, LP_Hessian, LP_Hessian_structured
import cvxpy as cp
import matplotlib.pyplot as plt 

def rosenbrock(theta):
    v = 0
    d = theta.shape[0]
    for i in range(d-1):
        v += 100* (theta[i+1] - theta[i])**2 + (theta[i] - 1)**2 
    return v

def F(theta):
    if theta.ndim == 1:
        theta = np.expand_dims(theta, 0)
    return np.apply_along_axis(rosenbrock, 1, theta)

plt.figure(2)

np.random.seed(1)
initial_pt = np.random.uniform(-2,2,5)

res = ES_vanilla_gradient(F, alpha=1e-4, sigma=0.05, theta_0=initial_pt, num_samples = 100, time_steps = 200)
plt.plot(res[3], res[4], label = "ES_vanilla_gradient")

res = ES_Hessian(F, alpha=0.5, sigma=0.05, theta_0=initial_pt, num_samples = 100, time_steps = 100)
plt.plot(res[3], res[4], label = "ES_Hessian")

res = Hess_Aware(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 100, time_steps = 67)
plt.plot(res[3], res[4], label = "Hess_Aware")

res = LP_Hessian(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 100, time_steps = 100)
plt.plot(res[3], res[4], label = "LP_Hessian")

res = LP_Hessian_structured(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 100, time_steps = 200)
plt.plot(res[3], res[4], label = "LP_Hessian_structured")

plt.legend(loc="lower right")
plt.xlabel("# function calls")
plt.ylabel("function value")
plt.title("Rosenbrock function")

plt.show()
