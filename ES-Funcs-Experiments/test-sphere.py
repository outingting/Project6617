# Main file to test the 4 functions. 

import numpy as np
from scipy.stats import multivariate_normal
from methods import ES_Hessian, ES_vanilla_gradient, Hess_Aware, LP_Gradient, LP_Hessian, LP_Hessian_structured, LP_Hessian_structured_v2, LP_Hessian_structured_v3, LP_Hessian_structured_v4
import cvxpy as cp
import matplotlib.pyplot as plt 


def F(theta):
    if theta.ndim == 1:
        theta = np.expand_dims(theta, 0)
    return -np.sum((theta) ** 2, axis=tuple(range(theta.ndim)[1:]))

plt.figure(1)

seed = 1
np.random.seed(seed)
initial_pt = np.random.uniform(-2,2,5)

print("ES vanilla gradient ...")
res = ES_vanilla_gradient(F, alpha=0.005, sigma=0.05, theta_0=initial_pt, num_samples = 50, time_steps = 200, seed=1)
plt.plot(res[3], res[4], label = "ES_vanilla_gradient")

print("ES Hessian ...")
res = ES_Hessian(F, alpha=0.3, sigma=0.05, theta_0=initial_pt, num_samples = 50, time_steps = 100, seed=1)
plt.plot(res[3], res[4], label = "ES_Hessian")

print("Hess-Aware ...")
res = Hess_Aware(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 50, time_steps = 67, seed=1)
plt.plot(res[3], res[4], label = "Hess_Aware")

print("LP Hessian ...")
res = LP_Hessian(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 50, time_steps = 100, seed=1)
plt.plot(res[3], res[4], label = "LP_Hessian")

print("LP structured Hessian ...")
res = LP_Hessian_structured(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 50, time_steps = 100, seed=1)
plt.plot(res[3], res[4], label = "LP_Hessian_structured")

print("LP structured Hessian with PT inverse ...")
res = LP_Hessian_structured_v2(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 50, time_steps = 100, seed=1)
plt.plot(res[3], res[4], label = "LP_Hessian_structured_v2")

print("LP structured Hessian with PT inverse and antithetic samples ...")
res = LP_Hessian_structured_v3(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 50, time_steps = 100, seed=1)
plt.plot(res[3], res[4], label = "LP_Hessian_structured_v3")

print("LP structured Hessian with PT inverse, antithetic samples and backtracking ...")
res = LP_Hessian_structured_v4(F, alpha = 0.01, sigma = 0.05, theta_0=initial_pt, num_samples = 50, time_steps = 100, seed=1)
plt.plot(res[3], res[4], label = "LP_Hessian_structured_v4")

plt.legend(loc="lower right")
plt.xlabel("# function calls")
plt.ylabel("function value")
plt.title("Sphere function")

plt.show()
