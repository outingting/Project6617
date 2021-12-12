# Main file to test the 4 functions. 

import numpy as np
import math
from scipy.stats import multivariate_normal
from methods import ES_Hessian, ES_vanilla_gradient, Hess_Aware, LP_Gradient, LP_Hessian, LP_Hessian_structured, LP_Hessian_structured_v2, LP_Hessian_structured_v3, LP_Hessian_structured_v4, LP_Hessian_structured_v5, LP_Hessian_structured_v6, LP_Hessian_structured_v7
import cvxpy as cp
import matplotlib.pyplot as plt 

def rastrigin(theta):
    return 10*theta.shape[0] + np.sum(theta**2 - 10*np.cos(2*math.pi*theta))

def F(theta):
    if theta.ndim == 1:
        theta = np.expand_dims(theta, 0)
    return -np.apply_along_axis(rastrigin, 1, theta)

plt.figure(3)

np.random.seed(1)
initial_pt = np.random.uniform(-2,2,100)

# print("ES vanilla gradient ...")
# res = ES_vanilla_gradient(F, alpha=1e-4, sigma=0.05, theta_0=initial_pt, num_samples = 10, time_steps = 300 * 20)
# plt.plot(res[3], res[4], label = "ES_vanilla_gradient")

print("ES Hessian")
res = ES_Hessian(F, alpha=0.5, sigma=0.05, theta_0=initial_pt, num_samples = 10, time_steps = 100* 20)
plt.plot(res[3], res[4], label = "ES_Hessian")

print("Hess-Aware")
res = Hess_Aware(F, alpha = 0.1, sigma = 0.01, theta_0=initial_pt, num_samples = 10, time_steps = 67*20)
plt.plot(res[3], res[4], label = "Hess_Aware")
#
# print("LP Hessian")
# res = LP_Hessian(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 100, time_steps = 100)
# plt.plot(res[3], res[4], label = "LP_Hessian")
#
# print("LP Hessian structured")
# res = LP_Hessian_structured(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 100, time_steps = 100)
# plt.plot(res[3], res[4], label = "LP_Hessian_structured")
#
# print("LP Hessian structured with PT inverse")
# res = LP_Hessian_structured_v2(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 100, time_steps = 100)
# plt.plot(res[3], res[4], label = "LP_Hessian_structured_v2")
#
# print("LP Hessian structured with PT inverse and antithetic samples")
# res = LP_Hessian_structured_v3(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 100, time_steps = 100)
# plt.plot(res[3], res[4], label = "LP_Hessian_structured_v3")

print("LP Hessian structured with PT inverse, antithetic samples and backtracking")
res = LP_Hessian_structured_v4(F, alpha = 1e-6, sigma = 0.01, theta_0=initial_pt, PT_threshold=-1e1, num_samples = 10, time_steps = 100*10)
plt.plot(res[3], res[4], label = "LP_Hessian_structured_v4")

print("LP Hessian structured with PT inverse, antithetic samples and backtracking [simulation gradient estimator]")
res = LP_Hessian_structured_v5(F, alpha = 1e-6, sigma = 0.01, sigma_g =0.01, theta_0=initial_pt, PT_threshold=-1e1, num_samples = 10, time_steps = 100*10)
plt.plot(res[3], res[4], label = "LP_Hessian_structured_v5")
#
# print("LP Hessian structured with PT inverse, antithetic samples and backtracking [linear regression structured hessian estimator]")
# res = LP_Hessian_structured_v6(F, alpha = 1e-6, sigma = 0.01, sigma_g =0.05, theta_0=initial_pt, PT_threshold=-1e1, num_samples = 10, time_steps = 100)
# plt.plot(res[3], res[4], label = "LP_Hessian_structured_v6")

print("LP Hessian structured with PT inverse, antithetic samples and backtracking [unconstrained linear regression structured hessian estimator]")
res = LP_Hessian_structured_v7(F, alpha = 1e-6, sigma = 0.01, sigma_g =0.01, theta_0=initial_pt, PT_threshold=-1e1, num_samples = 10, time_steps = 100*10)
plt.plot(res[3], res[4], label = "LP_Hessian_structured_v7")

plt.legend(loc="lower right")
plt.xlabel("# function calls")
plt.ylabel("function value")
plt.title("Rastrigin function")

plt.show()
