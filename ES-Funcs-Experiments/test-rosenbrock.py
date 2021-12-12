# Main file to test the 4 functions. 

import numpy as np
from scipy.stats import multivariate_normal
from methods import ES_Hessian, ES_Hessian_v2, ES_vanilla_gradient, Hess_Aware, LP_Gradient, LP_Hessian, LP_Hessian_structured, LP_Hessian_structured_v2, LP_Hessian_structured_v3, LP_Hessian_structured_v4, LP_Hessian_structured_v6, LP_Hessian_structured_v5, LP_Hessian_structured_v7, asebo
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
    return -np.apply_along_axis(rosenbrock, 1, theta)

plt.figure(2)

np.random.seed(1)
initial_pt = np.random.uniform(-2,2,100)

# res = ES_vanilla_gradient(F, alpha=1e-4, sigma=0.05, theta_0=initial_pt, num_samples = 20, time_steps = 200)
# plt.plot(res[3], res[4], label = "ES_vanilla_gradient")

print("ES Hessian ... ")
res = ES_Hessian(F, alpha=0.5, sigma=0.05, theta_0=initial_pt, num_samples = 25, time_steps = 100*20)
plt.plot(res[3], res[4], label = "ES_Hessian")

print("ES Hessian - v2 ...")
res = ES_Hessian_v2(F, alpha=0.5, sigma=0.05, theta_0=initial_pt, num_samples = 25, time_steps = 100*20)
plt.plot(res[3], res[4], label = "ES_Hessian_v2")

print("Hess_Aware ...")
res = Hess_Aware(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 25, time_steps = 100*20)
plt.plot(res[3], res[4], label = "Hess_Aware")

# res = LP_Hessian(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 100, time_steps = 100)
# plt.plot(res[3], res[4], label = "LP_Hessian")
#
# res = LP_Hessian_structured(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 100, time_steps = 200)
# plt.plot(res[3], res[4], label = "LP_Hessian_structured")
#
# res = LP_Hessian_structured_v2(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 100, time_steps = 200)
# plt.plot(res[3], res[4], label = "LP_Hessian_structured v2")

# res = LP_Hessian_structured_v3(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, PT_threshold=-1e1, num_samples = 20, time_steps = 200)
# plt.plot(res[3], res[4], label = "LP_Hessian_structured v3")


print("LP Hessian structured with PT inverse, antithetic samples and backtracking ... ")
res = LP_Hessian_structured_v4(F, alpha = 1e-6, sigma = 0.01, theta_0=initial_pt, PT_threshold=-1e1, num_samples = 25, time_steps = 100*20)
plt.plot(res[3], res[4], label = "LP_Hessian_structured_v4")

print("LP Hessian structured with PT inverse, antithetic samples and backtracking [simulation gradient estimator] ... ")
res = LP_Hessian_structured_v5(F, alpha = 1e-6, sigma = 0.01, sigma_g =0.01, theta_0=initial_pt, PT_threshold=-1e1, num_samples = 25, time_steps = 100*20)
plt.plot(res[3], res[4], label = "LP_Hessian_structured_v5")

print("LP Hessian structured with PT inverse, antithetic samples and backtracking [linear regression structured hessian estimator] ... ")
res = LP_Hessian_structured_v6(F, alpha = 1e-6, sigma = 0.01, sigma_g =0.01, theta_0=initial_pt, PT_threshold=-1e1, num_samples = 25, time_steps = 100*20)
plt.plot(res[3], res[4], label = "LP_Hessian_structured_v6")

print("LP Hessian structured with PT inverse, antithetic samples and backtracking [unconstrained linear regression structured hessian estimator] ...")
res = LP_Hessian_structured_v7(F, alpha = 1e-6, sigma = 0.01, sigma_g =0.01, theta_0=initial_pt, PT_threshold=-1e1, num_samples = 25, time_steps = 100*20)
plt.plot(res[3], res[4], label = "LP_Hessian_structured_v7")

print("Running ASEBO ...")
res = asebo(F, sigma=0.05, learning_rate=0.05, decay=0.99, k=140, theta_0=initial_pt, min_samples=25, num_sensings=50, max_iter=1500, use_log=False, threshold=0.995)
plt.plot(res[3], res[4], label = "asebo")


plt.legend(loc="lower right")
plt.xlabel("# function calls")
plt.ylabel("function value")
plt.title("Rosenbrock function")

plt.show()
