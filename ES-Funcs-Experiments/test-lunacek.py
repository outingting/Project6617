# Main file to test the 4 functions. 
# https://al-roomi.org/benchmarks/unconstrained/n-dimensions/229-lunacek-s-bi-rastrigin-function

import numpy as np
import math
from scipy.stats import multivariate_normal
from methods import ES_Hessian, ES_vanilla_gradient, Hess_Aware, LP_Gradient, LP_Hessian, LP_Hessian_structured
import cvxpy as cp
import matplotlib.pyplot as plt 

def lunacek(theta):
    pdim = theta.shape[0]
    s = 1.0 - (1.0 / (2.0 * math.sqrt(pdim + 20.0) - 8.2))
    mu1 = 2.5
    mu2 = - math.sqrt(abs((mu1**2 - 1.0) / s))
    firstSum = secondSum = thirdSum = 0
    for i in range(pdim):
        firstSum += (theta[i]-mu1)**2
        secondSum += (theta[i]-mu2)**2
        thirdSum += 1.0 - math.cos(2*math.pi*(theta[i]-mu1))
    return min(firstSum, 1.0*pdim + s*secondSum)+10*thirdSum


def F(theta):
    if theta.ndim == 1:
        theta = np.expand_dims(theta, 0)
    return -np.apply_along_axis(lunacek, 1, theta)

plt.figure(4)

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

res = LP_Hessian_structured(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 100, time_steps = 100)
plt.plot(res[3], res[4], label = "LP_Hessian_structured")

plt.legend(loc="lower right")
plt.xlabel("# function calls")
plt.ylabel("function value")
plt.title("Lunacek function")

plt.show()
