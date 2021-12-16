# Driver to test the optimizers on the [lunacek] function.

import numpy as np
from scipy.stats import multivariate_normal
from methods import ES_Hessian, ES_Hessian_v2, ES_vanilla_gradient, Hess_Aware, LP_Gradient, LP_Hessian, LP_Hessian_structured, LP_Hessian_structured_v2, LP_Hessian_structured_v3, LP_Hessian_structured_v4, LP_Hessian_structured_v6, LP_Hessian_structured_v5, LP_Hessian_structured_v7, asebo
import cvxpy as cp
import matplotlib.pyplot as plt 
import math

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

plt.figure(figsize = (16,9))

np.random.seed(1)
initial_pt = np.random.uniform(-2,2,100)


print("Running ASEBO ...")
res = asebo(F, sigma=0.05, learning_rate=0.05, decay=0.99, k=140, theta_0=initial_pt, min_samples=25, num_sensings=50, max_iter=100, use_log=False, threshold=0.995)
plt.plot(res[3], res[4], label = "asebo", linewidth=4, color = 'blue')

print("Hess_Aware ...")
res = Hess_Aware(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 25, time_steps = 132)
plt.plot(res[3], res[4], label = "Hess_Aware", linewidth=4, color = 'green')

print("vanilla gradient ...")
res = ES_vanilla_gradient(F, alpha=1e-5, sigma=1, theta_0=initial_pt, num_samples = 50, time_steps = 200)
plt.plot(res[3], res[4], label = "ES_vanilla_gradient", linewidth=4, color = 'purple')


# methods # 

print("ES Hessian ... ")
res = ES_Hessian(F, alpha=0.5, sigma=0.05, theta_0=initial_pt, num_samples = 25, time_steps = 200)
plt.plot(res[3], res[4], label = "ES_Hessian", linewidth=4, color = 'orange')

print("ES Hessian - v2 ...")
res = ES_Hessian_v2(F, alpha=0.5, sigma=0.05, theta_0=initial_pt, num_samples = 25, time_steps = 200)
plt.plot(res[3], res[4], label = "ES_Hessian_v2", linewidth=4, color = 'gold')


#print("Original LP Hessian ...")
#res = LP_Hessian(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 25, time_steps = 100)
#plt.plot(res[3], res[4], label = "LP_Hessian")

# res = LP_Hessian_structured(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 100, time_steps = 200)
# plt.plot(res[3], res[4], label = "LP_Hessian_structured")

# res = LP_Hessian_structured_v2(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, num_samples = 100, time_steps = 200)
# plt.plot(res[3], res[4], label = "LP_Hessian_structured v2")

# res = LP_Hessian_structured_v3(F, alpha = 0.1, sigma = 0.05, theta_0=initial_pt, PT_threshold=-1e1, num_samples = 20, time_steps = 200)
# plt.plot(res[3], res[4], label = "LP_Hessian_structured v3")

print("LP Hessian structured with PT inverse, antithetic samples and backtracking ... ")
res = LP_Hessian_structured_v4(F, alpha = 1e-6, sigma = 0.01, theta_0=initial_pt, PT_threshold=-1e1, num_samples = 25, time_steps = 165)
plt.plot(res[3], res[4], label = "LP_Hessian_structured_v4", linewidth=4, color = 'red')

print("LP Hessian structured with PT inverse, antithetic samples and backtracking [simulation gradient estimator] ... ")
res = LP_Hessian_structured_v5(F, alpha = 1e-6, sigma = 0.01, sigma_g =0.01, theta_0=initial_pt, PT_threshold=-1e1, num_samples = 25, time_steps = 165)
plt.plot(res[3], res[4], label = "LP_Hessian_structured_v5", linewidth=4, color = 'magenta')

print("LP Hessian structured with PT inverse, antithetic samples and backtracking [linear regression structured hessian estimator] ... ")
res = LP_Hessian_structured_v6(F, alpha = 1e-6, sigma = 0.01, sigma_g =0.01, theta_0=initial_pt, PT_threshold=-1e1, num_samples = 25, time_steps = 165)
plt.plot(res[3], res[4], label = "LP_Hessian_structured_v6", linewidth=4, color = 'brown')

print("LP Hessian structured with PT inverse, antithetic samples and backtracking [unconstrained linear regression structured hessian estimator] ...")
res = LP_Hessian_structured_v7(F, alpha = 1e-6, sigma = 0.01, sigma_g =0.01, theta_0=initial_pt, PT_threshold=-1e1, num_samples = 25, time_steps = 165)
plt.plot(res[3], res[4], label = "LP_Hessian_structured_v7",linewidth=4, color = 'hotpink')


plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=20)
plt.xlabel("# function calls", fontsize=16)
plt.ylabel("function value", fontsize=16)
plt.title("Lunacek function (100 dimensional)", fontsize=20)
plt.tight_layout()

# plt.show()
plt.savefig('test_lunacek.png')
