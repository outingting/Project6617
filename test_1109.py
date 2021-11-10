# Nov 09 2021 

# packages
import numpy as np
from scipy.stats import multivariate_normal
from utils import benchmark_methods
import matplotlib.pyplot as plt

# Blackbox function
def F(theta):
    return -sum((theta - 0.5) ** 2)

# True gradient of the Blackbox function
# [unknown to testing methods]
def DF(theta):
    return -2 * (theta - 0.5)

# True Hessian of the Blackbox function
# [unknown to testing methods]
def D2F(theta):
    d = theta.shape[0]
    return -2 * np.identity(d)

# Main function
if __name__ == "__main__":
    np.random.seed(2)
#     print(benchmark_methods.ES_benchmark_gradient(F, alpha=0.001, sigma=0.01, theta_0 = np.array([1.0,1.0]), num_samples = 50, time_steps = 1000))
#     print(benchmark_methods.ES_hessian_play(F, alpha=1, sigma=0.01, theta_0=np.array([1.0,1.0]), num_samples = 50, time_steps = 1000, p = 1, H_lambda = 0))
    theta_t, res, H, F_val_lst_ours = benchmark_methods.ES_hessian_play(F, alpha=1, sigma=0.01, theta_0=np.array([1.0,1.0]), num_samples = 1e5, time_steps = 100, p = 1, H_lambda = 0)
#     print(benchmark_methods.ES_benchmark_hess_aware(F, alpha=1, sigma=0.01, theta_0=np.array([1.0,1.0]), num_samples = 50, time_steps = 100, p = 10, H_lambda = 0.00))
    theta_t_bm, res_bm, H_bm, F_val_lst_bm = benchmark_methods.ES_benchmark_hess_aware(F, alpha=1, sigma=0.01, theta_0=np.array([1.0,1.0]), num_samples = 50, time_steps = 100, p = 1, H_lambda = 0.00)
    plt.plot(range(len(F_val_lst_ours)), F_val_lst_ours)
    plt.plot(range(len(F_val_lst_bm)), F_val_lst_bm)
    plt.plot(range(len(F_val_lst_ours)), np.zeros((len(F_val_lst_ours),1)))
    plt.legend(['ours','bm','0'])
    plt.show()