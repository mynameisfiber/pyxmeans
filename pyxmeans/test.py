import numpy as np
import random

def generate_data(N, D, k, sigma=0.1):
    data = np.empty((N, D))
    distributions = [{"mean" : np.random.rand(D), "cov" : np.eye(D) * np.random.rand() * sigma} for i in xrange(k)]
    for i in xrange(N):
        params = random.choice(distributions)
        data[i, :] = np.random.multivariate_normal(**params)
    return data, distributions
    

