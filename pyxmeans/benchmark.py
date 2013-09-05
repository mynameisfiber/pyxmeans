import numpy as np
import random
import time
from contextlib import contextmanager
from pyxmeans import _minibatch
import pylab as py

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError:
    MiniBatchKMeans = None


@contextmanager
def TimerBlock(name):
    start = time.time()
    yield
    end = time.time()
    print "%s took %fs" % (name, end-start)

def generate_data(N, D, k, sigma=0.1):
    data = np.empty((N, D))
    distributions = [{"mean" : np.random.rand(D), "cov" : np.eye(D) * np.random.rand() * sigma} for i in xrange(k)]
    for i in xrange(N):
        params = random.choice(distributions)
        data[i, :] = np.random.multivariate_normal(**params)
    return data, distributions

def error(actual, test):
    err = 0.0
    for t in test:
        v = np.square(actual - test[:, np.newaxis]).sum(axis=1).min()
        err += v
    return err / float(len(test))


if __name__ == "__main__":
    print "Creating data"
    N = 10000
    D = 2
    k = 48

    data, actual = generate_data(N, D, k, sigma=0.0005)
    actual_data = np.asarray([x["mean"] for x in actual])
    clusters = _minibatch.kmeanspp_multi(data, np.empty((k, D)), 20, 4)
    print "Number of points: ", N
    print "Number of dimensions: ", D
    print "Number of clusters: ", k
    print "initial BIC: ", _minibatch.bic(data, clusters)
    print "initial variance: ", _minibatch.model_variance(data, clusters)
    print

    print "Clustering with single-threaded pyxmeans"
    clusters_pymeans_single = clusters.copy()
    with TimerBlock("singlethreaded pyxmeans"):
        clusters_pymeans_single = _minibatch.minibatch(data, clusters_pymeans_single, k*5, 100, -1.0)
    print "BIC of single-threaded pyxmeans: ", _minibatch.bic(data, clusters_pymeans_single)
    print "Variance of single-threaded pyxmeans: ", _minibatch.model_variance(data, clusters_pymeans_single)
    print "RMS Error: ", error(actual_data, clusters_pymeans_single)
    print
    
    print "Clustering with multi-threaded pyxmeans"
    clusters_pymeans_multi = clusters.copy()
    with TimerBlock("singlethreaded pyxmeans"):
        clusters_pymeans_multi = _minibatch.minibatch_multi(data, clusters_pymeans_multi, k*5, 100, 4, 4, -1.0)
    print "BIC of multi-threaded pyxmeans: ", _minibatch.bic(data, clusters_pymeans_multi)
    print "Variance of multi-threaded pyxmeans: ", _minibatch.model_variance(data, clusters_pymeans_multi)
    print "RMS Error: ", error(actual_data, clusters_pymeans_multi)
    print

    print "Clustering with sklearn"
    if MiniBatchKMeans:
        clusters_sklearn = clusters.copy()
        with TimerBlock("singlethreaded pyxmeans"):
            mbkmv = MiniBatchKMeans(k, max_iter=100, batch_size=k*5, init=clusters_sklearn, reassignment_ratio=0, compute_labels=False, max_no_improvement=None).fit(data)
        print "BIC of sklearn: ", _minibatch.bic(data, mbkmv.cluster_centers_)
        print "Variance of sklearn: ", _minibatch.model_variance(data, mbkmv.cluster_centers_)
        print "RMS Error: ", error(actual_data, clusters_sklearn)
    else:
        print "sklearn not found"


    py.figure()
    py.title("pyxmeans performance")
    py.scatter(data[:,0], data[:,1], label="data")
    py.scatter(clusters_pymeans_single[:,0], clusters_pymeans_single[:,1], c='m', s=75, alpha=0.75, label="pymeans single")
    py.scatter(clusters_pymeans_multi[:,0], clusters_pymeans_multi[:,1], c='y', s=75, alpha=0.75, label="pymeans multi")
    if MiniBatchKMeans:
        py.scatter(clusters_sklearn[:,0], clusters_sklearn[:,1], s=75, c='g', alpha=0.75, label="sklearn")
    py.scatter(actual_data[:,0], actual_data[:,1], c='r', s=75, alpha=0.75, label="actual center")
    py.legend()

    py.show()


