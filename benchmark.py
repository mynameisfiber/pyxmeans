import numpy as np
import random
import time
from contextlib import contextmanager
from pyxmeans import _minibatch
from pyxmeans.mini_batch import MiniBatch
from pyxmeans.xmeans import XMeans

try:
    import pylab as py
except ImportError:
    py = None
    print "Could not find matplotlib, not plotting"

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
    k = 32
    max_iter = 100
    n_samples = k * 10

    data, actual = generate_data(N, D, k, sigma=0.001)
    actual_data = np.asarray([x["mean"] for x in actual])
    clusters = _minibatch.kmeanspp_multi(data, np.empty((k, D), dtype=np.double), N / 100, 20, 4)
    print "Number of points: ", N
    print "Number of dimensions: ", D
    print "Number of clusters: ", k
    print "initial BIC: ", _minibatch.bic(data, clusters)
    print "initial variance: ", _minibatch.model_variance(data, clusters)
    print "initial RMS Error: ", error(actual_data, clusters)
    print

    print "Clustering with single-threaded pyxmeans"
    clusters_pymeans_single = clusters.copy()
    with TimerBlock("singlethreaded pyxmeans"):
        mbst = MiniBatch(k, n_samples=n_samples, max_iter=max_iter, n_runs=1, init=clusters_pymeans_single, n_jobs=1, compute_labels=False).fit(data)
        clusters_pymeans_single = mbst.cluster_centers_
    print "BIC: ", _minibatch.bic(data, clusters_pymeans_single)
    print "Variance: ", _minibatch.model_variance(data, clusters_pymeans_single)
    print "RMS Error: ", error(actual_data, clusters_pymeans_single)
    print
    
    print "Clustering with multi-threaded pyxmeans"
    clusters_pymeans_multi = clusters.copy()
    with TimerBlock("multithreaded pyxmeans"):
        mbmt = MiniBatch(k, n_samples=n_samples, max_iter=max_iter, n_runs=4, init=clusters_pymeans_multi, n_jobs=0, compute_labels=False).fit(data)
        clusters_pymeans_multi = mbmt.cluster_centers_
    print "BIC: ", _minibatch.bic(data, clusters_pymeans_multi)
    print "Variance: ", _minibatch.model_variance(data, clusters_pymeans_multi)
    print "RMS Error: ", error(actual_data, clusters_pymeans_multi)
    print

    k_init = int(k * 0.65)
    print "Clustering with multi-threaded pyxmeans (starting k at {})".format(k_init)
    with TimerBlock("multithreaded pyxmeans"):
        mxmt = XMeans(k_init, verbose=False).fit(data)
        clusters_xmeans = mxmt.cluster_centers_
    print "Num Clusters: ", len(clusters_xmeans)
    print "BIC: ", _minibatch.bic(data, clusters_xmeans)
    print "Variance: ", _minibatch.model_variance(data, clusters_xmeans)
    print "RMS Error: ", error(actual_data, clusters_xmeans)
    print

    print "Clustering with sklearn"
    if MiniBatchKMeans:
        clusters_sklearn = clusters.copy()
        with TimerBlock("scikitlearn"):
            mbkmv = MiniBatchKMeans(k, max_iter=max_iter, batch_size=n_samples, init=clusters_sklearn, reassignment_ratio=0, compute_labels=False, max_no_improvement=None).fit(data)
        print "BIC: ", _minibatch.bic(data, mbkmv.cluster_centers_)
        print "Variance: ", _minibatch.model_variance(data, mbkmv.cluster_centers_)
        print "RMS Error: ", error(actual_data, clusters_sklearn)
    else:
        print "sklearn not found"


    if py is not None:
        py.figure()
        py.title("pyxmeans performance")
        py.scatter(data[:,0], data[:,1], alpha=0.25, label="data")
        py.scatter(actual_data[:,0], actual_data[:,1], c='r', s=125, alpha=0.6, label="actual center")
        py.scatter(clusters_pymeans_single[:,0], clusters_pymeans_single[:,1], c='m', s=75, alpha=0.4, label="pymeans single")
        py.scatter(clusters_pymeans_multi[:,0], clusters_pymeans_multi[:,1], c='y', s=75, alpha=0.4, label="pymeans multi")
        py.scatter(clusters_xmeans[:,0], clusters_xmeans[:,1], c='w', s=75, alpha=0.4, label="pyxmeans")
        if MiniBatchKMeans:
            py.scatter(clusters_sklearn[:,0], clusters_sklearn[:,1], s=75, c='g', alpha=0.4, label="sklearn")
        py.legend()

        py.tight_layout()
        py.show()

