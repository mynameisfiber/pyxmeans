#!/usr/bin/env python2.7

import numpy as np
import _minibatch

import multiprocessing

class MiniBatch(object):
    def __init__(self, n_clusters, n_samples=None, max_iter=1000, n_runs=4, n_init=3, init='kmeans++', n_jobs=-1, bic_termination=-1.0, reassignment_ratio=0.0, compute_labels=False, verbose=False):
        """
        Create a MiniBatch model as described in http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf

        :param n_clusters: Number of clusters to classify
        :type n_clusters: int

        :param n_samples: Number of samples to consider for each iteration of
                          the fit
        :type n_samples: int

        :param max_iter: Maximum number of iterations of MiniBatch to perform
        :type max_iter: int

        :param n_runs: Number of independent MiniBatch runs to perform when
                       running multiple jobs. The result with the best variance
                       will be picked.
        :type n_runs: int

        :param n_init: Number of attempts at initializing the cluster centers
                       when using kmeans++. The cluster centers with the best
                       variance will be picked.
        :type n_init: int

        :param init: Method of cluster initialization.  Can be 'kmeans++',
                     'random' or an ndarray of pre-computed centers
        :type init: string or ndarray(double), shape = (n_clusters, n_features)

        :param bic_termination: Maximum change in BIC score that will trigger
                                an early termination of the mini batch run
        :type bic_termination: float

        :param reassignment_ratio: Maximum ratio from maximum number of cluster
                                   members that will cause a centroid's
                                   reassignment
        :type reassignment_ratio: float

        :param n_jobs: Number of concurrent jobs to run.  A value of 1 will run
                       things serially, a positive number will run with that
                       number of cores and a negative value will leave that
                       many cores free (ie: '0' will run on all cores, '-1'
                       will run on all but one core, etc..)
        :type n_jobs: int

        :param computer_labels: Whether to calculate the labels for the data
                                the model is fit with.  This data is available
                                in MiniBatch.labels_
        :type compute_labels: bool

        :param vebose: Toggle verbosity
        :type verbose: bool

        :returns: Trained MiniBatch instance
        :rtype: MiniBatch
        """
        self.n_clusters = n_clusters
        self.n_samples = n_samples or 10 * n_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_runs = n_runs
        self.n_init = n_init
        self.n_jobs = n_jobs
        self.compute_labels = compute_labels
        self.verbose = verbose

        self.bic_termination = bic_termination
        self.reassignment_ratio = reassignment_ratio

        self.cluster_centers_ = None
        self.labels_ = None

        if n_jobs <= 0:
            self.n_jobs = multiprocessing.cpu_count() - n_jobs


    def fit(self, data):
        """
        Fit the current model with the inputted data.

        :param data: Samples to fit model with
        :type data: ndarray(double), shape = (n_samples, n_features)
        :returns: Trained MiniBatch instance
        :rtype: MiniBatch
        """
        data = np.asarray(data, dtype=np.double)
        if self.verbose:
            print "Initializing clusters"
        if self.init == 'random':
            self.cluster_centers_ = np.random.random((self.n_clusters, data.shape[1]), dtype=np.double)
        elif self.init == 'kmeans++':
            self.cluster_centers_ = np.zeros((self.n_clusters, data.shape[1]), dtype=np.double)
            if self.n_jobs > 1:
                jobs = min(self.n_jobs, self.n_init)
                self.cluster_centers_ = _minibatch.kmeanspp_multi(data, self.cluster_centers_, self.n_samples, self.n_init, jobs)
            else:
                self.cluster_centers_ = _minibatch.kmeanspp(data, self.cluster_centers_, self.n_samples)
        elif isinstance(self.init, np.ndarray):
            if not self.init.flags['C_CONTIGUOUS']:
                raise Exception("init ndarray must be C_CONTIGUOUS")
            elif self.init.shape != (self.n_clusters, data.shape[1]):
                raise Exception("init cluster not of correct shape %r != (%d, %d)" % (self.init.shape, self.n_clusters, data.shape[1]))
            self.cluster_centers_ = self.init

        if self.verbose:
            print "Running minibatch"
        if self.n_jobs > 1:
            jobs = min(self.n_jobs, self.n_runs)
            self.cluster_centers_ =  _minibatch.minibatch_multi(data, self.cluster_centers_, self.n_samples, self.max_iter, self.n_runs, jobs, self.bic_termination, self.reassignment_ratio)
        else:
            self.cluster_centers_ =  _minibatch.minibatch(data, self.cluster_centers_, self.n_samples, self.max_iter, self.bic_termination, self.reassignment_ratio)

        if self.compute_labels:
            if self.verbose:
                print "Computing labels"
            self.labels_ = np.zeros((data.shape[0], ), dtype=np.intc)
            self.labels_ = _minibatch.assign_centroids(data, self.cluster_centers_, self.labels_, self.n_jobs)

        return self


    def predict(self, data):
        """
        Labels the data given the fitted mode.

        :param data: Samples to classify
        :type data: ndarray(double), shape = (n_samples, n_features)
        :returns: Index into MiniBatch.cluster_centers_ for each datapoint in
                  data
        :rtype: ndarray(intc), shape = (n_samples,)
        """
        assert self.cluster_centers_ is not None, "Model not yet fitted"
        labels = np.zeros((data.shape[0], ), dtype=np.intc)
        labels = _minibatch.assign_centroids(data, self.cluster_centers_, labels, self.n_jobs)
        return labels


    def variance(self, data):
        """
        Returns the variance, or inertia, of the fitted model given the data

        :param data: Data to use in calculation of the variance
        :type data: ndarray(double), shape = (n_samples, n_features)
        :rtype: float
        """
        assert self.cluster_centers_ is not None, "Model not yet fitted"
        return _minibatch.model_variance(data, self.cluster_centers_)


    def bic(self, data):
        """
        Returns the Bayesian information criterion for the fitted model given
        the data

        :param data: Data to use in calculation of the Bayesian information
                     criterion
        :type data: ndarray(double), shape = (n_samples, n_features)
        :rtype: float
        """
        assert self.cluster_centers_ is not None, "Model not yet fitted"
        return _minibatch.bic(data, self.cluster_centers_)

