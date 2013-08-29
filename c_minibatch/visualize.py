#!/usr/bin/env python2.7

import numpy as np
import pylab as py

data = np.loadtxt("data/cluster_data.dat")
for i in range(100):
    py.clf()
    try:
        centroids = np.loadtxt("data/centroids-%0.2d.dat" % i)
        samples = np.loadtxt("data/samples-%0.2d.dat" % i, dtype=np.int)
    except:
        break
    py.scatter(data[~samples, 0], data[~samples, 1])
    py.scatter(data[ samples, 0], data[ samples, 1], c='k')
    py.scatter(centroids[:,0], centroids[:,1], c="g", s=100, alpha=0.5)
    py.savefig("clusters-%0.2d.png" % i)
