## pyxmeans

This is a quick implementation of
[XMeans](http://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf) for using kmeans
type clustering with an unknown number of clusters.  In order to make this code
runnable, I chose to use MiniBatchKMeans instead of KMeans, but they should be
swappable. 

**Currently** it only supports basic MiniBatch kmeans and does not yet do XMeans.  However, the MiniBatch implementation is exceedingly fast and uses a small memory footprint.  On my 2011 MacBook Air, the following benchmarks were obtained:

```
In [1]: from sklearn.cluster import MiniBatchKMeans

In [2]: from pyxmeans import _minibatch

In [3]: from pyxmeans.text import generate_data

# Generate data with 50000 2D samples belonging to 128 gaussians
In [9]: data, actual = generate_data(50000, 2, 128, 0.0005)


In [28]: %%timeit 
   ....: clusters = np.empty((128,2))
   ....: clusters = _minibatch.kmeanspp(data, clusters)
   ....: _minibatch.minibatch(data, clusters, 128*5, 100)
   ....:
   1 loops, best of 3: 2.15 s per loop


In [30]: %%timeit 
   ....: clusters = np.empty((128,2))
   ....: clusters = _minibatch.kmeanspp(data, clusters)
   ....: _minibatch.minibatch_multi(data, clusters, 128*5, 100, 4, 4)
   ....:
   1 loops, best of 3: 2.31 s per loop
   

In [31]: %%timeit
   ....: kmv = MiniBatchKMeans(128, max_iter=100, batch_size=128*5, n_init=1, compute_labels=False, max_no_improvement=None).fit(data)
   ....:
   1 loops, best of 3: 88.7 s per loop
```

NOTE: `max_no_improvement` is set to `None` for MiniBatchKMeans to properly compare per-iteration speeds since we currently do not support early-stopping.



### Dependencies

* [numpy](http://numpy.org/)

### Todo:

* Optimize data layout when dealing with / comparing computed clusters
* Early stopping mechanism
