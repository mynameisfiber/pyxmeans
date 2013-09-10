## pyxmeans

This is a quick implementation of
[XMeans](http://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf) for using kmeans
type clustering with an unknown number of clusters.  In order to make this code
runnable, I chose to use MiniBatchKMeans instead of KMeans, but they should be
swappable. 

**Currently** it only supports basic MiniBatch kmeans and does not yet do
XMeans.  However, the MiniBatch implementation is exceedingly fast and uses a
small memory footprint.  On my 2011 MacBook Air, the following benchmarks were
obtained (by running `python -m pyxmeans.benchmark`):

```
$ python -m pyxmeans.benchmark
Creating data
Number of points:  10000
Number of dimensions:  2
Number of clusters:  32
initial BIC:  -49941.3646148
initial variance:  0.00149836505705
initial RMS Error:  2.6090552287

Clustering with single-threaded pyxmeans
singlethreaded pyxmeans took 0.024292s
BIC:  -51057.495506
Variance:  0.000901304299092
RMS Error:  2.60713773714

Clustering with multi-threaded pyxmeans
multithreaded pyxmeans took 0.067050s
BIC:  -50773.6912824
Variance:  0.000928799737286
RMS Error:  2.60608823957

Clustering with sklearn
scikitlearn took 20.308440s
BIC:  -50525.8407882
Variance:  0.000968057715029
RMS Error:  2.61356849059
```

![](benchmark.png)

NOTES: 
    * `max_no_improvement` is set to `None` for MiniBatchKMeans to properly
      compare per-iteration speeds since we currently do not support
      early-stopping.
    * RMS Error for the multi-threaded pymeans is higher because that function
      aims at minimizing the variance of the resulting model.


### Dependencies

* [numpy](http://numpy.org/)

### Todo:

* Optimize data layout when dealing with / comparing computed clusters
* Early stopping mechanism
