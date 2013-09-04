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
Creating data
Number of points:  10000
Number of dimensions:  2
Number of clusters:  48
initial BIC:  -54533.9853416
initial variance:  0.00119839264405

Clustering with single-threaded pyxmeans
singlethreaded pyxmeans took 0.018506s
BIC of single-threaded pyxmeans:  -54556.7397626
Variance of single-threaded pyxmeans:  0.000896105585028
RMS Error:  3.66005961722

Clustering with multi-threaded pyxmeans
singlethreaded pyxmeans took 0.043338s
BIC of multi-threaded pyxmeans:  -54951.6477002
Variance of multi-threaded pyxmeans:  0.000840786730204
RMS Error:  3.66460410732

Clustering with sklearn
singlethreaded pyxmeans took 29.838236s
BIC of sklearn:  -55225.3186563
Variance of sklearn:  0.000682348816436
RMS Error:  3.66329123839
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
