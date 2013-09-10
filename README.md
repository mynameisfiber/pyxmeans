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
initial BIC:  -50732.4386243
initial variance:  0.00132625903391
initial RMS Error:  2.8660516391

Clustering with single-threaded pyxmeans
singlethreaded pyxmeans took 0.023339s
BIC:  -50897.6672898
Variance:  0.000826657605777
RMS Error:  2.86707908932

Clustering with multi-threaded pyxmeans
multithreaded pyxmeans took 0.079182s
BIC:  -51117.8275529
Variance:  0.000801938412009
RMS Error:  2.8652907064

Clustering with sklearn
scikitlearn took 38.654962s
BIC:  -50979.7777762
Variance:  0.000767407020249
RMS Error:  2.86791602632
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
