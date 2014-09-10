## pyxmeans

This is a quick implementation of
[XMeans](http://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf) for using kmeans
type clustering with an unknown number of clusters.  In order to make this code
runnable, I chose to use MiniBatchKMeans instead of KMeans, but they should be
swappable. 

Currently MiniBatch and XMeans are supported.  XMeans uses multiple MiniBatch
and trial MiniBatch runs in order to infer how many clusters the data has.  It
does so by taking the population of a given cluster center, splitting it, and
seeing if the resulting labels have a better BIC (Bayesian Information
Criterion) than before.  This can be done successively until we find the number
of clusters.

In addition to providing XMeans, the MiniBatch implementation in this package is
exceedingly fast.  Below are benchmarks for all the provided clustering methods
and sklearn's MiniBatch routine.


```
$ python -m pyxmeans.benchmark
Creating data
Number of points:  10000
Number of dimensions:  2
Number of clusters:  32
initial BIC:  -50214.4559857
initial variance:  0.00164148581105
initial RMS Error:  2.31798411948

Clustering with single-threaded pyxmeans
singlethreaded pyxmeans took 0.043875s
BIC:  -50762.9827994
Variance:  0.00115765494439
RMS Error:  2.31035692593

Clustering with multi-threaded pyxmeans
multithreaded pyxmeans took 0.326129s
BIC:  -50982.8001508
Variance:  0.00104848000929
RMS Error:  2.3113536455

Clustering with multi-threaded pyxmeans (starting k at 20)
multithreaded pyxmeans took 79.005781s
Num Clusters:  30
BIC:  -50352.8461421
Variance:  0.00104986238957
RMS Error:  2.31100693171

Clustering with sklearn
scikitlearn took 9.241426s
BIC:  -50679.1763114
Variance:  0.00112580908789
RMS Error:  2.31050074192
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
* Better memory management in XMeans (we're copying things everywhere)
* Pool out children tests in XMeans
