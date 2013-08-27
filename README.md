## pyxmeans

This is a quick implementation of
[XMeans](http://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf) for using kmeans
type clustering with an unknown number of clusters.  In order to make this code
runnable, I chose to use MiniBatchKMeans instead of KMeans, but they should be
swappable.  This choice was primarily made because `sklearn`'s implementation of
kmeans is memory-inefficient.

### Dependencies

* [milk](http://pythonhosted.org/milk/) (with the [kmeans centroid](https://github.com/luispedro/milk/pull/11) patch)
* [scikit-learn](http://scikit-learn.org/) (soon to be removed)
* [numpy](http://numpy.org/)

### Todo:

* Use lower-level kmeans implemintation focues on memory efficiency in order to
  not use sklearn
* Optimize data layout when dealing with / comparing computed clusters

Inspired by [goxmeans](https://github.com/danielhfrank/goxmeans/)
