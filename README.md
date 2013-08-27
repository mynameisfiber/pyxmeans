## pyxmeans

This is a quick implementation of
[XMeans](http://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf) for using kmeans
type clustering with an unknown number of clusters.  In order to make this code
runnable, I chose to use MiniBatchKMeans instead of KMeans, but they should be
swappable.  This choice was primarily made because `sklearn`'s implementation of
kmeans is memory-inefficient.


### Todo:

* Use lower-level kmeans implemintation focues on memory efficiency in order to
  not use sklearn
* Optimize data layout when dealing with / comparing computed clusters

Inspired by [goxmeans](https://github.com/danielhfrank/goxmeans/)
