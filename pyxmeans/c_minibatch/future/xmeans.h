#ifndef __XMEANS_H

int generate_random_indicies_in_cluster(double *data, double *centroids, int *sample_indicies, int cluster, int n, int max_iter, int k, int N, int D);
int xmeans(double *data, double *centroids, int n_samples, int max_iter, int k_min, int k_max, int N, int D);

#define __XMEANS_H
#endif

