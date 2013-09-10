#ifndef __MINIBATCH_H

#define PI (3.141592653589793)

#ifdef VERBOSE
    /* macro using var args */
    #include <stdio.h>
    #include <time.h>
    #define _LOG(...) do { fprintf(stderr, "[%lu] ", time(NULL)); fprintf (stderr, ## __VA_ARGS__); } while(0)
#else
    /* when debug isn't defined all the macro calls do absolutely nothing */
    #define _LOG(...) do {;} while(0)
#endif

void minibatch_multi(double *data, double *centroids, int n_samples, int max_iter, int n_runs, int n_jobs, double bic_ratio_termination, double reassignment_ratio, int k, int N, int D);
void minibatch(double *data, double *centroids, int n_samples, int max_iter, double bic_ratio_termination, double reassignment_ratio, int k, int N, int D);
void minibatch_iteration(double *data, double *centroids, int *sample_indicies, int *centroid_counts, int *cluster_cache, int n_samples, int k, int N, int D);
void gradient_step(double *vector, double *centroid, int count, int D);

double model_variance(double *data, double *centroids, int k, int N, int D);
double bayesian_information_criterion(double *data, double *centroids, int k, int N, int D);
void reassign_centroids(double *data, double *centroids, int *reassign_clusters, int n_samples, int K, int k, int N, int D);
void kmeanspp(double *data, double *centroids, int n_samples, int k, int N, int D);
void kmeanspp_multi(double *data, double *centroids, int n_samples, int n_runs, int n_jobs, int k, int N, int D);

void save_double_matrix(double *data, char *filename, int N, int D);
void save_int_matrix(int *data, char *filename, int N, int D);

#define __MINIBATCH_H
#endif
