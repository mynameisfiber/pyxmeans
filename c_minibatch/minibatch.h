#ifndef __MINIBATCH_H

#define PI (3.141592653589793)

#ifdef VERBOSE
/* macro using var args */
#define _LOG(...) do { fprintf(stderr, "[%lu] ", time(NULL)); fprintf (stderr, ## __VA_ARGS__); } while(0)
#else
/* when debug isn't defined all the macro calls do absolutely nothing */
#define _LOG(...) do {;} while(0)
#endif

void generate_random_indexes(int N, int n, int *sample_indexes);
void gradient_step(double *vector, double *centroid, int count, int D);
void kmeanspp(double *data, double *centroids, int k, int N, int D);
void minibatch(double *data, double *centroids, int n_samples, int max_iter, int k, int N, int D);
void save_double_matrix(double *data, char *filename, int N, int D);
void save_int_matrix(int *data, char *filename, int N, int D);
double bayesian_information_criterion(double *data, double *centroids, int k, int N, int D);

#else
#define __MINIBATCH_H
#endif
