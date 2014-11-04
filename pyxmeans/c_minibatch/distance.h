#ifndef __DISTANCE_H

void assign_centroids_multi(double *data, double *centroids, int *assignments, int n_jobs, int k, int N, int D);
void assign_centroids(double *data, double *centroids, int *assignments, int k, int N, int D);
int closest_centroid(double *vector, double *centroids, int k, int D);
double euclidian_distance(double *A, double *B, int D);
double cosine_distance(double *A, double *B, int D);
double distance_to_closest_centroid(double *vector, double *centroids, int k, int D);

int set_distance_metric(int metric);
double (*distance_metric)(double*, double*, int);

#define __DISTANCE_H
#endif
