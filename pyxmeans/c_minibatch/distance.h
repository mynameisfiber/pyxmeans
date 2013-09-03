#ifndef __DISTANCE_H

void assign_centroids(double *data, double *centroids, int *assignments, int k, int N, int D);
int closest_centroid(double *vector, double *centroids, int k, int D);
double euclidian_distance(double *A, double *B, int D);
double distance_to_closest_centroid(double *vector, double *centroids, int k, int D);

#define __DISTANCE_H
#endif
