#ifndef __DISTANCE_H

int closest_centroid(double *vector, double *centroids, int k, int D);
double distance(double *A, double *B, int D);
double distance_to_closest_centroid(double *vector, double *centroids, int k, int D);

#else
#define __DISTANCE_H
#endif
