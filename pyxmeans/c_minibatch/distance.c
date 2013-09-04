#include "distance.h"

/*
 * Assigns centroids to each datapoint
 */
void assign_centroids(double *data, double *centroids, int *assignments, int k, int N, int D) {
    for(int i=0; i<N; i++) {
        assignments[i] = closest_centroid(data + i*D, centroids, k, D);
    }
}

/*
 * Returns the squared euclidian distance between vectors A and B of length D
 */
double euclidian_distance(double *A, double *B, int D) {
    double d = 0.0;
    double dx = 0.0;
    for (int i=0; i<D; i++) {
        dx = (*A++) - (*B++);
        d += dx * dx;
    }
    return d;
}

/*
 * Returns the index of the closest centroid to the inputted vector where k is
 * the number of centroids and D is the dimensionality of the space
 */
int closest_centroid(double *vector, double *centroids, int k, int D) {
    int c = -1;
    double min_distance, cur_distance;
    for(int i=0; i<k; i++) {
        cur_distance = euclidian_distance(vector, (centroids + i * D), D);
        if (c == -1 || cur_distance < min_distance) {
            c = i;
            min_distance = cur_distance;
        }
    }
    return c;
}

/*
 * Returns the distance to the closest centroid to the inputted vector where k is
 * the number of centroids and D is the dimensionality of the space
 */
double distance_to_closest_centroid(double *vector, double *centroids, int k, int D) {
    double min_distance = -1.0;
    double cur_distance;
    for(int i=0; i<k; i++) {
        cur_distance = euclidian_distance(vector, (centroids + i * D), D);
        if (min_distance < 0 || cur_distance < min_distance) {
            min_distance = cur_distance;
        }
    }
    return min_distance;
}
