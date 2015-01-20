#include "distance.h"
#include <math.h>

int set_distance_metric(int metric) {
    switch (metric) {
        case 0:
            distance_metric = euclidian_distance;
            break;
        case 1:
            distance_metric = cosine_distance;
            break;
        default:
            return 0;
    }
    return 1;
}

/*
 * Assigns centroids to each datapoint
 */
void assign_centroids(double *data, double *centroids, int *assignments, int k, int N, int D) {
    assign_centroids_multi(data, centroids, assignments, 1, k, N, D);
}

/*
 * Assigns centroids to each datapoint using multiple threads
 */
void assign_centroids_multi(double *data, double *centroids, int *assignments, int n_jobs, int k, int N, int D) {
    #pragma omp parallel shared(data, centroids, assignments) num_threads(n_jobs)
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
 * Returns the cosine distance between vectors A and B of length D.  We
 * renormalize the distance such that a value of 0 is returned when the vectors
 * are pointing in the same direction and a distance of 2 is returned when they
 * are pointing in opposite directions
 */
double cosine_distance(double *A, double *B, int D) {
    double dot = 0.0;
    double lenA = 0.0, lenB = 0.0;
    for (int i=0; i<D; i++) {
        dot += (*A) * (*B);
        lenA += (*A) * (*A);
        lenB += (*B) * (*B);
        A++;
        B++;
    }
    return 1.0 - dot / (sqrt(lenA) * sqrt(lenB));
}

/*
 * Returns the index of the closest centroid to the inputted vector where k is
 * the number of centroids and D is the dimensionality of the space
 */
int closest_centroid(double *vector, double *centroids, int k, int D) {
    int c = -1;
    double min_distance, cur_distance;
    for(int i=0; i<k; i++) {
        cur_distance = distance_metric(vector, (centroids + i * D), D);
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
        cur_distance = distance_metric(vector, (centroids + i * D), D);
        if (min_distance < 0 || cur_distance < min_distance) {
            min_distance = cur_distance;
        }
    }
    return min_distance;
}
