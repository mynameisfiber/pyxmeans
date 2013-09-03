#include <stdlib.h>
#include <math.h>
#include "generate_data.h"

void random_matrix(double *data, int N, int D) {
    for(int i=0; i<N; i++) {
        for(int j=0; j<D; j++) {
            data[i*D + j] = rand() / (double)RAND_MAX;
        }
    }
}

void gaussian_data(double *data, int K, int N, int D) {
    double *center = (double*) malloc(D * sizeof(double));
    for(int i=0; i<N; i++) {
        if (i % (N/K) == 0) {
            random_matrix(center, N, 1);
        }

        for(int j=0; j<D; j++) {
            double dx = center[j] - (rand() / (double)RAND_MAX - 0.5)/ 2.0;
            data[i*D + j] = exp(-1.0 * dx * dx);
        }
    }
    free(center);
}

