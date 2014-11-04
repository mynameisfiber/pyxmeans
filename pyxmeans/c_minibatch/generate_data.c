#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "generate_data.h"

bool relatively_prime(int A, int B){
    // assert A > 0
    // assert B > 0
    for ( ; ; ) {
        if (!(A%=B)) {
            return B == 1;
        }
        if (!(B%=A)) {
            return A == 1;
        }
    }
}

/*
 * Will calculate a list of n unique integers in [0,N) and fill sample_indicies
 * with the result
 */
void generate_random_indicies(int N, int n, int *sample_indicies) {
    // This uses a very simple LCG in order to quickly get n unique numbers in
    // the space [0,N)
    
    unsigned int seed = (int) clock() * (omp_get_thread_num() + 1);
    srand(seed);
    // Pick a random starting prime in the set [2,N)
    unsigned int rel_prime = (rand() / ((double)RAND_MAX+1)) * (N-3) + 2;

    while (!relatively_prime(rel_prime, N)) {
        rel_prime += 1;
    }

    int current = rand() & (N-1);
    for(int i=0; i<n; i++) {
        sample_indicies[i] = current;
        current = (current + rel_prime) % N;
    }
}

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

