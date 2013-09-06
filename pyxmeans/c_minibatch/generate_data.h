#ifndef __GENERATE_DATA_H

#include <stdbool.h>

void generate_random_indicies(int N, int n, int *sample_indicies);
bool relatively_prime(int A, int B);
void gaussian_data(double *data, int K, int N, int D);
void random_matrix(double *data, int N, int D);

#define __GENERATE_DATA_H
#endif
