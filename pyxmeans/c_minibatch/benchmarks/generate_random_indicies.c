#include <time.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

void generate_random_indicies_lcg(int N, int n, int *sample_indicies);
void generate_random_indicies_brute(int N, int n, int *sample_indicies);
bool relatively_prime(int A, int B);

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

void generate_random_indicies_lcg(int N, int n, int *sample_indicies) {
    unsigned int seed = clock();
    // Pick a random starting prime in the set [2,N)
    unsigned int prime = rand_r(&seed) % (N-3) + 2;

    while (!relatively_prime(prime, N)) {
        prime += 1;
    }

    int current = rand_r(&seed) & (N-1);
    for(int i=0; i<n; i++) {
        sample_indicies[i] = current;
        current = (current + prime) % N;
    }
}

void generate_random_brute(int N, int n, int *sample_indicies) {
    /* Parameters:
     *      N - size of array to pick samples from
     *      n - number of samples to pick
     *      sample_indicies - array of the sample indicies (len(sample_indicies) == n)
     *
     * TODO: generate the sample indicies with a LCG
     */
    
    unsigned int seed = clock();
    for(int i=0; i<n; i++) {
        int index;
        for(int j=-1; j<i; j++) {
            if (j == -1 || sample_indicies[j] == index) {
                index = rand_r(&seed) % (N - 1);
                j = 0;
            }
        }
        sample_indicies[i] = index;
    }
}

int main(void) {
    int T = 10000;
    int n =  1 << 9;
    int N =  1 << 20;
    int *indicies = (int*) malloc(n * sizeof(int));
    clock_t start, stop;
    
    printf("Iterations: %d\n", T);
    printf("N: %d\n", N);
    printf("n: %d\n", n);

    printf("Testing LCG method\n");
    start = clock();
    for(int i=0; i<T; i++) {
        generate_random_indicies_lcg(N, n, indicies);
    }
    stop = clock();
    printf("Total time: %0.5f\n", (double) (stop-start) / CLOCKS_PER_SEC);
    printf("Time per iteration: %0.5f\n", (double) (stop-start) / CLOCKS_PER_SEC / T);

    printf("\n");

    printf("Testing brute method\n");
    start = clock();
    for(int i=0; i<T; i++) {
        generate_random_brute(N, n, indicies);
    }
    stop = clock();
    printf("Total time: %0.5f\n", (double) (stop-start) / CLOCKS_PER_SEC);
    printf("Time per iteration: %0.5f\n", (double) (stop-start) / CLOCKS_PER_SEC / T);

    return 0;
}
