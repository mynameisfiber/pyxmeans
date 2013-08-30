#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "minibatch.h"
#include "generate_data.h"
#include "distance.h"

void save_double_matrix(double *data, char *filename, int N, int D) {
    FILE *fd = fopen(filename, "w+");
    for(int i=0; i<N; i++) {
        for(int j=0; j<D; j++) {
            fprintf(fd, "%f\t", data[i*D + j]);
        }
        fprintf(fd, "\n");
    }
    fclose(fd);
}
void save_int_matrix(int *data, char *filename, int N, int D) {
    FILE *fd = fopen(filename, "w+");
    for(int i=0; i<N; i++) {
        for(int j=0; j<D; j++) {
            fprintf(fd, "%d\t", data[i*D + j]);
        }
        fprintf(fd, "\n");
    }
    fclose(fd);
}


/*
 * Moves the given centroid closer to the given vector with a learning rate
 * proportional to the number of vectors already in the centroid
 */
void gradient_step(double *vector, double *centroid, int count, int D) {
    double eta = 1.0 / count;
    double eta_compliment = 1.0 - eta;
    for(int i=0; i<D; i++) {
        centroid[i] = eta_compliment * centroid[i] + eta * vector[i];
    }
}

/*
 * Calculates the bayesian information criterion for clustered data which
 * represents how good a model the centroids represents.
 */
double bayesian_information_criterion(double *data, double *centroids, int k, int N, int D) {
    /* Calculate the variance of the model and the centroid counts */
    int *centroid_count = (int*) malloc(k * sizeof(int));
    for(int c=0; c<k; c++) {
        centroid_count[c] = 0;
    }

    double variance_distance = 0.0;
    for(int i=0; i<N; i++) {
        int c = closest_centroid(data + i*D, centroids, k, D);
        centroid_count[c] += 1;
        variance_distance += distance(data + i*D, centroids + c*D, D);
    }
    double variance = variance_distance / (double)(N - k);
    if (variance == 0) {
        variance = nextafter(0, 1);
    }

    /* Calculate the log likelihood */
    double log_likelihood = 0.0;
    double t1, t2, t3, t4;
    double ccount;
    for(int c=0; c<k; c++) {
        ccount = (double) centroid_count[c];
        if (ccount == 0) {
            ccount = nextafter(0, 1);
        }
        t1 = ccount * log(ccount);
        t2 = ccount * log(N);
        t3 = (ccount * D) / 2.0 + log(2.0 * PI) * variance;
        t4 = (ccount - 1.0) / 2.0;
        log_likelihood += t1 - t2 - t3 - t4;
    }
    /* calculate the BIC with the number of free parameters = k * (D + 1) */
    double bic = log_likelihood - k * (D + 1) * 2.0 * log(N);

    free(centroid_count);
    return bic;
}


/*
 * Does max_iter iterations of minibatch on the given data.  The centroids
 * should already be initialized and each batch will consist of n_samples
 * samples from the data.
 */
void minibatch(double *data, double *centroids, int n_samples, int max_iter, int k, int N, int D)  {
    // assert(k < n_samples < N)
    // assert(data.shape == (N, D)
    // assert(centoids.shape == (k, D)

    _LOG("Initializing\n");
    char filename[128];
    int *sample_indexes = (int*) malloc(n_samples * sizeof(int));
    int *centroid_counts = (int*) malloc(k * sizeof(int));
    double *cluster_cache = (double*) malloc(n_samples * sizeof(double));
    double eta;
    int idx, cur_cluster;

    for (int i=0; i<k; i++) {
        centroid_counts[i] = 0;
    }

    _LOG("Starting minibatch\n");
    for(int iter=0; iter<max_iter; iter++) {
        _LOG("Iteration %d\n", iter);

        _LOG("\tGenerating samples\n");
        generate_random_indexes(N, n_samples, sample_indexes);

        _LOG("\tGenerating cache\n");
        for(int i=0; i<n_samples; i++) {
            idx = sample_indexes[i];
            cluster_cache[i] = closest_centroid(data + idx * D, centroids, k, D);
        }

        _LOG("\tUpdating centroids\n");
        for(int i=0; i<n_samples; i++) {
            idx = sample_indexes[i];
            cur_cluster = cluster_cache[i];
            centroid_counts[cur_cluster] += 1;
            gradient_step(data + idx * D, centroids + cur_cluster * D, centroid_counts[cur_cluster], D);
        }

#ifdef DEBUG_OUTPUT
        sprintf(filename, "data/centroids-%02d.dat", iter);
        save_double_matrix(centroids, filename, k, D);

        sprintf(filename, "data/samples-%02d.dat", iter);
        save_int_matrix(sample_indexes, filename, n_samples, 1);

        _LOG("\tBIC of current model: %f\n", bayesian_information_criterion(data, centroids, k, N, D));
#endif
    }

    
    _LOG("Cleaning up\n");
    free(centroid_counts);
    free(sample_indexes);
    free(cluster_cache);
}

/*
 * Will calculate a list of n unique integers in [0,N) and fill sample_indexes
 * with the result
 */
void generate_random_indexes(int N, int n, int *sample_indexes) {
    /* Parameters:
     *      N - size of array to pick samples from
     *      n - number of samples to pick
     *      sample_indexes - array of the sample indexes (len(sample_indexes) == n)
     *
     * TODO: generate the sample indexes with a LCG
     */
    
    for(int i=0; i<n; i++) {
        int index;
        for(int j=-1; j<i; j++) {
            if (j == -1 || sample_indexes[j] == index) {
                index = (int)(rand() / (double)RAND_MAX * N);
                j = 0;
            }
        }
        sample_indexes[i] = index;
    }
}

/*
 * Initialize centroids using the k-means++ algorithm over the given data.
 */
void kmeanspp(double *data, double *centroids, int k, int N, int D) {
    /* The first cluster is centered from a randomly chosen point in the data */
    int index = rand() / (double)RAND_MAX * N;
    for(int i=0; i<D; i++) {
        centroids[i] = data[index*D + i];
    }

    /*
     * Now we pick random data points to use for centroids using a weighted
     * probability propotional to the datapoints squared distance to the
     * closest centroid
     */
    double distance, total_distance2;
    double *distances = (double*) malloc(N * sizeof(double));
    for(int c=1; c<k; c++) {
        total_distance2 = 0.0;
        for(int i=0; i<N; i++) {
            distance = distance_to_closest_centroid(data + D*i, centroids, c, D);
            distances[i] = distance;
            total_distance2 += distance * distance;
        }
        
        int index;
        double d = rand() / (double)RAND_MAX * total_distance2;
        for(index = 0; index < N && d > 0; index++) {
            d -= distances[index];
        }
        index--;
            
        for(int i=0; i<D; i++) {
            centroids[c*D + i] = data[index*D + i];
        }
    }

    free(distances);
}


int main(void) {
    int N = 10000;
    int D = 2;
    int k = 256;
    int n_samples = 1000;
    int max_iter = 1000;

    printf("Allocating test data\n");
    double *data = (double*) malloc(N * D * sizeof(double));
    double *centroids = (double*) malloc(k * D * sizeof(double));

    printf("Creating synthetic data\n");
    srand((unsigned)time(NULL));
    gaussian_data(data, 500, N, D);
    kmeanspp(data, centroids, k, N, D);

#ifdef DEBUG_OUTPUT
    save_double_matrix(data, "data/cluster_data.dat", N, D);
#endif

    clock_t start_clock = clock();
    minibatch(data, centroids, n_samples, max_iter, k, N, D);
    clock_t end_clock = clock();
    printf("BIC of resulting model: %f\n", bayesian_information_criterion(data, centroids, k, N, D));
    printf("Time to run: %fs\n", (end_clock - start_clock) / (double)CLOCKS_PER_SEC);

    free(data);
    free(centroids);
    return 1;
}
