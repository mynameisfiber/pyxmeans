#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

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
        variance_distance += euclidian_distance(data + i*D, centroids + c*D, D);
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
 * Runs multiple minibatches (as given by n_runs) and returns the centroids
 * that have the best BIC
 */
void minibatch_multi(double *data, double *centroids, int n_samples, int max_iter, int n_runs, int n_jobs, int k, int N, int D) {
    double *all_centroids;
    double *all_bics;
    
    if (n_jobs > 1) {
        all_centroids = (double*) malloc(k * D * n_jobs * sizeof(double));
        all_bics = (double*) malloc(n_jobs * sizeof(double));
    } else {
        all_centroids = centroids;
        all_bics = (double*) malloc(sizeof(double));
    }

    #pragma omp parallel shared(all_centroids) num_threads(n_jobs)
    {
        int id = omp_get_thread_num();
        double lowest_bic, cur_bic;
        double *current_centroid = (double*) malloc(k * D * sizeof(double));

        #pragma omp for
        for(int i=0; i<n_runs; i++) {
            for(int j=0; i<k*D; i++) {
                current_centroid[j] = centroids[j];
            }

            minibatch(data, current_centroid, n_samples, max_iter, k, N, D);
            cur_bic = bayesian_information_criterion(data, current_centroid, k, N, D);

            if (cur_bic < lowest_bic) {
                lowest_bic = cur_bic;
                all_bics[id] = cur_bic;
                for(int j=0; j<k*D; j++) {
                    all_centroids[id * D * k + j] = current_centroid[j];
                }
            }
        }
        free(current_centroid);
        _LOG("Thread %d is done\n", id);
    }

    if (n_jobs > 1) {
        double min_bic;
        int min_bic_index;
        _LOG("Finding min BIC\n");
        for(int i=0; i<n_jobs; i++) {
            _LOG("BIC[%d] = %f\n", i, all_bics[i]);
            if (i == 0 || all_bics[i] < min_bic) {
                min_bic = all_bics[i];
                min_bic_index = i;
            }
        }
        _LOG("Min BIC = %f\n", min_bic);

        for(int i=0; i<k*D; i++) {
            centroids[i] = all_centroids[min_bic_index*k*D + i];
        }
    }

    free(all_centroids);
    free(all_bics);
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
    int *sample_indicies = (int*) malloc(n_samples * sizeof(int));
    int *centroid_counts = (int*) malloc(k * sizeof(int));
    int *cluster_cache = (int*) malloc(n_samples * sizeof(int));

    for (int i=0; i<k; i++) {
        centroid_counts[i] = 0;
    }

    _LOG("Starting minibatch\n");
    for(int iter=0; iter<max_iter; iter++) {
        _LOG("Iteration %d\n", iter);

        _LOG("\tGenerating samples\n");
        generate_random_indicies(N, n_samples, sample_indicies);

        minibatch_iteration(data, centroids, sample_indicies, centroid_counts, cluster_cache, n_samples, k, N, D);

#ifdef DEBUG_OUTPUT
        char filename[128];
        sprintf(filename, "data/centroids-%02d.dat", iter);
        save_double_matrix(centroids, filename, k, D);

        sprintf(filename, "data/samples-%02d.dat", iter);
        save_int_matrix(sample_indicies, filename, n_samples, 1);

        _LOG("\tBIC of current model: %f\n", bayesian_information_criterion(data, centroids, k, N, D));
#endif
    }

    
    _LOG("Cleaning up\n");
    free(centroid_counts);
    free(sample_indicies);
    free(cluster_cache);
}

/*
 * Does a single iteration of minibatch on the given data.
 * Parameters:
 *      data: the data to cluster centroids: location of the centroids
 *      sample_indicies: list of indexes into data that should be used for the
 *          clustering 
 *      centroid_counts: a count of the number of datapoints found
 *          in each centroid 
 *      cluster_cache: a cache of which cluster a sample belongs to.
 */
void minibatch_iteration(double *data, double *centroids, int *sample_indicies, int *centroid_counts, int *cluster_cache, int n_samples, int k, int N, int D)  {
    // assert(k < n_samples < N)
    // assert(data.shape == (N, D)
    // assert(centoids.shape == (k, D)
    // assert(sample_indicies.shape == (n_samples,)
    // assert(centroid_counts.shape == (k, )
    // assert(cluster_cache.shape == (n_samples, )

    int idx, cur_cluster;

    _LOG("\tGenerating cache\n");
    for(int i=0; i<n_samples; i++) {
        idx = sample_indicies[i];
        cluster_cache[i] = closest_centroid(data + idx * D, centroids, k, D);
    }

    _LOG("\tUpdating centroids\n");
    for(int i=0; i<n_samples; i++) {
        idx = sample_indicies[i];
        cur_cluster = cluster_cache[i];
        centroid_counts[cur_cluster] += 1;
        gradient_step(data + idx * D, centroids + cur_cluster * D, centroid_counts[cur_cluster], D);
    }
}

/*
 * Will calculate a list of n unique integers in [0,N) and fill sample_indicies
 * with the result
 */
void generate_random_indicies(int N, int n, int *sample_indicies) {
    /* Parameters:
     *      N - size of array to pick samples from
     *      n - number of samples to pick
     *      sample_indicies - array of the sample indicies (len(sample_indicies) == n)
     *
     * TODO: generate the sample indicies with a LCG
     */
    
    for(int i=0; i<n; i++) {
        int index;
        for(int j=-1; j<i; j++) {
            if (j == -1 || sample_indicies[j] == index) {
                index = (int)(rand() / (double)RAND_MAX * N);
                j = 0;
            }
        }
        sample_indicies[i] = index;
    }
}

/*
 * Initialize centroids using the k-means++ algorithm over the given data.
 */
void kmeanspp(double *data, double *centroids, int k, int N, int D) {
    /* The first cluster is centered from a randomly chosen point in the data */
    int index = (int) (rand() / (double)RAND_MAX * N);
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
    int n_samples = k*3;
    int max_iter = 1000;

    printf("Allocating test data\n");
    double *data = (double*) malloc(N * D * sizeof(double));
    double *centroids = (double*) malloc(k * D * sizeof(double));

    printf("Creating synthetic data\n");
    srand((unsigned)time(NULL));
    gaussian_data(data, 20, N, D);
    kmeanspp(data, centroids, k, N, D);

#ifdef DEBUG_OUTPUT
    save_double_matrix(data, "data/cluster_data.dat", N, D);
#endif

    clock_t start_clock = clock();
    /*minibatch(data, centroids, n_samples, max_iter, k, N, D);*/
    minibatch_multi(data, centroids, n_samples, max_iter, 10, 4, k, N, D);
    clock_t end_clock = clock();
    printf("BIC of resulting model: %f\n", bayesian_information_criterion(data, centroids, k, N, D));
    printf("Time to run: %fs\n", (end_clock - start_clock) / (double)CLOCKS_PER_SEC);

    free(data);
    free(centroids);
    return 1;
}
