#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "minibatch.h"
#include "generate_data.h"
#include "distance.h"

#define EARLY_TERM_WINDOW (10)

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
 * Calculate the variance of the model given the current centroids
 */
double model_variance(double *data, double *centroids, int k, int N, int D) {
    double variance_distance = 0.0;
    for(int i=0; i<N; i++) {
        int c = closest_centroid(data + i*D, centroids, k, D);
        variance_distance += distance_metric(data + i*D, centroids + c*D, D);
    }
    double variance = variance_distance / (double)(N - k);
    if (variance == 0) {
        variance = nextafter(0, 1);
    }
    return variance;
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
        variance_distance += distance_metric(data + i*D, centroids + c*D, D);
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
 * Runs multiple kmeanspp (as given by n_runs) and returns the centroids
 * that have the best variance
 */
void kmeanspp_multi(double *data, double *centroids, int n_samples, int n_runs, int n_jobs, int k, int N, int D) {
    double *all_centroids;
    double *all_variances = (double*) malloc(n_jobs * sizeof(double));
    
    if (n_jobs > 1) {
        all_centroids = (double*) malloc(k * D * n_jobs * sizeof(double));
    } else {
        all_centroids = centroids;
    }

    #pragma omp parallel shared(all_centroids, all_variances, data, distance_metric) num_threads(n_jobs)
    {
        int id = omp_get_thread_num();
        double minimum_variance, cur_variance;
        double *current_centroid = (double*) malloc(k * D * sizeof(double));
        int local_iter = 0;

        #pragma omp for
        for(int i=0; i<n_runs; i++) {
            kmeanspp(data, current_centroid, n_samples, k, N, D);
            cur_variance = model_variance(data, current_centroid, k, N, D);

            if (local_iter == 0 || cur_variance < minimum_variance) {
                minimum_variance = cur_variance;
                all_variances[id] = cur_variance;
                for(int j=0; j<k*D; j++) {
                    all_centroids[id * D * k + j] = current_centroid[j];
                }
            }
            local_iter++;
        }
        free(current_centroid);
        _LOG("Thread %d is done\n", id);
    }

    if (n_jobs > 1) {
        double min_variance;
        int min_variance_index;
        _LOG("Finding min variance\n");
        for(int i=0; i<n_jobs; i++) {
            _LOG("variance[%d] = %e\n", i, all_variances[i]);
            if (i == 0 || all_variances[i] < min_variance) {
                min_variance = all_variances[i];
                min_variance_index = i;
            }
        }
        _LOG("Min variance = %f\n", min_variance);

        for(int i=0; i<k*D; i++) {
            centroids[i] = all_centroids[min_variance_index*k*D + i];
        }
    }

    free(all_centroids);
    free(all_variances);
}

/*
 * Runs multiple minibatches (as given by n_runs) and returns the centroids
 * that have the best variance
 */
void minibatch_multi(double *data, double *centroids, int n_samples, int max_iter, int n_runs, int n_jobs, double bic_ratio_termination, double reassignment_ratio, int k, int N, int D) {
    double *all_centroids;
    double *all_variances = (double*) malloc(n_jobs * sizeof(double));
    
    if (n_jobs > 1) {
        all_centroids = (double*) malloc(k * D * n_jobs * sizeof(double));
    } else {
        all_centroids = centroids;
    }

    #pragma omp parallel shared(all_centroids, all_variances, data, distance_metric) num_threads(n_jobs)
    {
        int id = omp_get_thread_num();
        double minimum_variance, cur_variance;
        double *current_centroid = (double*) malloc(k * D * sizeof(double));
        int local_iter = 0;

        #pragma omp for
        for(int i=0; i<n_runs; i++) {
            for(int j=0; j<k*D; j++) {
                current_centroid[j] = centroids[j];
            }

            minibatch(data, current_centroid, n_samples, max_iter, bic_ratio_termination, reassignment_ratio, k, N, D);
            cur_variance = model_variance(data, current_centroid, k, N, D);

            if (local_iter == 0 || cur_variance < minimum_variance) {
                minimum_variance = cur_variance;
                all_variances[id] = cur_variance;
                for(int j=0; j<k*D; j++) {
                    all_centroids[id * D * k + j] = current_centroid[j];
                }
            }
            local_iter++;
        }
        free(current_centroid);
        _LOG("Thread %d is done\n", id);
    }

    if (n_jobs > 1) {
        double min_variance;
        int min_variance_index;
        _LOG("Finding min variance\n");
        for(int i=0; i<n_jobs; i++) {
            _LOG("variance[%d] = %f\n", i, all_variances[i]);
            if (i == 0 || all_variances[i] < min_variance) {
                min_variance = all_variances[i];
                min_variance_index = i;
            }
        }
        _LOG("Min variance = %f\n", min_variance);

        for(int i=0; i<k*D; i++) {
            centroids[i] = all_centroids[min_variance_index*k*D + i];
        }
    }

    free(all_centroids);
    free(all_variances);
}


/*
 * Does max_iter iterations of minibatch on the given data.  The centroids
 * should already be initialized and each batch will consist of n_samples
 * samples from the data.
 */
void minibatch(double *data, double *centroids, int n_samples, int max_iter, double bic_ratio_termination, double reassignment_ratio, int k, int N, int D)  {
    // assert(k < n_samples < N)
    // assert(data.shape == (N, D)
    // assert(centoids.shape == (k, D)

    _LOG("Initializing\n");
    int *sample_indicies = (int*) malloc(n_samples * sizeof(int));
    int *centroid_counts = (int*) malloc(k * sizeof(int));
    int *cluster_cache = (int*) malloc(n_samples * sizeof(int));

    int *last_centroid_counts = (int*) malloc(k * sizeof(int));
    int *reassign_centroid_indicies = (int*) malloc(k * sizeof(int));
    int count_diff = 0, reassign_num = 0, max_count_diff = 0;

    double current_bic, bic_sum = 0.0;
    double *historical_bic;
    int historical_bic_idx = 0;
    if (bic_ratio_termination > 0.0) {
        historical_bic = (double*) malloc(EARLY_TERM_WINDOW  * sizeof(double));
    }

    for (int i=0; i<k; i++) {
        centroid_counts[i] = 0;
        last_centroid_counts[i] = 0;
    }

    _LOG("Starting minibatch\n");
    for(int iter=0; iter<max_iter; iter++) {
        _LOG("Iteration %d\n", iter);

        _LOG("\tGenerating samples\n");
        generate_random_indicies(N, n_samples, sample_indicies);

        minibatch_iteration(data, centroids, sample_indicies, centroid_counts, cluster_cache, n_samples, k, N, D);

        reassign_num = 0;
        max_count_diff = 0;
        for(int i=0; i<k; i++) {
            count_diff = centroid_counts[i] - last_centroid_counts[i];
            if (count_diff == 0) {
                reassign_centroid_indicies[reassign_num] = i;
                reassign_num += 1;
            }
            if (count_diff > max_count_diff) {
                max_count_diff = count_diff;
            }
        }
        for(int i=0; i<k; i++) {
            count_diff = centroid_counts[i] - last_centroid_counts[i];
            if (count_diff > 0 && count_diff < max_count_diff * reassignment_ratio) {
                reassign_centroid_indicies[reassign_num] = i;
                reassign_num += 1;
            }
            last_centroid_counts[i] = centroid_counts[i];
        }
        if (reassign_num > 0) {
            _LOG("Reassigning %d centroids\n", reassign_num);
            reassign_centroids(data, centroids, reassign_centroid_indicies, n_samples, reassign_num, k, N, D);
        }

        if (bic_ratio_termination > 0.0) {
            _LOG("\tChecking for early termination condition\n");
            current_bic = bayesian_information_criterion(data, centroids, k, N, D);
            if (iter > EARLY_TERM_WINDOW) {
                _LOG("Current bic ratio: %f\n", fabs(1.0 - current_bic * EARLY_TERM_WINDOW / bic_sum));
                if (fabs(1.0 - current_bic * EARLY_TERM_WINDOW / bic_sum) < bic_ratio_termination) {
                    _LOG("Finishing early at iteration %d. ratio = %f, threshold = %f\n", 
                            iter, 
                            fabs(1.0 - current_bic * EARLY_TERM_WINDOW / bic_sum), 
                            bic_ratio_termination
                    );
                    break;
                }
            }

            bic_sum += current_bic;
            bic_sum -= historical_bic[historical_bic_idx];
            historical_bic[historical_bic_idx] = current_bic;
            historical_bic_idx = (historical_bic_idx + 1) % EARLY_TERM_WINDOW;
        }
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
    free(last_centroid_counts);

    if (bic_ratio_termination > 0.0) {
        free(historical_bic);
    }
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

void reassign_centroids(double *data, double *centroids, int *reassign_clusters, int n_samples, int K, int k, int N, int D) {
    unsigned int seed = (int) clock() * (omp_get_thread_num() + 1);
    srand(seed);

    double distance, total_distance2;
    double *distances2 = (double*) malloc(n_samples * sizeof(double));
    int *sample_indicies = (int*) malloc(n_samples * sizeof(int));

    generate_random_indicies(N, n_samples, sample_indicies);
    for(int i=0; i<n_samples; i++) {
        int idx = sample_indicies[i];
        distance = distance_to_closest_centroid(data + D*idx, centroids, k, D);
        distances2[i] = distance * distance;
        total_distance2 += distances2[i];
    }

    for(int c=0; c<K; c++) {
        total_distance2 = 0.0;
        
        int index;
        double d = (rand() / ((double)RAND_MAX+1)) * total_distance2;
        for(index = 0; index < n_samples && d >= 0; index++) {
            d -= distances2[index];
        }
        if (index) index--;
            
        int data_index = sample_indicies[index];
        int centroid_idx = reassign_clusters[c];
        for(int i=0; i<D; i++) {
            centroids[centroid_idx*D + i] = data[data_index*D + i];
        }

        total_distance2 -= distances2[index];
        distances2[index] = 0;
    }

    free(distances2);
    free(sample_indicies);
}

/*
 * Initialize centroids using the k-means++ algorithm over the given data.
 */
void kmeanspp(double *data, double *centroids, int n_samples, int k, int N, int D) {
    /* The first cluster is centered from a randomly chosen point in the data */
    unsigned int seed = (int) clock() * (omp_get_thread_num() + 1);
    srand(seed);

    int index = (int) (rand() / ((double)RAND_MAX+1) * N);
    for(int i=0; i<D; i++) {
        centroids[i] = data[index*D + i];
    }
    _LOG("Fitted clusters: 1 / %d\n", k);

    /*
     * Now we pick random data points to use for centroids using a weighted
     * probability propotional to the datapoints squared distance to the
     * closest centroid
     */
    double distance, total_distance2;
    double *distances2 = (double*) malloc(n_samples * sizeof(double));
    int *sample_indicies = (int*) malloc(n_samples * sizeof(int));
    for(int c=1; c<k; c++) {
        total_distance2 = 0.0;

        generate_random_indicies(N, n_samples, sample_indicies);
        for(int i=0; i<n_samples; i++) {
            int idx = sample_indicies[i];
            distance = distance_to_closest_centroid(data + D*idx, centroids, c, D);
            distances2[i] = distance * distance;
            total_distance2 += distances2[i];
        }
        
        int index;
        double d = (rand() / ((double)RAND_MAX+1)) * total_distance2;
        for(index = 0; index < N && d >= 0; index++) {
            d -= distances2[index];
        }
        if(index) index--;
            
        int data_index = sample_indicies[index];
        for(int i=0; i<D; i++) {
            centroids[c*D + i] = data[data_index*D + i];
        }
        _LOG("Fitted clusters: %d / %d\n", c, k);
    }

    free(distances2);
    free(sample_indicies);
}


int main(void) {
    int N = 1000;
    int D = 2;
    int k = 256;
    int n_samples = k*5;
    int max_iter = 1000;

    printf("Allocating test data\n");
    double *data = (double*) malloc(N * D * sizeof(double));
    double *centroids = (double*) malloc(k * D * sizeof(double));

    printf("Creating synthetic data\n");
    gaussian_data(data, 20, N, D);
    kmeanspp(data, centroids, n_samples, k, N, D);

#ifdef DEBUG_OUTPUT
    save_double_matrix(data, "data/cluster_data.dat", N, D);
#endif

    clock_t start_clock = clock();
    minibatch(data, centroids, n_samples, max_iter, 0.001, 0.1, k, N, D);
    /*minibatch_multi(data, centroids, n_samples, max_iter, 10, 4, -1.0, k, N, D);*/
    clock_t end_clock = clock();
    printf("BIC of resulting model: %f\n", bayesian_information_criterion(data, centroids, k, N, D));
    printf("Time to run: %fs\n", (end_clock - start_clock) / (double)CLOCKS_PER_SEC);

    free(data);
    free(centroids);
    return 1;
}
