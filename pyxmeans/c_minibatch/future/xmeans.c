#include <stdlib.h>
#include <stdbool.h>

#include "xmeans.h"
#include "minibatch.h"
#include "distance.h"

int xmeans(double *data, double *centroids, int n_samples, int max_iter, int k_min, int k_max, int N, int D)  {
    // assert(k < n_samples < N)
    // assert(data.shape == (N, D)
    // assert(centoids.shape == (k, D)

    _LOG("Initializing\n");
    int *sample_indicies = (int*) malloc(n_samples * sizeof(int));
    int *centroid_counts = (int*) malloc(k_max * sizeof(int));
    int *cluster_cache = (int*) malloc(n_samples * sizeof(int));

    int *test_sample_indicies = (int*) malloc(n_samples * sizeof(int));
    double *test_vector = (double*) malloc(D * sizeof(double));
    double *test_centroids = (double*) malloc(2 * D * sizeof(double));
    double *centroid_distances = (double*) malloc(k_max * sizeof(double));

    double distance;
    int new_k = -1;
    int k = k_min;

    for (int i=0; i<2*D; i++) {
        test_centroids[i] = 0.0;
    }

    _LOG("Starting xmeans\n");
    while (k < k_max && new_k != 0) {
        _LOG("Iteration k=%d\n", k);

        _LOG("\tRunning MiniBatch over full set\n");
        minibatch(data, centroids, n_samples, max_iter, k, N, D);

        _LOG("\tGetting centroid distances\n");
        // TODO: optimize this distance calculation
        for(int c1=0; c1<k; c1++) {
            centroid_distances[c1] = -1;
            for(int c2=0; c2<k; c2++) {
                if (c1 != c2) {
                    distance = euclidian_distance(centroids + c1*D, centroids + c2*D, D);
                    if (centroid_distances[c1] == -1 || distance < centroid_distances[c1]) {
                        centroid_distances[c1] = distance;
                    }
                }
            }
        }

        new_k = 0;
        for(int c=0; c<k; c++) {
            _LOG("\tRunning 2means on cluster c=%d\n", c);
            if (k + new_k >= k_max) {
                _LOG("\tNot continuing with splitting clusters\n");
                break;
            }
            for(int j=0; j<D; j++) {
                test_vector[j] = rand() / (double)RAND_MAX;
            }
            int dist = centroid_distances[c] / 4.0;
            for(int j=0; j<D; j++) {
                test_centroids[    j] =      dist * test_vector[j];
                test_centroids[D + j] = -1 * dist * test_vector[j];
            }

            int n = generate_random_indicies_in_cluster(data, centroids, test_sample_indicies, c, n_samples, 1000, k, N, D);
            for (int i=0; i<n; i++) {
                centroid_counts[i] = 0;
            }
            minibatch_iteration(data, test_centroids, test_sample_indicies, centroid_counts, cluster_cache, n, 2, N, D);

            double parent_bic = bayesian_information_criterion(data, centroids, k, N, D);
            double children_bic = bayesian_information_criterion(data, test_centroids, 2, N, D);
            _LOG("\t\tParent BIC: %f, Child BIC: %f\n", parent_bic, children_bic);
            if (children_bic > parent_bic) {
                _LOG("\t\tUsing children\n");
                int empty_k = k+new_k;
                for(int i=0; i<D; i++) {
                    centroids[c*D + i] = test_centroids[i];
                    centroids[empty_k*D + i] = test_centroids[D + i];
                }
                new_k += 1;
            } else {
                _LOG("\t\tUsing parents\n");
            }
            
        }
        k += new_k;
    }

    
    _LOG("Cleaning up\n");
    free(sample_indicies);
    free(centroid_counts);
    free(cluster_cache);

    free(test_sample_indicies);
    free(test_vector);
    free(test_centroids);
    free(centroid_distances);
    return k;
}

/*
 * Will calculate a list of n unique integers in [0,N) where each integer is an
 * index to a vector in data inside of cluster c and fill sample_indicies with
 * the result
 */
int generate_random_indicies_in_cluster(double *data, double *centroids, int *sample_indicies, int cluster, int n, int max_iter, int k, int N, int D) {
    int index, cur_c;
    int iter=0, num_found=0;
    bool not_unique;
    while (num_found < n) {
        iter = 0;
        not_unique = true;
        cur_c = -1;
        while (iter < max_iter && not_unique && cur_c != cluster) {
            index = (int)(rand() / (double)RAND_MAX * N);
            not_unique = false;
            for(int i=0; i<num_found; i++) {
                if (index == sample_indicies[i]) {
                    not_unique = true;
                    break;
                }
            }
            if (!not_unique) {
                cur_c = closest_centroid(data + index * D, centroids, k, D);
            }
            iter++;
        }
        if (!not_unique) {
            sample_indicies[num_found] = index;
            num_found++;
        }
        if (iter >= max_iter) {
            break;
        }
    }
    return num_found;
}
