#pragma once
#include <iostream>
#include <vector>
#include <armadillo>
#include "utils.h"

void load_indexes(char * index_path, IndexMap &indexes, float * centroids_list, int * assignments_list, long int dataset_size, int kmeans_dim, int subspace_num, int kmeans_num_centroid);
void gen_indexes(std::vector<arma::mat> & data_list, IndexMap &indexes, long int dataset_size, float * centroids_list, int * assignments_list, int kmeans_dim, int subspace_num, int kmeans_num_centroid, int kmeans_num_iters, long int &index_time);
void save_indexes(char * index_path, float * centroids_list, int * assignments_list, long int dataset_size, int kmeans_dim, int subspace_num, int kmeans_num_centroid);