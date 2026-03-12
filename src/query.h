#pragma once
#include <iostream>
#include <vector>
#include <utility>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <armadillo>
#include <unordered_map>
#include <omp.h>
#include <sys/time.h>
#include <cstring>

#include "dist_calculation.h"
#include "utils.h"

struct Compare {
    bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first > b.first;
    }
};

struct AnnQueryParams {
    float ** dataset;
    int ** queryknn_results;
    long int dataset_size;
    int data_dimensionality;
    int query_size;
    int k_size;
    float ** querypoints;
    IndexMap * indexes;
    float * centroids_list;
    int subspace_num;
    int subspace_dimensionality;
    int kmeans_num_centroid;
    int kmeans_dim;
    int collision_num;
    int candidate_num;
    int number_of_threads;
    float ** rotated_querypoints;
    long int ** gt;
};

void ann_query(AnnQueryParams &params, long int &query_time);

void dynamic_activate(IndexMap &indexes, std::vector<std::pair<int, int>> &retrieved_cell, std::vector<float> &first_half_dists, std::vector<int> &first_half_idx, std::vector<float> &second_half_dists, std::vector<int> &second_half_idx, int collision_num, int kmeans_num_centroid, int subspace_idx);

void scalable_dynamic_activate(IndexMap &indexes, std::vector<std::pair<int, int>> &retrieved_cell, std::vector<float> &first_half_dists, std::vector<int> &first_half_idx, std::vector<float> &second_half_dists, std::vector<int> &second_half_idx, int collision_num, int kmeans_num_centroid, int subspace_idx);