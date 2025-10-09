#pragma once
#include <iostream>
#include <armadillo>
#include <getopt.h>

using namespace std;

void load_data(float ** &dataset, char * dataset_path, long int dataset_size, int data_dimensionality);
void load_query(float ** &querypoints, char * query_path, int query_size, int data_dimensionality);
void load_groundtruth(long int ** &gt, char * groundtruth_path, int query_size, int k_size);

void save_query(float ** &querypoints, char * query_path, int query_size, int query_dimensionality);

void transfer_data(arma::mat &dataset, vector<arma::mat> &data_list, long int dataset_size, int subspace_num, int subspace_dimensionality);

int eigval_subspace_select(vector<double>& eigval_product_subspace, vector<int> eigval_number_subspace, int subspace_num, int subspace_dimensionality);