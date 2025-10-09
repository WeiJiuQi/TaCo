#include "dist_calculation.h"
#include "utils.h"
#include "index.h"
#include "query.h"
#include "preprocess.h"
#include "evaluate.h"
#include <mlpack/methods/pca/pca.hpp>

using namespace std;

void INThandler(int sig)
{
    char  c;
    signal(sig, SIG_IGN);
    fprintf(stderr, "Do you really want to quit? [y/n] ");
    c = getchar();
    if (c == 'y' || c == 'Y') {
    	exit(0);
    } else {
        signal(SIGINT, INThandler);
        getchar(); // Get new line character
    }  
}

int main (int argc, char **argv)
{
	signal(SIGINT, INThandler);

    static char * dataset_path;
    static char * query_path;
    static char * groundtruth_path;
    static char * index_path;
    static char * projected_query_path;
    
    static long int dataset_size = 1000000;
    static int query_size = 100;
    static int k_size = 50;

    static int data_dimensionality = 128;
    static int subspace_dimensionality = 16;
    static int subspace_num = 8;

    float candidate_ratio = 0.05;
    float collision_ratio = 0.1;
    
    static int kmeans_num_centroid = 50;
    static int kmeans_num_iters = 2;

    static int load_index=0;

    // Parse input
    while (1)
    {
        static struct option long_options[] =  {
            {"dataset-path", required_argument, 0, 'a'},
            {"query-path", required_argument, 0, 'b'},
            {"groundtruth-path", required_argument, 0, 'c'},
            {"index-path", required_argument, 0, 'd'},
            {"dataset-size", required_argument, 0, 'e'},
            {"query-size", required_argument, 0, 'f'},
            {"k-size", required_argument, 0, 'g'},
            {"data-dimensionality", required_argument, 0, 'h'},
            {"subspace-dimensionality", required_argument, 0, 'i'},
            {"subspace-num", required_argument, 0, 'j'},
            {"candidate-ratio", required_argument, 0, 'k'},
            {"collision-ratio", required_argument, 0, 'l'},
            {"kmeans-num-centroid", required_argument, 0, 'm'},
            {"kmeans-num-iters", required_argument, 0, 'n'},
            {"load-index", no_argument, 0, 'o'},
            {"projected-query-path", required_argument, 0, 'p'},
            {NULL, 0, NULL, 0}
        };

        /* getopt_long stores the option index here. */
        int option_index = 0;
        int c = getopt_long (argc, argv, "",
                             long_options, &option_index);
        if (c == -1)
            break;
        switch (c)
        {
            case 'a':
                dataset_path = optarg;
                break;

            case 'b':
                query_path = optarg;
                break;
            
            case 'c':
                groundtruth_path = optarg;
                break;
            
            case 'd':
                index_path = optarg;
                break;
            
            case 'e':
                dataset_size = atoi(optarg);
                break;

            case 'f':
                query_size = atoi(optarg);
                break;

            case 'g':
                k_size = atoi(optarg);
                break; 

            case 'h':
                data_dimensionality = atoi(optarg);
                break;

            case 'i':
            	subspace_dimensionality = atoi(optarg);

            case 'j':
            	subspace_num = atoi(optarg);
            
            case 'k':
                candidate_ratio = atof(optarg);
                break;

            case 'l':
                collision_ratio = atof(optarg);
                break;

            case 'm':
                kmeans_num_centroid = atoi(optarg);
                break;

            case 'n':
                kmeans_num_iters = atoi(optarg);
                break;   
            
            case 'o':
                load_index = 1;
                break;

            case 'p':
                projected_query_path = optarg;
                break;

            default:
                exit(-1);
                break;
        }
    }

    // Load data
    dataset_size = dataset_size - 100;
    float ** dataset;
    load_data(dataset, dataset_path, dataset_size, data_dimensionality);

    // Load query
    float ** querypoints;
    load_query(querypoints, query_path, query_size, data_dimensionality);

    // Load groundtruth
    long int ** gt;
    load_groundtruth(gt, groundtruth_path, query_size, k_size);

    arma::mat projectedData(subspace_dimensionality * subspace_num, dataset_size + query_size, arma::fill::zeros);
    vector<arma::mat> data_list;

    if (load_index == 0) {
        // Prepare data for PCA
        arma::mat dataset_armamat(data_dimensionality, dataset_size + query_size, arma::fill::zeros);
        for (int i = 0; i < dataset_size; i++) {
            for (int j = 0; j < data_dimensionality; j++) {
                dataset_armamat(j, i) = dataset[i][j];
            }
        }
        for (int i = dataset_size; i < dataset_size + query_size; i++) {
            for (int j = 0; j < data_dimensionality; j++) {
                dataset_armamat(j, i) = querypoints[i - dataset_size][j];
            }
        }

        // Perform PCA to project the data
        arma::mat transformedData(data_dimensionality, dataset_size + query_size, arma::fill::zeros);
        arma::vec eigVal;
        arma::mat eigvec(data_dimensionality, data_dimensionality, arma::fill::zeros);
        mlpack::pca::PCA pca;
        pca.Apply(dataset_armamat, transformedData, eigVal, eigvec);

        dataset_armamat.clear();

        // Scale the eigenvalues to avoid numerical issues
        int power = 0;
        while (eigVal[data_dimensionality - 1] < 1) {
            eigVal[data_dimensionality - 1] *= 10;
            power++;
        }
        for (int i = 0; i < data_dimensionality - 1; i++) {
            eigVal[i] *= pow(10, power);
        }
        
        // Eigensystem allocation
        vector<double> eigval_product_subspace(subspace_num, 1.0);
        vector<int> eigval_number_subspace(subspace_num, 0);

        for (int i = 0; i < subspace_num; i++) {
            projectedData.row(i * subspace_dimensionality) = transformedData.row(i);
            eigval_product_subspace[i] *= eigVal[i];
            eigval_number_subspace[i]++;
        }
        
        for (int i = subspace_num; i < subspace_dimensionality * subspace_num; i++) {
            int selected_subspace = eigval_subspace_select(eigval_product_subspace, eigval_number_subspace, subspace_num, subspace_dimensionality);
            projectedData.row(selected_subspace * subspace_dimensionality + eigval_number_subspace[selected_subspace]) = transformedData.row(i);
            eigval_product_subspace[selected_subspace] *= eigVal[i];
            eigval_number_subspace[selected_subspace]++;
        }

        // Preprocess dataset to fit the data format required by mlpack
        transfer_data(projectedData, data_list, dataset_size, subspace_num, subspace_dimensionality);

        transformedData.clear();
    }

    // Generate PCA query points
    float ** projected_querypoints = new float * [query_size];
    if (load_index == 1) {
        load_query(projected_querypoints, projected_query_path, query_size, subspace_dimensionality * subspace_num);
    } else {
        for (long i = 0; i < query_size; i++){
            projected_querypoints[i] = new float[subspace_dimensionality * subspace_num];
            for (int j = 0; j < subspace_dimensionality * subspace_num; j++) {
                projected_querypoints[i][j] = projectedData(j, i + dataset_size);
            }
        }
        save_query(projected_querypoints, projected_query_path, query_size, subspace_dimensionality * subspace_num);
    }

    projectedData.clear();
    
    // Indexing phase
    size_t RSS_before_indexing = getCurrentRSS() / 1000000; 

    long int index_time = 0;
    int kmeans_dim = subspace_dimensionality / 2;
    
    int * assignments_list = new int[dataset_size * subspace_num * 2];
    float * centroids_list = new float [kmeans_num_centroid * kmeans_dim * subspace_num * 2];
    vector<unordered_map<pair<int, int>, vector<int>, hash_pair>> indexes;

    if (load_index == 1) { // load index from index_path
        load_indexes(index_path, indexes, centroids_list, assignments_list, dataset_size, kmeans_dim, subspace_num, kmeans_num_centroid);
    } else { // need to generate index and save it to index_path
        // generate index
        gen_indexes(data_list, indexes, dataset_size, centroids_list, assignments_list, kmeans_dim, subspace_num, kmeans_num_centroid, kmeans_num_iters, index_time);
        // save index
        save_indexes(index_path, centroids_list, assignments_list, dataset_size, kmeans_dim, subspace_num, kmeans_num_centroid);
    }

    delete []assignments_list;
    size_t RSS_after_indexing = getCurrentRSS() / 1000000; 


    // Query phase
    long int query_time = 0;
    
    int collision_num = (int) (collision_ratio * dataset_size);
    int candidate_num = (int) (candidate_ratio * dataset_size);

    int ** queryknn_results = new int*[query_size];
    for (int i = 0; i < query_size; i++) {
        queryknn_results[i] = new int[k_size];
    }
    

    int number_of_threads = get_nprocs_conf() / 2;

    ann_query(dataset, queryknn_results, dataset_size, data_dimensionality, query_size, k_size, querypoints, indexes, centroids_list, subspace_num, subspace_dimensionality, kmeans_num_centroid, kmeans_dim, collision_num, candidate_num, number_of_threads, query_time, projected_querypoints, gt);
    
    if (load_index == 0) {
        cout << "The indexing time is: " << index_time / 1000.0 << "ms." << endl;
    }
    cout << "The indexing footprint is: " << RSS_after_indexing - RSS_before_indexing << "MB" << endl;
    cout << "The average query time is " << query_time / query_size / 1000.0 << "ms." << endl;
    
    // Evaluate the query accuracy (recall and ratio)
    recall_and_ratio(dataset, querypoints, data_dimensionality, queryknn_results, gt, query_size);

}