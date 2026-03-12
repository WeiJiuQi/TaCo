#include "dist_calculation.h"
#include "utils.h"
#include "index.h"
#include "query.h"
#include "preprocess.h"
#include "evaluate.h"
#include <mlpack/methods/pca/pca.hpp>
#include <string>

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
                break;

            case 'j':
            	subspace_num = atoi(optarg);
                break;
            
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

    vector<arma::mat> data_list;

    // PCA model for query-time projection (filled when load_index==0, or loaded from file when load_index==1)
    arma::vec dataset_mean;
    arma::mat eigvec;
    arma::vec eigVal;
    vector<int> row_to_dim;
    bool pca_model_ready = false;

    if (load_index == 0) {
        // 1) Prepare data for PCA
        arma::mat dataset_armamat(data_dimensionality, dataset_size, arma::fill::zeros);
        for (long int i = 0; i < dataset_size; i++) {
            for (int j = 0; j < data_dimensionality; j++) {
                dataset_armamat(j, i) = dataset[i][j];
            }
        }
        dataset_mean = arma::mean(dataset_armamat, 1);

        // 2) Perform PCA
        arma::mat transformedData(data_dimensionality, dataset_size, arma::fill::zeros);
        eigVal.set_size(data_dimensionality);
        eigvec.set_size(data_dimensionality, data_dimensionality);
        mlpack::pca::PCA pca;
        pca.Apply(dataset_armamat, transformedData, eigVal, eigvec);
        dataset_armamat.clear();

        // 3) Scale the eigenvalues to avoid numerical issues
        int power = 0;
        while (eigVal[data_dimensionality - 1] < 1) {
            eigVal[data_dimensionality - 1] *= 10;
            power++;
        }
        for (int i = 0; i < data_dimensionality - 1; i++) {
            eigVal[i] *= pow(10, power);
        }

        // 4) Project dataset to subspaces with eigensystem allocation
        arma::mat projectedData(subspace_dimensionality * subspace_num, dataset_size, arma::fill::zeros);
        row_to_dim.resize(subspace_dimensionality * subspace_num);
        vector<double> eigval_product_subspace(subspace_num, 1.0);
        vector<int> eigval_number_subspace(subspace_num, 0);

        for (int i = 0; i < subspace_num; i++) {
            int row = i * subspace_dimensionality;
            projectedData.row(row) = transformedData.row(i);
            row_to_dim[row] = i;
            eigval_product_subspace[i] *= eigVal[i];
            eigval_number_subspace[i]++;
        }
        for (int i = subspace_num; i < subspace_dimensionality * subspace_num; i++) {
            int selected_subspace = eigval_subspace_select(eigval_product_subspace, eigval_number_subspace, subspace_num, subspace_dimensionality);
            int row = selected_subspace * subspace_dimensionality + eigval_number_subspace[selected_subspace];
            projectedData.row(row) = transformedData.row(i);
            row_to_dim[row] = i;
            eigval_product_subspace[selected_subspace] *= eigVal[i];
            eigval_number_subspace[selected_subspace]++;
        }

        transfer_data(projectedData, data_list, dataset_size, subspace_num, subspace_dimensionality);
        transformedData.clear();
        projectedData.clear();
        pca_model_ready = true;
    }

    // Load PCA model from file
    if (load_index == 1) {
        string pca_path = string(index_path) + ".pca";
        if (!load_pca_model(pca_path.c_str(), data_dimensionality, subspace_dimensionality, subspace_num, dataset_mean, eigvec, eigVal, row_to_dim)) {
            cout << "PCA model file not found or dimension mismatch: " << pca_path << ". Cannot project query." << endl;
            exit(-1);
        }
        pca_model_ready = true;
    }
    
    // Indexing phase
    size_t RSS_before_indexing = getCurrentRSS() / 1000000; 

    long int index_time = 0;
    int kmeans_dim = subspace_dimensionality / 2;
    
    int * assignments_list = new int[dataset_size * subspace_num * 2];
    float * centroids_list = new float [kmeans_num_centroid * kmeans_dim * subspace_num * 2];
    IndexMap indexes;

    if (load_index == 1) { // load index from index_path
        load_indexes(index_path, indexes, centroids_list, assignments_list, dataset_size, kmeans_dim, subspace_num, kmeans_num_centroid);
    } else { // need to generate index and save it to index_path
        // generate index
        gen_indexes(data_list, indexes, dataset_size, centroids_list, assignments_list, kmeans_dim, subspace_num, kmeans_num_centroid, kmeans_num_iters, index_time);
        // save index
        save_indexes(index_path, centroids_list, assignments_list, dataset_size, kmeans_dim, subspace_num, kmeans_num_centroid);
        if (pca_model_ready) {
            string pca_path = string(index_path) + ".pca";
            save_pca_model(pca_path.c_str(), data_dimensionality, subspace_dimensionality, subspace_num, dataset_mean, eigvec, eigVal, row_to_dim);
        }
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

    // Project query points to subspaces
    float ** projected_querypoints = new float * [query_size];
    if (pca_model_ready) {
        int proj_dim = subspace_dimensionality * subspace_num;
        for (long i = 0; i < query_size; i++) {
            projected_querypoints[i] = new float[proj_dim];
            arma::vec q_col(data_dimensionality);
            for (int j = 0; j < data_dimensionality; j++) {
                q_col(j) = querypoints[i][j];
            }
            q_col -= dataset_mean;
            arma::vec q_trans = eigvec.t() * q_col;
            for (int r = 0; r < proj_dim; r++) {
                projected_querypoints[i][r] = (float)q_trans(row_to_dim[r]);
            }
        }
    }

    int number_of_threads = get_nprocs_conf() / 2;

    AnnQueryParams qparams = {
        dataset, queryknn_results, dataset_size, data_dimensionality,
        query_size, k_size, querypoints, &indexes, centroids_list,
        subspace_num, subspace_dimensionality, kmeans_num_centroid, kmeans_dim,
        collision_num, candidate_num, number_of_threads,
        projected_querypoints, gt
    };
    ann_query(qparams, query_time);
    
    if (load_index == 0) {
        cout << "The indexing time is: " << index_time / 1000.0 << "ms." << endl;
    }
    cout << "The indexing footprint is: " << RSS_after_indexing - RSS_before_indexing << "MB" << endl;
    cout << "The average query time is " << query_time / query_size / 1000.0 << "ms." << endl;
    
    // Evaluate the query accuracy (recall and ratio)
    recall_and_ratio(dataset, querypoints, data_dimensionality, queryknn_results, gt, query_size);

}