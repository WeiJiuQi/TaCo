#include "dist_calculation.h"
#include "utils.h"
#include "index.h"
#include "query.h"
#include "preprocess.h"
#include "evaluate.h"
#include <mlpack/methods/pca/pca.hpp>
#include <string>
#include <unistd.h>

using namespace std;

struct Config {
    char * dataset_path = nullptr;
    char * query_path = nullptr;
    char * groundtruth_path = nullptr;
    char * index_path = nullptr;
    long int dataset_size = 1000000;
    int query_size = 100;
    int k_size = 50;
    int data_dimensionality = 128;
    int subspace_dimensionality = 16;
    int subspace_num = 8;
    float candidate_ratio = 0.05f;
    float collision_ratio = 0.1f;
    int kmeans_num_centroid = 50;
    int kmeans_num_iters = 2;
    int load_index = 0;
};

void INThandler(int sig) {
    char c;
    signal(sig, SIG_IGN);
    fprintf(stderr, "Do you really want to quit? [y/n] ");
    c = getchar();
    if (c == 'y' || c == 'Y') {
        exit(0);
    } else {
        signal(SIGINT, INThandler);
        getchar();
    }
}

static bool parse_args(int argc, char **argv, Config &c) {
    static struct option long_options[] = {
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
    optind = 1;
    int option_index = 0;
    int ch;
    while ((ch = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
        switch (ch) {
            case 'a': c.dataset_path = optarg; break;
            case 'b': c.query_path = optarg; break;
            case 'c': c.groundtruth_path = optarg; break;
            case 'd': c.index_path = optarg; break;
            case 'e': c.dataset_size = atoi(optarg); break;
            case 'f': c.query_size = atoi(optarg); break;
            case 'g': c.k_size = atoi(optarg); break;
            case 'h': c.data_dimensionality = atoi(optarg); break;
            case 'i': c.subspace_dimensionality = atoi(optarg); break;
            case 'j': c.subspace_num = atoi(optarg); break;
            case 'k': c.candidate_ratio = static_cast<float>(atof(optarg)); break;
            case 'l': c.collision_ratio = static_cast<float>(atof(optarg)); break;
            case 'm': c.kmeans_num_centroid = atoi(optarg); break;
            case 'n': c.kmeans_num_iters = atoi(optarg); break;
            case 'o': c.load_index = 1; break;
            default: return false;
        }
    }
    return c.dataset_path && c.query_path && c.groundtruth_path && c.index_path;
}

static void load_data_step(const Config &c, float **&dataset, float **&querypoints, long int **&gt) {
    long int n = c.dataset_size - 100;
    load_data(dataset, c.dataset_path, n, c.data_dimensionality);
    load_query(querypoints, c.query_path, c.query_size, c.data_dimensionality);
    load_groundtruth(gt, c.groundtruth_path, c.query_size, c.k_size);
}

static bool run_pca_or_load(const Config &c, float **dataset, std::vector<arma::mat> &data_list,
                            arma::vec &dataset_mean, arma::mat &eigvec, arma::vec &eigVal,
                            std::vector<int> &row_to_dim) {
    long int n = c.dataset_size - 100;
    if (c.load_index == 1) {
        string pca_path = string(c.index_path) + ".pca";
        if (!load_pca_model(pca_path.c_str(), c.data_dimensionality, c.subspace_dimensionality,
                            c.subspace_num, dataset_mean, eigvec, eigVal, row_to_dim)) {
            cout << "PCA model file not found or dimension mismatch: " << pca_path << ". Cannot project query." << endl;
            exit(-1);
        }
        return true;
    }
    arma::mat dataset_armamat(c.data_dimensionality, n, arma::fill::zeros);
    for (long int i = 0; i < n; i++) {
        for (int j = 0; j < c.data_dimensionality; j++)
            dataset_armamat(j, i) = dataset[i][j];
    }
    dataset_mean = arma::mean(dataset_armamat, 1);
    arma::mat transformedData(c.data_dimensionality, n, arma::fill::zeros);
    eigVal.set_size(c.data_dimensionality);
    eigvec.set_size(c.data_dimensionality, c.data_dimensionality);
    mlpack::pca::PCA pca;
    pca.Apply(dataset_armamat, transformedData, eigVal, eigvec);
    dataset_armamat.clear();
    int power = 0;
    while (eigVal[c.data_dimensionality - 1] < 1) {
        eigVal[c.data_dimensionality - 1] *= 10;
        power++;
    }
    for (int i = 0; i < c.data_dimensionality - 1; i++)
        eigVal[i] *= pow(10, power);
    arma::mat projectedData(c.subspace_dimensionality * c.subspace_num, n, arma::fill::zeros);
    row_to_dim.resize(c.subspace_dimensionality * c.subspace_num);
    vector<double> eigval_product_subspace(c.subspace_num, 1.0);
    vector<int> eigval_number_subspace(c.subspace_num, 0);
    for (int i = 0; i < c.subspace_num; i++) {
        int row = i * c.subspace_dimensionality;
        projectedData.row(row) = transformedData.row(i);
        row_to_dim[row] = i;
        eigval_product_subspace[i] *= eigVal[i];
        eigval_number_subspace[i]++;
    }
    for (int i = c.subspace_num; i < c.subspace_dimensionality * c.subspace_num; i++) {
        int selected_subspace = eigval_subspace_select(eigval_product_subspace, eigval_number_subspace, c.subspace_num, c.subspace_dimensionality);
        int row = selected_subspace * c.subspace_dimensionality + eigval_number_subspace[selected_subspace];
        projectedData.row(row) = transformedData.row(i);
        row_to_dim[row] = i;
        eigval_product_subspace[selected_subspace] *= eigVal[i];
        eigval_number_subspace[selected_subspace]++;
    }
    transfer_data(projectedData, data_list, n, c.subspace_num, c.subspace_dimensionality);
    transformedData.clear();
    projectedData.clear();
    return true;
}

static long int build_or_load_index(const Config &c, std::vector<arma::mat> &data_list,
                                    const arma::vec &dataset_mean, const arma::mat &eigvec,
                                    const arma::vec &eigVal, const std::vector<int> &row_to_dim,
                                    IndexMap &indexes, float *&centroids_list) {
    long int n = c.dataset_size - 100;
    int kmeans_dim = c.subspace_dimensionality / 2;
    int *assignments_list = new int[n * c.subspace_num * 2];
    centroids_list = new float[kmeans_dim * c.kmeans_num_centroid * c.subspace_num * 2];
    long int index_time = 0;
    size_t rss_before = getCurrentRSS() / 1000000;

    if (c.load_index == 1) {
        load_indexes(c.index_path, indexes, centroids_list, assignments_list, n, kmeans_dim, c.subspace_num, c.kmeans_num_centroid);
    } else {
        gen_indexes(data_list, indexes, n, centroids_list, assignments_list, kmeans_dim, c.subspace_num, c.kmeans_num_centroid, c.kmeans_num_iters, index_time);
        save_indexes(c.index_path, centroids_list, assignments_list, n, kmeans_dim, c.subspace_num, c.kmeans_num_centroid);
        string pca_path = string(c.index_path) + ".pca";
        save_pca_model(pca_path.c_str(), c.data_dimensionality, c.subspace_dimensionality, c.subspace_num, dataset_mean, eigvec, eigVal, row_to_dim);
    }
    delete[] assignments_list;
    size_t rss_after = getCurrentRSS() / 1000000;
    if (c.load_index == 0)
        cout << "The indexing time is: " << index_time / 1000.0 << "ms." << endl;
    cout << "The indexing footprint is: " << (rss_after - rss_before) << "MB" << endl;
    return index_time;
}

static long int run_query_step(const Config &c, float **dataset, float **querypoints, long int **gt,
                               IndexMap &indexes, float *centroids_list,
                               const arma::vec &dataset_mean, const arma::mat &eigvec,
                               const std::vector<int> &row_to_dim, int **&queryknn_results) {
    long int n = c.dataset_size - 100;
    int collision_num = static_cast<int>(c.collision_ratio * n);
    int candidate_num = static_cast<int>(c.candidate_ratio * n);
    int kmeans_dim = c.subspace_dimensionality / 2;

    queryknn_results = new int*[c.query_size];
    for (int i = 0; i < c.query_size; i++)
        queryknn_results[i] = new int[c.k_size];

    float **projected_querypoints = new float*[c.query_size];
    int proj_dim = c.subspace_dimensionality * c.subspace_num;
    for (int i = 0; i < c.query_size; i++) {
        projected_querypoints[i] = new float[proj_dim];
        arma::vec q_col(c.data_dimensionality);
        for (int j = 0; j < c.data_dimensionality; j++)
            q_col(j) = querypoints[i][j];
        q_col -= dataset_mean;
        arma::vec q_trans = eigvec.t() * q_col;
        for (int r = 0; r < proj_dim; r++)
            projected_querypoints[i][r] = static_cast<float>(q_trans(row_to_dim[r]));
    }

    int number_of_threads = get_nprocs_conf() / 2;
    long int query_time = 0;
    AnnQueryParams qparams = {
        dataset, queryknn_results, n, c.data_dimensionality,
        c.query_size, c.k_size, querypoints, &indexes, centroids_list,
        c.subspace_num, c.subspace_dimensionality, c.kmeans_num_centroid, kmeans_dim,
        collision_num, candidate_num, number_of_threads,
        projected_querypoints, gt
    };
    ann_query(qparams, query_time);

    for (int i = 0; i < c.query_size; i++) {
        delete[] projected_querypoints[i];
    }
    delete[] projected_querypoints;

    cout << "The average query time is " << (query_time / c.query_size / 1000.0) << "ms." << endl;
    return query_time;
}

static void run_evaluate(float **dataset, float **querypoints, int data_dimensionality,
                         int **queryknn_results, long int **gt, int query_size) {
    recall_and_ratio(dataset, querypoints, data_dimensionality, queryknn_results, gt, query_size);
}

int main(int argc, char **argv) {
    signal(SIGINT, INThandler);

    Config c;
    if (!parse_args(argc, argv, c)) {
        cerr << "Invalid or missing arguments." << endl;
        exit(-1);
    }

    float **dataset = nullptr;
    float **querypoints = nullptr;
    long int **gt = nullptr;
    load_data_step(c, dataset, querypoints, gt);

    std::vector<arma::mat> data_list;
    arma::vec dataset_mean;
    arma::mat eigvec;
    arma::vec eigVal;
    std::vector<int> row_to_dim;
    (void)run_pca_or_load(c, dataset, data_list, dataset_mean, eigvec, eigVal, row_to_dim);

    IndexMap indexes;
    float *centroids_list = nullptr;
    (void)build_or_load_index(c, data_list, dataset_mean, eigvec, eigVal, row_to_dim, indexes, centroids_list);

    int **queryknn_results = nullptr;
    (void)run_query_step(c, dataset, querypoints, gt, indexes, centroids_list, dataset_mean, eigvec, row_to_dim, queryknn_results);

    run_evaluate(dataset, querypoints, c.data_dimensionality, queryknn_results, gt, c.query_size);

    delete[] centroids_list;
    for (int i = 0; i < c.query_size; i++) {
        delete[] queryknn_results[i];
    }
    delete[] queryknn_results;
    return 0;
}
