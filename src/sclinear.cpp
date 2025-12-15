#include "sclinear.h"

using namespace std;

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

    static int data_dimensionality = 256;
    static int subspace_dimensionality = 32;
    static int subspace_num = 8;

    float candidate_ratio = 0.05;
    float collision_ratio = 0.1;
    
    static int kmeans_num_centroid = 128;
    static int kmeans_num_iters = 10;

    static int load_index=0;

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

            default:
                exit(-1);
                break;
        }
    }

    // Load data
    dataset_size = dataset_size - 100;

    int fread_return;
    
    cout << ">>> Loading dataset from: " << dataset_path << endl;

    FILE * ifile_dataset;
    ifile_dataset = fopen(dataset_path,"rb");
    if (ifile_dataset == NULL) {
        cout << "File " << dataset_path << "not found!" << endl;
        exit(-1);
    }

    cout << "Cardinality of dataset is: " << dataset_size << endl;

    float ** dataset = new float*[dataset_size];
    for (long int i = 0; i < dataset_size; i++)
    {
        dataset[i] = new float[data_dimensionality];
        fread_return = fread(dataset[i], sizeof(float), data_dimensionality, ifile_dataset);                
    }
    fclose(ifile_dataset);

    // Load query
    cout << ">>> Loading query from: " << query_path << endl;
    FILE *ifile_query;
    ifile_query = fopen(query_path,"rb");
    if (ifile_query == NULL) {
        cout << "File " << query_path << "not found!" << endl;
        exit(-1);
    }

    float ** querypoints = new float*[query_size];
    for (int i = 0; i < query_size; i++)
    {
        querypoints[i] = new float[data_dimensionality];
        fread_return = fread(querypoints[i], sizeof(float), data_dimensionality, ifile_query);                
    }
    fclose(ifile_query);

    // Load benchmark
    cout << ">>> Loading groundtruth from: " << groundtruth_path << endl;
    FILE *ifile_groundtruth;
    ifile_groundtruth = fopen(groundtruth_path,"rb");
    if (ifile_groundtruth == NULL) {
        cout << "File " << groundtruth_path << "not found!" << endl;
        exit(-1);
    }

    long int ** result = new long int*[query_size];
    for (int i = 0; i < query_size; i++)
    {
        result[i] = new long int[k_size];
        fread_return = fread(result[i], sizeof(long int), k_size, ifile_groundtruth);                
    }

    vector<long int> sc_score(dataset_size, 0);

    int number_of_threads = get_nprocs_conf() / 2;

    long int collision_num = dataset_size * collision_ratio;
    long int candidate_num = dataset_size * candidate_ratio;

    long int ** queryknn_indexes = new long int*[query_size];

    progress_display pd_query(query_size);

    struct timeval start_query, end_query;
    long int query_phase = 0;

    for (int q_num = 0; q_num < query_size; q_num++) {

        queryknn_indexes[q_num] = new long int[k_size];

        gettimeofday(&start_query, NULL);

        for (int i = 0; i < subspace_num; i++) {
            float * dists = new float [dataset_size];
            // float * query_i = new float [subspace_dimensionality];

            // for (int j = 0; j < subspace_dimensionality; j++) {
            //     query_i[j] = querypoints[q_num][i * subspace_dimensionality + j];
            // }

            #pragma omp parallel for num_threads(number_of_threads)
            for (int j = 0; j < dataset_size; j++) {
                // float * data_j = new float [subspace_dimensionality];
                // for (int z = 0; z < subspace_dimensionality; z++) {
                //     data_j[z] = dataset[j][i * subspace_dimensionality + z];
                // }
                // dists[j] = euclidean_distance_SIMD(&querypoints[q_num][i * subspace_dimensionality], &dataset[j][i * subspace_dimensionality], subspace_dimensionality);
                dists[j] = faiss::fvec_L2sqr_avx512(&querypoints[q_num][i * subspace_dimensionality], &dataset[j][i * subspace_dimensionality], subspace_dimensionality);
                // delete data_j;
            }

            vector<long int> point_idx(dataset_size);
            iota(point_idx.begin(), point_idx.end(), 0);
            partial_sort(point_idx.begin(), point_idx.begin() + collision_num, point_idx.end(), [&dists](long int i1, long int i2) {return dists[i1] < dists[i2];});

            #pragma omp parallel for num_threads(number_of_threads)
            for (long int j = 0; j < collision_num; j++) {
                sc_score[point_idx[j]]++;
            }

            // delete dists;
            // delete query_i;
        }

        vector<long int> sc_point_idx(dataset_size);
        iota(sc_point_idx.begin(), sc_point_idx.end(), 0);
        partial_sort(sc_point_idx.begin(), sc_point_idx.begin() + candidate_num, sc_point_idx.end(), [&sc_score](long int i1, long int i2) {return sc_score[i1] > sc_score[i2];});

        float * candidate_dists = new float [candidate_num];
        #pragma omp parallel for num_threads(number_of_threads)
        for (long int i = 0; i < candidate_num; i++) {
            // candidate_dists[i] = euclidean_distance_SIMD(querypoints[q_num], dataset[sc_point_idx[i]], data_dimensionality);
            candidate_dists[i] = faiss::fvec_L2sqr_avx512(querypoints[q_num], dataset[sc_point_idx[i]], data_dimensionality);
        }

        vector<long int> result_idx(candidate_num);
        iota(result_idx.begin(), result_idx.end(), 0);
        partial_sort(result_idx.begin(), result_idx.begin() + k_size,  result_idx.end(), [&candidate_dists](long int i1, long int i2) {return candidate_dists[i1] < candidate_dists[i2];});
        
        gettimeofday(&end_query, NULL);
        query_phase += (1000000 * (end_query.tv_sec - start_query.tv_sec) + end_query.tv_usec - start_query.tv_usec);

        for (int i = 0; i < k_size; i++) {
            queryknn_indexes[q_num][i] = sc_point_idx[result_idx[i]];
        }
        

        delete candidate_dists;

        fill(sc_score.begin(), sc_score.end(), 0);

        ++pd_query;
    }

    cout << "The average query time is " << query_phase / query_size / 1000.0 << "ms." << endl;

    // Evaluate the query accuracy (recall and ratio)
    int ks[6] = {1, 10, 20, 30, 40, 50};

    for (int k_index = 0; k_index < sizeof(ks) / sizeof(ks[0]); k_index++) {
        int retrived_data_num = 0;

        for (int i = 0; i < query_size; i++)
        {
            for (int j = 0; j < ks[k_index]; j++)
            {
                for (int z = 0; z < ks[k_index]; z++) {
                    if (queryknn_indexes[i][j] == result[i][z]) {
                        retrived_data_num++;
                        break;
                    }
                }
            }
        }

        float ratio = 0.0f;
        for (int i = 0; i < query_size; i++)
        {
            for (int j = 0; j < ks[k_index]; j++)
            {
                float groundtruth_square_dist = euclidean_distance(querypoints[i], dataset[result[i][j]], data_dimensionality);
                float otbained_square_dist = euclidean_distance(querypoints[i], dataset[queryknn_indexes[i][j]], data_dimensionality);
                if (groundtruth_square_dist == 0) {
                    ratio += 1.0f;
                } else {
                    ratio += sqrt(otbained_square_dist) / sqrt(groundtruth_square_dist);
                }
            }
        }

        float recall_value = float(retrived_data_num) / (query_size * ks[k_index]);
        float overall_ratio = ratio / (query_size * ks[k_index]);

        cout << "When k = " << ks[k_index] << ", (recall, ratio) = (" << recall_value << ", " << overall_ratio << ")" << endl;
    }

    return 0;
}
        
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

float euclidean_distance(float * t, float * s, int size) {
    float distance = 0;
    while (size > 0) {
        size--;
        distance += (t[size] - s[size]) * (t[size] - s[size]);
    }  
    return distance;
}

float euclidean_distance_SIMD(float * t, float * s, int size) {
    float distance = 0;
    int i =0;
    float distancef[8];

    __m256 v_fd,v_t,v_s,v_d,distancev;

    v_fd=_mm256_setzero_ps ();
    while (size > 0) {
        v_t=_mm256_loadu_ps (&t[i]);
        v_s=_mm256_loadu_ps (&s[i]);
        
        v_d= _mm256_sub_ps (v_t,v_s);

        v_fd=_mm256_add_ps (v_fd,_mm256_mul_ps (v_d,v_d));
        size-=8;

        i=i+8;
    }

    distancev = _mm256_hadd_ps (v_fd, v_fd);
    distancev = _mm256_hadd_ps (distancev, distancev);
    _mm256_storeu_ps (distancef ,distancev);
    distance +=distancef[0]+distancef[4];
    return distance;
}

namespace faiss {

// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read (int d, const float *x) {
    assert (0 <= d && d < 4);
    __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
    switch (d) {
        case 3:
            buf[2] = x[2];
        case 2:
            buf[1] = x[1];
        case 1:
            buf[0] = x[0];
    }
    return _mm_load_ps(buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}
uint8_t lookup8bit[256];
//extern uint8_t lookup8bit[256];

float
fvec_inner_product_avx512(const float* x, const float* y, size_t d) {
    __m512 msum0 = _mm512_setzero_ps();

    while (d >= 16) {
        __m512 mx = _mm512_loadu_ps (x); x += 16;
        __m512 my = _mm512_loadu_ps (y); y += 16;
        msum0 = _mm512_add_ps (msum0, _mm512_mul_ps (mx, my));
        d -= 16;
    }

    __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
    msum1 +=       _mm512_extractf32x8_ps(msum0, 0);

    if (d >= 8) {
        __m256 mx = _mm256_loadu_ps (x); x += 8;
        __m256 my = _mm256_loadu_ps (y); y += 8;
        msum1 = _mm256_add_ps (msum1, _mm256_mul_ps (mx, my));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 +=       _mm256_extractf128_ps(msum1, 0);

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps (x); x += 4;
        __m128 my = _mm_loadu_ps (y); y += 4;
        msum2 = _mm_add_ps (msum2, _mm_mul_ps (mx, my));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read (d, x);
        __m128 my = masked_read (d, y);
        msum2 = _mm_add_ps (msum2, _mm_mul_ps (mx, my));
    }

    msum2 = _mm_hadd_ps (msum2, msum2);
    msum2 = _mm_hadd_ps (msum2, msum2);
    return  _mm_cvtss_f32 (msum2);
}

float
fvec_L2sqr_avx512(const float* x, const float* y, size_t d) {
    __m512 msum0 = _mm512_setzero_ps();

    while (d >= 16) {
        __m512 mx = _mm512_loadu_ps (x); x += 16;
        __m512 my = _mm512_loadu_ps (y); y += 16;
        const __m512 a_m_b1 = mx - my;
        msum0 += a_m_b1 * a_m_b1;
        d -= 16;
    }

    __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
    msum1 +=       _mm512_extractf32x8_ps(msum0, 0);

    if (d >= 8) {
        __m256 mx = _mm256_loadu_ps (x); x += 8;
        __m256 my = _mm256_loadu_ps (y); y += 8;
        const __m256 a_m_b1 = mx - my;
        msum1 += a_m_b1 * a_m_b1;
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 +=       _mm256_extractf128_ps(msum1, 0);

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps (x); x += 4;
        __m128 my = _mm_loadu_ps (y); y += 4;
        const __m128 a_m_b1 = mx - my;
        msum2 += a_m_b1 * a_m_b1;
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read (d, x);
        __m128 my = masked_read (d, y);
        __m128 a_m_b1 = mx - my;
        msum2 += a_m_b1 * a_m_b1;
    }

    msum2 = _mm_hadd_ps (msum2, msum2);
    msum2 = _mm_hadd_ps (msum2, msum2);
    return  _mm_cvtss_f32 (msum2);
}

float
fvec_L1_avx512(const float* x, const float* y, size_t d) {
    __m512 msum0 = _mm512_setzero_ps();
    __m512 signmask0 = __m512(_mm512_set1_epi32 (0x7fffffffUL));

    while (d >= 16) {
        __m512 mx = _mm512_loadu_ps (x); x += 16;
        __m512 my = _mm512_loadu_ps (y); y += 16;
        const __m512 a_m_b = mx - my;
        msum0 += _mm512_and_ps(signmask0, a_m_b);
        d -= 16;
    }

    __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
    msum1 +=       _mm512_extractf32x8_ps(msum0, 0);
    __m256 signmask1 = __m256(_mm256_set1_epi32 (0x7fffffffUL));

    if (d >= 8) {
        __m256 mx = _mm256_loadu_ps (x); x += 8;
        __m256 my = _mm256_loadu_ps (y); y += 8;
        const __m256 a_m_b = mx - my;
        msum1 += _mm256_and_ps(signmask1, a_m_b);
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 +=       _mm256_extractf128_ps(msum1, 0);
    __m128 signmask2 = __m128(_mm_set1_epi32 (0x7fffffffUL));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps (x); x += 4;
        __m128 my = _mm_loadu_ps (y); y += 4;
        const __m128 a_m_b = mx - my;
        msum2 += _mm_and_ps(signmask2, a_m_b);
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read (d, x);
        __m128 my = masked_read (d, y);
        __m128 a_m_b = mx - my;
        msum2 += _mm_and_ps(signmask2, a_m_b);
    }

    msum2 = _mm_hadd_ps (msum2, msum2);
    msum2 = _mm_hadd_ps (msum2, msum2);
    return  _mm_cvtss_f32 (msum2);
}

float
fvec_Linf_avx512(const float* x, const float* y, size_t d) {
    __m512 msum0 = _mm512_setzero_ps();
    __m512 signmask0 = __m512(_mm512_set1_epi32 (0x7fffffffUL));

    while (d >= 16) {
        __m512 mx = _mm512_loadu_ps (x); x += 16;
        __m512 my = _mm512_loadu_ps (y); y += 16;
        const __m512 a_m_b = mx - my;
        msum0 = _mm512_max_ps(msum0, _mm512_and_ps(signmask0, a_m_b));
        d -= 16;
    }

    __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
    msum1 = _mm256_max_ps (msum1, _mm512_extractf32x8_ps(msum0, 0));
    __m256 signmask1 = __m256(_mm256_set1_epi32 (0x7fffffffUL));

    if (d >= 8) {
        __m256 mx = _mm256_loadu_ps (x); x += 8;
        __m256 my = _mm256_loadu_ps (y); y += 8;
        const __m256 a_m_b = mx - my;
        msum1 = _mm256_max_ps(msum1, _mm256_and_ps(signmask1, a_m_b));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 = _mm_max_ps (msum2, _mm256_extractf128_ps(msum1, 0));
    __m128 signmask2 = __m128(_mm_set1_epi32 (0x7fffffffUL));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps (x); x += 4;
        __m128 my = _mm_loadu_ps (y); y += 4;
        const __m128 a_m_b = mx - my;
        msum2 = _mm_max_ps(msum2, _mm_and_ps(signmask2, a_m_b));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read (d, x);
        __m128 my = masked_read (d, y);
        __m128 a_m_b = mx - my;
        msum2 = _mm_max_ps(msum2, _mm_and_ps(signmask2, a_m_b));
    }

    msum2 = _mm_max_ps(_mm_movehl_ps(msum2, msum2), msum2);
    msum2 = _mm_max_ps(msum2, _mm_shuffle_ps (msum2, msum2, 1));
    return  _mm_cvtss_f32 (msum2);
}

uint64_t
_mm256_hsum_epi64(__m256i v) {
    return _mm256_extract_epi64(v, 0) +
           _mm256_extract_epi64(v, 1) +
           _mm256_extract_epi64(v, 2) +
           _mm256_extract_epi64(v, 3);
}

uint64_t _mm512_hsum_epi64(__m512i v) {
    const __m256i t0 = _mm512_extracti64x4_epi64(v, 0);
    const __m256i t1 = _mm512_extracti64x4_epi64(v, 1);

    return _mm256_hsum_epi64(t0) + _mm256_hsum_epi64(t1);
}

int
popcnt_AVX512VBMI_lookup(const uint8_t* data, const size_t n) {

    size_t i = 0;

    const __m512i lookup = _mm512_setr_epi64(
        0x0302020102010100llu, 0x0403030203020201llu,
        0x0302020102010100llu, 0x0403030203020201llu,
        0x0302020102010100llu, 0x0403030203020201llu,
        0x0302020102010100llu, 0x0403030203020201llu
    );

    const __m512i low_mask = _mm512_set1_epi8(0x0f);

    __m512i acc = _mm512_setzero_si512();

    while (i + 64 < n) {

        __m512i local = _mm512_setzero_si512();

        for (int k=0; k < 255/8 && i + 64 < n; k++, i += 64) {
            const __m512i vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data + i));
            const __m512i lo  = _mm512_and_si512(vec, low_mask);
            const __m512i hi  = _mm512_and_si512(_mm512_srli_epi32(vec, 4), low_mask);

            const __m512i popcnt1 = _mm512_shuffle_epi8(lookup, lo);
            const __m512i popcnt2 = _mm512_shuffle_epi8(lookup, hi);

            local = _mm512_add_epi8(local, popcnt1);
            local = _mm512_add_epi8(local, popcnt2);
        }

        acc = _mm512_add_epi64(acc, _mm512_sad_epu8(local, _mm512_setzero_si512()));
    }


    int result = _mm512_hsum_epi64(acc);
    for (/**/; i < n; i++) {
        result += lookup8bit[data[i]];
    }

    return result;
}

int
xor_popcnt_AVX512VBMI_lookup(const uint8_t* data1, const uint8_t* data2, const size_t n) {

    size_t i = 0;

    const __m512i lookup = _mm512_setr_epi64(
        0x0302020102010100llu, 0x0403030203020201llu,
        0x0302020102010100llu, 0x0403030203020201llu,
        0x0302020102010100llu, 0x0403030203020201llu,
        0x0302020102010100llu, 0x0403030203020201llu
    );

    const __m512i low_mask = _mm512_set1_epi8(0x0f);

    __m512i acc = _mm512_setzero_si512();

    while (i + 64 < n) {

        __m512i local = _mm512_setzero_si512();

        for (int k=0; k < 255/8 && i + 64 < n; k++, i += 64) {
            const __m512i s1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data1 + i));
            const __m512i s2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data2 + i));
            const __m512i vec = _mm512_xor_si512(s1, s2);
            const __m512i lo  = _mm512_and_si512(vec, low_mask);
            const __m512i hi  = _mm512_and_si512(_mm512_srli_epi32(vec, 4), low_mask);

            const __m512i popcnt1 = _mm512_shuffle_epi8(lookup, lo);
            const __m512i popcnt2 = _mm512_shuffle_epi8(lookup, hi);

            local = _mm512_add_epi8(local, popcnt1);
            local = _mm512_add_epi8(local, popcnt2);
        }

        acc = _mm512_add_epi64(acc, _mm512_sad_epu8(local, _mm512_setzero_si512()));
    }


    int result = _mm512_hsum_epi64(acc);
    for (/**/; i < n; i++) {
        result += lookup8bit[data1[i]^data2[i]];
    }

    return result;
}

int
or_popcnt_AVX512VBMI_lookup(const uint8_t* data1, const uint8_t* data2, const size_t n) {

    size_t i = 0;

    const __m512i lookup = _mm512_setr_epi64(
        0x0302020102010100llu, 0x0403030203020201llu,
        0x0302020102010100llu, 0x0403030203020201llu,
        0x0302020102010100llu, 0x0403030203020201llu,
        0x0302020102010100llu, 0x0403030203020201llu
    );

    const __m512i low_mask = _mm512_set1_epi8(0x0f);

    __m512i acc = _mm512_setzero_si512();

    while (i + 64 < n) {

        __m512i local = _mm512_setzero_si512();

        for (int k=0; k < 255/8 && i + 64 < n; k++, i += 64) {
            const __m512i s1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data1 + i));
            const __m512i s2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data2 + i));
            const __m512i vec = _mm512_or_si512(s1, s2);
            const __m512i lo  = _mm512_and_si512(vec, low_mask);
            const __m512i hi  = _mm512_and_si512(_mm512_srli_epi32(vec, 4), low_mask);

            const __m512i popcnt1 = _mm512_shuffle_epi8(lookup, lo);
            const __m512i popcnt2 = _mm512_shuffle_epi8(lookup, hi);

            local = _mm512_add_epi8(local, popcnt1);
            local = _mm512_add_epi8(local, popcnt2);
        }

        acc = _mm512_add_epi64(acc, _mm512_sad_epu8(local, _mm512_setzero_si512()));
    }


    int result = _mm512_hsum_epi64(acc);
    for (/**/; i < n; i++) {
        result += lookup8bit[data1[i]|data2[i]];
    }

    return result;
}

int
and_popcnt_AVX512VBMI_lookup(const uint8_t* data1, const uint8_t* data2, const size_t n) {

    size_t i = 0;

    const __m512i lookup = _mm512_setr_epi64(
        0x0302020102010100llu, 0x0403030203020201llu,
        0x0302020102010100llu, 0x0403030203020201llu,
        0x0302020102010100llu, 0x0403030203020201llu,
        0x0302020102010100llu, 0x0403030203020201llu
    );

    const __m512i low_mask = _mm512_set1_epi8(0x0f);

    __m512i acc = _mm512_setzero_si512();

    while (i + 64 < n) {

        __m512i local = _mm512_setzero_si512();

        for (int k=0; k < 255/8 && i + 64 < n; k++, i += 64) {
            const __m512i s1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data1 + i));
            const __m512i s2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data2 + i));
            const __m512i vec = _mm512_and_si512(s1, s2);
            const __m512i lo  = _mm512_and_si512(vec, low_mask);
            const __m512i hi  = _mm512_and_si512(_mm512_srli_epi32(vec, 4), low_mask);

            const __m512i popcnt1 = _mm512_shuffle_epi8(lookup, lo);
            const __m512i popcnt2 = _mm512_shuffle_epi8(lookup, hi);

            local = _mm512_add_epi8(local, popcnt1);
            local = _mm512_add_epi8(local, popcnt2);
        }

        acc = _mm512_add_epi64(acc, _mm512_sad_epu8(local, _mm512_setzero_si512()));
    }


    int result = _mm512_hsum_epi64(acc);
    for (/**/; i < n; i++) {
        result += lookup8bit[data1[i]&data2[i]];
    }

    return result;
}

float
jaccard_AVX512(const uint8_t * a, const uint8_t * b, size_t n) {
    int accu_num = and_popcnt_AVX512VBMI_lookup(a,b,n);
    int accu_den = or_popcnt_AVX512VBMI_lookup(a,b,n);
    return (accu_den == 0) ? 1.0 : ((float)(accu_den - accu_num) / (float)(accu_den));
}

} // namespace faiss
