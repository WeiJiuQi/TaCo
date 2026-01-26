# Instruction for running competitors and experiments

## 1. Obtain and run competitors

### 1.1 Subspace collision-based method

#### 1.1.1 SuCo (SOTA baseline)

+ Open source code: https://github.com/WeiJiuQi/SuCo

+ Parameter Settings: $N_s\in [4,10]$, $s\in [6,12]$, $\alpha \in \left[0.01, 0.1\right]$, $\beta \in \left[0.001, 0.05\right]$

+ Command (Detailed explanations are available on GitHub): 

  ```
  ./suco --dataset-path $PATH_TO_DATASET$ --query-path $PATH_TO_QUERY$ --groundtruth-path $PATH_TO_GROUNDTRUTH$ --dataset-size $n$ --query-size $QUERY_SIZE$ --k-size $k$ --data-dimensionality $DIMENSIONALITY$ --subspace-dimensionality $SUBSPACE_DIMENSIONALITY$ --subspace-num $SUBSPACE_NUMBER$ --candidate-ratio $beta$ --collision-ratio $alpha$ --kmeans-num-centroid $K$ --kmeans-num-iters $ITERATION$ --index-path $PATH_TO_INDEX$ --load-index
  ```

+ Notes: The current open-source version of SuCo code already includes some query optimizations (query.cpp), such as Query-aware Candidates Selection and Scalable Dynamic Activation. Please take note when running experiments.

#### 1.1.2 SuCo-DT,  SuCo-CS, SuCo-QS (Ablation baselines)

+ Open source code: https://anonymous.4open.science/r/TaCo-9AB5

+ Parameter Settings: $N_s\in [4,10]$, $s\in [6,12]$, $\alpha \in \left[0.01, 0.1\right]$, $\beta \in \left[0.001, 0.05\right]$

+ Command (Detailed explanations are available on GitHub): 

  ```
  ./taco --dataset-path $PATH_TO_DATASET$ --query-path $PATH_TO_QUERY$ --projected-query-path $PATH_TO_PROJECTED_QUERY$ --groundtruth-path $PATH_TO_GROUNDTRUTH$ --dataset-size $n$ --query-size $QUERY_SIZE$ --k-size $k$ --data-dimensionality $DIMENSIONALITY$ --subspace-dimensionality $SUBSPACE_DIMENSIONALITY$ --subspace-num $SUBSPACE_NUMBER$ --candidate-ratio $beta$ --collision-ratio $alpha$ --kmeans-num-centroid $K$ --kmeans-num-iters $ITERATION$ --index-path $PATH_TO_INDEX$ --load-index
  ```

+ Notes: Since the code for our proposed TaCo method is developed based on SuCo; SuCo-DT, SuCo-CS, and SuCo-QS, which serve as the ablation baseline, can be obtained by modifying the TaCo open-source code. Specifically:
  + SuCo-DT: in the query.cpp file, comment out 92-98 lines, uncomment the 90, 100-108, 113-120 lines; invoke the dynamic_activate function at line 38 instead of the scalable_dynamic_activate function at line 39.
  + SuCo-CS: comment out the data transformation strategy in the main.cpp file (163-220 lines); in the query.cpp file, invoke the dynamic_activate function at line 38 instead of the scalable_dynamic_activate function at line 39.
  + SuCo-QS: comment out the data transformation strategy in the main.cpp file (163-220 lines); in the query.cpp file, invoke the scalable_dynamic_activate function at line 39 instead of the dynamic_activate function at line 38.

#### 1.1.3 SC-Linear (Naive baseline, without index structure)

+ Open source code: https://anonymous.4open.science/r/TaCo-9AB5

+ Parameter Settings: $N_s\in [4,10]$, $s\in [6,12]$, $\alpha \in \left[0.01, 0.1\right]$, $\beta \in \left[0.001, 0.05\right]$

+ Compile

  ```
  g++ sclinear.cpp -o sclinear -O3 -std=c++11 -larmadillo -lmlpack -lboost_serialization -fopenmp -march=native -mavx512f
  ```

+ Command

  ```
  ./sclinear --dataset-path $PATH_TO_DATASET$ --query-path $PATH_TO_QUERY$ --groundtruth-path $PATH_TO_GROUNDTRUTH$ --dataset-size $n$ --query-size $QUERY_SIZE$ --k-size $k$ --data-dimensionality $DIMENSIONALITY$ --subspace-dimensionality $SUBSPACE_DIMENSIONALITY$ --subspace-num $SUBSPACE_NUMBER$ --candidate-ratio $beta$ --collision-ratio $alpha$ --kmeans-num-centroid $K$ --kmeans-num-iters $ITERATION$
  ```

+ Notes: We have uploaded the SC-Linear code (sclinear.h, sclinear.cpp) to the directory https://anonymous.4open.science/r/TaCo-9AB5/src for readers' convenience in running the experiments.

### 1.2 Non-subspace collision-based method

#### 1.2.1 DET-LSH (LSH-based)

+ Open source code: https://github.com/WeiJiuQi/DET-LSH

+ Parameter Settings: $L=4$, $K=16$, $\beta \in \left[0.005,0.2\right]$, $c=1.5$, $--cpu-type = 81$, $--queue-number = 4$

+ Command (Detailed explanations are available on GitHub): 

  ```
  cd ./bin
  ./DETLSH --dataset $PATH_TO_DATASET$ --leaf-size $MAX_SIZE$ --dataset-size $n$  --queries $PATH_TO_QUERY$ --queries-size $QUERY_SIZE$ --sample-size $SAMPLE_SIZE$ --sample-type $SAMPLE_TYPE$ --k-size $k$ --data-dimensionality $DIMENSIONALITY$ --l-size $L$ --search-radius $r$ --max-candidate-size $beta*n$ --benchmark $PATH_TO_BENCHMARK$ --cpu-type $ct$ --queue-number $qn$
  ```

+ Notes: The original DET-LSH paper did not provide a parallelized version design, but a parallelized version is provided in its GitHub code. We tested the parallelized version of the code and found that the algorithm achieved its best performance when the number of queues (--cpu-type) was half the number of threads (--queue-number). 

#### 1.2.2 IMI-OPQ (Quantization-based)

+ Open source code: the FAISS library (https://github.com/facebookresearch/faiss)

+ Parameter Settings

  ```
  OPQ16_64,IMI2x8,PQ8+16
  ```

+ Command: We use the demo test code provided by the FAISS repository, so after compilation, we can directly go to the `faiss/build/demos` directory and run the executable file. Due to the loss caused by quantization error, this method struggles to achieve high recall on some datasets. Therefore, we provide a rerank function for IMI-OPQ, enabling it to strike a balance between efficiency and accuracy.

+ Notes: IMI-OPQ has appeared as a strong baseline in previous papers until the advent of RaBitQ (SIGMOD 2024 & 2025). 

#### 1.2.3 IVF-RaBitQ (Quantization-based)

+ Open source code: the RaBitQ-Library (https://github.com/VectorDB-NTU/RaBitQ-Library)

+ Parameter Settings (default in code)

  ```
  K = 4096, B = 3
  ```

+ Command: We use the example script provided in the `example.sh` file of the RaBitQ-Library to separately perform IVF construction, quantized index construction, and query execution. Specifically, 

  ```
  # IVF construction
  python ./python/ivf.py ./data/gist/gist_base.fvecs 4096 ./data/gist/gist_centroids_4096.fvecs ./data/gist/gist_clusterids_4096.ivecs
  
  # Quantized index construction
  ./bin/ivf_rabitq_indexing ./data/gist/gist_base.fvecs ./data/gist/gist_centroids_4096.fvecs ./data/gist/gist_clusterids_4096.ivecs 3 ./data/gist/ivf_4096_3.index
  
  # Query execution
  ./bin/ivf_rabitq_querying ./data/gist/ivf_4096_3.index ./data/gist/gist_query.fvecs ./data/gist/gist_groundtruth.ivecs
  ```

+ Notes: RabitQ , as the state-of-the-art scalar quantization method, achieves unbiased estimation and has been widely used in the industry.

#### 1.2.4 HNSW (Graph-based)

+ Open source code: the HNSWlib library (https://github.com/nmslib/hnswlib)

+ Parameter Settings: $efConstruction=200$, $M=25$, $efSearch \in \left[300,3000\right]$.

+ Demo: We rewrote the `tests/cpp/sift_1b.cpp` file to test HNSW's performance comprehensively:  

  ```
  #include <iostream>
  #include <fstream>
  #include <queue>
  #include <chrono>
  #include "../../hnswlib/hnswlib.h"
  
  
  #include <unordered_set>
  #include <sys/sysinfo.h>
  
  using namespace std;
  using namespace hnswlib;
  
  float * rawfile;
  float * queries;
  long int ** result_gt;
  
  void load_data(char * dataset, int dataset_size, int data_dimensionality) {
      fprintf(stderr, ">>> Loading file: %s\n", dataset);
  
      FILE * ifiled;
  
      ifiled = fopen (dataset,"rb");
  
      if (ifiled == NULL) {
          fprintf(stderr, "File %s not found!\n", dataset);
          exit(-1);
      }
  
      fseek(ifiled, 0L, SEEK_END);
      unsigned long long sz = (unsigned long long) ftell(ifiled);
      unsigned long long total_records = sz/data_dimensionality/sizeof(float);
      fseek(ifiled, 0L, SEEK_SET);
  
      // if (total_records < dataset_size) {
      //     fprintf(stderr, "Dataset file %s has only %llu records!\n", dataset, total_records);
      //     exit(-1);
      // }
  
      std::cout << "Total records of dataset is: " << total_records << std::endl;
      std::cout << "Cardinality of dataset is: " << dataset_size << std::endl;
      rawfile=(float*)malloc(sizeof(float) * data_dimensionality * dataset_size);
      int data_read_number=fread(rawfile, sizeof(float), data_dimensionality * dataset_size, ifiled);
      fclose(ifiled);
  }
  
  void load_query(char * file_path, int query_size, int data_dimensionality) {
      fprintf(stderr, ">>> Loading file: %s\n", file_path);
      FILE *ifile;
      ifile = fopen (file_path,"rb");
  
      if (ifile == NULL) {
          fprintf(stderr, "File %s not found!\n", file_path);
          exit(-1);
      }
  
      fseek(ifile, 0L, SEEK_END);
      unsigned long long sz = (unsigned long long) ftell(ifile);
      unsigned long long total_records = sz/data_dimensionality/sizeof(float);
      fseek(ifile, 0L, SEEK_SET);
  
      if (total_records < query_size) {
          fprintf(stderr, "File %s has only %llu records!\n", file_path, total_records);
          exit(-1);
      }
  
      queries = (float*)malloc(sizeof(float) * data_dimensionality * query_size);
      int query_read_number=fread(queries, sizeof(float), data_dimensionality * query_size, ifile);
      fclose(ifile);
  }
  
  void load_groundtruth(char * file_path, int query_size, int k_size) {
      fprintf(stderr, ">>> Loading groundtruth: %s\n", file_path);
  
      FILE *ifile_groundtruth;
      ifile_groundtruth = fopen (file_path,"rb");
      if (ifile_groundtruth == NULL) {
          fprintf(stderr, "Groundtruth file %s not found!\n", file_path);
          exit(-1);
      }
      result_gt = (long int **) malloc(sizeof(long int *) * query_size);
      for (int i = 0; i < query_size; i++)
      {
          result_gt[i] = (long int *) malloc(sizeof(long int) * k_size);
          fread(result_gt[i], sizeof(long int), k_size, ifile_groundtruth);                
      }
      fclose(ifile_groundtruth);
  }
  
  class progress_display
  {
  public:
      explicit progress_display(
          unsigned long expected_count,
          std::ostream& os = std::cout,
          const std::string& s1 = "\n",
          const std::string& s2 = "",
          const std::string& s3 = "")
          : m_os(os), m_s1(s1), m_s2(s2), m_s3(s3)
      {
          restart(expected_count);
      }
      void restart(unsigned long expected_count)
      {
          //_count = _next_tic_count = _tic = 0;
          _expected_count = expected_count;
          m_os << m_s1 << "0%   10   20   30   40   50   60   70   80   90   100%\n"
              << m_s2 << "|----|----|----|----|----|----|----|----|----|----|"
              << std::endl
              << m_s3;
          if (!_expected_count)
          {
              _expected_count = 1;
          }
      }
      unsigned long operator += (unsigned long increment)
      {
          std::unique_lock<std::mutex> lock(mtx);
          if ((_count += increment) >= _next_tic_count)
          {
              display_tic();
          }
          return _count;
      }
      unsigned long  operator ++ ()
      {
          return operator += (1);
      }
  
      //unsigned long  operator + (int x)
      //{
      //	return operator += (x);
      //}
  
      unsigned long count() const
      {
          return _count;
      }
      unsigned long expected_count() const
      {
          return _expected_count;
      }
  private:
      std::ostream& m_os;
      const std::string m_s1;
      const std::string m_s2;
      const std::string m_s3;
      std::mutex mtx;
      std::atomic<size_t> _count{ 0 }, _expected_count{ 0 }, _next_tic_count{ 0 };
      std::atomic<unsigned> _tic{ 0 };
      void display_tic()
      {
          unsigned tics_needed = unsigned((double(_count) / _expected_count) * 50.0);
          do
          {
              m_os << '*' << std::flush;
          } while (++_tic < tics_needed);
          _next_tic_count = unsigned((_tic / 50.0) * _expected_count);
          if (_count == _expected_count)
          {
              if (_tic < 51) m_os << '*';
              m_os << std::endl;
          }
      }
  };
  
  class StopW {
      std::chrono::steady_clock::time_point time_begin;
   public:
      StopW() {
          time_begin = std::chrono::steady_clock::now();
      }
  
      float getElapsedTimeMicro() {
          std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
          return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
      }
  
      void reset() {
          time_begin = std::chrono::steady_clock::now();
      }
  };
  
  
  
  /*
  * Author:  David Robert Nadeau
  * Site:    http://NadeauSoftware.com/
  * License: Creative Commons Attribution 3.0 Unported License
  *          http://creativecommons.org/licenses/by/3.0/deed.en_US
  */
  
  #if defined(_WIN32)
  #include <windows.h>
  #include <psapi.h>
  
  #elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
  
  #include <unistd.h>
  #include <sys/resource.h>
  
  #if defined(__APPLE__) && defined(__MACH__)
  #include <mach/mach.h>
  
  #elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
  #include <fcntl.h>
  #include <procfs.h>
  
  #elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
  
  #endif
  
  #else
  #error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
  #endif
  
  
  /**
  * Returns the peak (maximum so far) resident set size (physical
  * memory use) measured in bytes, or zero if the value cannot be
  * determined on this OS.
  */
  static size_t getPeakRSS() {
  #if defined(_WIN32)
      /* Windows -------------------------------------------------- */
      PROCESS_MEMORY_COUNTERS info;
      GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
      return (size_t)info.PeakWorkingSetSize;
  
  #elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
      /* AIX and Solaris ------------------------------------------ */
      struct psinfo psinfo;
      int fd = -1;
      if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
          return (size_t)0L;      /* Can't open? */
      if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
          close(fd);
          return (size_t)0L;      /* Can't read? */
      }
      close(fd);
      return (size_t)(psinfo.pr_rssize * 1024L);
  
  #elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
      /* BSD, Linux, and OSX -------------------------------------- */
      struct rusage rusage;
      getrusage(RUSAGE_SELF, &rusage);
  #if defined(__APPLE__) && defined(__MACH__)
      return (size_t)rusage.ru_maxrss;
  #else
      return (size_t) (rusage.ru_maxrss * 1024L);
  #endif
  
  #else
      /* Unknown OS ----------------------------------------------- */
      return (size_t)0L;          /* Unsupported. */
  #endif
  }
  
  
  /**
  * Returns the current resident set size (physical memory use) measured
  * in bytes, or zero if the value cannot be determined on this OS.
  */
  static size_t getCurrentRSS() {
  #if defined(_WIN32)
      /* Windows -------------------------------------------------- */
      PROCESS_MEMORY_COUNTERS info;
      GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
      return (size_t)info.WorkingSetSize;
  
  #elif defined(__APPLE__) && defined(__MACH__)
      /* OSX ------------------------------------------------------ */
      struct mach_task_basic_info info;
      mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
      if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
          (task_info_t)&info, &infoCount) != KERN_SUCCESS)
          return (size_t)0L;      /* Can't access? */
      return (size_t)info.resident_size;
  
  #elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
      /* Linux ---------------------------------------------------- */
      long rss = 0L;
      FILE *fp = NULL;
      if ((fp = fopen("/proc/self/statm", "r")) == NULL)
          return (size_t) 0L;      /* Can't open? */
      if (fscanf(fp, "%*s%ld", &rss) != 1) {
          fclose(fp);
          return (size_t) 0L;      /* Can't read? */
      }
      fclose(fp);
      return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);
  
  #else
      /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
      return (size_t)0L;          /* Unsupported. */
  #endif
  }
  
  static void
  test_vs_recall(
      size_t dataset_size,
      size_t query_size,
      HierarchicalNSW<float> &appr_alg,
      size_t data_dimensionality,
      size_t k_size) {
      vector<size_t> efs;  // = { 10,10,10,10,10 };
      for (int i = k_size; i < 100; i += 10) {
          efs.push_back(i);
      }
      for (int i = 100; i < 1000; i += 50) {
          efs.push_back(i);
      }
      for (int i = 1000; i < 2000; i += 100) {
          efs.push_back(i);
      }
      // efs.push_back(1000);
      std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
      std::chrono::duration<float> duration;
      for (size_t ef : efs) {
          appr_alg.setEf(ef);
          // StopW stopw = StopW();
          start = std::chrono::high_resolution_clock::now();
  
          size_t correct = 0;
          size_t total = query_size * k_size;
  
          for (int i = 0; i < query_size; i++) {
              float * querypoint = (float *)malloc(sizeof(float) * data_dimensionality);
              memcpy(querypoint,&(queries[i*data_dimensionality]), sizeof(float)* data_dimensionality);
  
              std::priority_queue<std::pair<float, long unsigned int>> result = appr_alg.searchKnn(querypoint, k_size);
  
              long unsigned int * search_result = new long unsigned int[k_size];
              for (int j = 0; j < k_size; j++) {
                  search_result[j] = result.top().second;
                  result.pop();
              }
              
              // float kth_dist_gt = result_gt[i][k_size - 1];
  
              // while(result.size()){
              //     if(result.top().first <= kth_dist_gt){
              //         correct += result.size();
              //         break;
              //     } else {
              //         result.pop();
              //     }
              // }
  
              for (int j = 0; j < k_size; j++)
              {
                  for (int z = 0; z < k_size; z++) {
                      if (search_result[j] == result_gt[i][z]) {
                          correct++;
                          break;
                      }
                  }
              }
          }
  
          float recall = 1.0f * correct / total;
  
          end = std::chrono::high_resolution_clock::now();
          duration = end - start;
  
          // float time_ms_per_query = stopw.getElapsedTimeMicro() / query_size * 1000.0f;
  
          cout << ef << "\t" << recall << "\t" << duration.count() * 1000.0f / query_size << " ms\n";
          if (recall > 1.0) {
              cout << recall << "\t" << duration.count() * 1000.0f << " ms\n";
              break;
          }
      }
  }
  
  void sift_test1B() {
  
      char dataset_path[] = "/home/dataset/gist1m-960/gist.bin";
      char query_path[] = "/home/dataset/benchmark/query_gist1m_size100.bin";
      char gt_path[] = "/home/dataset/benchmark/benchmark_gist1m_size100_50knn.bin";
      long int dataset_size = 1000000;
      int data_dimensionality = 960;
  
      int efConstruction = 200;
      int M = 25;
      int query_size = 100;
      int k_size = 50;
  
      load_data(dataset_path, dataset_size, data_dimensionality);
  
      load_query(query_path, query_size, data_dimensionality);
  
      load_groundtruth(gt_path, query_size, k_size);
  
      int memory_before_indexing = getCurrentRSS() / 1000000;
  
      L2Space l2space(data_dimensionality);
  
      std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
      std::chrono::duration<float> index_all;
  
      HierarchicalNSW<float> *appr_alg;
      start = std::chrono::high_resolution_clock::now();
      int number_of_threads = get_nprocs_conf() / 2;
  
      cout << "Building index:\n";
      appr_alg = new HierarchicalNSW<float>(&l2space, dataset_size * 1.1, M, efConstruction);
  
      progress_display pd_hnsw(dataset_size);
  
      float * datapoint = (float *)malloc(sizeof(float) * data_dimensionality);
      memcpy(datapoint,&(rawfile[0]), sizeof(float)* data_dimensionality);
      appr_alg->addPoint((void *) datapoint, (size_t) 0);
      free(datapoint);
  
  #pragma omp parallel for num_threads(number_of_threads)
      for (long i = 1; i < dataset_size; i++){
          float * datapoint = (float *)malloc(sizeof(float) * data_dimensionality);
          memcpy(datapoint,&(rawfile[i*data_dimensionality]), sizeof(float)* data_dimensionality);
          appr_alg->addPoint((void *) datapoint, (size_t) i);
          free(datapoint);
          ++pd_hnsw;
      }
      end = std::chrono::high_resolution_clock::now();
      index_all = end - start;
  
      int memory_after_indexing = getCurrentRSS() / 1000000;
  
      cout << "Start query:\n";
  
      test_vs_recall(dataset_size, query_size, *appr_alg, data_dimensionality, k_size);
  
      std::cout << "The total time of indexing phase is: " << index_all.count() * 1000.0f << "ms." << std::endl;
      std::cout << "The indexing footprint is: " << memory_after_indexing - memory_before_indexing << "MB" << std::endl;
  
      return;
  }
  
  ```

+ Command: We use the tests code provided by the HNSWlib repository (`tests/cpp/main.cpp`), so after compilation, we can directly go to the `hnswlib/build` directory and run the executable file.

+ Notes: HNSW is a strong and representative graph-based baseline that all relevant studies compare to, and is widely used in industry, making it easy for readers to interpret the results.

#### 1.2.5 MIRAGE (Graph-based)

+ Open source code: https://github.com/dsg-uwaterloo/mirage

+ Parameter Settings (default in code): $S=32$, $R=4$, $iter=15$.

+ Demo: We rewrote the `demos/demo_mirage.cpp` file to test MIRAGE's performance comprehensively:  

  ```
  /**
  * Copyright (c) Facebook, Inc. and its affiliates.
  *
  * This source code is licensed under the MIT license found in the
  * LICENSE file in the root directory of this source tree.
  */
  
  #include <iostream>
  #include <chrono>
  #include <cstdio>
  #include <cstdlib>
  #include <random>
  
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <unistd.h>
  
  #include <sys/time.h>
  #include <sys/sysinfo.h>
  #include <atomic>
  #include <algorithm>
  
  #include <faiss/IndexFlat.h>
  #include <faiss/IndexNNDescent.h>
  #include <faiss/IndexMIRAGE.h>
  
  using namespace std::chrono;
  
  static size_t getCurrentRSS() {
  #if defined(_WIN32)
      /* Windows -------------------------------------------------- */
      PROCESS_MEMORY_COUNTERS info;
      GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
      return (size_t)info.WorkingSetSize;
  
  #elif defined(__APPLE__) && defined(__MACH__)
      /* OSX ------------------------------------------------------ */
      struct mach_task_basic_info info;
      mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
      if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
          (task_info_t)&info, &infoCount) != KERN_SUCCESS)
          return (size_t)0L;      /* Can't access? */
      return (size_t)info.resident_size;
  
  #elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
      /* Linux ---------------------------------------------------- */
      long rss = 0L;
      FILE *fp = NULL;
      if ((fp = fopen("/proc/self/statm", "r")) == NULL)
          return (size_t) 0L;      /* Can't open? */
      if (fscanf(fp, "%*s%ld", &rss) != 1) {
          fclose(fp);
          return (size_t) 0L;      /* Can't read? */
      }
      fclose(fp);
      return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);
  
  #else
      /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
      return (size_t)0L;          /* Unsupported. */
  #endif
  }
  
  float euclidean_distance(float * t, float * s, int size) {
      float distance = 0;
      while (size > 0) {
          size--;
          distance += (t[size] - s[size]) * (t[size] - s[size]);
      }  
      return distance;
  }
  
  int main(int argc, char **argv) {
      static char * dataset_path;
      static char * query_path;
      static char * groundtruth_path;
  
  	int dim = 128;
  
  	static long int dataset_size = 1000000;
      static int query_size = 100;
      static int k_size = 50;
  
  	int load_index = 0;
  
  	dim = atoi(argv[1]);
  	dataset_size = atoi(argv[2]);
  	dataset_path = argv[3];
  	query_path = argv[4];
  	groundtruth_path = argv[5];
  
  	// Load data
      dataset_size = dataset_size - 100;
  
      int fread_return;
      
      std::cout << ">>> Loading dataset from: " << dataset_path << std::endl;
  
      FILE * ifile_dataset;
      ifile_dataset = fopen(dataset_path,"rb");
      if (ifile_dataset == NULL) {
          std::cout << "File " << dataset_path << "not found!" << std::endl;
          exit(-1);
      }
  
      std::cout << "Cardinality of dataset is: " << dataset_size << std::endl;
      
      float * dataset = new float[dataset_size*dim];
      fread_return = fread(dataset, sizeof(float), dataset_size*dim, ifile_dataset); 
      fclose(ifile_dataset);
      std::vector<float> database(dataset_size * dim);
      for (int i = 0; i < dataset_size*dim; i++) {
          database[i] = dataset[i];
      }
      delete[] dataset;
  
      size_t RSS_before_indexing = getCurrentRSS() / 1000000; 
  
      struct timeval t_start, t_end;
  	long int index_all = 0, query_all = 0;
  
     // make the index object
     faiss::IndexMirage index(dim);
     index.mirage.S = 32;
     index.mirage.R = 4;
     index.mirage.iter = 15;
     index.verbose = true;
  
     { // populating the index
         gettimeofday(&t_start, NULL);
         index.add(dataset_size, database.data());
         auto end_index = high_resolution_clock::now();
         gettimeofday(&t_end, NULL);
         index_all = (1000000 * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_usec - t_start.tv_usec);
     }
  
     size_t RSS_after_indexing = getCurrentRSS() / 1000000; 
  
     std::cout << "Actual memory usage before indexing is : " << RSS_before_indexing << " MB" << std::endl;
      std::cout << "Actual memory usage after indexing is: " << RSS_after_indexing << " MB" << std::endl;
      std::cout << "The total time of indexing phase is: " << index_all / 1000.0 << "ms." << std::endl;
      std::cout << "Actual memory usage is: " << RSS_after_indexing - RSS_before_indexing << " MB" << std::endl;
  
     // Load query
      std::cout << ">>> Loading query from: " << query_path << std::endl;
      FILE *ifile_query;
      ifile_query = fopen(query_path,"rb");
      if (ifile_query == NULL) {
          std::cout << "File " << query_path << "not found!" << std::endl;
          exit(-1);
      }
      
      float * queries = new float[query_size*dim];
      fread_return = fread(queries, sizeof(float), query_size*dim, ifile_query);
      fclose(ifile_query);
      std::vector<float> querypoints(query_size*dim);
      for (int i = 0; i < query_size*dim; i++) {
          querypoints[i] = queries[i];
      }
      delete[] queries;
  
  
      // Load benchmark
      std::cout << ">>> Loading groundtruth from: " << groundtruth_path << std::endl;
      FILE *ifile_groundtruth;
      ifile_groundtruth = fopen(groundtruth_path,"rb");
      if (ifile_groundtruth == NULL) {
          std::cout << "File " << groundtruth_path << "not found!" << std::endl;
          exit(-1);
      }
  
      long int * result = new long int[query_size*k_size];
      fread_return = fread(result, sizeof(long int), query_size*k_size, ifile_groundtruth);
      fclose(ifile_groundtruth);          
      faiss::idx_t* gt;
      gt = new faiss::idx_t[query_size * k_size];
      for (int i = 0; i < query_size * k_size; i++) {
          gt[i] = result[i];
      }
  
      std::vector<size_t> efSearchs;
      for (int i = k_size; i < 100; i += 10) {
          efSearchs.push_back(i);
      }
      for (int i = 100; i < 1000; i += 50) {
          efSearchs.push_back(i);
      }
      for (int i = 1000; i < 2000; i += 100) {
          efSearchs.push_back(i);
      }
  
      std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
      std::chrono::duration<float> duration;
  
      for (size_t efSearch : efSearchs)
      { // searching the database
          printf("Searching ...\n");
          index.hierarchy.hnsw.efSearch = efSearch;
  
          std::vector<faiss::idx_t> nns(k_size * query_size);
          std::vector<float> dis(k_size * query_size);
  
          start = std::chrono::high_resolution_clock::now();
          index.search(query_size, querypoints.data(), k_size, dis.data(), nns.data());
          end = std::chrono::high_resolution_clock::now();
          duration = end - start;
  
          int recalls = 0;
          for (size_t i = 0; i < query_size; ++i) {
              for (int n = 0; n < k_size; n++) {
                  for (int m = 0; m < k_size; m++) {
                      if (nns[i * k_size + n] == gt[i * k_size + m]) {
                          recalls += 1;
                      }
                  }
              }
          }
  
          float ratio = 0.0f;
          for (int i = 0; i < query_size; i++)
          {
              for (int j = 0; j < k_size; j++)
              {
                  float otbained_square_dist = euclidean_distance(&querypoints[i * dim], &database[nns[i * k_size + j] * dim], dim);
                  float groundtruth_square_dist = euclidean_distance(&querypoints[i * dim], &database[gt[i * k_size + j] * dim], dim);
                  if (groundtruth_square_dist == 0) {
                      ratio += 1.0f;
                  } else {
                      ratio += sqrt(otbained_square_dist) / sqrt(groundtruth_square_dist);
                  }
              }
          }
  
          float recall = 1.0f * recalls / (k_size * query_size);
          std::cout << efSearch << "\t" << recall << "\t" << duration.count() * 1000.0f / query_size << " ms" << "\t" << ratio / (k_size * query_size) << std::endl;
      }
  
  }
  
  ```

+ Command:

  ```
  ./build/demos/demo_mirage $DIMENSIONALITY$ $n$ $PATH_TO_DATASET$ $PATH_TO_QUERY$ $PATH_TO_GROUNDTRUTH$
  ```

#### 1.2.6 SHG (Graph-based)

+ Open source code: https://github.com/gzyhkust/SHG-Index

+ Parameter Settings (default in code): $efConstruction=80$, $M=48$.

+ Demo: We rewrote the `examples/cpp/example_search.cpp` file to test SHG's performance comprehensively:  

  ```
  #include "../../hnswlib/hnswlib.h"
  
  #include <iostream>
  #include <fstream>
  #include <queue>
  #include <chrono>
  #include <sys/sysinfo.h>
  
  
  class progress_display
  {
  public:
      explicit progress_display(
          unsigned long expected_count,
          std::ostream& os = std::cout,
          const std::string& s1 = "\n",
          const std::string& s2 = "",
          const std::string& s3 = "")
          : m_os(os), m_s1(s1), m_s2(s2), m_s3(s3)
      {
          restart(expected_count);
      }
      void restart(unsigned long expected_count)
      {
          //_count = _next_tic_count = _tic = 0;
          _expected_count = expected_count;
          m_os << m_s1 << "0%   10   20   30   40   50   60   70   80   90   100%\n"
              << m_s2 << "|----|----|----|----|----|----|----|----|----|----|"
              << std::endl
              << m_s3;
          if (!_expected_count)
          {
              _expected_count = 1;
          }
      }
      unsigned long operator += (unsigned long increment)
      {
          std::unique_lock<std::mutex> lock(mtx);
          if ((_count += increment) >= _next_tic_count)
          {
              display_tic();
          }
          return _count;
      }
      unsigned long  operator ++ ()
      {
          return operator += (1);
      }
  
      //unsigned long  operator + (int x)
      //{
      //	return operator += (x);
      //}
  
      unsigned long count() const
      {
          return _count;
      }
      unsigned long expected_count() const
      {
          return _expected_count;
      }
  private:
      std::ostream& m_os;
      const std::string m_s1;
      const std::string m_s2;
      const std::string m_s3;
      std::mutex mtx;
      std::atomic<size_t> _count{ 0 }, _expected_count{ 0 }, _next_tic_count{ 0 };
      std::atomic<unsigned> _tic{ 0 };
      void display_tic()
      {
          unsigned tics_needed = unsigned((double(_count) / _expected_count) * 50.0);
          do
          {
              m_os << '*' << std::flush;
          } while (++_tic < tics_needed);
          _next_tic_count = unsigned((_tic / 50.0) * _expected_count);
          if (_count == _expected_count)
          {
              if (_tic < 51) m_os << '*';
              m_os << std::endl;
          }
      }
  };
  
  class StopW {
      std::chrono::steady_clock::time_point time_begin;
   public:
      StopW() {
          time_begin = std::chrono::steady_clock::now();
      }
  
      float getElapsedTimeMicro() {
          std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
          return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
      }
  
      void reset() {
          time_begin = std::chrono::steady_clock::now();
      }
  };
  
  
  
  /*
  * Author:  David Robert Nadeau
  * Site:    http://NadeauSoftware.com/
  * License: Creative Commons Attribution 3.0 Unported License
  *          http://creativecommons.org/licenses/by/3.0/deed.en_US
  */
  
  #if defined(_WIN32)
  #include <windows.h>
  #include <psapi.h>
  
  #elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
  
  #include <unistd.h>
  #include <sys/resource.h>
  
  #if defined(__APPLE__) && defined(__MACH__)
  #include <mach/mach.h>
  
  #elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
  #include <fcntl.h>
  #include <procfs.h>
  
  #elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
  
  #endif
  
  #else
  #error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
  #endif
  
  
  /**
  * Returns the peak (maximum so far) resident set size (physical
  * memory use) measured in bytes, or zero if the value cannot be
  * determined on this OS.
  */
  static size_t getPeakRSS() {
  #if defined(_WIN32)
      /* Windows -------------------------------------------------- */
      PROCESS_MEMORY_COUNTERS info;
      GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
      return (size_t)info.PeakWorkingSetSize;
  
  #elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
      /* AIX and Solaris ------------------------------------------ */
      struct psinfo psinfo;
      int fd = -1;
      if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
          return (size_t)0L;      /* Can't open? */
      if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
          close(fd);
          return (size_t)0L;      /* Can't read? */
      }
      close(fd);
      return (size_t)(psinfo.pr_rssize * 1024L);
  
  #elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
      /* BSD, Linux, and OSX -------------------------------------- */
      struct rusage rusage;
      getrusage(RUSAGE_SELF, &rusage);
  #if defined(__APPLE__) && defined(__MACH__)
      return (size_t)rusage.ru_maxrss;
  #else
      return (size_t) (rusage.ru_maxrss * 1024L);
  #endif
  
  #else
      /* Unknown OS ----------------------------------------------- */
      return (size_t)0L;          /* Unsupported. */
  #endif
  }
  
  
  /**
  * Returns the current resident set size (physical memory use) measured
  * in bytes, or zero if the value cannot be determined on this OS.
  */
  static size_t getCurrentRSS() {
  #if defined(_WIN32)
      /* Windows -------------------------------------------------- */
      PROCESS_MEMORY_COUNTERS info;
      GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
      return (size_t)info.WorkingSetSize;
  
  #elif defined(__APPLE__) && defined(__MACH__)
      /* OSX ------------------------------------------------------ */
      struct mach_task_basic_info info;
      mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
      if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
          (task_info_t)&info, &infoCount) != KERN_SUCCESS)
          return (size_t)0L;      /* Can't access? */
      return (size_t)info.resident_size;
  
  #elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
      /* Linux ---------------------------------------------------- */
      long rss = 0L;
      FILE *fp = NULL;
      if ((fp = fopen("/proc/self/statm", "r")) == NULL)
          return (size_t) 0L;      /* Can't open? */
      if (fscanf(fp, "%*s%ld", &rss) != 1) {
          fclose(fp);
          return (size_t) 0L;      /* Can't read? */
      }
      fclose(fp);
      return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);
  
  #else
      /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
      return (size_t)0L;          /* Unsupported. */
  #endif
  }
  
  
  float euclidean_distance(float * t, float * s, int size) {
      float distance = 0;
      while (size > 0) {
          size--;
          distance += (t[size] - s[size]) * (t[size] - s[size]);
      }  
      return distance;
  }
  
  int main(int argc, char** argv) {
  
  
      // ------------------------------  STEP 1: Loading the data ------------------------------  
      int dim = 128;               // Dimension of the elements
      // int dataset_size = -1;   // Maximum number of elements, should be known beforehand
      int M = 48;                 // Tightly connected with internal dimensionality of the data
                                  // strongly affects the memory consumption
      int ef_construction = 80;  // Controls index search speed/build speed tradeoff
  
      static char * dataset_path;
      static char * query_path;
      static char * groundtruth_path;
  
      static char * dataset_name;
  
  	static int dataset_size = 1000000;
      static int query_size = 100;
      static int k_size = 50;
  
      dim = atoi(argv[1]);
      dataset_name = argv[2];
  	dataset_size = atoi(argv[3]);
  	dataset_path = argv[4];
  	query_path = argv[5];
  	groundtruth_path = argv[6];
  
      dataset_size -= 100;
  
      // ------------------------------  STEP 1.1: Load the dataset ------------------------------
  
      float** data = new float*[dataset_size];
  
      fprintf(stderr, ">>> Loading file: %s\n", dataset_path);
      FILE* ifile_data = fopen(dataset_path, "rb");
      if (!ifile_data) {
          fprintf(stderr, "File %s not found!\n", dataset_path);
          exit(-1);
      }
  
      for (int i = 0; i < dataset_size; i++) {
          data[i] = new float[dim];
          size_t n = fread(data[i], sizeof(float), dim, ifile_data);
          if (n != (size_t)dim) {
              fprintf(stderr, "Read error at data %d\n", i);
              exit(-1);
          }
      }
  
      fclose(ifile_data);
  
      // ------------------------------  STEP 1.2: Load the query ------------------------------
      float** queries = new float*[query_size];
  
      fprintf(stderr, ">>> Loading file: %s\n", query_path);
      FILE* ifile_query = fopen(query_path, "rb");
      if (!ifile_query) {
          fprintf(stderr, "File %s not found!\n", query_path);
          exit(-1);
      }
  
      for (int i = 0; i < query_size; i++) {
          queries[i] = new float[dim];
          size_t n = fread(queries[i], sizeof(float), dim, ifile_query);
          if (n != (size_t)dim) {
              fprintf(stderr, "Read error at query %d\n", i);
              exit(-1);
          }
      }
  
      fclose(ifile_query);
  
      // ------------------------------  STEP 1.3: Load the groundtruth ------------------------------
      long int** result_gt = new long int*[query_size];
  
      fprintf(stderr, ">>> Loading groundtruth: %s\n", groundtruth_path);
  
      FILE* ifile_groundtruth = fopen(groundtruth_path, "rb");
      if (!ifile_groundtruth) {
          fprintf(stderr, "Groundtruth file %s not found!\n", groundtruth_path);
          exit(-1);
      }
  
      for (int i = 0; i < query_size; i++) {
          result_gt[i] = new long int[k_size];
          size_t n = fread(result_gt[i], sizeof(long int),
                          k_size, ifile_groundtruth);
          if (n != (size_t)k_size) {
              fprintf(stderr, "Read error at query %d\n", i);
              exit(-1);
          }
      }
  
      fclose(ifile_groundtruth);
  
      // ------------------------------  STEP 2: Initial the index ------------------------------ 
  
      std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
      std::chrono::duration<float> index_all, query_all;
  
      int number_of_threads = get_nprocs_conf() / 2;
  
      int memory_before_indexing = getCurrentRSS() / 1000000;
  
      start = std::chrono::high_resolution_clock::now();
      // Initing index
      hnswlib::L2Space space(dim);
      hnswlib::HEDS<float>* alg_hnsw = new hnswlib::HEDS<float>(&space, dim, dataset_size + query_size, M, ef_construction);
  
      // ------------------------------  STEP 3: Build the index ------------------------------ 
      Performance per;
      Timer t;
      progress_display pd_hnsw(dataset_size);
  
      t.restart();
      // #pragma omp parallel for num_threads(number_of_threads)
      for(int i = 0; i< dataset_size; ++i){
          alg_hnsw->addDataPoint(data[i], i, 1, per);
          ++pd_hnsw;
      }
  
      per.setTimeBuildindex(t.elapsed());
  
      std::string temp = dataset_name;
      std::string construction_res_path = "/home/jiuqi/SHG-Index/" + temp + "_construction.txt";
      std::ofstream construction_res(construction_res_path);
  
      std::cout << "Build Index Time : " << per.getTimeBuildindex() << " [s]" << std::endl;
      construction_res << "Build Index Time : " << per.getTimeBuildindex() << " [s]" << std::endl;
      std::cout << "Preprocessing Time: " << per.getTimePreprocessing() << " [s]" << std::endl;
      construction_res << "Preprocessing Time: " << per.getTimePreprocessing() << " [s]" << std::endl;
  
      t.restart();
      //std::cout << "max Level: " << *std::max_element(alg_hnsw->element_levels_.begin(),alg_hnsw->element_levels_.end()) << " min Level: " << *std::min_element(alg_hnsw->element_levels_.begin(),alg_hnsw->element_levels_.end()) << " maxFix" << alg_hnsw->maxFixLevel_ << std::endl;
      alg_hnsw->buildShortcuts(dataset_size);
      per.setTimeShortcut(t.elapsed());
  
      // std::cout << "Tree height: " << alg_hnsw->maxlevel_ << std::endl;
      // std::cout << "Num of elements: " << dataset_size << std::endl;
      //std::cout << "Build shortcut time: " << per.getTimeShortcut() << std::endl;
      construction_res << "Levels of HNSW: " << alg_hnsw->maxlevel_ << std::endl;
      construction_res << "Build shortcut time: " << per.getTimeShortcut() << std::endl;
      //std::cout << "Memory cost: " << (alg_hnsw->indexFileSize(dataset_size))/ (1024.0 * 1024.0) << std::endl;
      construction_res << "Memory cost: " << (alg_hnsw->indexFileSize(dataset_size))/ (1024.0 * 1024.0) << std::endl;
      construction_res << "Memory cost of shortcuts: " << (alg_hnsw->Shortcuts.size_in_bytes()) << std::endl;
      construction_res.close();
  
  
      end = std::chrono::high_resolution_clock::now();
      index_all = end - start;
  
      int memory_after_indexing = getCurrentRSS() / 1000000;
  
  
      // ------------------------------  STEP 4: Search ------------------------------
      int correct = 0;
      int total = query_size * k_size;
      // float ratio = 0.0f;
  
      alg_hnsw->resultsProcessing.assign(dataset_size * (alg_hnsw->maxlevel_+1),-1);
  
      std::vector<int> efSearchs;  // = { 10,10,10,10,10 };
      for (int i = k_size; i < 100; i += 10) {
          efSearchs.push_back(i);
      }
      for (int i = 100; i < 1000; i += 50) {
          efSearchs.push_back(i);
      }
      for (int i = 1000; i < 2000; i += 100) {
          efSearchs.push_back(i);
      }
      for (int i = 2000; i < 4000; i += 200) {
          efSearchs.push_back(i);
      }
  
      for (int efSearch : efSearchs) {
          correct = 0;
          // ratio = 0.0f;
          query_all = std::chrono::duration<float>::zero();
  
          alg_hnsw->setEf(efSearch);
  
          for(int i = 0; i< query_size; i++){
              alg_hnsw->addDataPoint(queries[i], dataset_size+i, -1, per);
  
              Query query(dataset_size+i, k_size);
              t.restart();
  
              start = std::chrono::high_resolution_clock::now();
              std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
              result = alg_hnsw->searchKnnShortcuts(query);
              end = std::chrono::high_resolution_clock::now();
              query_all += end - start;
  
              std::vector<int> queryResults;
              for (int j = 0; j < k_size; j++) {
                  std::pair<float, int> rez = result.top();
                  queryResults.push_back(rez.second);
                  result.pop();     
              }
  
              std::vector<int> queryGroundtruth(result_gt[i],result_gt[i]+k_size);
  
              for(int id: queryResults){
                  auto it = std::find(queryGroundtruth.begin(), queryGroundtruth.end(), id);
                  if (it != queryGroundtruth.end()){ correct++ ;} 
              }
  
              // for (int j = 0; j < k_size; j++)
              // {
              //     float otbained_square_dist = euclidean_distance(queries[i], data[queryResults[j]], dim);
              //     float groundtruth_square_dist = euclidean_distance(queries[i], data[result_gt[i][j]], dim);
              //     if (groundtruth_square_dist == 0) {
              //         ratio += 1.0f;
              //     } else {
              //         ratio += sqrt(otbained_square_dist) / sqrt(groundtruth_square_dist);
              //     }
              // }
  
              std::fill(alg_hnsw->resultsProcessing.begin(), alg_hnsw->resultsProcessing.end(),-1);
          }
          
  
          float recall = 1.0f * correct / total;
  
          // std::cout << efSearch << "\t" << recall << "\t" << query_all.count() * 1000.0f / query_size << " ms" << "\t" << ratio / (query_size * k_size) << std::endl;
  
          std::cout << efSearch << "\t" << recall << "\t" << query_all.count() * 1000.0f / query_size << " ms"<< std::endl;
      }
  
  
      std::cout << "The total time of indexing phase is: " << index_all.count() * 1000.0f << "ms." << std::endl;
      std::cout << "The indexing footprint is: " << memory_after_indexing - memory_before_indexing << "MB" << std::endl;
  
      delete alg_hnsw;
      return 0;
  }
  ```

+ Command: 

  ```
  ./build/example_search $DIMENSIONALITY$ $DATASET_NAME$ $n$ $PATH_TO_DATASET$ $PATH_TO_QUERY$ $PATH_TO_GROUNDTRUTH$
  ```

+ Notes: Although SHG is optimized based on HNSW, it does not support parallel index building, possibly because view errors can occur when building shortcuts in parallel.

## 2. How each experiment was conducted

### 2.1 Self-evaluation of TaCo (Section 5.2)

#### 2.1.1 TaCo vs. SC-Linear (Section 5.2.1)

We compare the query performance between TaCo and SC-Linear under the same parameter settings that are independent of subspace ($\alpha = 0.05$, $\beta = 0.005$). For subspace-related parameter settings, since TaCo has a dimensionality reduction effect while SC-Linear does not, each selects its optimal parameters ($N_s$ and $s$) for different datasets (Table 3 for TaCo; $N_s=8$, $s=32(DEEP1M)/16(SIFT10M)$ for SC-Linear).

#### 2.1.2 Scalable Dynamic Activation vs. Dynamic Activation (Section 5.2.2)

We compare the efficiency of the Scalable Dynamic Activation and the original Dynamic Activation algorithms on the SIFT10M dataset under varying numbers of clusters $K$ and collision ratios $\alpha$. $K$ and $\alpha$ are key parameters affecting the workload of query on Inverted Multi-Indexes (IMI).

It is important to note that the two algorithms return exactly the same results, so we measure the efficiency of the two algorithms by recording the time required to query the IMI index in the TaCo method.

#### 2.1.3 Parameter study on $N_s$ and $s$ (Section 5.2.3)

We evaluate parameters $N_s$ (number of subspaces) and $s$ (subspace dimensionality), which critically influence the indexing and query performance of TaCo. To avoid being affected by other parameters, we fix the other parameters: $\alpha = 0.05$, $\beta = 0.005$, $K=50^2$. 

For TaCo, $N_s*s$ can be less than $d$ (data dimensionality), which means that TaCo has a dimensionality reduction effect, and the higher the dimensionality reduction ratio ($1-\frac{N_s \cdot s}{d}$), the lower the workload required to index construction and query answering. 

### 2.2 TaCo vs. Subspace Collision Methods (Section 5.3)

#### 2.2.1 Indexing performance (Section 5.3.1)

Index performance includes two aspects: indexing time and index memory footprint.

Owing to the use of subspace-oriented data transformation in both TaCo and SuCo-DT, these two methods achieve same indexing performance. Similarly, SuCo-QS, SuCo-CS, SuCo, which do not incorporate this transformation, also exhibit same indexing performance.

For TaCo, SuCo-DT, SuCo-QS, SuCo-CS, and SuCo, we use the system's memory access tool (`/proc/self/statm`) to returns the resident set size (physical memory use) measured in bytes before and after the index construction. In addition, we use C++'s built-in clock function to record the time before and after the index construction. The index memory footprint and indexing time are obtained by measuring the difference. 

#### 2.2.2 Query performance (Section 5.3.2)

Query performance also includes two aspects: efficiency and accuracy. Queries per second (QPS) can reflect the query efficiency. Recall and mean relative error (MRE) can reflect the query accuracy.  For all methods, there is a trade-off between query efficiency and accuracy, that is, spending longer query time can get higher quality results (higher recall/lower MRE), while excessive pursuit of query efficiency will affect the quality of the results (lower recall/higher MRE). Therefore, it is necessary to evaluate query performance by combining efficiency and accuracy.

For TaCo, SuCo-DT, SuCo-QS, SuCo-CS, and SuCo, once the index is built, two key parameters affect their query performance: the collision ratio $\alpha$ and the re-rank ratio $\beta$. We dynamically change the values of two parameters ($\alpha \in \left[0.01, 0.1\right]$, $\beta \in \left[0.001, 0.05\right]$) to obtain the balance point between efficiency and accuracy under different query conditions, and draw the Recall-QPS and MRE-QPS curves shown in Figure 8.

#### 2.2.3 Performance under different $k$ (Section 5.3.3)

In $k$-ANNS tasks, the parameter $k$ denotes the number of nearest neighbors to be retrieved. A larger $k$ indicates a more difficult query task. The ability of a $k$-ANNS method to maintain stable performance across varying values of $k$ is an important aspect of its scalability.

For TaCo, SuCo-DT, SuCo-QS, SuCo-CS, and SuCo, we fix $\alpha = 0.05$ and $\beta = 0.005$, and then adjust $k \in [1,100]$. Since changing $k$ has little impact on query time, and has no impact on indexing time, we only report the results on query accuracy, i,e., recall and MRE, as shown in Figure 9.

### 2.3 TaCo vs. Non-Subspace Collision methods (Section 5.4)

#### 2.3.1 Indexing performance (Section 5.4.1)

As described in 2.2.1, index performance includes two aspects: indexing time and index memory footprint.

For DET-LSH, IMI-OPQ, IVF-RaBitQ, HNSW, MIRAGE, and SHG we set them to their optimal parameters given in 1.2.1~1.2.6. 

Then, we use the system's memory access tool (`/proc/self/statm`) to returns the resident set size (physical memory use) measured in bytes before and after the index construction. In addition, we use C++'s built-in clock function to record the time before and after the index construction. The index memory footprint and indexing time are obtained by measuring the difference.

#### 2.3.2 Query performance (Section 5.4.2)

As described in 2.2.2, query performance encompasses both efficiency and accuracy, and both aspects need to be evaluated comprehensively.

For DET-LSH, IMI-OPQ, IVF-RaBitQ, HNSW, MIRAGE, and SHG, we dynamically change their query parameters (given in 1.2.1~1.2.6) and probe the intermediate states of the query, thereby obtaining the balance point between efficiency and accuracy under different query conditions, and draw the Recall-QPS and MRE-QPS curves shown in Figure 11.

#### 2.3.3 Overall Evaluation (Section 5.4.3)

Given the considerable differences in indexing and query performance across different categories of ANNS methods, it is essential to conduct a comprehensive evaluation that integrates both indexing and query metrics.

We achieve an overall evaluation of indexing and query efficiency by counting cumulative query costs starting from the indexing time, as shown in Figure 12. 

To ensure fairness in the comparison, we set the same recall target for all comparison methods (e.g., 0.8 or 0.9), thus measuring indexing and query efficiency under the premise of identical query accuracy. We need to probe each method to obtain the average query time under a fixed recall.