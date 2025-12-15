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

#### 1.1.2 SuCo-DT & SuCo-QS (Ablation baseline)

+ Open source code: https://anonymous.4open.science/r/TaCo-9AB5

+ Parameter Settings: $N_s\in [4,10]$, $s\in [6,12]$, $\alpha \in \left[0.01, 0.1\right]$, $\beta \in \left[0.001, 0.05\right]$

+ Command (Detailed explanations are available on GitHub): 

  ```
  ./taco --dataset-path $PATH_TO_DATASET$ --query-path $PATH_TO_QUERY$ --projected-query-path $PATH_TO_PROJECTED_QUERY$ --groundtruth-path $PATH_TO_GROUNDTRUTH$ --dataset-size $n$ --query-size $QUERY_SIZE$ --k-size $k$ --data-dimensionality $DIMENSIONALITY$ --subspace-dimensionality $SUBSPACE_DIMENSIONALITY$ --subspace-num $SUBSPACE_NUMBER$ --candidate-ratio $beta$ --collision-ratio $alpha$ --kmeans-num-centroid $K$ --kmeans-num-iters $ITERATION$ --index-path $PATH_TO_INDEX$ --load-index
  ```

+ Notes: Since the code for our proposed TaCo method is developed based on SuCo, both SuCo-DT and SuCo-QS, which serve as the ablation baseline, can be obtained by modifying the TaCo open-source code. Specifically:
  + SuCo-DT: in the query.cpp file, comment out 92-98 lines, uncomment the 90, 100-108, 113-120 lines; invoke the dynamic_activate function at line 38 instead of the scalable_dynamic_activate function at line 39.
  + SuCo-QS: comment out the data transformation strategy in the main.cpp file (163-220 lines), restore this part to the same logic as SuCo.

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

#### 1.2.1 DET-LSH (LSH-based strong baseline)

+ Open source code: https://github.com/WeiJiuQi/DET-LSH

+ Parameter Settings: $L=4$, $K=16$, $\beta \in \left[0.005,0.2\right]$, $c=1.5$, $--cpu-type = 81$, $--queue-number = 4$

+ Command (Detailed explanations are available on GitHub): 

  ```
  cd ./bin
  ./DETLSH --dataset $PATH_TO_DATASET$ --leaf-size $MAX_SIZE$ --dataset-size $n$  --queries $PATH_TO_QUERY$ --queries-size $QUERY_SIZE$ --sample-size $SAMPLE_SIZE$ --sample-type $SAMPLE_TYPE$ --k-size $k$ --data-dimensionality $DIMENSIONALITY$ --l-size $L$ --search-radius $r$ --max-candidate-size $beta*n$ --benchmark $PATH_TO_BENCHMARK$ --cpu-type $ct$ --queue-number $qn$
  ```

+ Notes: The original DET-LSH paper did not provide a parallelized version design, but a parallelized version is provided in its GitHub code. We tested the parallelized version of the code and found that the algorithm achieved its best performance when the number of queues (--cpu-type) was half the number of threads (--queue-number). 

#### 1.2.2 IMI-OPQ (Quantization-based strong baseline)

+ Open source code: the FAISS library (https://github.com/facebookresearch/faiss)

+ Parameter Settings

  ```
  OPQ16_64,IMI2x8,PQ8+16
  ```

+ Command: We use the demo test code provided by the FAISS repository, so after compilation, we can directly go to the `faiss/build/demos` directory and run the executable file. Due to the loss caused by quantization error, this method struggles to achieve high recall on some datasets. Therefore, we provide a rerank function for IMI-OPQ, enabling it to strike a balance between efficiency and accuracy.

+ Notes: IMI-OPQ has appeared as a strong baseline in previous papers. RabitQ (SIGMOD 2024 & 2025), as the state-of-the-art scalar quantization method, achieves unbiased estimation and has been widely used in the industry. Therefore, we would like to add RabitQ into our non-subspace collision-based methods experiments in the revised version of our paper.

#### 1.2.3 HNSW (Graph-based strong baseline)

+ Open source code: the HNSWlib library (https://github.com/nmslib/hnswlib)
+ Parameter Settings: $efConstruction=200$, $M=25$, $efSearch \in \left[300,3000\right]$.
+ Command: We use the tests code provided by the HNSWlib repository (`tests/cpp/main.cpp`), so after compilation, we can directly go to the `hnswlib/build` directory and run the executable file.
+ Notes: HNSW is a strong and representative graph-based baseline that all relevant studies compare to, and is widely used in industry, making it easy for readers to interpret the results. Graph-based methods suffer from the issue of query efficiency being limited by random memory access. SymphonyQG (SIGMOD 2025) combines quantization-based and graph-based methods to address the random memory access issue, belonging to the category of hybrid indexing methods. This is a type of comparative method that our experiments currently lack. Therefore, we would like to add SymphonyQG  into our non-subspace collision-based methods experiments in the revised version of our paper. We believe it will make the experiment more complete.

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

Owing to the use of subspace-oriented data transformation in both TaCo and SuCo-DT, these two methods achieve same indexing performance. Similarly, SuCo-QS and SuCo, which do not incorporate this transformation, also exhibit same indexing performance.

For TaCo, SuCo-DT, SuCo-QS, and SuCo, we use the system's memory access tool (`/proc/self/statm`) to returns the resident set size (physical memory use) measured in bytes before and after the index construction. In addition, we use C++'s built-in clock function to record the time before and after the index construction. The index memory footprint and indexing time are obtained by measuring the difference. 

#### 2.2.2 Query performance (Section 5.3.2)

Query performance also includes two aspects: efficiency and accuracy. Queries per second (QPS) can reflect the query efficiency. Recall and mean relative error (MRE) can reflect the query accuracy.  For all methods, there is a trade-off between query efficiency and accuracy, that is, spending longer query time can get higher quality results (higher recall/lower MRE), while excessive pursuit of query efficiency will affect the quality of the results (lower recall/higher MRE). Therefore, it is necessary to evaluate query performance by combining efficiency and accuracy.

For TaCo, SuCo-DT, SuCo-QS, and SuCo, once the index is built, two key parameters affect their query performance: the collision ratio $\alpha$ and the re-rank ratio $\beta$. We dynamically change the values of two parameters ($\alpha \in \left[0.01, 0.1\right]$, $\beta \in \left[0.001, 0.05\right]$) to obtain the balance point between efficiency and accuracy under different query conditions, and draw the Recall-QPS and MRE-QPS curves shown in Figure 7.

#### 2.2.3 Performance under different $k$ (Section 5.3.3)

In $k$-ANNS tasks, the parameter $k$ denotes the number of nearest neighbors to be retrieved. A larger $k$ indicates a more difficult query task. The ability of a $k$-ANNS method to maintain stable performance across varying values of $k$ is an important aspect of its scalability.

For TaCo, SuCo-DT, SuCo-QS, and SuCo, we fix $\alpha = 0.05$ and $\beta = 0.005$, and then adjust $k \in [1,100]$. Since changing $k$ has little impact on query time, and has no impact on indexing time, we only report the results on query accuracy, i,e., recall and MRE, as shown in Figure 8.

### 2.3 TaCo vs. Non-Subspace Collision methods (Section 5.4)

#### 2.3.1 Indexing performance (Section 5.4.1)

As described in 2.2.1, index performance includes two aspects: indexing time and index memory footprint.

For DET-LSH, IMI-OPQ, and HNSW, we set them to their optimal parameters given in 1.2.1, 1.2.2, and 1.2.3. 

Then, we use the system's memory access tool (`/proc/self/statm`) to returns the resident set size (physical memory use) measured in bytes before and after the index construction. In addition, we use C++'s built-in clock function to record the time before and after the index construction. The index memory footprint and indexing time are obtained by measuring the difference.

#### 2.3.2 Query performance (Section 5.4.2)

As described in 2.2.2, query performance encompasses both efficiency and accuracy, and both aspects need to be evaluated comprehensively.

For DET-LSH, IMI-OPQ, and HNSW, we dynamically change their query parameters (given in 1.2.1, 1.2.2, and 1.2.3) and probe the intermediate states of the query, thereby obtaining the balance point between efficiency and accuracy under different query conditions, and draw the Recall-QPS and MRE-QPS curves shown in Figure 10.

#### 2.3.3 Overall Evaluation (Section 5.4.3)

Given the considerable differences in indexing and query performance across different categories of ANNS methods, it is essential to conduct a comprehensive evaluation that integrates both indexing and query metrics.

We achieve an overall evaluation of indexing and query efficiency by counting cumulative query costs starting from the indexing time, as shown in Figure 11. 

To ensure fairness in the comparison, we set the same recall target for all comparison methods (e.g., 0.8 or 0.9), thus measuring indexing and query efficiency under the premise of identical query accuracy. We need to probe each method to obtain the average query time under a fixed recall.