#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include "../../../include/cudaDMA.h"
#include <chrono>
#include <string>
#include <tuple>

#define DMA_THREADS_COMPUTE 32
#define DMA_THREADS_DMA 32

using namespace std;


__global__ void matmul(float *A, float *B, float *C, int m, int n, int p){
    __shared__ int B_shared[32][32];
    __shared__ int A_shared[32][32];

    cudaDMASequential<true, 4, 4*32> dma_ld_0(1, DMA_THREADS_DMA, DMA_THREADS_COMPUTE, DMA_THREADS_COMPUTE);
    int iterations = n/32;
    int x = blockIdx.x*32;
    int y = blockIdx.y*32;

    if(threadIdx.x < DMA_THREADS_COMPUTE){
        dma_ld_0.start_async_dma();

        for(int i=0 ; i < iterations ; i++){
            dma_ld_0.wait_for_dma_finish();
            for(int x_ind=0 ; x_ind < 32 ; x_ind++){
                int res = 0;
                for(int j = 0 ; j < 32 ; j++){
                    res += A_shared[x_ind][j]*B_shared[j][threadIdx.x];
                }
                C[(x+x_ind)*n + y + threadIdx.x] += res;
            }
            dma_ld_0.start_async_dma();
        }
    }
    else if(dma_ld_0.owns_this_thread()){
        dma_ld_0.wait_for_dma_start();
        int ind = threadIdx.x - DMA_THREADS_COMPUTE;
        for(int i = 0 ; i < iterations ; i++){
            for(int j = 0 ; j < 32 ; j++){
                B_shared[ind][j] = B[i*32*n + y + ind*n + j];
                A_shared[ind][j] = A[i*32 + x*n + ind*n + j];
            }
            dma_ld_0.finish_async_dma();
            dma_ld_0.wait_for_dma_start();
        }
    }
}

tuple<float *, int, int> readCSV(const string &filename) {
    // Open the CSV file
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "File Error" << endl;
        return {nullptr, 0, 0};
    }

    vector<float> data;
    string line;
    int dim = 0;
    int rows = 0;
    int padding = 0;
    int padded_dim = 0;

    // Read the CSV file line by line
    while (getline(file, line)) {
        stringstream ss(line);
        string value;

        // cout << "jjjjj"<< endl;
        // cout << line << endl;
        // cout << "jjjjj"<< endl;

        // Split the line by commas, convert to float and store in the vector
        while (getline(ss, value, ',')) {
            data.push_back(stof(value));
        }

        if (dim == 0) {
            dim = data.size();
            padding = 32 - (dim % 32);
            if (padding == 32) {
                padding = 0;
            }
            padded_dim = dim + padding;
        }

        // Do the required padding
        if (padding == 0) {
            padding = 32 - (data.size() % 32);
        }

        if (padding != 32) {
            for (int i = 0; i < padding; ++i) {
                data.push_back(0.0f);
            }
        }

        rows++;
    }

    file.close();

    // cout << "Rows: " << rows << ", Columns: " << dim << endl;
    // for(size_t i = 0; i < data.size(); i++){
    //     cout << data[i] << " ";
    // }

    // convert the vector to a float array
    float *arr = new float[data.size()];
    for (size_t i = 0; i < data.size() ; i++)
    {
        arr[i] = data[i];
    }

    // return the float array, number of rows and columns
    return {arr, rows, padded_dim};
}

// void printMat(float *mat, int m, int n){
//     for (int i = 0; i < m; i++){
//         for (int j = 0; j < n; j++){
//             cout << mat[i*n + j] << " ";
//         }
//         cout << endl;
//     }
// }

int main(int argc, char* argv[]){

    float *H_c, *D_a, *D_b, *D_c;
    int m, n, p;
    std::chrono::time_point<std::chrono::system_clock> start,  end;

    auto [A_data, rows_a, cols_a] = readCSV(argv[1]);
    
    
    auto [B_data, rows_b, cols_b] = readCSV(argv[2]);
    // for(int i = 0 ; i < rows_a; i++){
    //     for(int j = 0 ; j < cols_a ; j++){
    //         int temp = i*cols_a+j;
    //         cout << B_data[temp] << " ";
    //     }
    //     cout << endl;
    // }

    
    if (!A_data || !B_data) {
        cerr << "Error reading input files" << endl;
        return 1;
    }
    
    // if (cols_a/32 != rows_b) {
    //     cerr << "Matrix dimension mismatch: A has " << cols_a 
    //     << " cols, B has " << rows_b << " rows" << endl;
    //     return 1;
    // }
    
    m = rows_a;
    n = cols_a;
    p = cols_b;

    // printMat(A_data, rows_a, cols_a);
    // printMat(B_data, rows_b, cols_b);

    // cout << "Done" << endl;
    // return 0;

    // for(int i = 0 ; i < n ; i++){
    //     for(int j = 0 ; j < p ; j++){
    //         int temp = i*n+j;
    //         H_b[i*p+j] = temp;
    //     }
    // }

    cudaMalloc(&D_c, sizeof(float)*m*p);
    H_c = (float*)malloc(sizeof(float)*m*p);
    for(int i=0;i<m*p;i++){
        H_c[i] = 0;
    }
    cudaMemcpy(D_c, H_c, m*p*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc( &D_a,  n*m*sizeof(float));
    cudaMemcpy( D_a,  A_data,  n*m*sizeof(float),  cudaMemcpyHostToDevice);

    cudaMalloc( &D_b,  n*p*sizeof(float));
    cudaMemcpy( D_b,  B_data,  n*p*sizeof(float),  cudaMemcpyHostToDevice);
    
    start = std::chrono::system_clock::now();
    matmul<<<dim3(n/32, n/32, 1), dim3(DMA_THREADS_COMPUTE+DMA_THREADS_DMA, 1, 1)>>>(D_a, D_b, D_c, m, n, p);
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Error Name: %s\n",  cudaGetErrorName(err));
        printf("Error: %s\n",  cudaGetErrorString(err));
    }

    std::chrono::duration<double> elapsed_seconds = end - start; 

    cudaMemcpy( H_c,  D_c,  m*p*sizeof(int),  cudaMemcpyDeviceToHost);
    // for(int i = 0 ; i < m ; i++){
    //     for(int j = 0 ; j < p ; j++){
    //         int temp = i*p+j;
    //         cout << H_c[temp] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << "Done" << endl;
    printf("%f\n", elapsed_seconds.count());

    return 0;
}
