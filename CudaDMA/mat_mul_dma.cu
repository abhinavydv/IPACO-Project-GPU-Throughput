#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <tuple>
#include <ctime>
#include <stdio.h>
#include "./CudaDMA/include/cudadma.h"

using namespace std;

#define WARP_SIZE 32
#define VEC_ELMTS 32
#define COMPUTE_THREADS 32
#define DMA_THREADS_SEQ 4
#define DMA_THREADS_STRD 4



// GPU function to multiply two matrices `mat1` and `mat2`
// Tiling is done using shared memory to reduce global memory accesses
__global__ void matmul(float *d_mat1, float *d_mat2, float *d_result, int m1, int paddedN, int m2) {
    __shared__ float buff [ VEC_ELMTS ];
    __shared__ float mat [ VEC_ELMTS ][ COMPUTE_THREADS ];

    cudaDMASequential<sizeof(float) * VEC_ELMTS / DMA_THREADS_SEQ> dma_ld_0(1, DMA_THREADS_SEQ, COMPUTE_THREADS,
             COMPUTE_THREADS, sizeof(float) * VEC_ELMTS);

    cudaDMAStrided<sizeof(float) * VEC_ELMTS>
    dma_ld_1(2, DMA_THREADS_STRD, COMPUTE_THREADS,
             COMPUTE_THREADS + DMA_THREADS_SEQ,
             sizeof(float) * COMPUTE_THREADS,
             VEC_ELMTS, sizeof(float) * n,
             sizeof(float) * COMPUTE_THREADS);

    int ind = blockIdx.x * COMPUTE_THREADS + threadIdx.x;

    if ( threadIdx .x < COMPUTE_THREADS ) {
        dma_ld_0 . start_async_dma ();
        dma_ld_1 . start_async_dma ();
        float res = 0. f ;
        for ( int i =0; i < n; i += VEC_ELMTS ) {
            dma_ld_0 . wait_for_dma_finish ();
            dma_ld_1 . wait_for_dma_finish ();
            for ( int j =0; j < VEC_ELMTS ; j ++) {
                res += mat [j ][ threadIdx .x ]* buff [ j ];
            }
            dma_ld_0 . start_async_dma ();
            dma_ld_1 . start_async_dma ();
        }
        ind = blockIdx . x* COMPUTE_THREADS + threadIdx .x ;
        if ( ind < n ) 
            y[ ind ] = alpha * res ;
    }
    else if ( dma_ld_0 . owns_this_thread ()) {
        dma_ld_0 . wait_for_dma_start ();
        for ( int idx =0; idx < n; idx += VEC_ELMTS ) {
            dma_ld_0 . execute_dma (x , buff );
            dma_ld_0 . finish_async_dma ();
            dma_ld_0 . wait_for_dma_start ();
            x += VEC_ELMTS ;
        }
    }
    else if ( dma_ld_1 . owns_this_thread ()) {
        dma_ld_1 . wait_for_dma_start ();
        for ( int idx =0; idx < n; idx += VEC_ELMTS ) {
            dma_ld_1 . execute_dma (
            A+ idx *m+ blockIdx . x* COMPUTE_THREADS , mat );
            dma_ld_1 . finish_async_dma ();
            dma_ld_1 . wait_for_dma_start ();
        }
    }
}

// Function to read a CSV file and return the data as a float array
// The function also returns the number of rows and columns in the CSV file
// The function pads the data to make sure the number of columns is a multiple of WARP_SIZE
tuple<float *, int, int> readCSV(const string &filename)
{
    // Open the CSV file
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return {nullptr, 0, 0};
    }

    vector<float> data;
    string line;
    int dim = 0;
    int rows = 0;
    int padding = 0;

    // Read the CSV file line by line
    while (getline(file, line))
    {
        stringstream ss(line);
        string value;

        // Split the line by commas, convert to float and store in the vector
        while (getline(ss, value, ','))
        {
            data.push_back(stof(value));
        }

        if (dim == 0)
        {
            dim = data.size();
        }

        // Do the required padding
        if (padding == 0)
        {
            padding = WARP_SIZE - (data.size() % WARP_SIZE);
        }

        if (padding != WARP_SIZE)
        {
            for (int i = 0; i < padding; ++i)
            {
                data.push_back(0.0f);
            }
        }

        rows++;
    }

    file.close();

    // convert the vector to a float array
    float *arr = new float[data.size()];
    for (size_t i = 0; i < data.size(); ++i)
    {
        arr[i] = data[i];
    }

    // return the float array, number of rows and columns
    return {arr, rows, dim};
}

void writeCSV(const string &filename, float *data, int rows, int cols)
{
    ofstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    // Write the data to the CSV file
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            file << data[i * cols + j];
            if (j < cols - 1)
                file << ",";
        }
        file << endl;
    }

    file.close();
}

void printMat(float *mat, int m, int n){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            cout << mat[i*n + j] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char *argv[])
{
    // write your code here
    //  input validation
    if (argc != 4)
    {
        cout << "usage: " << argv[0] << " <path to pointSetA> <path to pointSetB> <output>" << endl;
        return -1;
    }

    // Required initializations
    float *mat1;
    float *mat2;
    float *d_mat1;
    float *d_mat2;
    float *result;
    float *d_result;
    int m1, n1, m2, n2, padX, paddedN;
    double elapsed_time = 0;

    // Load the pointsetA.csv into memory
    auto csvData = readCSV(argv[1]);
    mat1 = get<0>(csvData);
    m1 = get<1>(csvData);
    n1 = get<2>(csvData);

    // Load the pointsetB.csv into memory
    csvData = readCSV(argv[2]);
    mat2 = get<0>(csvData);
    m2 = get<1>(csvData);
    n2 = get<2>(csvData);

    // Check if the number of columns in pointSetA and pointSetB are the same
    if (n1 != n2) {
        cout << "Error: The number of columns in pointSetA and pointSetB must be the same." << endl;
        return -1;
    }

    // Calculate the padded size of the point sets
    if (n1 % WARP_SIZE != 0)
        padX = WARP_SIZE - (n1 % WARP_SIZE);
    else
        padX = 0;

    paddedN = n1 + padX;

    // printMat(mat1, m1, paddedN);
    // printMat(mat2, m2, paddedN);

    // Allocate memory for the result
    result = new float[m1 * m2];

    // Do the required cuda malloc's
    cudaMalloc(&d_mat1, m1 * paddedN * sizeof(float));
    cudaMalloc(&d_mat2, m2 * paddedN * sizeof(float));
    cudaMalloc(&d_result, m1 * m2 * sizeof(float));

    double start_time = (double)clock() / CLOCKS_PER_SEC;
    // Copy the dataset to GPU
    cudaMemcpy(d_mat1, mat1, m1 * paddedN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, m2 * paddedN * sizeof(float), cudaMemcpyHostToDevice);

    dim3 thread(WARP_SIZE, WARP_SIZE);  // 32x32 threads per block
    dim3 block((m2 + thread.x - 1) / thread.x, (m1 + thread.y - 1) / thread.y);
    // Launch the kernel
    matmul<<<block, thread>>>(d_mat1, d_mat2, d_result, m1, paddedN, m2);
    cudaMemcpy(result, d_result, m1 * m2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate the elapsed time
    double end_time = (double)clock() / CLOCKS_PER_SEC;
    elapsed_time = end_time - start_time;

    // print the elapsed_time
    cout << "Time taken to multiply: " << elapsed_time*1000 << " milliseconds" << endl;

    // Write the result to a CSV file
    writeCSV(argv[3], result, m1, m2);

    // Do the required cuda free and clean-ups
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);

    // free the host memory
    delete[] mat1;
    delete[] mat2;

    return 0;
}
