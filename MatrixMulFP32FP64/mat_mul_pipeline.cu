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
#include <cuda/pipeline>
#include <cooperative_groups.h>


using namespace std;

#define WARP_SIZE 32
#define NUM_STAGES 2

// Disables `pipeline_shared_state` initialization warning.
#pragma nv_diag_suppress static_var_with_dynamic_init

__always_inline __device__ float mult_row_col(float *A, float *B, int row, int col, int n)
{
    float tmp = 0.0f;
    for (int i = 0; i < n; i++)
    {
        tmp += A[row * n + i] * B[i * n + col];
    }
    return tmp;
}

// GPU function to multiply two matrices `mat1` and `mat2`
// Tiling is done using shared memory to reduce global memory accesses
__global__ void matmul(float *mat1, float *mat2, float *result, int m1, int paddedN, int n2)
{
    // Create shared memory for the two matrices
    // 2 buffers will be used for each matrix
    __shared__ float sharedA[NUM_STAGES][WARP_SIZE * WARP_SIZE];
    __shared__ float sharedB[NUM_STAGES][WARP_SIZE * WARP_SIZE];

    // Calculate the index into the shared memory for this thread
    int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int buffer = 0;

    // Create a pipeline
    auto group = cooperative_groups::this_thread_block();
    constexpr auto scope = cuda::thread_scope_block;

    __shared__ cuda::pipeline_shared_state<scope, NUM_STAGES> shared_state;
    auto pipeline = cuda::make_pipeline(group, &shared_state);

    // fill the first buffer
    int globalIdxA = paddedN * (blockDim.y * blockIdx.y + threadIdx.y) + threadIdx.x;
    int globalIdxB = paddedN * threadIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    pipeline.producer_acquire();
    // Load mat1 and mat2 into shared memory
    cuda::memcpy_async(
        sharedA[buffer] + localIdx,
        mat1 + globalIdxA,
        sizeof(float),
        pipeline
    );
    cuda::memcpy_async(
        sharedB[buffer] + localIdx,
        mat2 + globalIdxB,
        sizeof(float),
        pipeline
    );
    pipeline.producer_commit();

    // Initialize the temporary result to 0
    float tmp = 0.0f;

    for (int i = 1; i < paddedN / blockDim.x; i++)
    {
        // Load mat1 and mat2 into shared memory
        int globalIdxA = paddedN * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * i + threadIdx.x;
        // int globalIdxB = paddedN * (blockDim.y * blockIdx.x + threadIdx.y) + blockDim.x * i + threadIdx.x;
        int globalIdxB = paddedN * (blockDim.y * i + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;

        // Copy a tile of mat1 and mat2 into shared memory
        // Each thread copies one element
        pipeline.producer_acquire();

        buffer = i % NUM_STAGES;

        if (globalIdxA < m1 * paddedN)
            cuda::memcpy_async(
                sharedA[buffer] + localIdx,
                mat1 + globalIdxA,
                sizeof(float),
                pipeline
            );
        else
            sharedA[buffer][localIdx] = 0.0f;
        if (globalIdxB < paddedN * n2)
            cuda::memcpy_async(
                sharedB[buffer] + localIdx,
                mat2 + globalIdxB,
                sizeof(float),
                pipeline
            );
        else
            sharedB[buffer][localIdx] = 0.0f;

        pipeline.producer_commit();
        float tmp2 = sharedB[buffer][localIdx];
        if (threadIdx.y == 0 && i == 1 && blockIdx.x == 0 && blockIdx.y == 0)
        {
            printf("Buffer %d: A[%d] = %f, B[%d] = %f\n", buffer, globalIdxA, sharedA[buffer][localIdx], globalIdxB, tmp2);
        }

        // Compute the temporary result
        buffer = (i - 1) % NUM_STAGES;
        pipeline.consumer_wait();
        tmp += mult_row_col(
            sharedA[buffer],
            sharedB[buffer],
            threadIdx.y,
            threadIdx.x,
            blockDim.x
        );
        pipeline.consumer_release();
    }

    // Compute last buffer
    buffer = (paddedN / blockDim.x - 1) % NUM_STAGES;
    pipeline.consumer_wait();
    tmp += mult_row_col(
        sharedA[buffer],
        sharedB[buffer],
        threadIdx.y,
        threadIdx.x,
        blockDim.x
    );
    pipeline.consumer_release();

    // Check if the thread is within the bounds
    if (blockIdx.x * blockDim.x + threadIdx.x >= n2)
        return;
    if (blockIdx.y * blockDim.y + threadIdx.y >= m1)
        return;

    // Store the computed distance in the global memory
    int idx = n2 * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    result[idx] = tmp;
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
        cout << "usage: " << argv[0] << " <path to mat1> <path to mat2> <output>" << endl;
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

    // Load the mat1.csv into memory
    auto csvData = readCSV(argv[1]);
    mat1 = get<0>(csvData);
    m1 = get<1>(csvData);
    n1 = get<2>(csvData);

    // Load the mat2.csv into memory
    csvData = readCSV(argv[2]);
    mat2 = get<0>(csvData);
    m2 = get<1>(csvData);
    n2 = get<2>(csvData);

    // Check if the number of columns in mat1 and number of rows in mat2 are the same
    if (n1 != m2){
        cout << "Error: The number of columns in mat1 and number of rows in mat2 must be the same." << endl;
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
    cudaMalloc(&d_result, m1 * n2 * sizeof(float));

    double start_time = (double)clock() / CLOCKS_PER_SEC;
    // Copy the dataset to GPU
    cudaMemcpy(d_mat1, mat1, m1 * paddedN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, n2 * paddedN * sizeof(float), cudaMemcpyHostToDevice);

    dim3 thread(WARP_SIZE, WARP_SIZE);  // 32x32 threads per block
    dim3 block((n2 + thread.x - 1) / thread.x, (m1 + thread.y - 1) / thread.y);
    // Launch the kernel
    matmul<<<block, thread>>>(d_mat1, d_mat2, d_result, m1, paddedN, n2);
    cudaMemcpy(result, d_result, m1 * n2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate the elapsed time
    double end_time = (double)clock() / CLOCKS_PER_SEC;
    elapsed_time = end_time - start_time;

    // print the elapsed_time
    cout << "Time taken to multiply: " << elapsed_time*1000 << " milliseconds" << endl;

    // Write the result to a CSV file
    writeCSV(argv[3], result, m1, n2);

    // Do the required cuda free and clean-ups
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);

    // free the host memory
    delete[] mat1;
    delete[] mat2;

    return 0;
}
