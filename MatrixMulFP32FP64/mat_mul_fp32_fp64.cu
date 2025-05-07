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

using namespace std;

#define WARP_SIZE 32

#define mult(j, var) int indA##j = threadIdx.y * blockDim.x + j; \
                int indB##j = j * blockDim.x + threadIdx.x; \
                var += sharedA[indA##j] * sharedB[indB##j];

// GPU function to multiply two matrices `mat1` and `mat2`
// Tiling is done using shared memory to reduce global memory accesses
__global__ void matmul(float *mat1, float *mat2, float *result, int m1, int paddedN, int n2)
{
    // Create shared memory for the two point sets
    __shared__ float sharedA[WARP_SIZE * WARP_SIZE];
    __shared__ float sharedB[WARP_SIZE * WARP_SIZE];

    // Requirement: blockDim.x == WARP size
    // assert(blockDim.x == WARP_SIZE);

    // Calculate the index into the shared memory
    int localIdx = threadIdx.y * blockDim.x + threadIdx.x;

    // Initialize the temporary result to 0
    float tmp = 0.0f;
    double tmp2 = 0.0;

    for (int i = 0; i < paddedN / blockDim.x; i++)
    {
        // Load mat1 and mat2 into shared memory
        int globalIdxA = paddedN * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * i + threadIdx.x;
        int globalIdxB = paddedN * (blockDim.y * i + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;

        // Copy a tile of mat1 and mat2 into shared memory
        // Each thread copies one element

        if (globalIdxA < m1 * paddedN)
            sharedA[localIdx] = mat1[globalIdxA];
        else
            sharedA[localIdx] = 0.0f;

        if (globalIdxB < paddedN * n2)
            sharedB[localIdx] = mat2[globalIdxB];
        else
            sharedB[localIdx] = 0.0f;

        // Synchronize threads to ensure all data is loaded
        __syncthreads();

        // When tmp1 is used then it is FP32 and when tmp2 in used then it is FP64
        mult(0, tmp2);    //FP 32 replaced by FP64 in later lines
        mult(1, tmp);
        mult(2, tmp);
        mult(3, tmp);
        mult(4, tmp2);
        mult(5, tmp);
        mult(6, tmp);
        mult(7, tmp);
        mult(8, tmp2);
        mult(9, tmp);
        mult(10, tmp);
        mult(11, tmp);
        mult(12, tmp2);
        mult(13, tmp);
        mult(14, tmp);
        mult(15, tmp);
        mult(16, tmp2);
        mult(17, tmp);
        mult(18, tmp);
        mult(19, tmp);
        mult(20, tmp2);
        mult(21, tmp);
        mult(22, tmp);
        mult(23, tmp);
        mult(24, tmp2);
        mult(25, tmp);
        mult(26, tmp);
        mult(27, tmp);
        mult(28, tmp2);
        mult(29, tmp);
        mult(30, tmp);
        mult(31, tmp);

        // next load should happen only after this iteration's computation
        __syncthreads();
    }

    // Check if the thread is within the bounds of mat2
    if (blockIdx.x * blockDim.x + threadIdx.x >= n2)
        return;

    // Store the computed result in the global memory
    int idx = n2 * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    result[idx] = tmp + tmp2;
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
    if (n1 != m2){
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
    result = new float[m1 * n2];

    // Do the required cuda malloc's
    cudaMalloc(&d_mat1, m1 * paddedN * sizeof(float));
    cudaMalloc(&d_mat2, n2 * paddedN * sizeof(float));
    cudaMalloc(&d_result, m1 * n2 * sizeof(float));

    double start_time = (double)clock() / CLOCKS_PER_SEC;
    // Copy the dataset to GPU
    cudaMemcpy(d_mat1, mat1, m1 * paddedN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, n2 * paddedN * sizeof(float), cudaMemcpyHostToDevice);

    dim3 thread(WARP_SIZE, WARP_SIZE);  // 32x32 threads per block
    dim3 block((n2 + thread.x - 1) / thread.x, (m1 + thread.y - 1) / thread.y);

    double comp_start = (double)clock() / CLOCKS_PER_SEC;
    // Launch the kernel
    matmul<<<block, thread>>>(d_mat1, d_mat2, d_result, m1, paddedN, n2);
    cudaDeviceSynchronize();
    double comp_time = (double)clock() / CLOCKS_PER_SEC - comp_start;

    cudaMemcpy(result, d_result, m1 * n2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate the elapsed time
    double end_time = (double)clock() / CLOCKS_PER_SEC;
    elapsed_time = end_time - start_time;

    // print the elapsed_time
    cout << comp_time*1000 << endl;
    // cout << "Time taken to multiply: " << comp_time*1000 << " milliseconds" << endl;
    cout << "Total time: " << elapsed_time*1000 << " milliseconds" << endl;

    // Print error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cout << "CUDA error: " << cudaGetErrorString(error) << endl;
    }

    // Write the result to a CSV file
    // writeCSV(argv[3], result, m1, n2);

    // Do the required cuda free and clean-ups
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);

    // free the host memory
    delete[] mat1;
    delete[] mat2;

    return 0;
}
