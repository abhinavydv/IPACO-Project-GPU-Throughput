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


// GPU function to multiply two matrices `mat1` and `mat2`
// Tiling is done using shared memory to reduce global memory accesses
__global__ void matmul(float *mat1, float *mat2, float *result, int m1, int paddedN, int n2)
{
    // Create shared memory for the two point sets
    __shared__ float sharedA[WARP_SIZE * WARP_SIZE];
    __shared__ float sharedB[WARP_SIZE * WARP_SIZE];

    // Calculate the index into the shared memory
    int localIdx = threadIdx.y * blockDim.x + threadIdx.x;

    // Initialize the temporary result to 0
    float tmp = 0.0f;

    // // Check if the thread is within the bounds of pointSetA
    // if (blockIdx.y * blockDim.y + threadIdx.y >= m1)
    //     return;

    for (int i = 0; i < paddedN / blockDim.x; i++)
    {
        // Load pointSetA and pointSetB into shared memory
        int globalIdxA = paddedN * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * i + threadIdx.x;
        // int globalIdxB = paddedN * (blockDim.y * blockIdx.x + threadIdx.y) + blockDim.x * i + threadIdx.x;
        int globalIdxB = paddedN * (blockDim.y * i + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;

        // Copy a tile of pointSetA and pointSetB into shared memory
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

        // Compute the temporary result
        for (int j = 0; j < blockDim.x; j++)
        {
            int indA = threadIdx.y * blockDim.x + j;
            int indB = threadIdx.x + blockDim.x * j;
            tmp += sharedA[indA] * sharedB[indB];
            // if (threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0)
            // {
            //     printf("tmp: %f, indA: %d, indB: %d, a: %f, b: %f\n", tmp, indA, indB, sharedA[indA], sharedB[indB]);
            // }
        }

        // next load should happen only after this iteration's computation
        __syncthreads();
    }

    // Check if the thread is within the bounds of pointSetB
    if (blockIdx.x * blockDim.x + threadIdx.x >= n2)
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

    // Copy the dataset to GPU
    cudaMemcpy(d_mat1, mat1, m1 * paddedN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, m2 * paddedN * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 thread(WARP_SIZE, WARP_SIZE);  // 32x32 threads per block
    dim3 block((m2 + thread.x - 1) / thread.x, (m1 + thread.y - 1) / thread.y);
    // Launch the kernel
    double start_time = (double)clock() / CLOCKS_PER_SEC;
    matmul<<<block, thread>>>(d_mat1, d_mat2, d_result, m1, paddedN, m2);
    cudaDeviceSynchronize();
    double end_time = (double)clock() / CLOCKS_PER_SEC;
    cudaMemcpy(result, d_result, m1 * m2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate the elapsed time
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
