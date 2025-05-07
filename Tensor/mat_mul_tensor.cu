#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <mma.h>
// #include <mma.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
using namespace std;
using namespace nvcuda;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int pad_dim(int dim) {
    return (dim + 15) / 16 * 16;
}

void write_matrix_to_csv(const char* filename, float* matrix, int rows, int cols) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        return;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            fprintf(fp, "%.6f", matrix[i * cols + j]);
            if (j < cols - 1) fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

// Reads a CSV matrix into a flat float vector and returns rows and cols by reference
vector<float> read_csv_matrix(const char* filename, int &rows, int &cols) {
    ifstream file(filename);
    string line;
    vector<float> data;
    rows = 0;
    cols = -1;

    while (getline(file, line)) {
        stringstream ss(line);
        string val;
        int cur_cols = 0;
        while (getline(ss, val, ',')) {
            data.push_back(stof(val));
            cur_cols++;
        }
        if (cols == -1) cols = cur_cols;
        else if (cols != cur_cols) {
            fprintf(stderr, "Inconsistent column count in CSV file\n");
            exit(EXIT_FAILURE);
        }
        rows++;
    }

    return data;
}

// CUDA kernel for regular GEMM
__global__ void gemm_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// WMMA kernel for Tensor Core GEMM (assumes padded inputs)
__global__ void wmma_gemm(half *A, half *B, float *C, int M, int N, int K) {
    using namespace nvcuda::wmma;
    const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y;

    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, col_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        if (k + WMMA_K <= K) {
            load_matrix_sync(a_frag, A + warpM * WMMA_M * K + k, K);
            load_matrix_sync(b_frag, B + k * N + warpN * WMMA_N, N);
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    store_matrix_sync(C + warpM * WMMA_M * N + warpN * WMMA_N, acc_frag, N, mem_row_major);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s A.csv B.csv\n", argv[0]);
        return 1;
    }

    int M, K1, K2, N;
    vector<float> A_host = read_csv_matrix(argv[1], M, K1);
    vector<float> B_host = read_csv_matrix(argv[2], K2, N);

    if (K1 != K2) {
        fprintf(stderr, "Inner matrix dimensions must match.\n");
        return 1;
    }

    int K = K1;

    // Padded dimensions for Tensor Core
    int M_pad = pad_dim(M), N_pad = pad_dim(N), K_pad = pad_dim(K);

    // Allocate device memory
    float *d_A_cuda, *d_B_cuda, *d_C_cuda;
    half *d_A_tc, *d_B_tc;
    float *d_C_tc;

    CHECK_CUDA(cudaMalloc(&d_A_cuda, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_cuda, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_cuda, M * N * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_A_tc, M_pad * K_pad * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_tc, K_pad * N_pad * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_tc, M_pad * N_pad * sizeof(float)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_A_cuda, A_host.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_cuda, B_host.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Convert A and B to half for Tensor Core (pad and fill with 0)
    vector<half> A_tc(M_pad * K_pad, __float2half(0.0f));
    vector<half> B_tc(K_pad * N_pad, __float2half(0.0f));

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            A_tc[i * K_pad + j] = __float2half(A_host[i * K + j]);

    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j)
            B_tc[i * N_pad + j] = __float2half(B_host[i * N + j]);

    CHECK_CUDA(cudaMemcpy(d_A_tc, A_tc.data(), M_pad * K_pad * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_tc, B_tc.data(), K_pad * N_pad * sizeof(half), cudaMemcpyHostToDevice));

    // CUDA GEMM
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gemm_kernel<<<grid, block>>>(d_A_cuda, d_B_cuda, d_C_cuda, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_cuda = 0;
    cudaEventElapsedTime(&time_cuda, start, stop);

    // Tensor Core GEMM
    dim3 grid_tc(M_pad / 16, N_pad / 16);
    dim3 block_tc(32, 4);

    cudaEventRecord(start);
    wmma_gemm<<<grid_tc, block_tc>>>(d_A_tc, d_B_tc, d_C_tc, M_pad, N_pad, K_pad);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_tc = 0;
    cudaEventElapsedTime(&time_tc, start, stop);

    printf("Regular CUDA Time: %.3f ms\n", time_cuda);
    printf("Tensor Core Time: %.3f ms\n", time_tc);

    // Copy results back
    float *C_cuda = (float*)malloc(M * N * sizeof(float));
    float *C_tc = (float*)malloc(M * N * sizeof(float));

    cudaMemcpy(C_cuda, d_C_cuda, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; ++i)
        cudaMemcpy(C_tc + i * N, d_C_tc + i * N_pad, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Write results
    write_matrix_to_csv("output_cuda.csv", C_cuda, M, N);
    write_matrix_to_csv("output_tensorcore.csv", C_tc, M, N);

    // Cleanup
    cudaFree(d_A_cuda); cudaFree(d_B_cuda); cudaFree(d_C_cuda);
    cudaFree(d_A_tc); cudaFree(d_B_tc); cudaFree(d_C_tc);
    free(C_cuda); free(C_tc);

    return 0;
}