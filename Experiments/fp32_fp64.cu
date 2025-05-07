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
#include <thrust/extrema.h>

using namespace std;

#ifndef NUM_OPS
#define NUM_OPS 16;
#endif

__global__ void test_speed(float *data){
    double a = 0;
    float b = 0.0;

    __shared__ float sdata[32];

    if (threadIdx.x < 32){
            sdata[threadIdx.x] = 1.0f;
    }
   __syncthreads();

    for (int i=0; i<1000000; i++){
        #ifdef FP32_ONLY
        for (int j = 0; j < NUM_OPS; j++){
            b *= sdata[(i+j)%32];
        }
        #else
        a *= sdata[i%32];
        for (int j = 0; j < NUM_OPS-1; j++){
            b *= sdata[(i+j)%32];
        }
        #endif
    }

    data[threadIdx.x] = a + b;
}

int main(int argc, char *argv[])
{
    float elapsed_time = 0;
    float *data, *data2;

    data2 = new float[64 * 64];
    // cudaEvent_t start, stop;

    cudaMalloc((void**)&data, sizeof(float) * 64 * 64);

    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start, 0);
    // cudaEventSynchronize(start);

    double start_time = (double)clock() / CLOCKS_PER_SEC;

    dim3 block_dim(32, 32, 1);
    test_speed<<<64, block_dim>>>(data);
    cudaDeviceSynchronize();

    double end_time = (double)clock() / CLOCKS_PER_SEC;
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    cudaMemcpy(data2, data, sizeof(float) * 64 * 64, cudaMemcpyDeviceToHost);

    elapsed_time = (end_time - start_time) * 1000.0f;

    // cudaEventElapsedTime(&elapsed_time, start, stop);

    // Calculate the elapsed time
    cout << elapsed_time << endl;

    // Print error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cout << "CUDA error: " << cudaGetErrorString(error) << endl;
    }

    float tmp = 0;

    for (int i = 0; i < 64 * 64; i++) {
        tmp += data2[i];
    }

    cout << "tmp: " << tmp << endl;
    #ifdef FP32_ONLY
    printf("FP32_ONLY\n");
    #endif

    return 0;
}
