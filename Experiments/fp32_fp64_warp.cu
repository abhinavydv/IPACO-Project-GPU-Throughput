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

#ifndef NUM_FP64
#define NUM_FP64 4 // This is the number of fp64 warps
#endif

__global__ void test_speed(float *data){
    double a = 1;
    float b = 1.0;

    __shared__ float sdata[32];

    if (threadIdx.x < 32){
            sdata[threadIdx.x] = 1.0f;
    }
    __syncthreads();

    // if (threadIdx.y < NUM_FP64){
    //     for (int i=0; i<1000000; i++){
    //         a *= sdata[i%32];
    //     }
    // } else {
        for (int i=0; i<1000000; i++){
            b *= sdata[i%32];
        }
    // }

    data[threadIdx.x] = a + b;
}

int main(int argc, char *argv[])
{
    float elapsed_time = 0;
    float *data, *data2;

    data2 = new float[64 * 64];
    // cudaEvent_t start, stop;

    cudaMalloc((void**)&data, sizeof(float) * 64 * 64);

    double start_time = (double)clock() / CLOCKS_PER_SEC;

    dim3 block_dim(32, 32, 1);
    test_speed<<<64, block_dim>>>(data);
    cudaDeviceSynchronize();

    double end_time = (double)clock() / CLOCKS_PER_SEC;
    cudaMemcpy(data2, data, sizeof(float) * 64 * 64, cudaMemcpyDeviceToHost);

    elapsed_time = (end_time - start_time) * 1000.0f;

    // Calculate the elapsed time
    cout << elapsed_time << endl;

    // Print error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cout << "CUDA error: " << cudaGetErrorString(error) << endl;
    }

    float var = 0;

    for (int i = 0; i < 64 * 64; i++) {
        var += data2[i];
    }

    cout << "var: " << var << endl;

    return 0;
}
