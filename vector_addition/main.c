#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// __global__ Executed on `device` callable from `device`
// __device__ Executed on `device` callable from `host`
// __host__ Executed on `host` callable from `host`

__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<n)
        C[i] = A[i] + B[i];
}

__host__
void vecAdd(float* h_A, float* h_B, float* h_C, int n) {
    int size = sizeof(float) * n;
    float *d_A, *d_B, *d_C;

    // 1) Alocate A, B and C into the Device Memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 2) Copy h_A and h_B to d_A and d_B
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 3) Launch Kernel
    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    // 4) Copy d_C to h_C
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // 5) Free from device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int n = 1000;
    
    float *h_A = (float*) malloc(sizeof(float) * n);
    float *h_B = (float*) malloc(sizeof(float) * n);
    float *h_C = (float*) malloc(sizeof(float) * n);

    for (int i = 0; i < n; i += 1) {
        h_A[i] = i * 10;
        h_B[i] = i * 20;
    }

    vecAdd(h_A, h_B, h_C, n);
}
