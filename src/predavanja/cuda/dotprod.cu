#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define N 1024*1024

// Host
float *h_va;
float *h_vb;
float *h_vc;

// Device (GPU)
float *d_va;
float *d_vb;
float *d_vc;


// Code for GPU device
// __global__ (KERNEL) function which runs on device (GPU), executed from host
// __device__ function which runs on device (GPU), executed from device

__global__ void vec_mul(float *veca, float *vecb, float *vecc)
{
    // Block index * size of block
    // + sequential thread id of block
    // Global index of all threads
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // V1
    // block = 1k threads
    // N / block_size = 1M blocks
    // 1 thread process 1 element
    //vecc[tid] = veca[tid] * vecb[tid];

    // V2
    // block = 1k threads
    // Num of blocks = 1k

    // 1M threads --> stride (offset) of 1M 
    // thread 0: 0, 1M, 2M...
    // thread 1: 1, 1M+1, 2M+1...
    // thread n-1: 1M-1, 2M-1...
    // 1 thread need to process 1k elements
    // NT = number of threads
    // thread 0 elements: 0, NT, 2NT, 3NT, ... 1023NT

    while (tid < N)
    {
        vecc[tid] = veca[tid] * vecb[tid];

        // gridDim -> number of blocks (1k)
        // blockDim -> number of threads in block (1k)
        // 1k * 1k = 1M threads => 1M stide (offset)
        // tid + 1M (next itteration)
        tid = tid + gridDim.x * blockDim.x;
    }

}

// Host code (CPU)
int main(int argc, char *argv[])
{
    printf("Matrix addition started...\n");
    
    // Allocate host memory
    h_va = (float *)malloc(N * sizeof(float));
    h_vb = (float *)malloc(N * sizeof(float));
    h_vc = (float *)malloc(N * sizeof(float));

    // Allocate device memory
    cudaMalloc(&d_va, N * sizeof(float));
    cudaMalloc(&d_vb, N * sizeof(float));
    cudaMalloc(&d_vc, N * sizeof(float));

    // Vector initialization
    for (size_t i = 0; i < N; i++)
    {
        h_va[i] = 3.0;
        *(h_vb + i) = 2.0;
    }
    printf("Initialization done.\n");

    // cudaMemcpy
    // Destination address
    // Source address
    // Size (number of Bytes)
    // Data transfer direction (H -> D, D -> H)

    // Transfer data from host memory to device memory (GPU)
    printf("Transfering data from host to GPU...\n");
    cudaMemcpy((void *)d_va,
                (void *)h_va,
                N * sizeof(float),
                cudaMemcpyHostToDevice);

    cudaMemcpy((void *)d_vb,
                (void *)h_vb,
                N * sizeof(float),
                cudaMemcpyHostToDevice);

    // dim3 CUDA type for specifiying dimension 
    // Thread → Warp → Block → Grid
    // 1 thread (part of warp) executed on single SP
    // 32 threads in Warp (32 SP used to process 1 warp at same time)
    // Block contains multiple warps (common sizes: 128, 256, 512 threads)
    // Grid = collection of blocks
    // Blocks are sent on SM 
    // all threads in warp executed on multiple SPs in SM
    // x = 1024, y=1, z=1
    dim3 threadsInBlock(1024, 1, 1);

    //dim3 numOfBlocks(N / threadsInBlock.x, 1, 1); // 1G / 1K = 1M blocks
    dim3 numOfBlocks(1024, 1, 1); // 1K blocks -> each thread process 1K elements

    // Kernel lunch on GPU
    printf("Computing vector multiplication...\n");
    vec_mul<<<numOfBlocks, threadsInBlock>>>(d_va, d_vb, d_vc);


    // Transfer data from GPU to host
    cudaMemcpy((void *)h_vc,
                (void *)d_vc,
                N * sizeof(float),
                cudaMemcpyDeviceToHost);


    // Print result (should be 6)
    printf("Element: %f\n", h_vc[14]);
    printf("Element: %f\n", h_vc[N-1]);
    printf("Element (NOT OK): %f\n", h_vc[N]);

    // Sum up the products
    float result = 0.0;
    for (int i = 0; i < N; i++) {
        result += h_vc[i];
    }
    printf("Result: %f\n", result);

    // Sprostimo prostor gostitelja
    free(h_va);
    free(h_vb);
    free(h_vc);

    // Sprostimo prostor na naprvi (GPU)
    cudaFree(d_va);
    cudaFree(d_vb);
    cudaFree(d_vc);

    return 0;
}

