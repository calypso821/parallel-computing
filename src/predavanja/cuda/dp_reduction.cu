#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define N 1024*1024*1024

#define BLOCKDIM 512
#define GRIDDIM 216

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
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory decleration
    // Internal index for block
    __shared__ float vecC_perBlock[BLOCKDIM];

    // Init VecC_perBlock
    vecC_perBlock[threadIdx.x] = 0.0;

    // 1. Phase - partial products of vector A and B
    while (tid < N)
    {
        // Global memory
        //vecc[tid] = veca[tid] * vecb[tid];
        // Shared memory
        vecC_perBlock[threadIdx.x] += veca[tid] * vecb[tid];

        // gridDim -> number of blocks (N / BLOCKDIM)
        // blockDim -> number of threads in block (512)
        tid = tid + gridDim.x * blockDim.x;
    }

    // Sinhornization (for threads in same block)
    // Guarantee that all threads finish multiplication and write to vecC_block
    __syncthreads();

    // 2. Phase - sum of partial products 
    // Naive approach - 1 thread sum all partial products in shared memory
    // Parallel approach (Reducation) - use of multiple threads 

    // Number of i (working threads)
    int i = blockDim.x / 2;

    // Until there is only 1 thread left (1 sum => 1 result)
    while (i >= 1) {
        // Only working threads doing i
        // Half every itteration
        if (threadIdx.x < i) {
            vecC_perBlock[threadIdx.x] += vecC_perBlock[threadIdx.x + i]; 
        }

        // Half working threads
        i /= 2;

        // Sync all working threads -> partial i
        __syncthreads();
    }

    // Store result of reduction back to global memory
    // Result for 1 block of thread is in vecC_perBlock[0]
    // Number of all results = GRIDDIM

    if (threadIdx.x == 0) {
        // Only thread 0
        vecc[blockIdx.x] = vecC_perBlock[0];
    }

}

// Host code (CPU)
int main(int argc, char *argv[])
{
    // Allocate host memory
    h_va = (float *)malloc(N * sizeof(float));
    h_vb = (float *)malloc(N * sizeof(float));
    h_vc = (float *)malloc(GRIDDIM * sizeof(float));

    // Allocate device memory
    cudaMalloc(&d_va, N * sizeof(float));
    cudaMalloc(&d_vb, N * sizeof(float));
    cudaMalloc(&d_vc, GRIDDIM * sizeof(float));

    // Vector initialization
    for (size_t i = 0; i < N; i++)
    {
        h_va[i] = 3.0;
        *(h_vb + i) = 2.0;
    }
    printf("Initialization done.\n");

    // cudaMemcpy
    // Naslov ponor podatkov (destination)
    // Naslov izvora podatkov (source)
    // Stevilo B za prenos
    // Smer prenosa (H -> D, D -> H)

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

    dim3 threadsInBlock(BLOCKDIM, 1, 1); // B = 512
    // Number of blocks
    dim3 numOfBlocks(GRIDDIM, 1, 1); // N / B 

    // Kernel lunch on GPU
    printf("Computing vector multiplication (reduction)...\n");
    vec_mul<<<numOfBlocks, threadsInBlock>>>(d_va, d_vb, d_vc);


    // Transfer data from GPU to host
    // ONLY number of BLOCKS! (GRIDDIM)
    // partial sums already calculated per block (use of reduction)
    cudaMemcpy((void *)h_vc,
                (void *)d_vc,
                GRIDDIM * sizeof(float),
                cudaMemcpyDeviceToHost);


    // Print result
    printf("Element: %f\n", h_vc[10]);
    printf("Element: %f\n", h_vc[GRIDDIM-1]);
    printf("Element (NOT OK): %f\n", h_vc[GRIDDIM]);

    float result = 0;
    for (int i = 0; i < GRIDDIM; i++)
    {
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

