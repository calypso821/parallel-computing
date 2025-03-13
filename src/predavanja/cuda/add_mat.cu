#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define ROWS 1024
#define COLS 512
#define MAT_SIZE ROWS * COLS
#define BLOCK_SIZE 16 // 2^4

// Host
float *h_ma;
float *h_mb;
float *h_mc;

// Device (GPU)
float *d_ma;
float *d_mb;
float *d_mc;


// Code for GPU device
// __global__ (KERNEL) function which runs on device (GPU), executed from host
// __device__ function which runs on device (GPU), executed from device

// function is executed on threads of GPU
__global__ void add_mat(float *mata, float *matb, float *matc)
{
    // paramters = pointers to GPU memory

    // blockIdx: block index (which block in the grid)
    // blockDim: block dimensions (threads per block)
    // threadIdx: thread index within its block

    // Block index * size of block
    // + sequential thread id of block
    // Global index of all threads
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Addition of matrix a and b into c

    // Row - major ordering
    matc[y * COLS + x] = mata[y * COLS + x]  + matb[y * COLS + x];
    // Col - major ordering
    //matc[x * ROWS + y] = mata[x * ROWS + y]  + matb[x * ROWS + y];
}

// Host code (CPU)
int main(int argc, char *argv[])
{
    printf("Matrix addition started...\n");
    
    // Allocate host memory
    h_ma = (float *)malloc(MAT_SIZE * sizeof(float));
    h_mb = (float *)malloc(MAT_SIZE * sizeof(float));
    h_mc = (float *)malloc(MAT_SIZE * sizeof(float));
    
    // Allocate device memory
    cudaMalloc(&d_ma, MAT_SIZE * sizeof(float));
    cudaMalloc(&d_mb, MAT_SIZE * sizeof(float));
    cudaMalloc(&d_mc, MAT_SIZE * sizeof(float));
    
    // Matrix initialization
    // ROWS = number of elements in column
    // COLS = number of elements in row
    for (size_t i = 0; i < ROWS; i++) {
        for (size_t j = 0; j < COLS; j++) {
            // Row-major ordering (row by row)
            h_ma[i * COLS + j] = 9.0f;
            h_mb[i * COLS + j] = 4.0f;

            // Column-major ordering (col by col)
            // h_ma[j * ROWS + i] = 9.0f;
            // h_mb[j * ROWS + i] = 4.0f;
        }
    }
    printf("Initialization done.\n");

    // cudaMemcpy
    // Destination address
    // Source address
    // Size (number of Bytes)
    // Data transfer direction (H -> D, D -> H)

    // Transfer data from host memory to device memory (GPU)
    printf("Transfering data from host to GPU...\n");
    // Matrix A
    cudaMemcpy((void *)d_ma,
                (void *)h_ma,
                MAT_SIZE * sizeof(float),
                cudaMemcpyHostToDevice);
    // Matrix B
    cudaMemcpy((void *)d_mb,
                (void *)h_mb,
                MAT_SIZE * sizeof(float),
                cudaMemcpyHostToDevice);

    // dim3 CUDA type for specifiying dimension 
    // Thread → Warp → Block → Grid
    // 1 thread (part of warp) executed on single SP
    // 32 threads in Warp (32 SP used to process 1 warp at same time)
    // Block contains multiple warps (common sizes: 128, 256, 512 threads)
    // Grid = collection of blocks
    // Blocks are sent on SM 
    // all threads in warp executed on multiple SPs in SM

    // Block size: 256 (2^8)
    // rows: 2^4, cols: 2^4 (16x16)
    // rows: 2^5, cols: 2^3 (32x8)
    dim3 threadsInBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Number of blocks
    // Y (rows): (2^20 / 2^4 = 2^16) 64K (1024 * 1024 elements)
    // X (cols): (2^10 / 2^4 = 2^6) 64 (1024 elements)

    // Blocks in x dimension: COLS (num of ele in row) / BLOCK_SIZE 
    // Blocks in y dimension: ROWS (num of ele in col) / BLOCK_SIZE 
    dim3 numOfBlocks(COLS/BLOCK_SIZE, ROWS/BLOCK_SIZE, 1);
    //dim3 numOfBlocks(ROWS/BLOCK_SIZE, COLS/BLOCK_SIZE, 1);

    // Kernel lunch on GPU
    printf("Computing matrix addition...\n");
    add_mat<<<numOfBlocks, threadsInBlock>>>(d_ma, d_mb, d_mc);

    // Transfer data from GPU to host
    printf("Transfering data from GPU to host...\n");
    cudaMemcpy((void *)h_mc,
                (void *)d_mc,
                MAT_SIZE * sizeof(float),
                cudaMemcpyDeviceToHost);


    // Result test (should be 13)

    // Row-major
    printf("Element: %f\n", h_mc[1023 * COLS + 250]);
    printf("Element: %f (BAD ROW)\n", h_mc[1024 * COLS + 511]);
    printf("Element: %f\n", h_mc[1023 * COLS + 511]);
    printf("Element: %f (BAD COL)\n", h_mc[1023 * COLS + 512]);

    // Column-major
    // printf("Element: %f\n", h_mc[511 * ROWS + 511]);
    // printf("Element: %f (BAD COL)\n", h_mc[512 * ROWS + 511]);
    // printf("Element: %f\n", h_mc[511 * ROWS + 1023]);
    // printf("Element: %f (BAD ROW)\n", h_mc[511 * ROWS + 1024]);

    // Free host memory
    free(h_ma);
    free(h_mb);
    free(h_mc);

    // Free device memory (GPU)
    cudaFree(d_ma);
    cudaFree(d_mb);
    cudaFree(d_mc);

    return 0;
}

