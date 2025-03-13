#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define ROWS 1024*32
#define COLS ROWS // Cols = Rows

#define BLOCKDIM 32 // 2^5
#define N ROWS

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

__global__ void mul_mat_naive(float *mata, float *matb, float *matc)
{
    // Block index * size of block
    // + sequential thread id of block
    // Global index of all threads
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Number of threads = size of matrix C (ROWS * COLS)
    // Every thread its own element

    matc[row * COLS + col] = 0.0;

    // Traverse trough all elemnts in row, col
    for (size_t i = 0; i < N; i++ )
    {
        // Row --> mata[row * COLS + i]
        // Column --> matb[i * COLS + col]
        // matC[row, col] = matA[row, i:0..N] * matB[i:0..N, col]
        matc[row * COLS + col] += mata[row * COLS + i] * matb[i * COLS + col];

        // Memory coalessing PROBLEM!!
        // Threads from block will access same elements (row, col) as previous thread 
        // Threads in block (access same row, diffrent col) - executing at same time
        // Reading same elements (LOAD) at same time - BAD
        // GOAL - threads from same block access neighbour elements
        // Use of shared memory
    }
}


__global__ void mul_mat_tiles(float *mata, float *matb, float *matc)
{
    // Block index * size of block
    // + sequential thread id of block
    // Global index of all threads
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    float c = 0.0;

    // Shared memory for tiles
    __shared__ float tileA[BLOCKDIM][BLOCKDIM];
    __shared__ float tileB[BLOCKDIM][BLOCKDIM];

    // Itterate over all tiles in row/col of matrix A and B, we need to calculate C
    for (int tile = 0; tile < N / BLOCKDIM; tile++)
    {
        // Read tile A and B from global memory into shared

        // Tile A
        // 2D array [x, y] --> linearize!
        // tileA[local_row][local_col] = mata[(blockIdx.y * blockDim.y + threadIdx.y),
        //                                    (blockIdx.x * blockDim.x + threadIdx.x)];
        // Linearized (row * COLS + col)
        // block: x = variable (loop), y = fixed
        tileA[local_row][local_col] = mata[(blockIdx.y * blockDim.y + local_row) * BLOCKDIM + 
                                           (tile * blockDim.x + local_col)];

        // Tile B
        // 2D array [x, y] --> linearize!
        // tileB[local_row][local_col] = matb[(blockIdx.y * blockDim.y + threadIdx.y),
        //                                    (blockIdx.x * blockDim.x + threadIdx.x)];
        // Linearized (row * COLS + col)
        // block: x = fixed, y = variable (loop)
        tileB[local_row][local_col] = matb[(tile * blockDim.y + local_row) * BLOCKDIM + 
                                           (blockIdx.x * blockDim.x + local_col)];


        // Wait for all threads to move their elements
        __syncthreads();

        // Multiply tiles A and B and save into C
        // Each thread its own C
        for (int i = 0; i < BLOCKDIM; i++)
        {
            c += tileA[local_row][i] * tileB[i][local_col];
        }

        // Wait till all multiplication is done
        // Than start loading next 2 tiles...
        __syncthreads();
    }

    // Transfer calculated element C back to global matrix 
    matc[row * BLOCKDIM + col] = c;
}

// Host code (CPU)
int main(int argc, char *argv[])
{
    // Allocate host memory
    h_ma = (float *)malloc(ROWS * COLS * sizeof(float));
    h_mb = (float *)malloc(ROWS * COLS * sizeof(float));
    h_mc = (float *)malloc(ROWS * COLS * sizeof(float));
    
    // Allocate device memory
    cudaMalloc(&d_ma, ROWS * COLS * sizeof(float));
    cudaMalloc(&d_mb, ROWS * COLS * sizeof(float));
    cudaMalloc(&d_mc, ROWS * COLS * sizeof(float));
    
    // Matrix initialization
    // ROWS = number of elements in column
    // COLS = number of elements in row
    for (size_t i = 0; i < ROWS; i++) {
        for (size_t j = 0; j < COLS; j++) {
            // Row-major ordering (row by row)
            h_ma[i * COLS + j] = 1.0f;
            h_mb[i * COLS + j] = 2.0f;

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
    // Matrix A
    cudaMemcpy((void *)d_ma,
                (void *)h_ma,
                ROWS * COLS * sizeof(float),
                cudaMemcpyHostToDevice);
    // Matrix B
    cudaMemcpy((void *)d_mb,
                (void *)h_mb,
                ROWS * COLS * sizeof(float),
                cudaMemcpyHostToDevice);

    // dim3 CUDA type for specifiying dimension 
    // Thread → Warp → Block → Grid
    // 1 thread (part of warp) executed on single SP
    // 32 threads in Warp (32 SP used to process 1 warp at same time)
    // Block contains multiple warps (common sizes: 128, 256, 512 threads)
    // Grid = collection of blocks
    // Blocks are sent on SM 
    // all threads in warp executed on multiple SPs in SM

    // Block size: 32x32
    dim3 threadsInBlock(BLOCKDIM, BLOCKDIM, 1);

    // Number of blocks
    dim3 numOfBlocks(COLS/BLOCKDIM, ROWS/BLOCKDIM, 1);

    printf("Computing matrix multiplication...\n");
    float miliseconds;

    // CUDA events for measuring times
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Start recording (Naive)
    cudaEventRecord(start);
    mul_mat_naive<<<numOfBlocks, threadsInBlock>>>(d_ma, d_mb, d_mc);

    // Stop recording
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    miliseconds = 0.0;
    cudaEventElapsedTime(&miliseconds, start, end);
    printf("NAIVE kernel execution time: %0.3f miliseconds\n", miliseconds);

    // Start recording (Tiles)
    cudaEventRecord(start);
    mul_mat_tiles<<<numOfBlocks, threadsInBlock>>>(d_ma, d_mb, d_mc);

    // Stop recording
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    miliseconds = 0.0;
    cudaEventElapsedTime(&miliseconds, start, end);
    printf("Tiles execution time: %0.3f miliseconds\n", miliseconds);

    // Transfer data from GPU to host
    cudaMemcpy((void *)h_mc,
                (void *)d_mc,
                ROWS * COLS * sizeof(float),
                cudaMemcpyDeviceToHost);
    // row, col
    printf("c[1023, 1023] = %f\n", h_mc[1023 * COLS + 1023]);
    printf("c[1024, 1023] = %f (BAD ROW)\n", h_mc[1024 * COLS + 1023]);
    printf("c[1023, 1024] = %f (BAD COL)\n", h_mc[1023 * COLS + 1024]);

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

