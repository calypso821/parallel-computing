#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "cuda_config.h"
#include "stb_config.h"

#define COLOR_CHANNELS 1

#define GRAYLEVELS 256
#define DESIRED_NCHANNELS 1
#define BLOCK_SIZE 16

// Kernel (on GPU executed from host)
__global__ void hist_naive(const unsigned char *image, const int width, const int height, unsigned int *histogram)
{
    // Block index * size of block
    // + sequential thread id of block
    // Global index of all threads
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        // 1. Get pixel intensity
        unsigned char intensity = image[row * width + col];

        // 2. Atomic add
        //histogram[intensity]++; // non atomic (BAD)
        atomicAdd(&histogram[intensity], 1); 

        // Every thread is accessing same global memory (histogram) 
        // PROBEM: memory coalescing
        // SOLUTION: local histogram per block!!
    }
}

// Kernel (on GPU executed from host)
__global__ void hist_local(const unsigned char *image, const int width, const int height, unsigned int *histogram)
{
    // Block index * size of block
    // + sequential thread id of block
    // Global index of all threads
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Local x, y
    int ly = threadIdx.y;
    int lx = threadIdx.x;
    int thread_bin = ly * blockDim.x + lx;

    // Local histogram
    __shared__ unsigned int local_hist[GRAYLEVELS];

    // BLOCK SIZE = 16 --> 256 threads
    // Set local_hist values to 0 (GREAYLEVELS = 256)
    // Each thread clear its own bin
    local_hist[thread_bin] = 0.0;
    __syncthreads();

    if (col < width && row < height)
    {
        // 1. Get pixel intensity
        unsigned char intensity = image[row * width + col];

        // 2. Atomic add
        //histogram[intensity]++; // non atomic (BAD)
        atomicAdd(&local_hist[intensity], 1); 
    }
    // Wait for all threads to update local histogram
    __syncthreads();

    // Transfer values from local hist to global 
    // Every thread its own bin + atomic operation
    atomicAdd(&histogram[thread_bin], local_hist[thread_bin]);
}

// Kernel (on GPU executed from host)
__global__ void hist_normalize(unsigned int *histogram, float *histogram_norm, int size)
{
    // 1 block of 256 threads
    int thread_bin = threadIdx.x;

    histogram_norm[thread_bin] = (float)histogram[thread_bin] / size;
}

// Kernel (on GPU executed from host)
__global__ void cdf_naive(const float *histogram_norm, float *cdf)
{
    // 1 block of 256 threads
    int tid = threadIdx.x;

    // Local histogram (double buffer) - use of previous values!
    __shared__ float local_hist[GRAYLEVELS * 2];

    // Variable for read and write buffer (swapped every itteration)
    int wbuff = 1;
    int rbuff = 0;

    // Transfer normalized histogram into local memory
    local_hist[tid] = histogram_norm[tid];
    __syncthreads();

    for (int offset = 1; offset < GRAYLEVELS; offset <<= 1)
    {
        // Working threads >= offset
        if (tid >= offset) {
            local_hist[wbuff * GRAYLEVELS + tid] = local_hist[rbuff * GRAYLEVELS + tid] + 
                                                   local_hist[rbuff * GRAYLEVELS + tid - offset];
        } else {
            // Copy existing value
            local_hist[wbuff * GRAYLEVELS + tid] =  local_hist[rbuff * GRAYLEVELS + tid];
        }
        __syncthreads();
        
        // Flip
        wbuff = 1 - wbuff;
        rbuff = 1 - rbuff;
    }

    // Transfer local histogram back to global memory
    cdf[tid] = local_hist[rbuff * GRAYLEVELS + tid];
}
__global__ void cdf_blelloch(const float *histogram_norm, float *cdf)
{
    // TODO:
    // // 1 block of 256 threads
    // int tid = threadIdx.x;

    // // Local histogram (double buffer) - use of previous values!
    // __shared__ float local_hist[GRAYLEVELS * 2];

    // // Variable for read and write buffer (swapped every itteration)
    // int wbuff = 1;
    // int rbuff = 0;

    // // Transfer normalized histogram into local memory
    // local_hist[tid] = histogram_norm[tid];
    // __syncthreads();

    // for (int offset = 1; offset < GRAYLEVELS; offset <<= 1)
    // {
    //     // Working threads >= offset
    //     if (tid >= offset) {
    //         local_hist[wbuff * GRAYLEVELS + tid] = local_hist[rbuff * GRAYLEVELS + tid] + 
    //                                                local_hist[rbuff * GRAYLEVELS + tid - offset];
    //     } else {
    //         // Copy existing value
    //         local_hist[wbuff * GRAYLEVELS + tid] =  local_hist[rbuff * GRAYLEVELS + tid];
    //     }
    //     __syncthreads();
        
    //     // Flip
    //     wbuff = 1 - wbuff;
    //     rbuff = 1 - rbuff;
    // }

    // // Transfer local histogram back to global memory
    // cdf[tid] = local_hist[rbuff * GRAYLEVELS + tid];
}

// Kernel (on GPU executed from host)
__global__ void map_equalize(const unsigned char *imageIn, const int width, const int height,
                             const float *cdf, unsigned char *imageOut)
{
    // Block index * size of block
    // + sequential thread id of block
    // Global index of all threads
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // cdf_i = comulative count of pixels with intenisty values from 0 to i
    // cdf_min = first non zero value in CDF (first comulative count of pixles with intenisty > 0)
    // cdf_max = all pixels (imageSize)!!!
    // range (max - min)

    // 1. Shift CDF to left (by cdf_min)
    // cdf_i - cdf_min -> shifts CDF to left by range (cdf_min)
    // (darkets pixel starts at 0)
    // 2. Normalize scale value to [0, 1]
    // cdf_max (imageSize) - cdf_min

    // cdf_i ... range from 0 to max pixels (imageSize)
    // Dividing cdf_i by imageSize = range from 0 to max
    // Normalizing cdf_i by max range --> linear distirbution
    // CDF from normalized hist ... 0 to 1

    // TODO: cdf_min (reduction)
    float cdf_min = 0.000044f;
    float cdf_max = 1.0f;

    if (col < width && row < height)
    {
        // 1. Get pixel from input image
        unsigned char pixelIn = imageIn[row * width + col];
        // cdf_i representing probability for selected intenisity to 0
        float cdf_i = cdf[pixelIn];

        // 2. Transfrom (equalize) pixel with CDF distribution
        float scale = (cdf_i - cdf_min) / (cdf_max - cdf_min);
        // Map back to values from 0-255
        float pixelOut = roundf(scale * (GRAYLEVELS-1));

        // 3. Map to output image
        imageOut[row * width + col] = pixelOut;
    }
}

__global__ void findMin(const float *cdf, float min)
{

}



int main(int argc, char *argv[]){

     // Read image from file
    int width, height, cpp;

    // read only DESIRED_NCHANNELS channels from the input image:
    unsigned char *h_imageIn = stbi_load("resources/input/kolesar-neq.jpg", &width, &height, &cpp, DESIRED_NCHANNELS);
    if(h_imageIn == NULL) {
        printf("Error in loading the image\n");
        return 1;
    }
    printf("Loaded image W= %d, H = %d, actual cpp = %d \n", width, height, cpp);

    int imageSize = height * width * DESIRED_NCHANNELS * sizeof(unsigned char);
    int histSize = GRAYLEVELS * sizeof(unsigned int);
    int histNormSize = GRAYLEVELS * sizeof(float);

    // Allocate memory for raw output image data, histogram, and CDF 
	unsigned char *h_imageOut = (unsigned char *)malloc(imageSize);
    unsigned int *h_histogram = (unsigned int *)malloc(histSize);
    float *h_histogram_norm = (float *)malloc(histNormSize);
    float *h_cdf = (float *)malloc(histNormSize);

    unsigned char *d_imageIn;
    unsigned char *d_imageOut;
    unsigned int *d_histogram;
    float *d_histogram_norm;
    float *d_cdf;

    // Allocate device memory
    checkCudaErrors(cudaMalloc(&d_imageIn, imageSize));
    checkCudaErrors(cudaMalloc(&d_imageOut, imageSize));
    checkCudaErrors(cudaMalloc(&d_histogram, histSize));
    checkCudaErrors(cudaMalloc(&d_histogram_norm, histNormSize));
    checkCudaErrors(cudaMalloc(&d_cdf, histNormSize));


    // Initialize histogram memory to zero
    checkCudaErrors(cudaMemset(d_histogram, 0, histSize));
    checkCudaErrors(cudaMemset(d_histogram_norm, 0, histNormSize));
    checkCudaErrors(cudaMemset(d_cdf, 0, histNormSize));

    printf("Initialization done.\n");

    // Block size: 16x16
    dim3 threadsInBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Number of blocks
    dim3 numOfBlocks(ceil(width / BLOCK_SIZE), ceil(height / BLOCK_SIZE), 1);

    printf("Computing histogram...\n");

    // CUDA events for measuring times
    float kernel_time = 0.0f; 
    //float total_time = 0.0f;
    cudaEvent_t kernel_start, kernel_end, total_start, total_end;

    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_end);

    // Start recording (total time -> kernel to data transfer)
    //cudaEventRecord(total_start);

    // Transfer data from host memory to device memory (GPU)
    // Input image
    checkCudaErrors(
        cudaMemcpy((void *)d_imageIn,
                    (void *)h_imageIn,
                    imageSize,
                    cudaMemcpyHostToDevice));

    // =================== HISTOGRAM =================
    // Naive + atomic
    cudaEventRecord(kernel_start);
    hist_naive<<<numOfBlocks, threadsInBlock>>>(d_imageIn, width, height, d_histogram);
    getLastCudaError("hist_naive() execution failed\n");
    cudaEventRecord(kernel_end);
    cudaEventSynchronize(kernel_end);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_end);
    printf("Hist (naive) execution time: %.5f milliseconds\n", kernel_time);
    checkCudaErrors(cudaMemset(d_histogram, 0, histSize)); // clear hist
    // local memory + atmoic
    cudaEventRecord(kernel_start);
    hist_local<<<numOfBlocks, threadsInBlock>>>(d_imageIn, width, height, d_histogram);
    getLastCudaError("hist_local_mem() execution failed\n");
    cudaEventRecord(kernel_end);
    cudaEventSynchronize(kernel_end);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_end);
    printf("Hist (local mem) execution time: %.5f milliseconds\n", kernel_time);

    dim3 threadsInBlockHist(GRAYLEVELS, 1, 1);
    dim3 numOfBlocksHist(1, 1, 1);
    // =================== Normalize hist ================= 
    hist_normalize<<<numOfBlocksHist, threadsInBlockHist>>>(d_histogram, d_histogram_norm, width * height);

    // =================== CDF =================
    // (Comulative distribution function) 
    // Naive (Algorithm Hillis and Steele)
    cudaEventRecord(kernel_start);
    cdf_naive<<<numOfBlocksHist, threadsInBlockHist>>>(d_histogram_norm, d_cdf);
    getLastCudaError("cdf_naive() execution failed\n");
    cudaEventRecord(kernel_end);
    cudaEventSynchronize(kernel_end);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_end);
    printf("CDF (naive) execution time: %.5f milliseconds\n", kernel_time);
    //checkCudaErrors(cudaMemset(d_cdf, 0, histNormSize)); // clear cdf
    // Work-Efficient Parallel Scan (Blelloch)
    // cudaEventRecord(kernel_start);
    // cdf_blelloch<<<numOfBlocksHist, threadsInBlockHist>>>(d_histogram_norm, d_cdf);
    // getLastCudaError("cdf_blelloch() execution failed\n");
    // cudaEventRecord(kernel_end);
    // cudaEventSynchronize(kernel_end);
    // cudaEventElapsedTime(&kernel_time, kernel_start, kernel_end);
    // printf("CDF (blelloch) execution time: %.5f milliseconds\n", kernel_time);

    // ============== Equlize & Map ===============
    map_equalize<<<numOfBlocks, threadsInBlock>>>(d_imageIn, width, height, d_cdf, d_imageOut);

    // Transfer data from GPU to host (Histogram)
    checkCudaErrors(
        cudaMemcpy((void *)h_histogram,
                    (void *)d_histogram,
                    histSize,
                    cudaMemcpyDeviceToHost));
    checkCudaErrors(
        cudaMemcpy((void *)h_histogram_norm,
                    (void *)d_histogram_norm,
                    histNormSize,
                    cudaMemcpyDeviceToHost));
    checkCudaErrors(
        cudaMemcpy((void *)h_cdf,
                    (void *)d_cdf,
                    histNormSize,
                    cudaMemcpyDeviceToHost));
    checkCudaErrors(
        cudaMemcpy((void *)h_imageOut,
                    (void *)d_imageOut,
                    imageSize,
                    cudaMemcpyDeviceToHost));

    // // Start recording (total time)
    // cudaEventRecord(total_end);
    // // Wait for the event to finish
    // cudaEventSynchronize(total_end);

    // cudaEventElapsedTime(&total_time, total_start, total_end);
    // // Total = kernel + data transfer
    // printf("Total execution time: %.5f milliseconds\n", total_time);

    int sum = 0;
    float normSum = 0.0;
    printf("Bin         Hist        Norm        CDF\n");
    for (size_t i = 0; i < GRAYLEVELS; i++)
    {
        printf("%3ld      %6d       %6f     %6f\n", i, h_histogram[i], h_histogram_norm[i], h_cdf[i]);
        sum += h_histogram[i];
        normSum += h_histogram_norm[i];
    }

    printf("Pixel count: %d\n", sum);
    printf("Norm sum: %f\n", normSum);
    printf("CDF last: %f\n", h_cdf[GRAYLEVELS-1]);
    printf("Image size: %d x %d (%d pixels)\n", width, height, width * height);

    // Write output image to file
    stbi_write_jpg("resources/output/kolesar-neq.jpg", width, height, DESIRED_NCHANNELS, h_imageOut, 100);

    // Free device memory (GPU)
    checkCudaErrors(cudaFree(d_imageOut));
    checkCudaErrors(cudaFree(d_imageIn));
    checkCudaErrors(cudaFree(d_histogram));
    checkCudaErrors(cudaFree(d_histogram_norm));
    checkCudaErrors(cudaFree(d_cdf));

    // Clean up the events
	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_end);
    cudaEventDestroy(total_start);
	cudaEventDestroy(total_end);

    // Free host memory
    free(h_imageIn);
    free(h_imageOut);
    free(h_histogram);
    free(h_histogram_norm);
    free(h_cdf);

    return 0;
}