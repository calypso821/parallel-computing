#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "cuda_config.h"
#include "stb_config.h"

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
    __shared__ float local_cdf[GRAYLEVELS * 2];

    // Variable for read and write buffer (swapped every itteration)
    int wbuff = 1;
    int rbuff = 0;

    // Transfer normalized histogram into local memory
    local_cdf[tid] = histogram_norm[tid];
    __syncthreads();

    for (int offset = 1; offset < GRAYLEVELS; offset <<= 1)
    {
        // Working threads >= offset
        if (tid >= offset) {
            local_cdf[wbuff * GRAYLEVELS + tid] = local_cdf[rbuff * GRAYLEVELS + tid] + 
                                                   local_cdf[rbuff * GRAYLEVELS + tid - offset];
        } else {
            // Copy existing value
            local_cdf[wbuff * GRAYLEVELS + tid] =  local_cdf[rbuff * GRAYLEVELS + tid];
        }
        __syncthreads();
        
        // Flip
        wbuff = 1 - wbuff;
        rbuff = 1 - rbuff;
    }

    // Transfer local histogram back to global memory
    cdf[tid] = local_cdf[rbuff * GRAYLEVELS + tid];
}
__global__ void cdf_blelloch(const float *histogram_norm, float *cdf)
{
    // Threads working based of blanaced binary tree (of partial sums)
    // 1 block of 256 threads
    int tid = threadIdx.x; // 0 to 255

    // Local cdf
    __shared__ float local_cdf[GRAYLEVELS + 1];
    // Transfer normalized histogram into local memory
    local_cdf[tid] = histogram_norm[tid];
    __syncthreads();

    // Up-sweep (building tree from leaves) - reduce phase
    // Sum 2 nodes (partial sums) and write into right (copy left)
    // offset 2,4,8...
    // start with offset - 1, unitl offset <= N
    int offset;
    // 128 to 1
    for (int i = GRAYLEVELS/2; i >= 1; i >>= 1)
    {
        // Thread offset
        offset = GRAYLEVELS / i; // starts with 2, 4, 8...
        // Working threads
        if (tid < i) {
            local_cdf[(tid + 1) * offset - 1] += local_cdf[(tid + 1) * offset - 1 - offset/2]; 
        }
        __syncthreads();
    }

    // Down-sweep 
    // Root = 0 (+ save root value buff +1)
    // On each level... 
    // 0. Save right node to temp
    // 1. Sum nodes on same level
    // 2. pass value from right node to left (temp)

    // Trasnfer root to last element and set 0
    if (tid == 0) {
        local_cdf[GRAYLEVELS] = local_cdf[GRAYLEVELS-1];
        local_cdf[GRAYLEVELS-1] = 0;
    }
    __syncthreads();

    // 1 to 128
    for (int i = 1; i < GRAYLEVELS; i <<= 1)
    {
        // Thread offset
        offset = GRAYLEVELS / i; // starts with 8, 4, 2
        // Working threads
        if (tid < i) {
            // Value from right node
            float temp = local_cdf[(tid + 1) * offset - 1];
            // Sum nodes on same level
            local_cdf[(tid + 1) * offset - 1] += local_cdf[(tid + 1) * offset - 1 - offset/2]; 
            // Pass value from right node to left (temp)
            local_cdf[(tid + 1) * offset - 1 - offset/2] = temp;
        }
        __syncthreads();
    }

    // Transfer cdf back to global memory
    // current cdf 0, x0 ... xn, + 1 without leading 0
    cdf[tid] = local_cdf[tid + 1];
}

// Kernel (on GPU executed from host)
__global__ void map_equalize(const unsigned char *imageIn, const int width, const int height,
                             const float *cdf, const float *cdf_min, unsigned char *imageOut)
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

    // cdf_min - find by reduction in previous step
    float cdf_max = 1.0f;

    if (col < width && row < height)
    {
        // 1. Get pixel from input image
        unsigned char pixelIn = imageIn[row * width + col];
        // cdf_i representing probability for selected intenisity to 0
        float cdf_i = cdf[pixelIn];

        // 2. Transfrom (equalize) pixel with CDF distribution
        float scale = (cdf_i - cdf_min[0]) / (cdf_max - cdf_min[0]);
        // Map back to values from 0-255
        float pixelOut = roundf(scale * (GRAYLEVELS-1));

        // 3. Map to output image
        imageOut[row * width + col] = pixelOut;
    }
}

__global__ void findMin(const float *cdf, float *min)
{
    // find Non-zero min using reduction!!
    // 1 block of 256 threads
    int tid = threadIdx.x;

    // Local cdf
    __shared__ float local_cdf[GRAYLEVELS];
    // Transfer cdf into shared memory
    local_cdf[tid] = cdf[tid];
    __syncthreads();

    float val1, val2;
    for (int i = blockDim.x / 2; i >= 1; i >>= 1)
    {
        // Working threads
        if (tid < i) {
            val1 = local_cdf[tid];
            val2 = local_cdf[tid + i];

            if (val1 > 0.0f && val2 > 0.0f) {
                // Both values are non-zero
                local_cdf[tid] = MIN(val1, val2);
            } else {
                // 1 or both values are 0.0
                local_cdf[tid] = val1 == 0.0f ? val2 : val1;
            }
        }
        __syncthreads();
    }

    // Transfer min value to memory 
    if (tid == 0) {
        min[0] = local_cdf[0];
    }
}

int main(int argc, char *argv[]){

    if (argc < 2)
    {
        printf("USAGE: prog input_image [output_image]\n");
        exit(EXIT_FAILURE);
    }

    char szImage_in_name[255];
    snprintf(szImage_in_name, 255, "%s", argv[1]);

    // Read image from file
    int width, height, cpp;

    // read only DESIRED_NCHANNELS channels from the input image:
    unsigned char *h_imageIn = stbi_load(szImage_in_name, &width, &height, &cpp, DESIRED_NCHANNELS);
    if(h_imageIn == NULL) {
        printf("Error in loading the image\n");
        return 1;
    }
    printf("Loaded image %s of size %dx%d, actual channels %d.\n", szImage_in_name, width, height, cpp);

    int imageSize = height * width * DESIRED_NCHANNELS * sizeof(unsigned char);
    int histSize = GRAYLEVELS * sizeof(unsigned int);
    int histNormSize = GRAYLEVELS * sizeof(float);

    // Allocate memory for raw output image data, histogram, and CDF 
	unsigned char *h_imageOut = (unsigned char *)malloc(imageSize);
    unsigned int *h_histogram = (unsigned int *)malloc(histSize);
    float *h_histogram_norm = (float *)malloc(histNormSize);
    float *h_cdf = (float *)malloc(histNormSize);
    float *h_cdf_naive = (float *)malloc(histNormSize);
    float *h_cdf_min = (float *)malloc(sizeof(float));

    unsigned char *d_imageIn;
    unsigned char *d_imageOut;
    unsigned int *d_histogram;
    float *d_histogram_norm;
    float *d_cdf;
    float *d_cdf_naive;
    float *d_cdf_min;

    // Allocate device memory with error checking
    checkCudaErrors(cudaMalloc(&d_imageIn, imageSize));
    checkCudaErrors(cudaMalloc(&d_imageOut, imageSize));
    checkCudaErrors(cudaMalloc(&d_histogram, histSize));
    checkCudaErrors(cudaMalloc(&d_histogram_norm, histNormSize));
    checkCudaErrors(cudaMalloc(&d_cdf, histNormSize));
    checkCudaErrors(cudaMalloc(&d_cdf_naive, histNormSize));
    checkCudaErrors(cudaMalloc(&d_cdf_min, sizeof(float)));

    // Initialize histogram memory to zero with error checking
    checkCudaErrors(cudaMemset(d_histogram, 0, histSize));
    checkCudaErrors(cudaMemset(d_histogram_norm, 0, histNormSize));
    checkCudaErrors(cudaMemset(d_cdf, 0, histNormSize));
    checkCudaErrors(cudaMemset(d_cdf_naive, 0, histNormSize));

    printf("Initialization done.\n");

    // Block size: 16x16
    // image processing
    dim3 threadsInBlock(BLOCK_SIZE, BLOCK_SIZE, 1); // 16x16
    dim3 numOfBlocks(ceil(width / BLOCK_SIZE), ceil(height / BLOCK_SIZE), 1);

    // Histogram/CDF processing
    dim3 threadsInBlockHist(GRAYLEVELS, 1, 1); 
    dim3 numOfBlocksHist(1, 1, 1);

    printf("============= KERNEL MEASUREMENT =============\n");
    float kernel_time = 0.0f; 
    cudaEvent_t kernel_start, kernel_end;

    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);

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
    getLastCudaError("hist_naive() execution failed");
    cudaEventRecord(kernel_end);
    cudaEventSynchronize(kernel_end);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_end);
    printf("Hist (naive): %.5f milliseconds\n", kernel_time);
    checkCudaErrors(cudaMemset(d_histogram, 0, histSize)); // clear hist
    // local memory + atmoic
    cudaEventRecord(kernel_start);
    hist_local<<<numOfBlocks, threadsInBlock>>>(d_imageIn, width, height, d_histogram);
    getLastCudaError("hist_local_mem() execution failed");
    cudaEventRecord(kernel_end);
    cudaEventSynchronize(kernel_end);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_end);
    printf("Hist (local mem): %.5f milliseconds\n", kernel_time);

    // =================== Normalize hist ================= 
    hist_normalize<<<numOfBlocksHist, threadsInBlockHist>>>(d_histogram, d_histogram_norm, width * height);
    getLastCudaError("hist_normalize() execution failed");

    // =================== CDF =================
    // (Comulative distribution function) 
    // Naive (Algorithm Hillis and Steele)
    cudaEventRecord(kernel_start);
    cdf_naive<<<numOfBlocksHist, threadsInBlockHist>>>(d_histogram_norm, d_cdf_naive);
    getLastCudaError("hist_cdf_naive() execution failed");
    cudaEventRecord(kernel_end);
    cudaEventSynchronize(kernel_end);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_end);
    printf("CDF (naive): %.5f milliseconds\n", kernel_time);
    //Work-Efficient Parallel Scan (Blelloch)
    cudaEventRecord(kernel_start);
    cdf_blelloch<<<numOfBlocksHist, threadsInBlockHist>>>(d_histogram_norm, d_cdf);
    getLastCudaError("hist_cdf_blelloch() execution failed");
    cudaEventRecord(kernel_end);
    cudaEventSynchronize(kernel_end);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_end);
    printf("CDF (blelloch): %.5f milliseconds\n", kernel_time);

    // ============== Find min ===============
    // (Reduction)
    findMin<<<numOfBlocksHist, threadsInBlockHist>>>(d_cdf, d_cdf_min);
    getLastCudaError("hist_find_min() execution failed");

    // ============== Equlize & Map ===============
    map_equalize<<<numOfBlocks, threadsInBlock>>>(d_imageIn, width, height, d_cdf, d_cdf_min, d_imageOut);
    getLastCudaError("hist_equalize_map() execution failed");

    // Transfer data from GPU to host (Histogram)
    checkCudaErrors(
        cudaMemcpy((void *)h_imageOut,
                (void *)d_imageOut,
                imageSize,
                cudaMemcpyDeviceToHost));

    // ============= DEBUG ONLY ===============
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
        cudaMemcpy((void *)h_cdf_naive,
                (void *)d_cdf_naive,
                histNormSize,
                cudaMemcpyDeviceToHost));

    checkCudaErrors(
        cudaMemcpy((void *)h_cdf_min,
                (void *)d_cdf_min,
                sizeof(float),
                cudaMemcpyDeviceToHost));
                    
    printf("==============================================\n");

    int sum = 0;
    float normSum = 0.0;
    printf("==================== DEBUGG ==================\n");
    printf("Bin         Hist        Norm        CDF-naive   CDF-Blelloch\n");
    printf("%3ld      %6d       %6f     %6f     %6f\n", 0, h_histogram[0], h_histogram_norm[0], h_cdf_naive[0], h_cdf[0]);
    printf("%3ld      %6d       %6f     %6f     %6f\n", 100, h_histogram[100], h_histogram_norm[100], h_cdf_naive[100], h_cdf[100]);
    printf("%3ld      %6d       %6f     %6f     %6f\n", 127, h_histogram[127], h_histogram_norm[127], h_cdf_naive[127], h_cdf[127]);
    printf("%3ld      %6d       %6f     %6f     %6f\n", 150, h_histogram[150], h_histogram_norm[150], h_cdf_naive[150], h_cdf[150]);
    printf("%3ld      %6d       %6f     %6f     %6f\n", 255, h_histogram[255], h_histogram_norm[255], h_cdf_naive[255], h_cdf[255]);
    for (size_t i = 0; i < GRAYLEVELS; i++)
    {
        //printf("%3ld      %6d       %6f     %6f     %6f\n", i, h_histogram[i], h_histogram_norm[i], h_cdf_naive[i], h_cdf[i]);
        sum += h_histogram[i];
        normSum += h_histogram_norm[i];
    }

    printf("Pixel count: %d\n", sum);
    printf("Norm sum: %f\n", normSum);
    printf("CDF last: %f\n", h_cdf[GRAYLEVELS-1]);
    printf("CDF min: %f\n", h_cdf_min[0]);
    printf("Histogram size (GREYLEVELS): %d\n", GRAYLEVELS);
    printf("Image size: %d x %d (%d pixels), channels: %d\n", width, height, width * height, DESIRED_NCHANNELS);
    printf("==============================================\n");

    printf("============== FULL MEASUREMENT ==============\n");
    printf("Full measurement = data transfer + kernels\n");
    printf("Data transfer (image): CPU to GPU, GPU to CPU\n");
    printf("Kernels: histogram, normalize, cdf (blelloch), min_cdf (reduction), map + equalize\n");

    float total_time = 0.0f;
    cudaEvent_t total_start, total_end;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_end);

    // Start recording (total time -> kernel to data transfer)
    cudaEventRecord(total_start);

    // Transfer data from host memory to device memory (Input image)
    checkCudaErrors(
        cudaMemcpy((void *)d_imageIn,
                    (void *)h_imageIn,
                    imageSize,
                    cudaMemcpyHostToDevice));

    cudaEventRecord(kernel_start);

    // =================== HISTOGRAM =================
    // local memory + atmoic
    checkCudaErrors(cudaMemset(d_histogram, 0, histSize)); // clear hist
    hist_local<<<numOfBlocks, threadsInBlock>>>(d_imageIn, width, height, d_histogram);
    getLastCudaError("hist_local_mem() execution failed");

    // =================== NORMALIZE hist ================= 
    hist_normalize<<<numOfBlocksHist, threadsInBlockHist>>>(d_histogram, d_histogram_norm, width * height);
    getLastCudaError("hist_normalize() execution failed");

    // =================== CDF =================
    // (Comulative distribution function) 
    // Work-Efficient Parallel Scan (Blelloch)
    cdf_blelloch<<<numOfBlocksHist, threadsInBlockHist>>>(d_histogram_norm, d_cdf);
    getLastCudaError("hist_cdf_blelloch() execution failed");

    // ============== Find min ===============
    // (Reduction)
    findMin<<<numOfBlocksHist, threadsInBlockHist>>>(d_cdf, d_cdf_min);
    getLastCudaError("hist_find_min() execution failed");

    // ============== Equlize & Map ===============
    map_equalize<<<numOfBlocks, threadsInBlock>>>(d_imageIn, width, height, d_cdf, d_cdf_min, d_imageOut);
    getLastCudaError("hist_equalize_map() execution failed");

    cudaEventRecord(kernel_end);
    cudaEventSynchronize(kernel_end);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_end);
    printf("Kernel execution time: %.5f milliseconds\n", kernel_time);

    // Transfer data from GPU to host (Output image)
    checkCudaErrors(
        cudaMemcpy((void *)h_imageOut,
                    (void *)d_imageOut,
                    imageSize,
                    cudaMemcpyDeviceToHost));
    // =====================================

    // Start recording (total time)
    cudaEventRecord(total_end);
    // Wait for the event to finish
    cudaEventSynchronize(total_end);
    cudaEventElapsedTime(&total_time, total_start, total_end);
    // Total = kernel + data transfer
    printf("Total execution time: %.5f milliseconds\n", total_time);
    printf("==============================================\n");

    // Write output image to file
    if (argc == 3) {
        char szImage_out_name[255];
        snprintf(szImage_out_name, 255, "%s", argv[2]);

        // Retrieve output file type
        char szImage_out_name_temp[255];
        strncpy(szImage_out_name_temp, szImage_out_name, 255);

        char *token = strtok(szImage_out_name_temp, ".");
        char *FileType = NULL;
        while (token != NULL)
        {
            FileType = token;
            token = strtok(NULL, ".");
        }

        // Write output image to file
        if (!strcmp(FileType, "png"))
            stbi_write_png(szImage_out_name, width, height, DESIRED_NCHANNELS, h_imageOut, width);
        else if (!strcmp(FileType, "jpg"))
            stbi_write_jpg(szImage_out_name, width, height, DESIRED_NCHANNELS, h_imageOut, 100);
        else if (!strcmp(FileType, "bmp"))
            stbi_write_bmp(szImage_out_name, width, height, DESIRED_NCHANNELS, h_imageOut);
        else
            printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", FileType);
    }

    // Free device memory (GPU)
    checkCudaErrors(cudaFree(d_imageOut));
    checkCudaErrors(cudaFree(d_imageIn));
    checkCudaErrors(cudaFree(d_histogram));
    checkCudaErrors(cudaFree(d_histogram_norm));
    checkCudaErrors(cudaFree(d_cdf));
    checkCudaErrors(cudaFree(d_cdf_naive));
    checkCudaErrors(cudaFree(d_cdf_min));

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
    free(h_cdf_naive);

    return 0;
}