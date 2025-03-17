#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "cuda_config.h"
#include "stb_config.h"

#define COLOR_CHANNELS 4
#define BLOCK_SIZE 16

// Device code (executed from device - GPU)
__device__ inline unsigned char getIntensity(const unsigned char *image, int row, int col,
                                            int channel, int height, int width, int cpp)
{
    if (row < 0 || row >= height) return 0;
    if (col < 0 || col >= width ) return 0;
    return image[(row * width + col) * cpp + channel];
}

// Kernel (on GPU executed from host)
__global__ void sharpen_img(const unsigned char *imageIn, unsigned char *imageOut,
                            const int width, const int height, const int cpp)
{
    // Block index * size of block
    // + sequential thread id of block
    // Global index of all threads
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Each thread will process 1 pixel (4 channels)
    // If width/height does not allign with BLOCK_SIZE 16 
    // x or y could exceed width/height
    if (x < width && y < height)
    {
        for (int c = 0; c < cpp; c++)
        {
            unsigned char px01 = getIntensity(imageIn, y - 1, x, c, height, width, cpp);
            unsigned char px10 = getIntensity(imageIn, y, x - 1, c, height, width, cpp);
            unsigned char px11 = getIntensity(imageIn, y, x, c, height, width, cpp);
            unsigned char px12 = getIntensity(imageIn, y, x + 1, c, height, width, cpp);
            unsigned char px21 = getIntensity(imageIn, y + 1, x, c, height, width, cpp);

            // Apply kernel
            short pxOut = 5 * px11 - px01 - px10 - px12 - px21;
            pxOut = MIN(pxOut, 255);
            pxOut = MAX(pxOut, 0);

            imageOut[(y * width + x) * cpp + c] = (unsigned char)pxOut;
        }
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

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *h_imageIn = stbi_load(szImage_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (h_imageIn == NULL)
    {
        printf("Error reading loading image %s!\n", szImage_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d, channels %d.\n", szImage_in_name, width, height, cpp);

    unsigned char *d_imageIn;
    unsigned char *d_imageOut;

    cpp = COLOR_CHANNELS;

    // Allocate host memory
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *h_imageOut = (unsigned char *)malloc(datasize);

    // Allocate device memory
    checkCudaErrors(cudaMalloc(&d_imageIn, datasize));
    checkCudaErrors(cudaMalloc(&d_imageOut, datasize));

    printf("Initialization done.\n");

    // Block size: 16x16
    dim3 threadsInBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Number of blocks
    dim3 numOfBlocks(ceil(width / BLOCK_SIZE), ceil(height / BLOCK_SIZE), 1);

    printf("Computing matrix multiplication...\n");

    // CUDA events for measuring times
    float kernel_time = 0.0f; 
    float total_time = 0.0f;
    cudaEvent_t kernel_start, kernel_end, total_start, total_end;

    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_end);

    // Start recording (total time -> kernel to data transfer)
    cudaEventRecord(total_start);

    // Transfer data from host memory to device memory (GPU)
    // Input image
    checkCudaErrors(
        cudaMemcpy((void *)d_imageIn,
                (void *)h_imageIn,
                datasize,
                cudaMemcpyHostToDevice));

    // Start recording (kernel)
    cudaEventRecord(kernel_start);

    // Run kernel
    sharpen_img<<<numOfBlocks, threadsInBlock>>>(d_imageIn, d_imageOut, width, height, cpp);
    getLastCudaError("sharpen() execution failed");

    // Stop recording (kernel)
    cudaEventRecord(kernel_end);

    // Transfer data from GPU to host (ImageOut - sharpen image)
    checkCudaErrors(
        cudaMemcpy((void *)h_imageOut,
                (void *)d_imageOut,
                datasize,
                cudaMemcpyDeviceToHost));
    
    // Start recording (total time)
    cudaEventRecord(total_end);

    // Wait for the event to finish
    cudaEventSynchronize(kernel_end);
    cudaEventSynchronize(total_end);

    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_end);
    cudaEventElapsedTime(&total_time, total_start, total_end);
    printf("Image size: %d x %d (%d pixels), channels: %d\n", width, height, width * height, COLOR_CHANNELS);
    printf("Kernel execution time: %.5f milliseconds\n", kernel_time);
    // Total = kernel + data transfer
    printf("Total execution time: %.5f milliseconds\n", total_time);

    if (argc == 3)
    {
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
            stbi_write_png(szImage_out_name, width, height, cpp, h_imageOut, width * cpp);
        else if (!strcmp(FileType, "jpg"))
            stbi_write_jpg(szImage_out_name, width, height, cpp, h_imageOut, 100);
        else if (!strcmp(FileType, "bmp"))
            stbi_write_bmp(szImage_out_name, width, height, cpp, h_imageOut);
        else
            printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", FileType);
    }

    // Free device memory (GPU)
    checkCudaErrors(cudaFree(d_imageOut));
    checkCudaErrors(cudaFree(d_imageIn));

    // Clean up the events
	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_end);
    cudaEventDestroy(total_start);
	cudaEventDestroy(total_end);

    // Free host memory
    free(h_imageOut);
    free(h_imageIn);

    return 0;
}