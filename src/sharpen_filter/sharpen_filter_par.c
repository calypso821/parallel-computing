#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>
#include <pthread.h>
#include <stddef.h>
#include "stb_config.h"

#define NTHREADS 4

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define COLOR_CHANNELS 4
#define BSX 16
#define BSY 16

typedef struct {
    int id;
} args_t;


struct timespec timeStart, timeEnd;
pthread_t threads[NTHREADS];
args_t arguments[NTHREADS];

int width;
int height;
int cpp;
unsigned char *h_imageOut;
unsigned char *h_imageIn;

int getIntensity(const unsigned char *image, int row, int col, int channel, int height, int width, int cpp)
{
    if (row < 0 || row >= height) return 0;
    if (col < 0 || col >= width ) return 0;
    return image[(row * width + col) * cpp + channel];
}

void *filter(void *args);


int main(int argc, char *argv[]){

    if (argc < 2)
    {
        printf("USAGE: prog input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char szImage_in_name[255];
    snprintf(szImage_in_name, 255, "%s", argv[1]);

    // Load image from file and allocate space for the output image
    // Last parameter req_cpp changes strucutre to fit desired number of channels
    h_imageIn = stbi_load(szImage_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (h_imageIn == NULL)
    {
        printf("Error reading loading image %s!\n", szImage_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d, channels %d.\n", szImage_in_name, width, height, cpp);

    // Set color channels number to 4
    cpp = COLOR_CHANNELS;

    // Allocate memory for output image
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    h_imageOut = (unsigned char *)malloc(datasize);

    // Start measuring time
    clock_gettime(CLOCK_REALTIME, &timeStart);

    for (size_t i = 0; i < NTHREADS; i++) {
        arguments[i].id = i;

        pthread_create(
            &threads[i],            // kazalec na nit, ki jo ustvarjamo
            NULL,
            filter,              // funkcija, ki jo izvede ustvarjena nit
            (void*)(&arguments[i])
        );
    }

   
    for (size_t i = 0; i < NTHREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_REALTIME, &timeEnd);

    clock_gettime(CLOCK_REALTIME, &timeEnd);
    double elapsed_time = (timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_nsec - timeStart.tv_nsec) * 1e-9;
    double elapsed_time_ms = elapsed_time * 1000.0;
    printf("CPU (par) execution time: %.5f milliseconds\n", elapsed_time_ms);

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

    // FREE resources
    free(h_imageOut);


    return 0;
}

void *filter(void *args)
{
      // 1. Pixels of imageare stored in 1D array 
    // Greyscale: image[y * Width + x]
    // RGBA: image[(y * Width + x) * cpp + channel]

    // 2. Sharepning kernel 
    //***************************************************
    // Image sharpening using a 3x3 kernel
    //
    //      |  0  -1   0 |
    // K =  | -1   5  -1 |
    //      |  0  -1   0 |
    //
    //***************************************************

    // 3. Traverse every pixel 
    //      3.1 Tarverse all channels
    //      3.2 Extract intensitiy for 8 neighbour pixles
    //      3.3 Apply sharpen kernel and sum
    //      3.4 Save value to h_imageOut (RGBA) A=1

    args_t *parguments = (args_t*)args;
    int id = parguments->id;
    // number of threads = NTHREADS

    for (int y = id; y < height; y+=NTHREADS) {
        for (int x = 0; x < width; x++) {

            for (int c = 0; c < cpp; c++)
            {
                unsigned char px01 = getIntensity(h_imageIn, y - 1, x, c, height, width, cpp);
                unsigned char px10 = getIntensity(h_imageIn, y, x - 1, c, height, width, cpp);
                unsigned char px11 = getIntensity(h_imageIn, y, x, c, height, width, cpp);
                unsigned char px12 = getIntensity(h_imageIn, y, x + 1, c, height, width, cpp);
                unsigned char px21 = getIntensity(h_imageIn, y + 1, x, c, height, width, cpp);

                // Apply kernel
                short pxOut = 5 * px11 - px01 - px10 - px12 - px21;
                pxOut = MIN(pxOut, 255);
                pxOut = MAX(pxOut, 0);

                h_imageOut[(y * width + x) * cpp + c] = (unsigned char)pxOut;
            }
        }
    }

}