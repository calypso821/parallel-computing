include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

/*
https://solarianprogrammer.com/2019/06/10/c-programming-reading-writing-images-stb_image-libraries/
*/
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 4
#define BSX 16
#define BSY 16

int main(int argc, char *argv[]){

    if (argc < 3)
    {
        printf("USAGE: prog input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char szImage_in_name[255];
    char szImage_out_name[255];
    snprintf(szImage_in_name, 255, "%s", argv[1]);
    snprintf(szImage_out_name, 255, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *h_imageIn = stbi_load(szImage_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (h_imageIn == NULL)
    {
        printf("Error reading loading image %s!\n", szImage_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d, channels %d.\n", szImage_in_name, width, height, cpp);
    //ne glede na dejansko število kanalov bomo vedno predpostavili 4 kanale:
    cpp = COLOR_CHANNELS;

    // rezerviraj prostor v pomnilniku za izhodno sliko:
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *h_imageOut = (unsigned char *)malloc(datasize);


    // TODO: vaja



    // shranimo izhodno sliko
    stbi_write_jpg(szImage_out_name, width, height, cpp, h_imageOut, 100);


    // TODO 
    // FREE resources
    free(h_imageOut);


    return 0;
}