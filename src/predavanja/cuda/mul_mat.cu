#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define ROWS 1024*8
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


// Koda za GPU napravo
// __global__ (KERNEL) funkcija, se izvaja na napravi, zaganan iz gostitelja 
// __device__ funkcija, se izvaja na napravi, zagana na napravi

__global__ void mul_mat_naive(float *mata, float *matb, float *matc)
{
    // Index bloka * veliksot bloka 
    // + nit znotraj bloka
    // Dobimo globalni index vseh niti (1M)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Stevilo niti = velikost matrike C (ROWS * COLS)
    // Vsaka nit racuna 1 element matrike C

    matc[row * ROWS + col] = 0.0;

    for (size_t i = 0; i < N; i++ )
    {
        // Memory coalessing PROBLEM!!
        // Sosednji element dosotopa do istih elemntov ko prejsnji (row, col)
        // V celotni vrstici / stolpcu dostopamo do enakih elemntov

        // matC[row, col] = matA[row, i:0..N] * matB[i:0..N, col]
        // Premik po vrstici --> mata[row * ROWS + i]
        // Premik po stolpcu --> matb[i * ROWS + col]
        matc[row * ROWS + col] += mata[row * ROWS + i] * matb[i * ROWS + col];
    }
}


__global__ void mul_mat_tiles(float *mata, float *matb, float *matc)
{
    // Index bloka * veliksot bloka 
    // + nit znotraj bloka
    // Dobimo globalni index vseh niti (1M)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    float c = 0.0;

    // Prostor za plscici v shread memory
    __shared__ float tileA[BLOCKDIM][BLOCKDIM];
    __shared__ float tileB[BLOCKDIM][BLOCKDIM];

    // Po vseh ploscicah (tilih) po matriki A in B, ki jih potrebujemo za izracun C
    // Izbiram in nalagam jih po parih

    for (int tile = 0; tile < (N / BLOCKDIM); tile++)
    {
        // preberimo poscici A in B iz globalnega pomnilnika
        // Vsaka nit v bloku prenese 1 element
        // Lokalni ploscicic 

        // * BLOCKDIM (linearizacija)?? 
        // mata[blockIdx.y * blockDim.y + local_row][tile * blockDim.x + local_col] NOT OK??? 
        // mata in matb ni matrika?? Just array?? 

        tileA[local_row][local_col] = mata[(blockIdx.y * blockDim.y + local_row) * BLOCKDIM + 
                                            (tile * blockDim.x + local_col)]
        tileB[local_row][local_col] = matb[(tile * blockDim.y + local_row) * BLOCKDIM + 
                                            (blockIdx.x * blockDim.x + local_col)]


        // Pocakaj, da vse niti prenesejo svoje elemente
        __syncthreads();

        // Zmnozi ploscici in rezultat pisi v plosico C
        for (int i = 0; i < BLOCKDIM; i++)
        {
            c += tileA[local_row][i] * tileB[i][local_col];
        }

        // Pocakaj, pred ponovnim nalaganjem novih ploscic
        __syncthreads();
    }

    // Prenesi element c nazaj v globalni pomnilnik v matriko C
    // What is rowN???
    // ROWS?? or
    matc[rowN, col] = c;

}

//(const float *a, const float *b, const float *c, const in n)



// Koda za gostitelja (CPU)
int main(int argc, char *argv[])
{
    // Rezerviramo prostor na pomnilniku gostitelja
    h_ma = (float *)malloc(MAT_SIZE * sizeof(float));
    h_mb = (float *)malloc(MAT_SIZE * sizeof(float));
    h_mc = (float *)malloc(MAT_SIZE * sizeof(float));

    // Rezerviramo prostor na pomnilniku GPU
    cudaMalloc(&d_va, MAT_SIZE * sizeof(float));
    cudaMalloc(&d_vb, MAT_SIZE * sizeof(float));
    cudaMalloc(&d_vc, MAT_SIZE * sizeof(float));

    // Vector initialization
    for (size_t i = 0; i < ROWS; i++) {
        for (size_t j = 0; i < COLS; j++)
        {
            h_ma[i * ROWS + j] = 1.0f;
            h_mb[i * ROWS + j] = 2.0f;
        }
    }

    // cudaMemcpy
    // Naslov ponor podatkov (destination)
    // Naslov izvora podatkov (source)
    // Stevilo B za prenos
    // Smer prenosa (H -> D, D -> H)

    // Prenos podatkov iz pomnilnika gostitelja v pomnilnik naprav3
    cudaMemecpy((void *)d_ma,
                (void *)h_ma,
                MAT_SIZE * sizeof(float),
                cudaMemcpyHostToDevice);

    cudaMemecpy((void *)d_mb,
                (void *)h_mb,
                MAT_SIZE * sizeof(float),
                cudaMemcpyHostToDevice);

    // Zazeni kernel na napravi (GPU)
    // Niti v 1 bloku --> tvorjenje snopov
    // Skupaj blok 256 (2^8)
    // rows: 2^4, cols: 2^4 (16x16)
    // rows: 2^5, cols: 2^3 (32x8)
    dim3 threadsInBlock[BLOCKDIM, BLOCKDIM, 1];
    // St blokov 
    // Y (rows): (2^20 / 2^4 = 2^16) 64K (1024 * 1024 elements)
    // X (cols): (2^10 / 2^4 = 2^6) 64 (1024 elements)
    dim3 numOfBlocks[ROWS/BLOCKDIM, COLS/BLOCKDIM, 1];

    // CUDA events for measuring times
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);


    // Start recording
    cudaEventRecorde(start);

    add_mat<<<numOfBlocks, threadsInBlock>>>>(d_ma, d_mb, d_mc);

    // Stop recording
    cudaEventRecorde(end);
    cudaEventSynchronize(end);

    float miliseconds = 0.0;
    cudaEventElapsedTime(&miliseconds, start, end);
    printf("Kernel execution time: %0.3f miliseconds\n", miliseconds);

    // Prevajanje programa 
    // srun --partition=gpu nvcc dotprod.cu -o dtoprod
    // srun --partition=gpu --ntask=1 --gpus=1 --mem-per-cpu=1600MB? 

    // Prenos rezultate iz naprave v gostitelja
    cudaMemecpy((void *)h_mc,
                (void *)d_mc,
                MAT_SIZE * sizeof(float),
                cudaMemcpyDeviceToHost);


    // Print result (should be 6)
    printf("Element: %d", h_mc[567 * ROWS + 120]);
    // Delni produkt bedno 2 (1 * 2)
    // Dimenzija = 1024 * 8 
    // 2 * 8K --> 16K
    // Element should be ~ 16k

    // Sprostimo prostor gostitelja
    free(h_ma);
    free(h_mb);
    free(h_mc);

    // Sprostimo prostor na naprvi (GPU)
    cudaFree(d_ma);
    cudaFree(d_mb);
    cudaFree(d_mc);

    return 0;
}

