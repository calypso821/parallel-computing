#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define N 1024*1024*1024

#define BLOCKDIM 512
#define GRIDDIM N / BLOCKDIM

// Host
float *h_va;
float *h_vb;
float *h_vc;

// Device (GPU)
float *d_va;
float *d_vb;
float *d_vc;


// Koda za GPU napravo
// __global__ (KERNEL) funkcija, se izvaja na napravi, zaganan iz gostitelja 
// __device__ funkcija, se izvaja na napravi, zagana na napravi

__global__ void vec_mul(float *veca, float *vecb, float *vecc)
{
    // Index bloka * veliksot bloka 
    // + nit znotraj bloka
    // Dobimo globalni index vseh niti (1M)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Dekleracija skupnega pomnilnika
    __shared__ float vecC_perBlock[BLOCKDIM];

    // Init VecC_perBlock
    vecC_perBlock[threadIdx.x] = 0.0;

    // 1. Iteracija - 1M elemntov (milijon niti)

    // While is number or itterations N / thredas 
    // 1Miljarda elemnotv / 1M nitit
    while (tid < N)
    {
        //vecc[tid] = veca[tid] * vecb[tid];
        vecC_perBlock[threadIdx.x] += veca[tid] * vecb[tid];

        // Stevilo blokov - gridDim
        // stevilo niti v bloku - blockDim
        // tid + 1M (next itteration)
        tid = tid + gridDim.x * blockDim.x
    }

    // Sinhornizacija
    // Samo za niti v istem bloku!!
    __syncthreads();

    // Redukcija
    int i = blockDim.x / 2; // max index niti, ki dela redukcijo

    while (i > 0)
    {
        // Check if thread is working on reduction
        if (threadIdx.x < i)
        {
            // Sestejem svoj element + elemnt na (index + i)
            vecC_perBlock[threadIdx.x] += vecC_perBlock[threadIdx.x + i];
        }


        i = i / 2; // Razpolovim niti
        // Zadnja nit 1/2 -> 0 konec

        // Pocakamo vse nitit (tudi tiste ki ne sestevajo)!!!
        __syncthredas();
    }

    // Rezultat redukcije za en blokc 
    // Za en blok niti je rezultat v vecC_perBlock[0]
    // Toliko rezultatov, kot je vseh blockov!!
    // Prenesemo v pomnilnik na gostitelja

    if (threadIdx.x == 0)
    {
        // Samo 1x
        vecc[blockId.x] = vecC_perBlock[0];
    }

}


(const float *a, const float *b, const float *c, const in n)



// Koda za gostitelja (CPU)
int main(int argc, char *argv[])
{
    // Rezerviramo prostor na pomnilniku gostitelja
    h_va = (float *)malloc(N * sizeof(float));
    h_vb = (float *)malloc(N * sizeof(float));
    h_vc = (float *)malloc(GRIDDIM * sizeof(float));

    // Rezerviramo prostor na pomnilniku GPU
    cudaMalloc(&d_va, N * sizeof(float));
    cudaMalloc(&d_vb, N * sizeof(float));
    cudaMalloc(&d_vc, GRIDDIM * sizeof(float));

    // Vector initialization
    for (size_t i = 0; i < N; i++)
    {
        h_va[i] = 3.0;
        *(h_ba + i) = 2.0;
    }

    // cudaMemcpy
    // Naslov ponor podatkov (destination)
    // Naslov izvora podatkov (source)
    // Stevilo B za prenos
    // Smer prenosa (H -> D, D -> H)

    // Prenos podatkov iz pomnilnika gostitelja v pomnilnik naprav3
    cudaMemecpy((void *)d_va,
                (void *)h_va,
                N * sizeof(float),
                cudaMemcpyHostToDevice);

    cudaMemecpy((void *)d_vb,
                (void *)h_vb,
                N * sizeof(float),
                cudaMemcpyHostToDevice);

    // Zazeni kernel na napravi (GPU)
    // Niti v 1 bloku --> tvorjenje snopov
    dim3 threadsInBlock[BLOCKDIM, 1, 1];
    // St blokov 
    //dim3 numOfBlocks[N / threadsInBlock.x, 1, 1]; // 1M blokov
    dim3 numOfBlocks[GRIDDIM, 1, 1];

    vec_mul<<<numOfBlocks, threadsInBlock>>>>(d_va, d_vb, d_vc);

    // Prevajanje programa 
    // srun --partition=gpu nvcc dotprod.cu -o dtoprod
    // srun --partition=gpu --ntask=1 --gpus=1 --mem-per-cpu=1600MB? 

    // Prenos rezultate iz naprave v gostitelja
    cudaMemecpy((void *)h_vc,
                (void *)d_vc,
                GRIDDIM * sizeof(float),
                cudaMemcpyDeviceToHost);


    // Print result (should be 6)
    printf("Element: %d", h_vc[14]);

    float dotproduct = 0;
    for (int i = 0; i < GRIDDIM; i++)
    {
        dotproduct += h_vc[i];
    }
    printf("Dot product: %d", dotproduct);

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

