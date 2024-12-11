#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include "omp.h"

#define NTHREADS 4
//#define NELEMENTS 1024*1024*32

#define NELEMENTS 64

int num_threads;

double* pvecA;
double* pvecB;
double* pvecC;
double dotproduct = 0.0;
double dps[NTHREADS];

// GCC: srun --reservation=fri gcc -fopenmp -o dp dotproduct.c
// srun --reservation=fri --cpus-per-taks=4 dp

int main(){

    // Ustvarim tri velike vektorje v pomnilniku (na kopici)
    pvecA = (double*) malloc(sizeof(double)*NELEMENTS);
    pvecB = (double*) malloc(sizeof(double)*NELEMENTS);
    pvecC = (double*) malloc(sizeof(double)*NELEMENTS);

    printf("MAX: %d \n", omp_get_max_threads());
    omp_set_num_threads(NTHREADS);

    // inicializiram vektorja A in B:

    #pragma omp parallel
    {
        printf("Sem nit %d od %d niti.\n", omp_get_thread_num(), omp_get_num_threads());
    }

    #pragma omp parallel for 
    for(int i = 0; i < NELEMENTS; i++) {
        *(pvecA+i) = 1.0;
        pvecB[i] = 2.0;
        
    }

    #pragma omp parallel for 
    for(int i = 0; i < NELEMENTS; i++){
        *(pvecC + i) = *(pvecA + i) * *(pvecB + i);
    }

    dotproduct = 0.0;
    //#pragma omp parallel for schedule(static)
    // schedule(static) - privzeto delovanje omp parallel for
    // Staticna razdelitev - najbolj pogosto uporabljena
    // razdeli na NTHREADS delov
    //#pragma omp parallel for schedule(dynamic,2)
    // Dinamicna razdelitv (0,1,2,3, 0,1,2,3..)
    // Vsaka nit izbere po n iteracij (2 -> nit1: 0,1, )
    // Guided -> zmanjsusje kose (najprej 4 iteracije ... na konuc vsaka po 1)

    // 3. Reduction
    // + uporaba rekukcije (sestevanje dotproduct)
    // reduction(+:dotproduct) operacija sestevanja nad dotproduct
    // atomic ni vec potrebno
    #pragma omp parallel for schedule(static) reduction(+:dotproduct) private(dotproduct)
    // What is firstprivate()
    for(int i = 0; i < NELEMENTS; i++){
        int id = omp_get_thread_num();
        printf("Nit %d, Iteracija %d\n", id, i);

        // 1. Critical section
        // #pragma omp critical
        // {
        //     // Kritican sekcija
        //     // samo 1 nit naenkrat lahko dostopa do dotprodut!
        //     // Serializacija (slabo!!)
        //     dotproduct += *(pvecC + i) ;
        // }
        
        // 2. Atomic
        // zaklene dotproduct (load, store)
        // #pragma omp atomic
        // dotproduct += *(pvecC + i) ;

    }
    // Merjenje casa znotraj omp 
    // built in fuctions

    // Spremenljivke znotraj OMP niti
    // Privatne (lokalne) - znotraj paralelnega obmocja
    // Vse kar je znotraj paralel obmocja (id, i) - avtomatsko private
    // private(dotproduct)

    // Za vsako nit svoja privatna
    // Shared (global) - izven obmocja
    // Za vse niti samo 1 skupna 
    // shared(dotproduct) - default at reduction
    printf("Skalarni produkt = %f\n", dotproduct);
}