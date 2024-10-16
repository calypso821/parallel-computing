#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NELEMENTS 1024*1024

typedef struct
{
    unsigned int thread_ID;
    float* pvecA;
    float* pvecB;
    float* pvecC;
} argumenti_t;


pthread_t nit1;
pthread_t nit2;

// Function decleration
void* funkcija_niti(void* arg);
// Variable decleration (pointer to function)
void* (*pfunkcija_niti)(void*);

// Both function and variable declerations 
// can be used as input to pthread_create() function
// If function is passed - automatically converted to address

argumenti_t args1;
argumenti_t args2;

float* pvecA;
float* pvecB;
float* pvecC;


int main(){

    pfunkcija_niti = funkcija_niti;
    // dinamično ustvarim 3 velike vektorje v pomnilniku
    // (float*) - cast return of malloc to float pointer
    pvecA =  (float*)malloc(NELEMENTS*sizeof(float));
    pvecB =  (float*)malloc(NELEMENTS*sizeof(float));
    pvecC =  (float*)malloc(NELEMENTS*sizeof(float));

    // init vektorjev A in B:
    for (int i = 0; i < NELEMENTS; i++)
    {
        // pvecA - address (pointer)
        // + i (of size float)
        // * - dereference, so value can be set
        *(pvecA + i) = 2.0;
        pvecB[i] = 3.0;
    }

    args1.thread_ID = 0;
    args1.pvecA = pvecA;
    args1.pvecB = pvecB;
    args1.pvecC = pvecC;

    pthread_create(&nit1,           // kazalec na nit, ki jo ustvarjamo
            NULL,
            pfunkcija_niti,          // funkcija, ki jo izvede ustvarjena nit
            (void*) &args1);        // argumenti funkcije, ki jo izvede ustvarjena nit - 
                                    // edini argument je navaden void naslov, zato moram naslov na strukturo args1
                                    // pretvorit v navaden void naslov

    args2.thread_ID = 1;
    args2.pvecA = pvecA;
    args2.pvecB = pvecB;
    args2.pvecC = pvecC;
    pthread_create(&nit2,
            NULL,
            funkcija_niti,
            (void*) &args2);


    // Počakajmo , da se obe niti zaključita. V ta namen uporabim funkcijo join:
    pthread_join(nit1, NULL);
    pthread_join(nit2, NULL);

    // šele sedaj je varno zaključiti main().

    // preverimo rezultat:
    float rezultat = 0.0;
    for (int i = 0; i < NELEMENTS; i++)
    {
        // 5+5+5+5+5... 1024*1024
        rezultat += pvecC[i];
    }

    printf("Rezultat = %f \n", rezultat);
    

    free(pvecA);
    free(pvecB);
    free(pvecC);
    return 0;
}


void* funkcija_niti(void* arg){
    // argument arg je navaden void nasalov, ki ga želim uporabiti za dostop do elementov strukture, na katero kaže.
    // Zato moram definirati nov kazalec na sterukturo in argument castat na ta tip
    argumenti_t* argumenti = (argumenti_t*) arg;

    // NELEMENTS / 2 -> 1/2 nit0, 2/2 nit1
    // / 2 <-- number of threads
    // Start index: thread_0 -> 0 
    // End index: thread_0 -> (1 * NELEMENTS/2) - 1  <-- if i < no -1 needed
    // Start index: thread_1 -> 1 * NELEMENTS/2
    // End index: thread_1 -> (2 * NELEMENTS/2) - 1  <-- if i < no -1 needed
    // Length of for loop: NELEMENTS / 2
    for (int i = argumenti->thread_ID * (NELEMENTS/2); i < (argumenti->thread_ID + 1) * (NELEMENTS/2); i++)
    {
        argumenti->pvecC[i] = argumenti->pvecA[i] + argumenti->pvecB[i];
    }
    
    
}
