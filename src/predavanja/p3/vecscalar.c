#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NELEMENTS 1024*1024
// ce imamo 1024*1024*512 uporabi double namesto float
#define NTHREADS 4

float* pvecA;
float* pvecB;
float* pvecC;

float dot_product = 0;

pthread_t nit[NTHREADS];
unsigned int id_niti[NTHREADS];

// Function decleration
void* mnozi(void* args);

// Both function and variable declerations 
// can be used as input to pthread_create() function
// If function is passed - automatically converted to address



int main(){
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
        *(pvecA + i) = 1.0;
        pvecB[i] = 2.0;
    }

    for (int i=0; i<NTHREADS; i++) {
        id_niti[i] = i;
        pthread_create(
            &nit[i],            // kazalec na nit, ki jo ustvarjamo
            NULL,
            mnozi,              // funkcija, ki jo izvede ustvarjena nit
            (void*)(&id_niti[i])
        );                      // argumenti funkcije, ki jo izvede ustvarjena nit - 
                                // edini argument je navaden void naslov, zato moram naslov na strukturo args1
                                // pretvorit v navaden void naslov

    }

    for (int i=0; i<NTHREADS; i++) {
        pthread_join(nit[i], NULL);
    }

    // Vse niti so se zakljucile in so izracunani vsi delni produkti v pvecC

    printf("Element 12450 vecC je %f \n", pvecC[12450]);

    // Sestevanje vseh delnih vrednsoti (neucinkovito)
    for (int i = 0; i < NELEMENTS; i++) {
        dot_product += pvecC[i];
    }

    printf("Skalarni produkt vecC je %f \n", dot_product);

    free(pvecA);
    free(pvecB);
    free(pvecC);
    return 0;
}

void* mnozi(void* args) {

    // args = void* 
    // pretvorimo v unsigned int* (pointer)
    // *() - dereferenciramo vrednsot
    unsigned int id = *((unsigned int*)args);

        // NELEMENTS / 2 -> 1/2 nit0, 2/2 nit1
    // / 2 <-- number of threads
    // Start index: thread_0 -> 0 
    // End index: thread_0 -> (1 * NELEMENTS/2) - 1  <-- if i < no -1 needed
    // Start index: thread_1 -> 1 * NELEMENTS/2
    // End index: thread_1 -> (2 * NELEMENTS/2) - 1  <-- if i < no -1 needed
    // Length of for loop: NELEMENTS / 2
    for (int i = 0; i < NELEMENTS/NTHREADS; i++)
    {
        pvecC[i+id*(NELEMENTS/NTHREADS)] = pvecA[i+id*(NELEMENTS/NTHREADS)] * pvecB[i+id*(NELEMENTS/NTHREADS)];
        // WRONG!! RWA nevarnost (read after write)
        // dot_product += pvecC[i+id*(NELEMENTS/NTHREADS)];
    }
}
