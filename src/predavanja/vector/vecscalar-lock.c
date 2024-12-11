#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define NELEMENTS 1024*1024*20
// ce imamo 1024*1024*512 uporabi double namesto float
#define NTHREADS 16

pthread_mutex_t lock;

struct timespec timeStart, timeEnd;

typedef struct {
    unsigned int id;
    double *psum;
} argument_thread_t;

double* pvecA;
double* pvecB;
double* pvecC;

double dot_product = 0;

pthread_t nit[NTHREADS];
argument_thread_t arguments[NTHREADS];
double local_sums[NTHREADS];

// Function decleration
void *multiplay(void *args);

// Both function and variable declerations 
// can be used as input to pthread_create() function
// If function is passed - automatically converted to address

int main(){
    pthread_mutex_init(&lock, NULL);

    // dinamično ustvarim 3 velike vektorje v pomnilniku
    // (double*) - cast return of malloc to double pointer
    pvecA =  (double*)malloc(NELEMENTS*sizeof(double));
    pvecB =  (double*)malloc(NELEMENTS*sizeof(double));
    pvecC =  (double*)malloc(NELEMENTS*sizeof(double));

    // init vektorjev A in B:
    for (int i = 0; i < NELEMENTS; i++)
    {
        // pvecA - address (pointer)
        // + i (of size double)
        // * - dereference, so value can be set
        *(pvecA + i) = 1.0;
        pvecB[i] = 2.0;
    }

    // Start measuring time
    clock_gettime(CLOCK_REALTIME, &timeStart);

    for (int i=0; i<NTHREADS; i++) {
        arguments[i].id = i;
        arguments[i].psum = &local_sums[i];
        pthread_create(
            &nit[i],            // kazalec na nit, ki jo ustvarjamo
            NULL,
            multiplay,              // funkcija, ki jo izvede ustvarjena nit
            (void*)(&arguments[i])
        );                      // argumenti funkcije, ki jo izvede ustvarjena nit - 
                                // edini argument je navaden void naslov (1B), zato moram naslov na strukturo args1
                                // pretvorit v navaden void naslov

    }

    for (int i=0; i<NTHREADS; i++) {
        pthread_join(nit[i], NULL);
    }

    // 1. Sestevanje vseh delnih vrednsoti (NEUCINKOVITO) - NOT OK
    // for (int i = 0; i < NELEMENTS; i++) {
    //     dot_product += pvecC[i];
    // }

    // 2. Sestevanje lokalnih vsot (UCINKOVITO) - OK
    // for (int i=0; i<NTHREADS; i++) {
    //     //printf("Local sum [%d]: %f\n", i, local_sums[i]);
    //     dot_product += local_sums[i];
    // }

    // 3. Blocking read/wrtie (direct dot_product access) - mutex
    // Sequentail dot_product addition (like in 1. example)


    clock_gettime(CLOCK_REALTIME, &timeEnd);

    // Vse niti so se zakljucile in so izracunani vsi delni produkti v pvecC

    printf("Element 12450 vecC je %f \n", pvecC[12450]);

    printf("Skalarni produkt vecC je %f\n", dot_product);

    // seconds (long int)
    // nano seconds (long int) / 1e9 (1 bill) -> convert to nano seconds (double)
    //double elapsed_time = (timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_nsec - timeStart.tv_nsec) / 1e9;
    // nano seconds (long int) * 1e-9 (multiply nano) -> convert to nano seconds (double)
    double elapsed_time = (timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_nsec - timeStart.tv_nsec) * 1e-9;
    printf("Elapsed time: %.9f seconds\n", elapsed_time);

    free(pvecA);
    free(pvecB);
    free(pvecC);
    pthread_mutex_destroy(&lock);

    return 0;
}


void *multiplay(void *args) {

    // args = void* 
    // pretvorimo v unsigned int* (pointer)
    // *() - dereferenciramo vrednsot
    argument_thread_t *argument = (argument_thread_t*)args;
    unsigned int id = argument->id;
    double *plocal_sum = argument->psum;

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
        // 1. WRONG!! RWA nevarnost (read after write)
        // dot_product += pvecC[i+id*(NELEMENTS/NTHREADS)];

        // 2. Local sum
        //*plocal_sum += pvecC[i+id*(NELEMENTS/NTHREADS)];

        // 3. Blocking read/write (mutex) - NO PARALLISM
        // Lock (blocking function)
        pthread_mutex_lock(&lock);
        // Critical section
        dot_product += pvecC[i+id*(NELEMENTS/NTHREADS)];
        // Unlock
        pthread_mutex_unlock(&lock);
    }
}