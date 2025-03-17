#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

// Stevilo niti
#define NMISLECEV 4

pthread_mutex_t lock;

struct timespec timeStart, timeEnd;

typedef struct {
    unsigned int id;
} argument_thread_t;


argument_thread_t arguments[NMISLECEV];

// Niti
pthread_t mislec[NMISLECEV];
// Kljucavnice
pthread_mutex_t bambusova_palcka[NMISLECEV];

// Function decleration
void *misli_in_jej(void *args);

// Both function and variable declerations 
// can be used as input to pthread_create() function
// Function always saved as address
// If function is passed - automatically converted to address

int main(){
    for (int i = 0; i < NMISLECEV; i++) {
        pthread_mutex_init(&bambusova_palcka[NMISLECEV], NULL);
    }



    for (int i = 0; i < NMISLECEV; i++) {
        arguments[i].id = i;
        pthread_create(
            &mislec[i],            // kazalec na nit, ki jo ustvarjamo
            NULL,
            misli_in_jej,              // funkcija, ki jo izvede ustvarjena nit
            (void*)(&arguments[i])
        );                      // argumenti funkcije, ki jo izvede ustvarjena nit - 
                                // edini argument je navaden void naslov (1B), zato moram naslov na strukturo args1
                                // pretvorit v navaden void naslov

    }

    for (int i = 0; i < NMISLECEV; i++) {
        pthread_join(mislec[i], NULL);
    }


    // Start measuring time
    //clock_gettime(CLOCK_REALTIME, &timeStart);
    //clock_gettime(CLOCK_REALTIME, &timeEnd);

    // seconds (long int)
    // nano seconds (long int) / 1e9 (1 bill) -> convert to nano seconds (double)
    //double elapsed_time = (timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_nsec - timeStart.tv_nsec) / 1e9;
    // nano seconds (long int) * 1e-9 (multiply nano) -> convert to nano seconds (double)
    // double elapsed_time = (timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_nsec - timeStart.tv_nsec) * 1e-9;
    // printf("Elapsed time: %.9f seconds\n", elapsed_time);

    
    for (int i = 0; i < NMISLECEV; i++) {
        pthread_mutex_destroy(&bambusova_palcka[i]);
    }

    return 0;
}


void *misli_in_jej(void *args) {
    argument_thread_t *argument = (argument_thread_t*)args;
    unsigned int id = argument->id;

    for (int i = 0; i< 100; i++) {
        // 1. Misli (pocakaj nakljucno )
        usleep(117*(id * i));

        // 2. Jej
        // Leva = id niti
        pthread_mutex_lock(&bambusova_palcka[id]);
        printf("Mislec %d vzame L palcko %d\n", id, id);
        // Desna = id niti - 1 mod NMISLEC
        usleep(35*(id * i) - 15 * id);
        pthread_mutex_lock(&bambusova_palcka[(id - 1) % NMISLECEV]);
        printf("Mislec %d vzame D palcko %d\n", id, (id - 1) % NMISLECEV);

        // Hmmm... how mod at 0 works?? 
        // 0 - 1 mod 5? res should be 4

        // Cas, ki je 
        usleep(319*(id * i) - 15 * id);

        // 3. Sprostitev palck
        // Desna = id niti - 1 mod NMISLEC
        pthread_mutex_unlock(&bambusova_palcka[(id - 1) % NMISLECEV]);
        printf("Mislec %d vrne D palcko %d\n", id, (id - 1) % NMISLECEV);
                usleep(71*(id * i) - 15 * id);
        // Leva = id niti
        pthread_mutex_unlock(&bambusova_palcka[id]);
        printf("Mislec %d vrne L palcko %d\n", id, id);

    }
}

 