#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// struct argumenti {
//     unsigned int thread_ID;
// };
// Usage struct argumenti arg;

// Define struct type 
typedef struct
{
    unsigned int thread_ID;
} argumenti_t;


pthread_t nit1;
pthread_t nit2;

// Function decelration 
// returns void pointer
// argument void pointer
void* funkcija_niti(void* arg);

argumenti_t args1;
argumenti_t args2;

// Variable decleration
// pointer pfunkcija_niti
// (*pfunkcija_niti) = pointer to a function
// points to function which takes void* argument
// points to function which returns void*
void* (*pfunkcija_niti)(void*) = &funkcija_niti;

// void* pfunkcija_niti(void*) = function decleartion
// void* *pfunkcija_niti(void*) = void** pfunkcija_niti(void*) 
// function (returns pointer to a void*)


int main(){


    args1.thread_ID = 1;
    pthread_create(&nit1,           // kazalec na nit, ki jo ustvarjamo
            NULL,
            funkcija_niti,          // funkcija, ki jo izvede ustvarjena nit
            (void*) &args1);        // argumenti funkcije, ki jo izvede ustvarjena nit - 
                                    // edini argument je navaden void naslov, zato moram naslov na strukturo args1
                                    // pretvorit v navaden void naslov

    args2.thread_ID = 2;
    pthread_create(&nit2,
            NULL,
            funkcija_niti,
            (void*) &args2);


    // Počakajmo , da se obe niti zaključita. V ta namen uporabim funkcijo join:
    pthread_join(nit1, NULL);
    pthread_join(nit2, NULL);

    // šele sedaj je varno zaključiti main().

    return 0;
}


void* funkcija_niti(void* arg){
    // argument arg je navaden void nasalov, ki ga želim uporabiti za dostop do elementov strukture, na katero kaže.
    // Zato moram definirati nov kazalec na sterukturo in argument castat na ta tip
    argumenti_t* argumenti = (argumenti_t*) arg;

    // Already casted
    if (argumenti->thread_ID == 1){
        printf("Sem nit 1 \n");
    }
    // Cast + access
    else if (((argumenti_t* )arg)->thread_ID == 2){
        printf("Sem nit 2 \n");
    }
    
}
