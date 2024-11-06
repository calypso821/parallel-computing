#include "config.h"

#define N 10        // Number of elements (naj bo sodo)

typedef struct {
    int id;
} args_t;


int *pseznam;
pthread_t threads[NTHREADS];
args_t arguments[NTHREADS];
pthread_barrier_t barrier;

bool finished = true;


void *sort(void *args);

void swap(int *pa, int *pb);
void even_pass();
void odd_pass();
void printSeznam(int*);

int main() {
    // Init barrier
    pthread_barrier_init(&barrier, NULL, NTHREADS);

    for (size_t i = 0; i < NTHREADS; i++) {
        arguments[i].id = i;
        pthread_create(
            &threads[i],            // kazalec na nit, ki jo ustvarjamo
            NULL,
            sort,              // funkcija, ki jo izvede ustvarjena nit
            (void*)(&arguments[i])
        );
    }

    // N elementov
    pseznam = (int*)malloc(N*sizeof(int));
    
    srand(time(NULL));
    // Init sezanm random N elements 
    for (int i = 0; i  < N; i++) {
        // *(pseznam + i)
        pseznam[i] = rand() % 100;
    }

//     printSeznam(pseznam);

//     for (int i = 0; i < N; i++) {
//         //printf("Prehod: %d\n", i);

//         // Sodi prehod
//         even_pass();

// #ifdef __PRINT__
//         printf("SODI prehod: ");
//         printSeznam(pseznam);
// #endif

//         // Lihi prehod
//         odd_pass();

// #ifdef __PRINT__
//         printf("LIHI prehod: ");
//         printSeznam(pseznam);
// #endif

//     }
//     printSeznam(pseznam);

    for (size_t i = 0; i < NTHREADS; i++) {
        pthread_join(threads[i], NULL);
    }


    return 0;
}

void *sort(void *args) {

    for (size_t i = 0; i < N; i++)
    {
        args_t *parguments = (args_t*)args;
        int id = parguments->id;

        // if (konec) break;
        finished = true;
        pthread_barrier_wait(&barrier);

        // Sodi prehod
        for (size_t j = id * 2; j < N; j=j + NTHREADS * 2) 
        {
            swap(pseznam + j, pseznam + j + 1);
        }
        // ========= POCAKAJ (barrier) ==========
        pthread_barrier_wait(&barrier);

        // Lihi prehod
        for (size_t j = (id * 2)+ 1; j < N - 1; j=j + NTHREADS * 2)
        {
            swap(pseznam + j, pseznam + j + 1);
        }
        // ========= POCAKAJ (barrier) ==========
        pthread_barrier_wait(&barrier);

        // if id == 0 and urejen == true --> set konec = true
        // ? WHYY 


        if (finished)
        {
            // Should not break??? Why?? 
            break;
        }
    }

    return NULL;
}

void swap(int *pa, int *pb) {
    if (*pb < *pa) {
        int tmp = *pa;
        *pa = *pb;
        *pb = tmp;

        finished = false;
    }
}

void even_pass() {
    for (int i = 0; i < N; i+=2) {
        swap(&pseznam[i], &pseznam[i+1]);
    }
}

void odd_pass() {
    for (int i = 1; i < N - 1; i+=2) {
        swap(&pseznam[i], &pseznam[i+1]);
    }
}

void printSeznam(int *pseznam) {
    for (int i = 0; i < N; i++) {
        printf("%d ", pseznam[i]);
    }
    printf("\n");
}