#include "config.h"

#define N 15      // Number of elements (naj bo sodo)

// NUmber of elements each thread should process
const size_t chunk = N/NTHREADS;
const bool even = (N % 2 == 0);


typedef struct {
    int id;
} args_t;


int *pseznam;
pthread_t threads[NTHREADS];
args_t arguments[NTHREADS];
pthread_barrier_t barrier;

bool sorted = true;


void *sort(void *args);

void swap(int *pa, int *pb, int id);
void process_even_chunk(int id, size_t chunk, int *pseznam, pthread_barrier_t *barrier);
void process_odd_chunk(int id, size_t chunk, int *pseznam, pthread_barrier_t *barrier);
void printSeznam(int*);

int main() {
    printf("NTHREADS: %d\n", NTHREADS);
    printf("N elements: %d\n", N);
    printf("CHUNK size: %d\n", chunk);


    // N elementov
    pseznam = (int*)malloc(N*sizeof(int));
    
    srand(time(NULL));
    // Init sezanm random N elements 
    for (int i = 0; i  < N; i++) {
        // *(pseznam + i)
        pseznam[i] = rand() % 100;
    }

    printSeznam(pseznam);
    printf("\n");

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
        printf("Waiting at start... thread %d\n", id);
        sorted = true;
        pthread_barrier_wait(&barrier);

        if (chunk % 2 == 0) {
            process_even_chunk(id, chunk, pseznam, &barrier);
        }
        else {
            process_odd_chunk(id, chunk, pseznam, &barrier);
        }
        pthread_barrier_wait(&barrier);
        printf("Thread %d sees sorted=0x%x before if\n", id, sorted);
        if (sorted) {
            printf("Thread %d sorted = 0x%x\n", id, sorted);
            printf("Exited thread %d\n", id);
            break;
        }
        // All threads need to check sorted 
        // before nex titteration where sorted is changed back to true
        pthread_barrier_wait(&barrier);




        // if id == 0 and urejen == true --> set konec = true
        // ? WHYY 


        // if (finished)
        // {
        //     // Should not break??? Why?? 
        //     break;
        // }
    }

    return NULL;
}

void swap(int *pa, int *pb, int id) {
    if (*pb < *pa) {
        int tmp = *pa;
        *pa = *pb;
        *pb = tmp;

        sorted = false;
        printf("Thread %d set sorted=false\n", id);
    }
}

void process_even_chunk(int id, size_t chunk, int *pseznam, pthread_barrier_t *barrier) {
        // Even pass
        for (size_t j = id * chunk; j < (id + 1) * chunk; j+=2) 
        {
            //printf("Thread %d, even pass, index:  %d\n", id, j);
            swap(pseznam + j, pseznam + j + 1, id);
        }
        pthread_barrier_wait(barrier);

#ifdef __PRINT__
        if (id == 0) {
            printf("Even pass: ");
            printSeznam(pseznam);
        }
#endif
        // Odd pass
        bool last = (id == NTHREADS - 1);
        for (size_t j = id * chunk + 1; j < (id + 1) * chunk - (last && even); j+=2)
        {
            // if even N + last thread: (id + 1) * CHUNK -1
            // if odd N + last thread: (id + 1) * CHUNK
            //printf("Thread %d, odd pass, index:  %d\n", id, j);
            swap(pseznam + j, pseznam + j + 1, id);
        }
        pthread_barrier_wait(barrier);

#ifdef __PRINT__
        if (id == 0) {
            printf("Odd pass: ");
            printSeznam(pseznam);
            printf("===========================================\n");
        }
#endif
        pthread_barrier_wait(barrier); // WHY HERE???? 
}

void process_odd_chunk(int id, size_t chunk, int *pseznam, pthread_barrier_t *barrier) {
        // Even pass
        bool last = (id == NTHREADS - 1);
        for (size_t j = id * chunk + (id % 2); j < (id + 1) * chunk - (last && !even); j+=2) 
        {
            //printf("Thread %d, even pass, index:  %d\n", id, j);
            swap(pseznam + j, pseznam + j + 1, id);
        }
        pthread_barrier_wait(barrier);

#ifdef __PRINT__
        if (id == 0) {
            printf("Even pass: ");
            printSeznam(pseznam);
        }
#endif

        // Odd pass
        for (size_t j = id * chunk + (id % 2 == 0); j < (id + 1) * chunk; j+=2)
        {
            // if even N + last thread: (id + 1) * CHUNK -1
            // if odd N + last thread: (id + 1) * CHUNK
            //printf("Thread %d, odd pass, index:  %d\n", id, j);
            swap(pseznam + j, pseznam + j + 1, id);
        }
        pthread_barrier_wait(barrier);

#ifdef __PRINT__
        if (id == 0) {
            printf("Odd pass: ");
            printSeznam(pseznam);
            printf("===========================================\n");
        }
#endif
        pthread_barrier_wait(barrier);


        return NULL;
}

void printSeznam(int *pseznam) {
    for (int i = 0; i < N; i++) {
        printf("%d ", pseznam[i]);
    }
    printf("\n");
}