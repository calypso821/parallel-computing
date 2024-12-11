#include "config.h"

#define N 15   // Number of elements (naj bo sodo)

// NUmber of elements each thread should process
const size_t chunk = N/NTHREADS;
const bool even_n = (N % 2 == 0);


typedef struct {
    int id;
} args_t;


int *pseznam;
pthread_t threads[NTHREADS];
args_t arguments[NTHREADS];
pthread_barrier_t barrier;

int pass_cnt = 0;
bool sorted = true;

void *sort(void *args);

void swap(int *pa, int *pb, int id);
void process_even_chunk(int id);
void process_odd_chunk(int id);
void printList(int *pseznam);

int main() {
    printf("NTHREADS: %d\n", NTHREADS);
    printf("N elements: %d\n", N);
    printf("CHUNK size: %ld\n", chunk);
    printf("Pass type: %s\n", (chunk % 2 == 0 ? "even" : "odd"));

    // N elementov
    pseznam = (int*)malloc(N*sizeof(int));
    
    srand(time(NULL));
    // Init sezanm random N elements 
    for (int i = 0; i  < N; i++) {
        // *(pseznam + i)
        pseznam[i] = rand() % 100;
    }
    printf("Unordered list: ");
    printList(pseznam);
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

    printf("Orderd list   : ");
    printList(pseznam);

    return 0;
}

void *sort(void *args) {
    args_t *parguments = (args_t*)args;
    int id = parguments->id;

    for (size_t i = 0; i < N; i++)
    {
        if (id == 0) {
            // Pass counter
            pass_cnt++;
        }
        // Do even + odd pass based of N/NTHREADS
        if (chunk % 2 == 0) {
            process_even_chunk(id);
        }
        else {
            process_odd_chunk(id);
        }

        // Wait for all threads to finish both even and odd passes
        // before reading from variable sorted
        pthread_barrier_wait(&barrier);

        if (sorted) {
            //printf("Thread %d exited\n", id);
            break;
        }
        // Wait for all threads to read sorted (process if)
        // before setting sorted back to true
        pthread_barrier_wait(&barrier);

        // Reset sorted to true before next itteration
        // Only thread with id == 0
        if (id == 0) {
            sorted = true;  
        }   

        // Thread 0 has to set sorted = true 
        // before any of threads start new itteration
        // where they can set sorted to false
        pthread_barrier_wait(&barrier);
    }

    if (id == 0) {
        printf("Number of passes: %d\n", pass_cnt);
    }

    return NULL;
}

void swap(int *pa, int *pb, int id) {
    if (*pb < *pa) {
        int tmp = *pa;
        *pa = *pb;
        *pb = tmp;
        
        sorted = false;
    }
}

void process_even_chunk(int id) {
        // Even pass
        for (size_t j = id * chunk; j < (id + 1) * chunk; j+=2) 
        {
            //printf("Thread %d, even pass, index:  %d\n", id, j);
            swap(pseznam + j, pseznam + j + 1, id);
        }

        // Wait all threads to finish even pass before starting odd pass
        pthread_barrier_wait(&barrier);

#ifdef __PRINT__
        if (id == 0) {
            printf("Even pass : ");
            printList(pseznam);
        }
#endif
        // Odd pass
        bool last = (id == NTHREADS - 1);
        for (size_t j = id * chunk + 1; j < (id + 1) * chunk - (last && even_n); j+=2)
        {
            // if even N + last thread: (id + 1) * CHUNK -1
            // if odd N + last thread: (id + 1) * CHUNK
            //printf("Thread %d, odd pass, index:  %d\n", id, j);
            swap(pseznam + j, pseznam + j + 1, id);
        }

#ifdef __PRINT__
        if (id == 0) {
            printf("Odd pass: ");
            printList(pseznam);
            printf("===========================================\n");
        }
#endif
}

void process_odd_chunk(int id) {
        // Even pass
        bool last = (id == NTHREADS - 1);
        // for (size_t j = id * chunk + (id % 2); j < (id + 1) * chunk - (last && !even_n); j+=2) 
        // {
        //     printf("Thread %d, even pass, index:  %d\n", id, j);
        //     swap(pseznam + j, pseznam + j + 1, id);
        // }
        for (size_t j = id * chunk + (id % 2); j < (id + 1) * chunk- (last && !even_n); j+=2) 
        {
            //printf("Thread %d, even pass, index:  %d\n", id, j);
            swap(pseznam + j, pseznam + j + 1, id);
        }

        // Wait all threads to finish even pass before starting odd pass
        pthread_barrier_wait(&barrier);

#ifdef __PRINT__
        if (id == 0) {
            printf("Even pass: ");
            printList(pseznam);
        }
#endif

        // Odd pass
        // for (size_t j = id * chunk + (id % 2 == 0); j < (id + 1) * chunk; j+=2)
        // {
        //     // if even N + last thread: (id + 1) * CHUNK -1
        //     // if odd N + last thread: (id + 1) * CHUNK
        //     printf("Thread %d, odd pass, index:  %d\n", id, j);
        //     swap(pseznam + j, pseznam + j + 1, id);
        // }

        //bool last = (id == NTHREADS - 1);
        for (size_t j = id * chunk + (id % 2 == 0); j < (id + 1) * chunk + (last && even_n); j+=2)
        {
            // if even N + last thread: (id + 1) * CHUNK -1
            // if odd N + last thread: (id + 1) * CHUNK
            //printf("Thread %d, odd pass, index:  %d\n", id, j);
            swap(pseznam + j, pseznam + j + 1, id);
        }

#ifdef __PRINT__
        if (id == 0) {
            printf("Odd pass: ");
            printList(pseznam);
            printf("===========================================\n");
        }
#endif
}

void printList(int *pseznam) {
    for (int i = 0; i < N; i++) {
        printf("%d ", pseznam[i]);
    }
    printf("\n");
}