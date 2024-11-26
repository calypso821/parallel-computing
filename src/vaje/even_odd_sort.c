#include "config.h"

// Config.h
// #define NTHREADS
// #define SEED
// #define N

// Build: srun --reservation=fri gcc -O2 -o sl even_odd_sort.c -pthread
// Run: srun --reservation=fri --cpus-per-task=8 sl

typedef struct {
    int id;
} args_t;

struct timespec timeStart, timeEnd;

int *pseznam;
pthread_t threads[NTHREADS];
args_t arguments[NTHREADS];
pthread_barrier_t barrier;

bool sorted = true;
int pass_cnt = 0;

void *sort(void *args);

void swap(int *pa, int *pb);
void printList(int*);

int main() {
    printf("Even-Odd sort\n");
    printf("NTHREADS: %d\n", NTHREADS);
    printf("N elements: %d\n", N);

    // N elementov
    pseznam = (int*)malloc(N*sizeof(int));
    
    srand(SEED);
    // Init sezanm random N elements 
    for (int i = 0; i  < N; i++) {
        // *(pseznam + i)
        pseznam[i] = rand() % 100;
    }

#ifdef __PRINT__
    printf("Unordered list: ");
    printList(pseznam);
    printf("\n");
#endif

    // Init barrier
    pthread_barrier_init(&barrier, NULL, NTHREADS);

    // Start measuring time
    clock_gettime(CLOCK_REALTIME, &timeStart);

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

    clock_gettime(CLOCK_REALTIME, &timeEnd);

#ifdef __PRINT__
    printf("Orderd list: ");
    printList(pseznam);
    printf("\n");
#endif


    double elapsed_time = (timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_nsec - timeStart.tv_nsec) * 1e-9;
    printf("Elapsed time: %.9f seconds\n", elapsed_time);

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

void swap(int *pa, int *pb) {
    if (*pb < *pa) {
        int tmp = *pa;
        *pa = *pb;
        *pb = tmp;

        sorted = false;
    }
}

void printList(int *pseznam) {
    for (int i = 0; i < N; i++) {
        printf("%d ", pseznam[i]);
    }
    printf("\n");
}