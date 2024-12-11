#include "config.h"

#define NTHREADS 4  // Move to header
#define N 30        // Number of elements (naj bo sodo)

int *pseznam;

void swap(int *pa, int *pb);
void even_pass();
void odd_pass();
void printSeznam(int*);

int main() {
    // N elementov
    pseznam = (int*)malloc(N*sizeof(int));
    
    srand(time(NULL));
    // Init sezanm random N elements 
    for (int i = 0; i  < N; i++) {
        // *(pseznam + i)
        pseznam[i] = rand() % 100;
    }

    printSeznam(pseznam);

    for (int i = 0; i < N; i++) {
        //printf("Prehod: %d\n", i);

        // Sodi prehod
        even_pass();

#ifdef __PRINT__
        printf("SODI prehod: ");
        printSeznam(pseznam);
#endif

        // Lihi prehod
        odd_pass();

#ifdef __PRINT__
        printf("LIHI prehod: ");
        printSeznam(pseznam);
#endif

    }
    printSeznam(pseznam);
    return 0;
}

void swap(int *pa, int *pb) {
    if (*pb < *pa) {
        int tmp = *pa;
        *pa = *pb;
        *pb = tmp;
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