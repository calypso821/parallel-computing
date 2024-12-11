#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NTHREADS 4  // Move to header
#define N 10        // Number of elements

unsigned int seznam[N];
void swap(unsigned int *pa, unsigned int *pb);
void izpisSeznam();

int main() {
    
    srand(time(NULL));
    // Init sezanm random N elements 
    for (int i = 0; i  < N; i++) {
            seznam[i] = rand() % 100;
    }
    printf("Seznam: ");
    izpisSeznam();

    for (int i = 0; i < N; i++) {
        // Sodi prehod
        for (int j = 0; j < N; j+=2) {
            swap(&seznam[j], &seznam[j+1]);
        }
        printf("SODI prehod: ");
        izpisSeznam();

        // Lihi prehod
        for (int j = 1; j < N-1; j+=2) {
            swap(&seznam[j], &seznam[j+1]);
        }
        printf("LIHI prehod: ");
        izpisSeznam();
    }
    return 0;
}

void swap(unsigned int *pa, unsigned int *pb) {
    unsigned int tmp;
    if (*pb < *pa) {
        tmp = *pa;
        *pa = *pb;
        *pb = tmp;
    }
}

void izpisSeznam(void) {
    for (int i = 0; i < N; i++) {
        printf("%d ", seznam[i]);
    }
    printf("\n\n");
}