#include <pthread.h>
#include <stdio.h>

// Definicija funkcij (header)
void *koda_za_nit1(void *);
void *koda_za_nit2(void *);


int main() {

    pthread_t nit1, nit2;

    // Create thread
    // kazalec na nit, arguemnti, koda (funkcija), init
    // pointer na kodo, ki jo izvaja nit
    // void *koda_za_nit1(void *), nic ne vraca, brez argumentov
    pthread_create(&nit1, NULL, koda_za_nit1, NULL);
    pthread_create(&nit2, NULL, koda_za_nit2, NULL);

    // vrstni red in cas zacetka izvajanja niti ni dolocen

    // blokira main
    pthread_join(nit1, NULL);
    pthread_join(nit2, NULL);


    return 0;
}

void *koda_za_nit1(void *) {
    printf("Nit1\n");
}

void *koda_za_nit2(void *) {
    printf("Nit2\n");
}