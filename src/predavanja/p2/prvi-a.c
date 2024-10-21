#include <pthread.h>
#include <stdio.h>

// Definicija funkcij (header)
void *koda_niti(void *);


int main() {

    pthread_t nit1, nit2;
    int tid[2]; // thread ID

    int* naslov_na_tid = &tid[0];
    naslov_na_tid = naslov_na_tid + 1; // +1 int
    // void* pointer, brez tipa
    // (void*) - cast kazalca

    // Create thread
    // kazalec na nit, init, koda (funkcija), arguemnts
    // pointer na kodo, ki jo izvaja nit
    // void *koda_za_nit1(void *), nic ne vraca
    tid[0] = 0;
    pthread_create(&nit1, NULL, koda_niti, (void*)(&tid[0]));
    tid[1] = 1;
    pthread_create(&nit2, NULL, koda_niti, (void*)(&tid[1]));

    // vrstni red in cas zacetka izvajanja niti ni dolocen

    // blokira main
    pthread_join(nit1, NULL);
    pthread_join(nit2, NULL);


    return 0;
}

void *koda_niti(void* argumenti) {
    int* mojid = (int*)argumenti; 
    // dereferenciranje pointerja arguemtna
    if (*mojid == 0) {
        printf("Nit1\n");
    }
    else if (*mojid == 1) {
        printf("Nit2\n");
    }
    
}
