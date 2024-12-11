

typedef struct {
    int tid; // i
    int tstart; // i * N / NTHREADS
    int tend; // (i+1) * N / NTHREADS
} args_t;


void *sort(void *args) {

    for (size_t i = 0; i < N; i++)
    {
        args_t *parguments = (args_t*)args;
        int id = parguments->tid;
        int start = parguments->tstart;
        int end = parguments->tend;
        // end - 1  




        // if (konec) break;
        finished = true;
        pthread_barrier_wait(&barrier);

        // Sodi prehod
        for (size_t j  = start + (start % 2); j < end; j+=2) 
        {
            // Check if right elemnt is still in array
            if (i + 1 < N)
            {

            }
            swap(pseznam + j, pseznam + j + 1);
        }
        // ========= POCAKAJ (barrier) ==========
        pthread_barrier_wait(&barrier);

        // Lihi prehod
        for (size_t j  = start + (start % 2 == 0    ); j < end; j+=2) 
        {
            // Check if right elemnt is still in array
            if (i + 1 < N)
            {
                
            }
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