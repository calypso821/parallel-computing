int main()
{
        pvecA =  (double*)malloc(NELEMENTS*sizeof(double));
    pvecB =  (double*)malloc(NELEMENTS*sizeof(double));
    pvecC =  (double*)malloc(NELEMENTS*sizeof(double));

    // init vektorjev A in B:
    #pragma omp for
    for (int i = 0; i < NELEMENTS; i++)
    {
        *(pvecA + i) = 1.0;
        pvecB[i] = 2.0;
    }

    #pragma omp for
    // Razdeli for zanko na NTHREADS
    for (int i = 0; i < N; i++)
    {
        *(pvecC + i) = *(pvecA + i) * *(pvecB + i);
    }

    #pragma omp for
    for (int i = 0; i < N; i++)
    {   
        #pragma omp critical 
        {
            dot_product += *(pvecC + i);
        }
    }


    int max_threads = omp_get_max_threads();
    printf("Max supproted threads: %d", max_threads);
    // Najvecje stevilo moznih niti
    if (max_threads < NTHREADS)
    {
        printf("Number of threads %d not supported", NTHREADS);
        return 1;
    }

    omp_set_num_threads(NTHREADS);
    #pragma omp prallel
    // Create threads
    {
        printf("Sem nit: %d od %d niti", omp_get_thread_num(), NHREADS);



    }
    // Join threads

    return 0;
}