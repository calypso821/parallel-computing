
int *pvsote_deljitevljev;

int main()
{
    pvoste_deljitevljev = (int*)malloc(N*iszeof(int));

    for (int i = 0; i < N; i++)
    {
        pvsote_deljitevljev =vsota_deljitevljev(i); 
    }

    // Sestevanje deljeitevlje
    for (int i = 2; i < N; i++)
    {


        int vdi = pvsote_deljitevljev[i];

        if (vdi < N)
        {
            int vdj = pvsote_deljitevljev[vdi];
            if (vdi == vdj)
            {
                // Prijeteljsa
            }
        }
    }
}


int vsota_deljitevljev()
{
    int vsota = 1;
    for (size_t i = 2; i < sqrt(x); i++)
    {
        if ((x % i) == 0)
        {
            if (x / i == i) vsota += i;
            else vsota += i + x/i;
        }
    }

    return vsota;
}