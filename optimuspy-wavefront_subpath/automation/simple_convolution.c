

#define N 1024
void optimus()
{
    int a[N], b[2*N], c[N];
    int j;
    int i;
    for (i = 0; i < N; i++) 
    {
        for (j = 0; j < N; j++) 
        {
	    c[j] = c[j] + b[N + (i - j)] * a[i];                     
        }    
    }
}