#define N 300
float matmul(const float *a, int m, int n, const float *b, int p, int q, float *r)
{
    if (n != p)
        return 0;
    float s = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < q; j++)
        {
            for (int k = 0; k < p; k++)
                r[i * m + j] += a[i * m + k] * b[k * p + j];
            s += r[i * m + j];
        }
    }
    return s;
}
void optimus()
{
    float a[N * N], b[N * N], c[N * N];
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i * N + j] = 1;
            b[i * N + j] = 2;
            c[i * N + j] = 0;
        }
    }
    volatile float r = matmul(a, N, N, b, N, N, c);
}