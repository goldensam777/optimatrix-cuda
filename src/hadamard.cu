/*
 * hadamard.cu — Produit element par element sur GPU
 *
 * out[i] = a[i] * b[i]
 *
 * Memory-bound : 1 flop pour 12 octets lus + 4 ecrits.
 * Le kernel est trivial, la perf depend de la bande passante memoire.
 */

#include "optimatrix.h"

__global__ void hadamard_kernel(const float *a, const float *b,
                                float *out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = a[tid] * b[tid];
}

void om_hadamard(const float *d_a, const float *d_b, float *d_out, int n)
{
    int blocks = (n + 255) / 256;
    hadamard_kernel<<<blocks, 256>>>(d_a, d_b, d_out, n);
}
