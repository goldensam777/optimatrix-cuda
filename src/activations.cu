/*
 * activations.cu — ReLU, Sigmoid, SiLU, Softplus sur GPU
 *
 * Chaque activation = 1 kernel trivial.
 * 1 thread par element, blockSize = 256.
 *
 * Pattern commun :
 *   - Calculer l'index global tid
 *   - Si tid < n, appliquer la fonction
 *   - Lancer avec (n + 255) / 256 blocs
 */

#include "optimatrix.h"
#include <math.h>

/* ── Kernels ─────────────────────────────────────────────────────── */

__global__ void relu_kernel(const float *in, float *out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = fmaxf(in[tid], 0.0f);
}

__global__ void sigmoid_kernel(const float *in, float *out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = 1.0f / (1.0f + expf(-in[tid]));
}

__global__ void silu_kernel(const float *in, float *out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float x = in[tid];
        out[tid] = x / (1.0f + expf(-x));  /* x * sigmoid(x) */
    }
}

__global__ void softplus_kernel(const float *in, float *out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = log1pf(expf(in[tid]));  /* ln(1 + e^x) */
}

/* ── API ─────────────────────────────────────────────────────────── */

static inline int grid(int n) { return (n + 255) / 256; }

void om_relu(const float *d_in, float *d_out, int n)
{
    relu_kernel<<<grid(n), 256>>>(d_in, d_out, n);
}

void om_sigmoid(const float *d_in, float *d_out, int n)
{
    sigmoid_kernel<<<grid(n), 256>>>(d_in, d_out, n);
}

void om_silu(const float *d_in, float *d_out, int n)
{
    silu_kernel<<<grid(n), 256>>>(d_in, d_out, n);
}

void om_softplus(const float *d_in, float *d_out, int n)
{
    softplus_kernel<<<grid(n), 256>>>(d_in, d_out, n);
}
